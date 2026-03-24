from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path
from typing import Iterable
from torch.optim import Adam
import torch
from safetensors.torch import load_model, save_model
from torch import Tensor, nn
from transformers import PreTrainedModel
import wandb
import einops


@dataclass
class SaeConfig:
    d_in: int
    num_latents: int
    hookpoint: str
    k: int
    transcode: bool = False


class Sae(nn.Module):
    def __init__(
        self,
        cfg: SaeConfig,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.cfg = cfg

        self.encoder = nn.Linear(
            self.cfg.d_in, self.cfg.num_latents, device=device, dtype=dtype
        )
        self.encoder.bias.data.zero_()

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=dtype, device=device)
        )

    @staticmethod
    def load_from_disk(path: Path | str, device: str | torch.device = "cpu") -> "Sae":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            cfg = SaeConfig(**cfg_dict)

        sae = Sae(cfg, device=device)
        load_model(
            model=sae, filename=str(path / "sae.safetensors"), device=str(device)
        )
        return sae

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(asdict(self.cfg), f)

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def encode(self, x: Tensor) -> Tensor:
        forward = self.encoder(x - self.b_dec)
        top_acts, top_indices = forward.topk(self.cfg.k, dim=-1)
        return top_acts, top_indices

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        batch_size = top_indices.shape[0]
        top_acts = top_acts.flatten(end_dim=1)
        top_indices = top_indices.flatten(end_dim=1)
        res = nn.functional.embedding_bag(
            top_indices, self.W_dec, per_sample_weights=top_acts, mode="sum"
        )
        res = einops.rearrange(res, "(b n) d -> b n d", b=batch_size)
        return res + self.b_dec

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(*self.encode(x))

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + 1e-5


@dataclass
class TrainConfig:
    wandb_project: str
    wandb_name: str
    mask_first_n_tokens: int
    model_batch_size: int = 8
    save_every_n_tokens: int = 10_000_000
    optimize_every_n_tokens: int = 8192
    save_repr_every_n_steps: int = 1000
    checkpoint_dir: str = ""  # si vacío, usa sae-ckpts/{wandb_name}


def train_sae(
    sae: Sae,
    model: PreTrainedModel,
    token_iterator: Iterable[Tensor],
    train_cfg: TrainConfig,
    use_wandb: bool = True,
):

    if use_wandb:
        wandb.init(
            name=train_cfg.wandb_name,
            project=train_cfg.wandb_project,
            config={"sae_config": asdict(sae.cfg), "train_config": asdict(train_cfg)},
            save_code=True,
        )

    hookpoint = model.get_submodule(sae.cfg.hookpoint)

    # Auto-select LR using 1 / sqrt(d) scaling law from Fig 3 of the paper
    lr = 2e-4 / (sae.cfg.num_latents / (2**14)) ** 0.5
    optimizer = Adam(sae.parameters(), lr=lr)

    global_inputs = None
    global_outputs = None

    def hook(module: nn.Module, inputs, outputs):
        nonlocal global_inputs, global_outputs
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        global_inputs = inputs
        global_outputs = outputs

        raise StopIteration("Stop here")

    handle = hookpoint.register_forward_hook(hook)

    # Directorio donde se guardarán checkpoints y representaciones
    if train_cfg.checkpoint_dir:
        save_dir = Path(train_cfg.checkpoint_dir)
    else:
        save_dir = Path("sae-ckpts") / train_cfg.wandb_name
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokens_seen_since_last_step = 0
        tokens_seen_since_last_save = 0
        total_examples = len(token_iterator) if hasattr(token_iterator, '__len__') else None
        last_log_time = time.time()
        last_loss = None
        batch = []
        for step, tokens in enumerate(token_iterator):

            if len(batch) < train_cfg.model_batch_size:
                batch.append(torch.tensor(tokens["input_ids"]))
                continue

            batch = torch.stack(batch).to(model.device)
            tokens_seen_since_last_step += batch.numel()
            tokens_seen_since_last_save += batch.numel()

            with torch.no_grad():
                try:
                    model(batch)
                except StopIteration:
                    pass

            sae_input = global_inputs.to(sae.dtype).to(sae.device)[
                :, train_cfg.mask_first_n_tokens :
            ]
            sae_output = global_outputs.to(sae.dtype).to(sae.device)[
                :, train_cfg.mask_first_n_tokens :
            ]

            if not sae.cfg.transcode:
                sae_input = sae_output

            predicted = sae(sae_input)
            error = predicted - sae_output
            loss = (error**2).sum()

            # Guarda representaciones cada cierto número de pasos
            # - gpt_repr: activaciones originales del modelo (media en tokens)
            # - sae_repr: salida reconstruida por la SAE (media en tokens)
            if (
                train_cfg.save_repr_every_n_steps > 0
                and step % train_cfg.save_repr_every_n_steps == 0
            ):
                with torch.no_grad():
                    gpt_repr = sae_input.mean(dim=1).detach().cpu()
                    sae_repr = predicted.mean(dim=1).detach().cpu()

                torch.save(
                    {
                        "step": step,
                        "gpt_repr": gpt_repr,
                        "sae_repr": sae_repr,
                    },
                    save_dir / f"repr_step{step}.pt",
                )

            loss /= ((sae_output - sae_output.mean(dim=1, keepdim=True)) ** 2).sum()
            loss.backward()

            if tokens_seen_since_last_step >= train_cfg.optimize_every_n_tokens:
                optimizer.step()
                optimizer.zero_grad()
                sae.set_decoder_norm_to_unit_norm()
                tokens_seen_since_last_step = 0
                if use_wandb:
                    wandb.log({"fvu": loss.item()}, step=step)

            if tokens_seen_since_last_save >= train_cfg.save_every_n_tokens:
                # Borrar checkpoint anterior antes de guardar el nuevo
                ckpt_safetensors = save_dir / "sae.safetensors"
                ckpt_cfg = save_dir / "cfg.json"
                if ckpt_safetensors.exists():
                    ckpt_safetensors.unlink()
                if ckpt_cfg.exists():
                    ckpt_cfg.unlink()
                sae.save_to_disk(save_dir)
                tokens_seen_since_last_save = 0

            last_loss = loss.item()
            now = time.time()
            if now - last_log_time >= 3600:  # cada hora
                elapsed_h = (now - last_log_time) / 3600
                pct = f"{step / total_examples * 100:.1f}%" if total_examples else "?"
                print(f"[SAE] step={step} | progreso={pct} | loss={last_loss:.4f} | +{elapsed_h:.1f}h")
                last_log_time = now
            batch = []
    finally:
        handle.remove()
        if use_wandb:
            wandb.finish()