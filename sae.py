
try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    is_notebook = True
except:
    is_notebook = False

import math
import dotenv

dotenv.load_dotenv()

# %%

import os

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tiny_sae import Sae, SaeConfig, train_sae, TrainConfig
from tqdm import tqdm

# %%

MODEL = "openai-community/gpt2"
dataset_name = "togethercomputer/RedPajama-Data-1T-Sample"
dataset = load_dataset(
    dataset_name,
    split="train",
    trust_remote_code=True,
)
dataset = dataset.shuffle(seed=43)
tokenizer = AutoTokenizer.from_pretrained(MODEL)


context_len = 1024
device = "cuda:0"
gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
)

# %%


def _tokenize_fn(x: dict[str, list]):
    output = tokenizer(
        x["text"], max_length=context_len, return_attention_mask=False, truncation=True
    )

    return output


data = dataset.map(
    _tokenize_fn, batched=True, batch_size=32, num_proc=16, load_from_cache_file=True
)

# %%

# Filter out sequences shorter than context_len
data = data.filter(
    lambda x: len(x["input_ids"]) == context_len,
    load_from_cache_file=True,
    num_proc=16,
)

# %%

sae_cfg = SaeConfig(
    d_in=768,
    num_latents=2**14,
    k=64,
    hookpoint="transformer.h.8",
)

sae = Sae(sae_cfg, device=device)

# %%
cfg = TrainConfig(
    wandb_project="tiny-sae",
    wandb_name="test",
    save_every_n_tokens=10_000_000,
    optimize_every_n_tokens=8192,
    model_batch_size=16,
    mask_first_n_tokens=1,
)
train_sae(sae=sae, model=gpt, token_iterator=data, train_cfg=cfg, use_wandb=True)
# %%

sae = Sae.load_from_disk("sae-ckpts/test")