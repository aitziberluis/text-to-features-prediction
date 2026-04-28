import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from interpretabilidad_sae_common import SaeInterpretabilityConfig, run_posthoc_analysis

def main() -> None:
    config = SaeInterpretabilityConfig(
        task_name="genero",
        dataset_kind="gender",
        cache_dir="/hdd/aitziber.l/activaciones_sae_gpt2_genero",
        splits_dir="data/splits_genero_70_15_15",
        output_dir="modelos/genero_sae_interpretabilidad",
        class_names=("female", "male"),
    )
    run_posthoc_analysis(config)

if __name__ == "__main__":
    main()