import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from interpretabilidad_sae_common import SaeInterpretabilityConfig, run_posthoc_analysis


def main() -> None:
    config = SaeInterpretabilityConfig(
        task_name="introverted",
        dataset_kind="mbti",
        cache_dir="/hdd/aitziber.l/activaciones_sae_gpt2_introverted",
        splits_dir="data/splits_introverted",
        output_dir="modelos/introverted_sae_interpretabilidad",
        class_names=("0", "1"),
        mbti_trait="introverted",
    )
    run_posthoc_analysis(config)


if __name__ == "__main__":
    main()