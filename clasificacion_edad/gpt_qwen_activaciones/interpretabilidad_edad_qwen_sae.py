import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from interpretabilidad_sae_qwen_common import SaeInterpretabilityConfig, run_posthoc_analysis

def main() -> None:
    config = SaeInterpretabilityConfig(
        task_name="edad_qwen",
        dataset_kind="age",
        cache_dir="/hdd/aitziber.l/activaciones_qwen_sae_edad",
        splits_dir="data/splits_edad",
        output_dir="modelos/edad_qwen_sae_interpretabilidad",
        class_names=("14_19", "20_29", "30_39", "40_plus"),
        age_groups=("14_19", "20_29", "30_39", "40_plus"),
        sae_results_path="modelos/edad_qwen_sae_activaciones/resultados_resumen.json",
    )
    run_posthoc_analysis(config)

if __name__ == "__main__":
    main()
