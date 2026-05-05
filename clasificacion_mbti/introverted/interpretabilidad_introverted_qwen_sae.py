import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from interpretabilidad_sae_qwen_common import SaeInterpretabilityConfig, run_posthoc_analysis

def main() -> None:
    config = SaeInterpretabilityConfig(
        task_name="introverted_qwen",
        dataset_kind="mbti",
        splits_dir="data/splits_introverted",
        output_dir="modelos/introverted_qwen_sae_interpretabilidad",
        class_names=("0", "1"),
        mbti_trait="introverted",
        sae_results_path="modelos/introverted_qwen_sae_activaciones/resultados_resumen.json",
    )
    run_posthoc_analysis(config)
if __name__ == "__main__":
    main()
