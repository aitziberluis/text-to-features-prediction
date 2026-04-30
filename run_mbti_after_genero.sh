#!/usr/bin/env bash
# Lanza todos los scripts de MBTI (4 rasgos x 3 scripts: SAE + interpretabilidad + GPT)
# en secuencia, despues de esperar a que termine el job de interpretabilidad de genero.
#
# Uso:
#   nohup bash run_mbti_after_genero.sh > /dev/null 2>&1 &

set -uo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled

ROOT_DIR="/home/aitziber.l/TFM"
HDD_DIR="/hdd/aitziber.l"
LOG_DIR="$ROOT_DIR/logs"
MASTER_LOG="$LOG_DIR/run_mbti_after_genero_$(date +%Y%m%d_%H%M%S).log"
LATEST_LINK="$LOG_DIR/run_mbti_after_genero_latest.log"

mkdir -p "$LOG_DIR" "$HDD_DIR"
cd "$ROOT_DIR"

ln -sfn "$MASTER_LOG" "$LATEST_LINK"
printf '%s\n' "$$" > "$LOG_DIR/run_mbti_after_genero.pid"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$MASTER_LOG"; }

# --- Esperar a que termine el job de genero en marcha ---
WAIT_PATTERN="interpretabilidad_genero_sae|run_genero_interpretabilidad_only"
log "Esperando a que termine el proceso de interpretabilidad de genero (patron: $WAIT_PATTERN)"
while pgrep -f "$WAIT_PATTERN" > /dev/null; do
    sleep 60
done
log "Proceso de genero finalizado. Iniciando pipeline MBTI."

run_script() {
    local SCRIPT_PATH="$1"
    log "--- Iniciando $SCRIPT_PATH ---"
    local START_SEC=$SECONDS
    local TRAIT=""
    local CLEANUP_DIRS=()

    if [[ "$SCRIPT_PATH" == *"clasificacion_mbti"* ]]; then
        TRAIT=$(echo "$SCRIPT_PATH" | sed 's|.*/clasificacion_mbti/\([^/]*\)/.*|\1|')
    fi

    if [[ "$SCRIPT_PATH" == *"_gpt_activaciones.py" ]]; then
        if [[ -n "$TRAIT" ]]; then
            local ACT_HDD="$HDD_DIR/activaciones_gpt2_${TRAIT}"
            local ACT_LOCAL="$ROOT_DIR/data/activaciones_gpt2_${TRAIT}"
            rm -rf "$ACT_LOCAL" "$ACT_HDD"
            mkdir -p "$ACT_HDD"
            ln -sfn "$ACT_HDD" "$ACT_LOCAL"
            log "  Activaciones GPT redirigidas a $ACT_HDD"
            CLEANUP_DIRS+=("$ACT_HDD" "$ACT_LOCAL")
        fi
    elif [[ "$SCRIPT_PATH" == *"_sae_activaciones.py" || "$SCRIPT_PATH" == *"interpretabilidad_"*"_sae.py" ]]; then
        if [[ -n "$TRAIT" ]]; then
            local SAE_CACHE_DIR="$HDD_DIR/activaciones_sae_gpt2_${TRAIT}"
            rm -rf "$SAE_CACHE_DIR"
            log "  Cache SAE eliminada: $SAE_CACHE_DIR"
        fi
    fi

    stdbuf -oL -eL python3 -u "$SCRIPT_PATH" 2>&1 | tee -a "$MASTER_LOG"
    local EXIT_CODE=${PIPESTATUS[0]}
    local ELAPSED=$(( SECONDS - START_SEC ))
    local HOURS=$(( ELAPSED / 3600 ))
    local MINS=$(( (ELAPSED % 3600) / 60 ))

    for d in "${CLEANUP_DIRS[@]}"; do
        if [[ -e "$d" ]]; then
            rm -rf "$d"
            log "  Activaciones temporales eliminadas: $d"
        fi
    done

    if [[ $EXIT_CODE -ne 0 ]]; then
        log "ERROR: $SCRIPT_PATH fallo con exit code $EXIT_CODE (${HOURS}h ${MINS}m)"
        return $EXIT_CODE
    fi
    log "OK: $SCRIPT_PATH completado (${HOURS}h ${MINS}m)"
    return 0
}

SCRIPTS=(
    "clasificacion_mbti/introverted/clasificador_introverted_gpt_activaciones.py"
    "clasificacion_mbti/introverted/clasificador_introverted_sae_activaciones.py"
    "clasificacion_mbti/introverted/interpretabilidad_introverted_sae.py"
    "clasificacion_mbti/intuitive/clasificador_intuitive_gpt_activaciones.py"
    "clasificacion_mbti/intuitive/clasificador_intuitive_sae_activaciones.py"
    "clasificacion_mbti/intuitive/interpretabilidad_intuitive_sae.py"
    "clasificacion_mbti/thinking/clasificador_thinking_gpt_activaciones.py"
    "clasificacion_mbti/thinking/clasificador_thinking_sae_activaciones.py"
    "clasificacion_mbti/thinking/interpretabilidad_thinking_sae.py"
    "clasificacion_mbti/perceiving/clasificador_perceiving_gpt_activaciones.py"
    "clasificacion_mbti/perceiving/clasificador_perceiving_sae_activaciones.py"
    "clasificacion_mbti/perceiving/interpretabilidad_perceiving_sae.py"
)

TOTAL=${#SCRIPTS[@]}
OK=0
FAIL=0
GLOBAL_START=$SECONDS

for i in "${!SCRIPTS[@]}"; do
    IDX=$(( i + 1 ))
    log "[$IDX/$TOTAL] ${SCRIPTS[$i]}"
    if run_script "${SCRIPTS[$i]}"; then
        OK=$(( OK + 1 ))
    else
        FAIL=$(( FAIL + 1 ))
        log "CONTINUANDO con el siguiente script pese al error..."
    fi
done

GLOBAL_ELAPSED=$(( SECONDS - GLOBAL_START ))
GLOBAL_H=$(( GLOBAL_ELAPSED / 3600 ))
GLOBAL_M=$(( (GLOBAL_ELAPSED % 3600) / 60 ))

log "COMPLETADO MBTI: $OK/$TOTAL exitosos, $FAIL fallidos (${GLOBAL_H}h ${GLOBAL_M}m)"
