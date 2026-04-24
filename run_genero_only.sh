#!/usr/bin/env bash
# Pipeline secuencial SOLO para tarea genero.
# Ejecuta en este orden:
#   1) clasificador_genero_gpt_activaciones.py
#   2) clasificador_genero_sae_activaciones.py
#   3) interpretabilidad_genero_sae.py
#
# Uso:
#   nohup bash run_genero_only.sh > /dev/null 2>&1 &

set -uo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled

ROOT_DIR="/home/aitziber.l/TFM"
HDD_DIR="/hdd/aitziber.l"
LOG_DIR="$ROOT_DIR/logs"
MASTER_LOG="$LOG_DIR/run_genero_only_$(date +%Y%m%d_%H%M%S).log"
LATEST_LINK="$LOG_DIR/run_genero_only_latest.log"

mkdir -p "$LOG_DIR" "$HDD_DIR"

cd "$ROOT_DIR"

ln -sfn "$MASTER_LOG" "$LATEST_LINK"
printf '%s\n' "$$" > "$LOG_DIR/run_genero_only.pid"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$MASTER_LOG"; }

log "PASO 0: Borrar splits antiguos de genero para forzar regeneracion"
for SPLIT_DIR in data/splits_genero*; do
    if [[ -d "$SPLIT_DIR" ]]; then
        rm -rf "$SPLIT_DIR"
        log "  Eliminado: $SPLIT_DIR"
    fi
done

log "PASO 1: Limpiar caches SAE/GPT antiguas de genero"
rm -rf "$HDD_DIR/activaciones_sae_gpt2_genero"
rm -rf "$HDD_DIR/activaciones_gpt2_genero"
rm -rf "$ROOT_DIR/data/activaciones_gpt2_genero"
log "  Caches limpiadas"

run_script() {
    local SCRIPT_PATH="$1"
    log "--- Iniciando $SCRIPT_PATH ---"
    local START_SEC=$SECONDS
    local CLEANUP_DIRS=()
    local TRAIT="genero"

    if [[ "$SCRIPT_PATH" == *"_gpt_activaciones.py" ]]; then
        local ACT_HDD="$HDD_DIR/activaciones_gpt2_${TRAIT}"
        local ACT_LOCAL="$ROOT_DIR/data/activaciones_gpt2_${TRAIT}"
        rm -rf "$ACT_LOCAL" "$ACT_HDD"
        mkdir -p "$ACT_HDD"
        ln -sfn "$ACT_HDD" "$ACT_LOCAL"
        log "  Activaciones GPT redirigidas a $ACT_HDD"
        CLEANUP_DIRS+=("$ACT_HDD" "$ACT_LOCAL")
    elif [[ "$SCRIPT_PATH" == *"_sae_activaciones.py" ]]; then
        local SAE_CACHE_DIR="$HDD_DIR/activaciones_sae_gpt2_${TRAIT}"
        rm -rf "$SAE_CACHE_DIR"
        log "  Ejecucion SAE en streaming; cache antigua eliminada: $SAE_CACHE_DIR"
    elif [[ "$SCRIPT_PATH" == *"interpretabilidad_"*"_sae.py" ]]; then
        local SAE_CACHE_DIR="$HDD_DIR/activaciones_sae_gpt2_${TRAIT}"
        rm -rf "$SAE_CACHE_DIR"
        log "  Interpretabilidad SAE en streaming; cache antigua eliminada: $SAE_CACHE_DIR"
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
    "clasificacion_genero/gpt_activaciones/clasificador_genero_gpt_activaciones.py"
    "clasificacion_genero/gpt_activaciones/clasificador_genero_sae_activaciones.py"
    "clasificacion_genero/gpt_activaciones/interpretabilidad_genero_sae.py"
)

TOTAL=${#SCRIPTS[@]}
OK=0
FAIL=0
GLOBAL_START=$SECONDS

log "PASO 2: Pipeline GENERO secuencial ($TOTAL scripts)"
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

log "COMPLETADO: $OK/$TOTAL exitosos, $FAIL fallidos (${GLOBAL_H}h ${GLOBAL_M}m pipeline genero)"
