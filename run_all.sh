#!/usr/bin/env bash
# Pipeline completo: entrenar SAE + 12 clasificadores.
#
# Uso:
#   nohup bash run_all.sh > /dev/null 2>&1 &

set -eu

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

ROOT_DIR="/home/aitziber.l/TFM"
HDD_DIR="/hdd/aitziber.l"
LOG_DIR="$ROOT_DIR/logs"
MASTER_LOG="$LOG_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"
LATEST_LINK="$LOG_DIR/run_all_latest.log"

mkdir -p "$LOG_DIR" "$HDD_DIR"

cd "$ROOT_DIR"

ln -sfn "$MASTER_LOG" "$LATEST_LINK"
printf '%s\n' "$$" > "$LOG_DIR/run_all.pid"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$MASTER_LOG"; }

# PASO 0: BORRAR SPLITS ANTIGUOS
log "PASO 0: Borrar splits antiguos para forzar regeneracion"

for SPLIT_DIR in data/splits_*; do
    if [[ -d "$SPLIT_DIR" ]]; then
        rm -rf "$SPLIT_DIR"
        log "  Eliminado: $SPLIT_DIR"
    fi
done
log "Splits borrados."

# PASO 1: ENTRENAR SAE
log "PASO 1: Entrenamiento SAE (sae_gpt.py)"

SAE_HDD_DIR="/hdd/aitziber.l/TFM/sae-ckpts/sae-gpt2-comments"
SAE_LOCAL_DIR="$ROOT_DIR/sae-ckpts/sae-gpt2-comments"

rm -rf "$SAE_HDD_DIR" "$SAE_LOCAL_DIR"
mkdir -p "$SAE_LOCAL_DIR"

SAE_START=$SECONDS

export SAE_DEVICE="cuda:0"
export SAE_CHECKPOINT_DIR="$SAE_HDD_DIR"
export SAE_SAVE_REPR_EVERY_N_STEPS=0

log "  Device: $SAE_DEVICE"
log "  Checkpoint dir: $SAE_CHECKPOINT_DIR"

python3 sae_gpt.py >> "$MASTER_LOG" 2>&1
SAE_EXIT=$?
SAE_ELAPSED=$(( SECONDS - SAE_START ))
SAE_H=$(( SAE_ELAPSED / 3600 ))
SAE_M=$(( (SAE_ELAPSED % 3600) / 60 ))

if [[ $SAE_EXIT -ne 0 ]]; then
    log "ERROR: sae_gpt.py fallo con exit code $SAE_EXIT (${SAE_H}h ${SAE_M}m)"
    exit 1
fi
log "SAE entrenada OK (${SAE_H}h ${SAE_M}m)"

if [[ -f "$SAE_HDD_DIR/sae.safetensors" ]]; then
    cp "$SAE_HDD_DIR/sae.safetensors" "$SAE_LOCAL_DIR/"
    cp "$SAE_HDD_DIR/cfg.json" "$SAE_LOCAL_DIR/"
    log "  Checkpoint copiado a $SAE_LOCAL_DIR"
    rm -f "$SAE_HDD_DIR"/repr_step*.pt 2>/dev/null
    rm -rf "$SAE_HDD_DIR"
    log "  Checkpoint HDD limpiado"
else
    log "ERROR: No se encontro sae.safetensors en $SAE_HDD_DIR"
    exit 1
fi

# PASO 2: CLASIFICADORES (12 scripts secuenciales)
log "PASO 2: 12 clasificadores"

run_script() {
    local SCRIPT_PATH="$1"
    log "--- Iniciando $SCRIPT_PATH ---"
    local START_SEC=$SECONDS
    local CLEANUP_DIRS=()

    if [[ "$SCRIPT_PATH" == *"_gpt_activaciones.py" ]]; then
        local TRAIT=""
        if [[ "$SCRIPT_PATH" == *"clasificacion_mbti"* ]]; then
            TRAIT=$(echo "$SCRIPT_PATH" | sed 's|.*/clasificacion_mbti/\([^/]*\)/.*|\1|')
        elif [[ "$SCRIPT_PATH" == *"clasificacion_genero"* ]]; then
            TRAIT="genero"
        elif [[ "$SCRIPT_PATH" == *"clasificacion_edad"* ]]; then
            TRAIT="edad"
        fi
        if [[ -n "$TRAIT" ]]; then
            local ACT_HDD="$HDD_DIR/activaciones_gpt2_${TRAIT}"
            local ACT_LOCAL="$ROOT_DIR/data/activaciones_gpt2_${TRAIT}"
            rm -rf "$ACT_LOCAL" "$ACT_HDD"
            mkdir -p "$ACT_HDD"
            ln -sfn "$ACT_HDD" "$ACT_LOCAL"
            log "  Activaciones GPT redirigidas a $ACT_HDD"
            CLEANUP_DIRS+=("$ACT_HDD" "$ACT_LOCAL")
        fi
    elif [[ "$SCRIPT_PATH" == *"_sae_activaciones.py" ]]; then
        if [[ "$SCRIPT_PATH" == *"clasificacion_genero"* ]]; then
            CLEANUP_DIRS+=("/hdd/aitziber.l/activaciones_sae_gpt2_genero")
        elif [[ "$SCRIPT_PATH" == *"clasificacion_edad"* ]]; then
            CLEANUP_DIRS+=("/hdd/aitziber.l/activaciones_sae_gpt2_edad")
        fi
        for d in "${CLEANUP_DIRS[@]}"; do
            rm -rf "$d"
            log "  Cache SAE previo eliminado: $d"
        done
    fi

    python3 "$SCRIPT_PATH" >> "$MASTER_LOG" 2>&1
    local EXIT_CODE=$?
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
    "clasificacion_genero/gpt_activaciones/clasificador_genero_sae_activaciones.py"
    "clasificacion_genero/gpt_activaciones/clasificador_genero_gpt_activaciones.py"
    "clasificacion_edad/gpt_activaciones/clasificador_edad_sae_activaciones.py"
    "clasificacion_edad/gpt_activaciones/clasificador_edad_gpt_activaciones.py"
    "clasificacion_mbti/introverted/clasificador_introverted_sae_activaciones.py"
    "clasificacion_mbti/introverted/clasificador_introverted_gpt_activaciones.py"
    "clasificacion_mbti/intuitive/clasificador_intuitive_sae_activaciones.py"
    "clasificacion_mbti/intuitive/clasificador_intuitive_gpt_activaciones.py"
    "clasificacion_mbti/thinking/clasificador_thinking_sae_activaciones.py"
    "clasificacion_mbti/thinking/clasificador_thinking_gpt_activaciones.py"
    "clasificacion_mbti/perceiving/clasificador_perceiving_sae_activaciones.py"
    "clasificacion_mbti/perceiving/clasificador_perceiving_gpt_activaciones.py"
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

log "COMPLETADO: $OK/$TOTAL exitosos, $FAIL fallidos (${GLOBAL_H}h ${GLOBAL_M}m clasificadores)"
