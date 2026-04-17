#!/usr/bin/env bash
# ===========================================================================
# Relanza TODOS los clasificadores (12 scripts) tras corregir SMOTE/ADASYN.
#
# Espera a que el proceso indicado (PID_WAIT) termine, y luego ejecuta
# secuencialmente cada clasificador.
#
# Los scripts GPT redirigen activaciones a /hdd para no gastar /home.
#
# Uso:
#   nohup bash run_smote_fix_all.sh > /dev/null 2>&1 &
# ===========================================================================

set -eu

PID_WAIT=430641

ROOT_DIR="/home/aitziber.l/TFM"
HDD_DIR="/hdd/aitziber.l"
LOG_DIR="$ROOT_DIR/logs"
MASTER_LOG="$LOG_DIR/smote_fix_all_$(date +%Y%m%d_%H%M%S).log"
LATEST_LINK="$LOG_DIR/smote_fix_all_latest.log"

mkdir -p "$LOG_DIR" "$HDD_DIR"

export CUDA_VISIBLE_DEVICES=0

cd "$ROOT_DIR"
ln -sfn "$MASTER_LOG" "$LATEST_LINK"
printf '%s\n' "$$" > "$LOG_DIR/smote_fix_all.pid"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$MASTER_LOG"; }

# ===========================================================================
# ESPERAR A QUE TERMINE EL PROCESO ANTERIOR
# ===========================================================================
if kill -0 "$PID_WAIT" 2>/dev/null; then
    log "Esperando a que termine PID $PID_WAIT (run_thinking_intuitive.sh)..."
    while kill -0 "$PID_WAIT" 2>/dev/null; do
        sleep 60
    done
    log "PID $PID_WAIT ha terminado. Comenzando relanzamiento SMOTE/ADASYN."
else
    log "PID $PID_WAIT ya no existe. Comenzando directamente."
fi

# ===========================================================================
# Funcion para lanzar un script con manejo de activaciones GPT en /hdd
# ===========================================================================
run_script() {
    local SCRIPT_PATH="$1"
    local SCRIPT_NAME
    SCRIPT_NAME=$(basename "$SCRIPT_PATH" .py)

    log "--- Iniciando $SCRIPT_PATH ---"
    local START_SEC=$SECONDS

    # Preparar directorios de activaciones segun tipo de script
    local CLEANUP_DIRS=()

    if [[ "$SCRIPT_PATH" == *"_gpt_activaciones.py" ]]; then
        # GPT: redirigir activaciones a /hdd via symlink
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
        # SAE genero/edad guardan memmaps en /hdd; MBTI SAE hace streaming (no guarda)
        if [[ "$SCRIPT_PATH" == *"clasificacion_genero"* ]]; then
            CLEANUP_DIRS+=("/hdd/aitziber.l/activaciones_sae_gpt2_genero")
        elif [[ "$SCRIPT_PATH" == *"clasificacion_edad"* ]]; then
            CLEANUP_DIRS+=("/hdd/aitziber.l/activaciones_sae_gpt2_edad")
        fi
        # Limpiar cache previo antes de ejecutar (fuerza re-extraccion)
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

    # Limpiar TODAS las activaciones temporales
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

# ===========================================================================
# LISTA DE LOS 12 CLASIFICADORES (secuencial)
# ===========================================================================
#
# Orden: genero, edad, mbti (introverted, intuitive, thinking, perceiving)
# Cada trait: primero SAE, luego GPT (SAE usa GPU streaming, GPT usa memmap)
#

SCRIPTS=(
    # --- GENERO ---
    "clasificacion_genero/gpt_activaciones/clasificador_genero_sae_activaciones.py"
    "clasificacion_genero/gpt_activaciones/clasificador_genero_gpt_activaciones.py"
    # --- EDAD ---
    "clasificacion_edad/gpt_activaciones/clasificador_edad_sae_activaciones.py"
    "clasificacion_edad/gpt_activaciones/clasificador_edad_gpt_activaciones.py"
    # --- MBTI INTROVERTED ---
    "clasificacion_mbti/introverted/clasificador_introverted_sae_activaciones.py"
    "clasificacion_mbti/introverted/clasificador_introverted_gpt_activaciones.py"
    # --- MBTI INTUITIVE ---
    "clasificacion_mbti/intuitive/clasificador_intuitive_sae_activaciones.py"
    "clasificacion_mbti/intuitive/clasificador_intuitive_gpt_activaciones.py"
    # --- MBTI THINKING ---
    "clasificacion_mbti/thinking/clasificador_thinking_sae_activaciones.py"
    "clasificacion_mbti/thinking/clasificador_thinking_gpt_activaciones.py"
    # --- MBTI PERCEIVING ---
    "clasificacion_mbti/perceiving/clasificador_perceiving_sae_activaciones.py"
    "clasificacion_mbti/perceiving/clasificador_perceiving_gpt_activaciones.py"
)

log "======================================================================"
log "RELANZAMIENTO COMPLETO: ${#SCRIPTS[@]} clasificadores con fix SMOTE/ADASYN"
log "======================================================================"

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

log "======================================================================"
log "COMPLETADO: $OK/$TOTAL exitosos, $FAIL fallidos (${GLOBAL_H}h ${GLOBAL_M}m total)"
log "======================================================================"
