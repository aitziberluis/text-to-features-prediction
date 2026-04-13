#!/usr/bin/env bash
# ===========================================================================
# Lanza secuencialmente:
#   1) thinking SAE
#   2) thinking GPT (activaciones en /hdd para no gastar espacio en /home)
#   3) intuitive SAE (SMOTE/ADASYN a nivel comentario corregido)
#
# GPU 0, sin PYTHONUNBUFFERED (sin flush forzado).
# Uso: nohup bash run_thinking_intuitive.sh > /dev/null 2>&1 &
# ===========================================================================

set -eu

ROOT_DIR="/home/aitziber.l/TFM"
HDD_DIR="/hdd/aitziber.l"
LOG_DIR="$ROOT_DIR/logs"
MASTER_LOG="$LOG_DIR/thinking_intuitive_$(date +%Y%m%d_%H%M%S).log"
LATEST_LINK="$LOG_DIR/thinking_intuitive_latest.log"

mkdir -p "$LOG_DIR" "$HDD_DIR"

export CUDA_VISIBLE_DEVICES=0
# Sin PYTHONUNBUFFERED (el usuario pidio sin flush)

cd "$ROOT_DIR"
ln -sfn "$MASTER_LOG" "$LATEST_LINK"
printf '%s\n' "$$" > "$LOG_DIR/thinking_intuitive.pid"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*" >> "$MASTER_LOG"; }

# -------------------------------------------------------------------
# 1) THINKING SAE
# -------------------------------------------------------------------
log "Iniciando clasificacion_mbti/thinking/clasificador_thinking_sae_activaciones.py"
python3 clasificacion_mbti/thinking/clasificador_thinking_sae_activaciones.py \
    >> "$MASTER_LOG" 2>&1
EXIT1=$?
if [[ $EXIT1 -ne 0 ]]; then
    log "ERROR: thinking SAE fallo con exit code $EXIT1"
    exit $EXIT1
fi
log "Finalizado correctamente: thinking SAE"

# -------------------------------------------------------------------
# 2) THINKING GPT  (activaciones redirigidas a /hdd para ahorrar espacio)
# -------------------------------------------------------------------
# Crear directorio en HDD y enlazar para que el script escriba ahi
ACT_HDD="$HDD_DIR/activaciones_gpt2_thinking"
ACT_LOCAL="$ROOT_DIR/data/activaciones_gpt2_thinking"
rm -rf "$ACT_LOCAL" "$ACT_HDD"
mkdir -p "$ACT_HDD"
ln -sfn "$ACT_HDD" "$ACT_LOCAL"

log "Iniciando clasificacion_mbti/thinking/clasificador_thinking_gpt_activaciones.py"
python3 clasificacion_mbti/thinking/clasificador_thinking_gpt_activaciones.py \
    >> "$MASTER_LOG" 2>&1
EXIT2=$?

# Limpiar activaciones grandes (modelos .pkl ya estan en modelos/)
rm -rf "$ACT_HDD" "$ACT_LOCAL"
log "Activaciones temporales de thinking GPT eliminadas"

if [[ $EXIT2 -ne 0 ]]; then
    log "ERROR: thinking GPT fallo con exit code $EXIT2"
    exit $EXIT2
fi
log "Finalizado correctamente: thinking GPT"

# -------------------------------------------------------------------
# 3) INTUITIVE SAE  (subsample estratificado -> SMOTE/ADASYN comentario)
# -------------------------------------------------------------------
log "Iniciando clasificacion_mbti/intuitive/clasificador_intuitive_sae_activaciones.py"
python3 clasificacion_mbti/intuitive/clasificador_intuitive_sae_activaciones.py \
    >> "$MASTER_LOG" 2>&1
EXIT3=$?
if [[ $EXIT3 -ne 0 ]]; then
    log "ERROR: intuitive SAE fallo con exit code $EXIT3"
    exit $EXIT3
fi
log "Finalizado correctamente: intuitive SAE"

# -------------------------------------------------------------------
log "Cola completada: thinking SAE + thinking GPT + intuitive SAE"
