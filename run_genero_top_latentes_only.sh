#!/usr/bin/env bash
# Solo el calculo de "top latentes mas activos por clase" para genero.
# No entrena clasificador, no hace ablacion ni usa coeficientes.
#
# Uso directo:
#   nohup bash run_genero_top_latentes_only.sh > /dev/null 2>&1 &

set -uo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled

ROOT_DIR="/home/aitziber.l/TFM"
HDD_DIR="/hdd/aitziber.l"
LOG_DIR="$ROOT_DIR/logs"
MASTER_LOG="$LOG_DIR/run_genero_top_latentes_$(date +%Y%m%d_%H%M%S).log"
LATEST_LINK="$LOG_DIR/run_genero_top_latentes_latest.log"
SCRIPT_PATH="clasificacion_genero/gpt_activaciones/top_latentes_activos_genero.py"

mkdir -p "$LOG_DIR" "$HDD_DIR"
cd "$ROOT_DIR"

ln -sfn "$MASTER_LOG" "$LATEST_LINK"
printf '%s\n' "$$" > "$LOG_DIR/run_genero_top_latentes.pid"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$MASTER_LOG"; }

SAE_CACHE_DIR="$HDD_DIR/activaciones_sae_gpt2_genero"
rm -rf "$SAE_CACHE_DIR"
log "Cache SAE eliminada: $SAE_CACHE_DIR"

log "--- Iniciando $SCRIPT_PATH (solo top latentes activos) ---"
START_SEC=$SECONDS
stdbuf -oL -eL python3 -u "$SCRIPT_PATH" 2>&1 | tee -a "$MASTER_LOG"
EXIT_CODE=${PIPESTATUS[0]}
ELAPSED=$(( SECONDS - START_SEC ))
HOURS=$(( ELAPSED / 3600 )); MINUTES=$(( (ELAPSED % 3600) / 60 ))

if [[ $EXIT_CODE -eq 0 ]]; then
    log "OK: $SCRIPT_PATH completado (${HOURS}h ${MINUTES}m)"
else
    log "FALLO: $SCRIPT_PATH exit=$EXIT_CODE (${HOURS}h ${MINUTES}m)"
fi

exit $EXIT_CODE
