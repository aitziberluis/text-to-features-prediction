#!/usr/bin/env bash
# Espera a que termine el pipeline de edad (run_edad_only.sh) y, cuando finalice,
# lanza solo la interpretabilidad SAE de genero (run_genero_interpretabilidad_only.sh).
#
# Uso:
#   nohup bash run_genero_interp_after_edad.sh > /dev/null 2>&1 &

set -uo pipefail

ROOT_DIR="/home/aitziber.l/TFM"
LOG_DIR="$ROOT_DIR/logs"
WAIT_LOG="$LOG_DIR/wait_genero_interp_after_edad.log"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$WAIT_LOG"; }

log "Esperando a que termine el pipeline de edad..."
while pgrep -f 'run_edad_only.sh|clasificador_edad_(gpt|sae)_activaciones.py|interpretabilidad_edad_sae.py' > /dev/null; do
    sleep 60
done
log "Pipeline de edad terminado. Lanzando interpretabilidad de genero."

nohup bash run_genero_interpretabilidad_only.sh > /dev/null 2>&1 &
disown
log "Interpretabilidad de genero lanzada en background (PID padre: $!)."
