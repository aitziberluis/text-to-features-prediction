#!/usr/bin/env bash
# Espera a que el pipeline de genero actualmente en ejecucion termine,
# y entonces lanza el pipeline de edad (GPT -> SAE -> interpretabilidad).
#
# Uso:
#   nohup bash run_edad_after_genero.sh > logs/run_edad_after_genero_wait.log 2>&1 &
#
# El proceso a esperar es el bash run_genero_only.sh (PID guardado en
# logs/run_genero_only.pid). Si no existe el PID o ya ha terminado, lanza
# directamente el pipeline de edad.

set -uo pipefail

ROOT_DIR="/home/aitziber.l/TFM"
LOG_DIR="$ROOT_DIR/logs"
WAIT_LOG="$LOG_DIR/run_edad_after_genero_wait.log"
GENERO_PID_FILE="$LOG_DIR/run_genero_only.pid"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$WAIT_LOG"; }

log "==> Watcher iniciado. Esperando a que termine el pipeline de genero."

if [[ -f "$GENERO_PID_FILE" ]]; then
    GENERO_PID=$(cat "$GENERO_PID_FILE" | tr -d '[:space:]')
    log "PID de genero leido del archivo: $GENERO_PID"
else
    log "No se encontro $GENERO_PID_FILE; intentando detectar por nombre."
    GENERO_PID=$(pgrep -f 'bash run_genero_only\.sh' | head -1 || true)
    if [[ -z "${GENERO_PID:-}" ]]; then
        log "No se detecto bash run_genero_only.sh; buscando script python."
        GENERO_PID=$(pgrep -f 'clasificador_genero_.*_activaciones\.py' | head -1 || true)
    fi
fi

if [[ -n "${GENERO_PID:-}" ]] && kill -0 "$GENERO_PID" 2>/dev/null; then
    log "Esperando a PID $GENERO_PID..."
    while kill -0 "$GENERO_PID" 2>/dev/null; do
        sleep 60
    done
    log "PID $GENERO_PID ha terminado."
else
    log "No hay proceso activo de genero; se lanza edad inmediatamente."
fi

# Margen extra: asegurar que tampoco quedan procesos python del pipeline genero
while pgrep -f 'clasificador_genero_.*_activaciones\.py|interpretabilidad_genero_sae\.py' > /dev/null; do
    log "Aun hay procesos de genero vivos; esperando 60s mas..."
    sleep 60
done

log "==> Lanzando pipeline EDAD: bash run_edad_only.sh"
exec bash "$ROOT_DIR/run_edad_only.sh"
