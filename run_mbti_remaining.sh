#!/usr/bin/env bash

set -u

ROOT_DIR="/home/aitziber.l/TFM"
HDD_DIR="/hdd/aitziber.l"
LOG_DIR="$ROOT_DIR/logs"
MASTER_LOG="$LOG_DIR/mbti_secuencial_$(date +%Y%m%d_%H%M%S).log"
LATEST_LOG_LINK="$LOG_DIR/mbti_secuencial_latest.log"
STATUS_FILE="$LOG_DIR/mbti_secuencial.pid"

mkdir -p "$LOG_DIR"
mkdir -p "$HDD_DIR"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Introverted GPT ya completado, se omite
TASKS=(
  "clasificacion_mbti/introverted/clasificador_introverted_sae_activaciones.py"
  "clasificacion_mbti/intuitive/clasificador_intuitive_gpt_activaciones.py"
  "clasificacion_mbti/intuitive/clasificador_intuitive_sae_activaciones.py"
  "clasificacion_mbti/perceiving/clasificador_perceiving_gpt_activaciones.py"
  "clasificacion_mbti/perceiving/clasificador_perceiving_sae_activaciones.py"
  "clasificacion_mbti/thinking/clasificador_thinking_gpt_activaciones.py"
  "clasificacion_mbti/thinking/clasificador_thinking_sae_activaciones.py"
)

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$MASTER_LOG"
}

clear_hdd_contents() {
  log "Limpiando contenido de $HDD_DIR"
  find "$HDD_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  sync || true
  log "Limpieza completada en $HDD_DIR"
}

run_task() {
  local rel_path="$1"
  local abs_path="$ROOT_DIR/$rel_path"

  log "Iniciando $rel_path"
  python3 -u "$abs_path" 2>&1 | tee -a "$MASTER_LOG"
  local exit_code=${PIPESTATUS[0]}

  if [[ $exit_code -ne 0 ]]; then
    log "ERROR: $rel_path fallo con exit code $exit_code"
    return $exit_code
  fi

  log "Finalizado correctamente: $rel_path"
  return 0
}

main() {
  cd "$ROOT_DIR" || exit 1

  ln -sfn "$MASTER_LOG" "$LATEST_LOG_LINK"
  printf '%s\n' "$$" > "$STATUS_FILE"

  log "Launcher MBTI secuencial arrancado (sin introverted GPT)"
  log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  log "Log maestro: $MASTER_LOG"
  log "Tareas pendientes: ${#TASKS[@]}"

  for task in "${TASKS[@]}"; do
    clear_hdd_contents
    run_task "$task" || exit $?
  done

  clear_hdd_contents
  log "Cola MBTI completada"
}

main
