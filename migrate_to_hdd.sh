#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Migrar carpetas pesadas de TFM al HDD y dejar symlinks.
#
# Uso:
#   1. Pide a un admin: sudo mkdir -p /hdd/aitziber.l/TFM && sudo chown -R aitziber.l:aitziber.l /hdd/aitziber.l
#   2. Ejecuta: bash migrate_to_hdd.sh
# ============================================================

SRC="/home/aitziber.l/TFM"
DST="/hdd/aitziber.l/TFM"

# Carpetas a mover (las mas pesadas)
DIRS_TO_MOVE=(
    "data"
    "sae-ckpts"
    "wandb"
    "modelos"
)

# Verificaciones previas
if [[ ! -d "$DST" ]]; then
    echo "ERROR: $DST no existe. Pide a un admin que lo cree."
    echo "  sudo mkdir -p $DST && sudo chown -R $(whoami):$(whoami) /hdd/$(whoami)"
    exit 1
fi

if [[ ! -w "$DST" ]]; then
    echo "ERROR: No tienes permisos de escritura en $DST."
    exit 1
fi

echo "=================================="
echo "Migrando carpetas pesadas al HDD"
echo "=================================="
echo "Origen: $SRC"
echo "Destino: $DST"
echo ""

for dir in "${DIRS_TO_MOVE[@]}"; do
    src_path="$SRC/$dir"
    dst_path="$DST/$dir"

    # Si ya es un symlink, saltar
    if [[ -L "$src_path" ]]; then
        echo "[OK] $dir ya es un symlink -> $(readlink -f "$src_path")"
        continue
    fi

    # Si no existe la carpeta origen, crear en destino y symlink
    if [[ ! -d "$src_path" ]]; then
        echo "[SKIP] $dir no existe en $SRC, creando en HDD directamente..."
        mkdir -p "$dst_path"
        ln -s "$dst_path" "$src_path"
        echo "  -> Symlink creado: $src_path -> $dst_path"
        continue
    fi

    echo "[MOVING] $dir ..."
    du -sh "$src_path" 2>/dev/null || true

    # Mover con rsync (resume-safe) y luego eliminar original
    rsync -a --progress "$src_path/" "$dst_path/"
    rm -rf "$src_path"
    ln -s "$dst_path" "$src_path"

    echo "  -> Movido y symlink creado: $src_path -> $dst_path"
    echo ""
done

echo ""
echo "=================================="
echo "Migracion completada"
echo "=================================="
echo ""
echo "Espacio liberado en /home:"
df -h /home/aitziber.l
echo ""
echo "Espacio en HDD:"
df -h /hdd
echo ""
echo "Symlinks creados:"
ls -la "$SRC" | grep '^l'
