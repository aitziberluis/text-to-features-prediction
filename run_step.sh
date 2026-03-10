#!/bin/bash
# Script para lanzar pasos individuales del pipeline con nohup

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

case "$1" in
    preprocesamiento|prep|1)
        echo "🚀 Lanzando preprocesamiento..."
        nohup python3 preprocesamiento.py > logs/preprocesamiento_${TIMESTAMP}.log 2>&1 &
        PID=$!
        echo "✓ PID: $PID"
        echo "  Log: logs/preprocesamiento_${TIMESTAMP}.log"
        echo "  Monitorear: tail -f logs/preprocesamiento_${TIMESTAMP}.log"
        ;;
    
    sae|2)
        echo "🚀 Lanzando entrenamiento de SAE..."
        echo "⚠️  Este proceso puede tardar HORAS"
        nohup python3 sae_genero.py > logs/sae_genero_${TIMESTAMP}.log 2>&1 &
        PID=$!
        echo "✓ PID: $PID"
        echo "  Log: logs/sae_genero_${TIMESTAMP}.log"
        echo "  Monitorear: tail -f logs/sae_genero_${TIMESTAMP}.log"
        echo "  Matar: kill $PID"
        ;;
    
    clasificador|clf|3)
        echo "🚀 Lanzando entrenamiento del clasificador..."
        nohup python3 clasificador_genero.py > logs/clasificador_genero_${TIMESTAMP}.log 2>&1 &
        PID=$!
        echo "✓ PID: $PID"
        echo "  Log: logs/clasificador_genero_${TIMESTAMP}.log"
        echo "  Monitorear: tail -f logs/clasificador_genero_${TIMESTAMP}.log"
        ;;
    
    inferencia|inf|4)
        echo "🚀 Lanzando inferencia..."
        nohup python3 inferencia_genero.py > logs/inferencia_genero_${TIMESTAMP}.log 2>&1 &
        PID=$!
        echo "✓ PID: $PID"
        echo "  Log: logs/inferencia_genero_${TIMESTAMP}.log"
        echo "  Monitorear: tail -f logs/inferencia_genero_${TIMESTAMP}.log"
        ;;
    
    *)
        echo "Uso: $0 {preprocesamiento|sae|clasificador|inferencia}"
        echo ""
        echo "Ejemplos:"
        echo "  $0 sae           # Lanza solo entrenamiento de SAE"
        echo "  $0 clasificador  # Lanza solo entrenamiento del clasificador"
        echo "  $0 inferencia    # Lanza solo inferencia"
        echo ""
        echo "También puedes usar números:"
        echo "  $0 1  # preprocesamiento"
        echo "  $0 2  # sae"
        echo "  $0 3  # clasificador"
        echo "  $0 4  # inferencia"
        exit 1
        ;;
esac

echo ""
echo "Proceso lanzado en background. Puedes cerrar la terminal."
echo "El proceso $PID seguirá ejecutándose."
