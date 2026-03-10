#!/bin/bash
# Script para ejecutar el pipeline completo de clasificación de género con SAE
# Usa nohup para ejecutar en background

set -e  # Salir si hay error

echo "============================================================"
echo "PIPELINE DE CLASIFICACIÓN DE GÉNERO CON SAE"
echo "============================================================"
echo ""

# Crear directorio para logs
mkdir -p logs

# Variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"

echo "📁 Logs se guardarán en: $LOG_DIR/"
echo ""

# =============================================================================
# PASO 1: PREPROCESAMIENTO (Opcional, ya está ejecutado, pero incluido por completitud)
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PASO 1/4: Preprocesamiento y EDA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏩ Saltando preprocesamiento (ya ejecutado previamente)"
echo "   Si quieres ejecutarlo de nuevo, descomenta las líneas siguientes:"
echo ""
# nohup python3 preprocesamiento.py > $LOG_DIR/preprocesamiento_${TIMESTAMP}.log 2>&1 &
# PREP_PID=$!
# echo "✓ PID: $PREP_PID"
# echo "  Log: $LOG_DIR/preprocesamiento_${TIMESTAMP}.log"
# echo "  Monitorear: tail -f $LOG_DIR/preprocesamiento_${TIMESTAMP}.log"
# echo ""
# wait $PREP_PID
# echo "✓ Preprocesamiento completado"
# echo ""

# =============================================================================
# PASO 2: ENTRENAR SAE
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PASO 2/4: Entrenamiento de SAE (Sparse Autoencoder)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Iniciando entrenamiento de SAE en background..."
echo "⚠️  Este paso puede tardar VARIAS HORAS dependiendo del dataset y GPU"
echo ""

nohup python3 sae_genero.py > $LOG_DIR/sae_genero_${TIMESTAMP}.log 2>&1 &
SAE_PID=$!

echo "✓ SAE lanzada con PID: $SAE_PID"
echo "  Log: $LOG_DIR/sae_genero_${TIMESTAMP}.log"
echo "  Monitorear: tail -f $LOG_DIR/sae_genero_${TIMESTAMP}.log"
echo "  Matar proceso: kill $SAE_PID"
echo ""
echo "⏳ Esperando a que termine la SAE..."
echo "   (Puedes cerrar esta terminal, el proceso continuará)"
echo ""

# Esperar a que termine el entrenamiento de SAE
wait $SAE_PID
SAE_EXIT=$?

if [ $SAE_EXIT -ne 0 ]; then
    echo "❌ Error en entrenamiento de SAE (exit code: $SAE_EXIT)"
    echo "   Revisa el log: $LOG_DIR/sae_genero_${TIMESTAMP}.log"
    exit 1
fi

echo "✓ SAE entrenada exitosamente"
echo ""

# =============================================================================
# PASO 3: ENTRENAR CLASIFICADOR
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PASO 3/4: Entrenamiento del Clasificador de Género"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Iniciando entrenamiento del clasificador en background..."
echo "⏱️  Este paso debería tardar unos minutos"
echo ""

nohup python3 clasificador_genero.py > $LOG_DIR/clasificador_genero_${TIMESTAMP}.log 2>&1 &
CLF_PID=$!

echo "✓ Clasificador lanzado con PID: $CLF_PID"
echo "  Log: $LOG_DIR/clasificador_genero_${TIMESTAMP}.log"
echo "  Monitorear: tail -f $LOG_DIR/clasificador_genero_${TIMESTAMP}.log"
echo ""

# Esperar a que termine el clasificador
wait $CLF_PID
CLF_EXIT=$?

if [ $CLF_EXIT -ne 0 ]; then
    echo "❌ Error en entrenamiento del clasificador (exit code: $CLF_EXIT)"
    echo "   Revisa el log: $LOG_DIR/clasificador_genero_${TIMESTAMP}.log"
    exit 1
fi

echo "✓ Clasificador entrenado exitosamente"
echo ""

# =============================================================================
# PASO 4: INFERENCIA DE EJEMPLO
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PASO 4/4: Inferencia en Textos de Ejemplo"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Ejecutando predicciones de ejemplo..."
echo ""

nohup python3 inferencia_genero.py > $LOG_DIR/inferencia_genero_${TIMESTAMP}.log 2>&1 &
INF_PID=$!

echo "✓ Inferencia lanzada con PID: $INF_PID"
echo "  Log: $LOG_DIR/inferencia_genero_${TIMESTAMP}.log"
echo ""

# Esperar a que termine la inferencia
wait $INF_PID
INF_EXIT=$?

if [ $INF_EXIT -ne 0 ]; then
    echo "❌ Error en inferencia (exit code: $INF_EXIT)"
    echo "   Revisa el log: $LOG_DIR/inferencia_genero_${TIMESTAMP}.log"
    exit 1
fi

echo "✓ Inferencia completada"
echo ""

# =============================================================================
# RESUMEN FINAL
# =============================================================================
echo "============================================================"
echo "✅ PIPELINE COMPLETADO EXITOSAMENTE"
echo "============================================================"
echo ""
echo "📊 Resultados guardados en:"
echo "  - SAE entrenada: sae-ckpts/sae-gpt2-genero/"
echo "  - Clasificador: modelos/clasificador_genero.pkl"
echo "  - Gráficos: figuras/"
echo "  - Logs: $LOG_DIR/"
echo ""
echo "📝 Logs generados:"
echo "  - $LOG_DIR/sae_genero_${TIMESTAMP}.log"
echo "  - $LOG_DIR/clasificador_genero_${TIMESTAMP}.log"
echo "  - $LOG_DIR/inferencia_genero_${TIMESTAMP}.log"
echo ""
echo "🎉 ¡Todo listo para usar el sistema de predicción de género!"
echo ""
