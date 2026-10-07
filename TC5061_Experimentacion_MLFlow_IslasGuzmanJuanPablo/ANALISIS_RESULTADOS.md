# Analisis breve de resultados

Se ejecutaron cinco configuraciones distintas en el experimento
`TC5061_Titanic_Logistic_Experiments`. La métrica principal para seleccionar
la configuración es F1, porque combina precision y recall en un problema
binario y evita elegir únicamente por la proporción global de aciertos.

| C | max_iter | random_state | accuracy | precision | recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 500 | 42 | 0.7834 | 0.8444 | 0.5846 | 0.6909 |
| 0.1 | 1000 | 42 | 0.8153 | 0.8000 | 0.7385 | 0.7680 |
| 1.0 | 1000 | 42 | 0.8280 | 0.7794 | 0.8154 | **0.7970** |
| 10.0 | 1000 | 42 | 0.8280 | 0.7794 | 0.8154 | **0.7970** |
| 1.0 | 2000 | 7 | **0.8408** | **0.8846** | 0.7077 | 0.7863 |

La configuración recomendada por F1 es `C=1.0`, `max_iter=1000` y
`random_state=42`. Si el criterio fuera únicamente accuracy, la corrida con
semilla 7 sería la mejor, con 0.8408; sin embargo, su recall y F1 son menores
que los de la configuración recomendada. El cambio más influyente fue `C`:
con regularización fuerte (`C=0.01`) el modelo obtuvo el peor recall y F1,
mientras que al aumentar `C` a 1.0 mejoró claramente. El cambio de 1.0 a
10.0 no produjo mejora, lo que sugiere rendimientos decrecientes en este
conjunto de variables.

Las principales limitaciones son el tamaño y origen del dataset, el uso de
una sola partición de entrenamiento/prueba y la comparación de un único tipo
de modelo. La semilla modifica la partición y, por tanto, puede cambiar la
conclusión si se evalúa solo una corrida. Para una siguiente iteración usaría
validación cruzada, un conjunto de prueba separado y una comparación con
árboles o ensambles. Las capturas solicitadas deben tomarse desde MLflow:
la tabla comparativa y el detalle de una corrida con `metrics.json`,
`confusion_matrix.png`, el reporte y el modelo.
