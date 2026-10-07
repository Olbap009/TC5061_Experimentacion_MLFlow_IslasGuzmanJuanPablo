# TC5061 - Experimentacion con MLflow

Este proyecto migra el baseline del notebook de Titanic a un script
parametrizado. El notebook se conserva como material de exploracion; las
corridas reproducibles se ejecutan con `train.py`.

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

## Ejecucion

La corrida base usa valores por defecto:

```bash
python train.py
```

Ejemplo parametrizado:

```bash
python train.py --C 0.1 --max-iter 1000 --test-size 0.2 --random-state 42
```

Los argumentos `--C`, `--max-iter`, `--test-size` y `--random-state`
se pueden cambiar sin editar el codigo. La semilla 42 se usa tanto en la
particion estratificada como en el modelo.

## MLflow

Cada ejecucion queda agrupada en el experimento
`TC5061_Titanic_Logistic_Experiments`, usando el almacenamiento local
predeterminado `mlruns`. Para consultar la interfaz:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Abrir `http://127.0.0.1:5000`, seleccionar el experimento y comparar las
corridas por `accuracy`, `f1`, `C`, `max_iter` y `random_state`.

Cada corrida registra parametros, accuracy, precision, recall, F1, una matriz
de confusion, `metrics.json`, el reporte de clasificacion y el modelo
scikit-learn.

## Cinco corridas sugeridas

```bash
python train.py --C 0.01 --max-iter 500  --random-state 42
python train.py --C 0.1  --max-iter 1000 --random-state 42
python train.py --C 1.0  --max-iter 1000 --random-state 42
python train.py --C 10.0 --max-iter 1000 --random-state 42
python train.py --C 1.0  --max-iter 2000 --random-state 7
```

Para la evidencia de entrega, tomar una captura de la tabla comparativa de
MLflow y otra del detalle de una corrida mostrando sus artefactos.
