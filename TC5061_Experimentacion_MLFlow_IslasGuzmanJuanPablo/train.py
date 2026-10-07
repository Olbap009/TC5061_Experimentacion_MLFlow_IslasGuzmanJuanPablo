"""Parametrized Titanic baseline training with MLflow Tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EXPERIMENT_NAME = "TC5061_Titanic_Logistic_Experiments"
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "alone"]
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked", "alone"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena y registra una regresion logistica sobre Titanic con MLflow."
    )
    parser.add_argument("--C", type=float, default=1.0, help="Inverso de la regularizacion.")
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="Numero maximo de iteraciones."
    )
    parser.add_argument(
        "--test-size", type=float, default=0.20, help="Proporcion reservada para prueba."
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Semilla para particion y modelo."
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help="Nombre del experimento de MLflow.",
    )
    return parser.parse_args()


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    data = sns.load_dataset("titanic").drop_duplicates()
    return data[FEATURES].copy(), data["survived"].astype(int)


def build_model(C: float, max_iter: int, random_state: int) -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(C=C, max_iter=max_iter, random_state=random_state),
            ),
        ]
    )


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, float], str]:
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }
    report = classification_report(
        y_test, predictions, target_names=["No sobrevivio", "Sobrevivio"], zero_division=0
    )
    return metrics, report


def save_artifacts(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: dict[str, float],
    report: str,
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X_test)
    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    (artifact_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_test, predictions)
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_xlabel("Prediccion")
    axis.set_ylabel("Valor real")
    axis.set_title("Matriz de confusion")
    figure.tight_layout()
    figure.savefig(artifact_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def train(args: argparse.Namespace) -> tuple[str, dict[str, float]]:
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size debe estar entre 0 y 1.")
    if args.C <= 0 or args.max_iter <= 0:
        raise ValueError("--C y --max-iter deben ser positivos.")

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    model = build_model(args.C, args.max_iter, args.random_state)
    model.fit(X_train, y_train)
    metrics, report = evaluate(model, X_test, y_test)

    artifact_dir = Path("artifacts")
    save_artifacts(model, X_test, y_test, metrics, report, artifact_dir)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(
        run_name=f"logistic_C{args.C:g}_iter{args.max_iter}_seed{args.random_state}"
    ) as run:
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "C": args.C,
                "max_iter": args.max_iter,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "feature_count": len(FEATURES),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(artifact_dir))
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model, "model", signature=signature, input_example=X_train.head(3)
        )
        return run.info.run_id, metrics


def main() -> None:
    args = parse_args()
    run_id, metrics = train(args)
    print(f"Run ID: {run_id}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
