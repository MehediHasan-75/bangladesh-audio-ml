"""
MLflow experiment tracking for the Bangladeshi audio classifier.

Usage:
    python experiments/track_experiment.py

This script wraps a training run with MLflow tracking so that every
experiment — parameters, metrics, and model artifacts — is logged
automatically and can be compared in the MLflow UI.

Start the UI with:
    mlflow ui --port 5000
Then open http://localhost:5000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from src.utils.helpers import get_logger

logger = get_logger(__name__)

MLFLOW_TRACKING_URI = "experiments/mlruns"
EXPERIMENT_NAME = "bangladeshi-audio-classifier"


def load_features(metadata_path: str = "ml_data/processing_metadata.csv") -> tuple:
    """
    Build a minimal feature matrix from processing metadata.
    Replace this with real audio features (MFCCs, spectrograms) for actual training.

    Returns:
        (X, y, label_encoder)
    """
    df = pd.read_csv(metadata_path)

    # Use dBFS and speech_percentage as proxy features for demo purposes
    feature_cols = ["dbfs", "speech_percentage"]
    X = df[feature_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df["category"])

    return X, y, le


def run_experiment(
    n_estimators: int = 100,
    max_depth: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train a RandomForest classifier and log everything to MLflow.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree.
        test_size: Fraction of data reserved for evaluation.
        random_state: Random seed for reproducibility.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    metadata = "ml_data/processing_metadata.csv"
    if not Path(metadata).exists():
        logger.error(
            "No processing metadata found at %s. "
            "Run the collection pipeline first.", metadata
        )
        return

    X, y, le = load_features(metadata)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "test_size": test_size,
            "random_state": random_state,
            "n_classes": len(le.classes_),
            "feature_cols": "dbfs,speech_percentage",
        })

        # Train
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log metrics
        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})

        # Log model artifact
        mlflow.sklearn.log_model(clf, artifact_path="model")

        logger.info(
            "Run complete — accuracy: %.4f, F1: %.4f | "
            "View at: mlflow ui --port 5000", acc, f1
        )

        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow run ID: %s", run_id)

    return {"accuracy": acc, "f1": f1}


if __name__ == "__main__":
    run_experiment()
