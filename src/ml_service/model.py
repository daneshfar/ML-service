from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

from .config import load_config


def train_and_save_model() -> None:
    config = load_config()

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    # MLflow logging
    mlflow.set_experiment("breast_cancer_classification")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", config.n_estimators)
        mlflow.log_param("max_depth", config.max_depth)
        mlflow.log_param("test_size", config.test_size)

        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)

        print(f"Validation accuracy: {acc:.3f}")

        model_path = Path(config.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        mlflow.log_artifact(str(model_path))

        # Optional: log entire model as MLflow model
        mlflow.sklearn.log_model(model, "model")

        print(f"Saved model to {model_path.resolve()}")


if __name__ == "__main__":
    train_and_save_model()
