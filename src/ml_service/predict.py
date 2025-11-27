from pathlib import Path
from typing import List

import joblib
from pydantic import BaseModel, Field, field_validator

from .config import load_config


class BreastCancerInput(BaseModel):
    features: List[float] = Field(
        ..., description="List of 30 numeric features in the correct order."
    )

    @field_validator("features")
    @classmethod
    def check_feature_length(cls, v: List[float]) -> List[float]:
        if len(v) != 30:
            raise ValueError(f"Expected 30 features, got {len(v)}")
        return v



class PredictionOutput(BaseModel):
    prediction: int
    probability: float


class ModelService:
    def __init__(self):
        self._config = load_config()
        self._model = self._load_model()

    def _load_model(self):
        path = Path(self._config.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found at {path}. Train the model first."
            )
        return joblib.load(path)

    def predict(self, inp: BreastCancerInput) -> PredictionOutput:
        import numpy as np

        X = np.array(inp.features, dtype=float).reshape(1, -1)
        pred = int(self._model.predict(X)[0])
        prob = float(self._model.predict_proba(X)[0, 1])  # probability of class 1
        return PredictionOutput(prediction=pred, probability=prob)
