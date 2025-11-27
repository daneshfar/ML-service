import logging
from fastapi import FastAPI, HTTPException

from .predict import BreastCancerInput, ModelService, PredictionOutput

logger = logging.getLogger("ml_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Breast Cancer ML Service", version="0.1.0")

model_service = None


@app.on_event("startup")
def load_model():
    global model_service
    try:
        model_service = ModelService()
        logger.info("Model loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Startup error: {e}")
        model_service = None


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: BreastCancerInput):
    if model_service is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return model_service.predict(input_data)
