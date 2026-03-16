from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from src.models.forecaster import PredictiveEngine
from src.models.anomaly_detector import AnomalyDetector
from src.utils.logger import logger

app = FastAPI(
    title="Lumina: Predictive Analytics API",
    description="High-performance forecasting and anomaly detection engine.",
    version="1.0.0"
)

# Global instances (simplified for demo)
forecaster = PredictiveEngine({"input_dim": 10})
detector = AnomalyDetector()

# In-memory baseline for demo
dummy_baseline = np.random.randn(100, 10)
detector.fit(dummy_baseline)

class ForecastRequest(BaseModel):
    data: List[List[float]] = Field(..., example=[[0.1]*10]*5)  # sequence of 5 steps, 10 dims

class AnomalyRequest(BaseModel):
    samples: List[List[float]] = Field(..., example=[[0.5]*10, [10.0]*10])

@app.get("/health")
async def health():
    return {"status": "operational", "engine": "Lumina"}

@app.post("/predict/forecast")
async def predict_forecast(request: ForecastRequest):
    try:
        # Convert to torch tensor
        x_tensor = torch.tensor(request.data).unsqueeze(0).float()
        prediction = forecaster.predict(x_tensor)
        return {"forecast": prediction.squeeze().tolist()}
    except Exception as e:
        logger.error(f"Forecasting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/anomalies")
async def predict_anomalies(request: AnomalyRequest):
    try:
        x_np = np.array(request.samples)
        results = detector.detect(x_np)
        return {
            "is_anomaly": (results["labels"] == -1).tolist(),
            "scores": results["scores"].tolist(),
            "summary": {"anomaly_rate": float(results["anomaly_rate"])}
        }
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
