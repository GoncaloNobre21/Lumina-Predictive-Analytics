from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from typing import Dict, Any, Union

class AnomalyDetector:
    """
    Multivariate Anomaly Detection using Isolation Forests and Robust Scaling.
    """
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """
        Fit the model with baseline data.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True

    def detect(self, X: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """
        Detect anomalies in a given set of samples.
        Returns:
            - Labels: -1 for anomalies, 1 for normal data.
            - Scores: Decision function score (lower means more anomalous).
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection.")
        
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        return {
            "labels": labels,
            "scores": scores,
            "anomaly_rate": np.mean(labels == -1)
        }
