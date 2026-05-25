"""
Model architectures for imbalanced network intrusion detection.
Cascaded architecture: anomaly detection → per-class specialists → calibration.
"""

from deepdefend.models.base import BaseDetector
from deepdefend.models.isolation_forest import IsolationForestAnomalyDetector
from deepdefend.models.autoencoder import AutoencoderAnomalyDetector
from deepdefend.models.xgboost_ensemble import XGBoostEnsemble
from deepdefend.models.calibrator import ProbabilityCalibrator

__all__ = [
    "BaseDetector",
    "IsolationForestAnomalyDetector",
    "AutoencoderAnomalyDetector",
    "XGBoostEnsemble",
    "ProbabilityCalibrator",
]