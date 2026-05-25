"""
Probability calibration for trustworthy confidence scores.

Model probabilities are often not well-calibrated. A 0.9 predicted probability
might actually mean 70% chance of being correct. For security applications,
we need probabilities we can trust — if we say 90% confidence, it should mean
90% of alerts at that level are true positives.

Platt scaling: Fits a logistic regression on the model's raw scores.
Isotonic regression: Non-parametric, fits monotonic function. Better when
sufficient data is available.
"""

import numpy as np
from typing import Optional
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from deepdefend.models.base import BaseDetector
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class ProbabilityCalibrator(BaseDetector):
    """
    Calibrates raw model scores into trustworthy probabilities.
    
    Wraps a detector and learns a calibration map from validation data.
    After calibration: P(attack | score) ≈ predicted probability.
    
    This is critical for:
    - Setting meaningful alert thresholds
    - Combining scores from different models
    - Giving analysts trustworthy confidence numbers
    """
    
    def __init__(self,
                 base_detector: BaseDetector,
                 method: str = 'platt',
                 name: str = "calibrator"):
        super().__init__(name=name, model_type="calibrator")
        
        self.base_detector = base_detector
        self.method = method
        self._calibrator = None
        self._calibration_map = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> 'ProbabilityCalibrator':
        """
        Fit calibration map using validation data.
        
        Args:
            X: Validation features
            y: Validation labels (binary: 1 for any attack, 0 for normal)
        """
        if not self.base_detector.is_fitted:
            raise RuntimeError("Base detector must be fitted before calibration.")
        
        # Get raw scores from base detector
        raw_scores = self.base_detector.score_samples(X)
        
        # Make binary: any attack = 1, normal = 0
        y_binary = (y != 'normal').astype(int)
        
        if self.method == 'platt':
            self._calibrator = LogisticRegression(
                C=1.0, 
                solver='lbfgs',
                class_weight='balanced'
            )
        elif self.method == 'isotonic':
            self._calibrator = IsotonicRegression(
                y_min=0.0, 
                y_max=1.0, 
                out_of_bounds='clip'
            )
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        # Reshape for sklearn
        scores_2d = raw_scores.reshape(-1, 1)
        
        self._calibrator.fit(scores_2d, y_binary)
        
        # Store calibration map for debugging
        test_scores = np.linspace(0, 1, 100).reshape(-1, 1)
        if hasattr(self._calibrator, 'predict_proba'):
            calibrated = self._calibrator.predict_proba(test_scores)[:, 1]
        else:
            calibrated = self._calibrator.predict(test_scores)
        
        self._calibration_map = {
            "raw_scores": test_scores.flatten().tolist(),
            "calibrated": calibrated.tolist()
        }
        
        self._fitted = True
        logger.logger.info(f"Calibrator fitted using {self.method} method")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using calibrated probabilities."""
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        raw_scores = self.base_detector.score_samples(X)
        scores_2d = raw_scores.reshape(-1, 1)
        
        if hasattr(self._calibrator, 'predict_proba'):
            # Fix for scikit-learn version mismatch
            if isinstance(self._calibrator, LogisticRegression) and not hasattr(self._calibrator, 'multi_class'):
                self._calibrator.multi_class = 'auto'
            return self._calibrator.predict_proba(scores_2d)
        else:
            calibrated = self._calibrator.predict(scores_2d)
            return np.column_stack([1 - calibrated, calibrated])
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated scores."""
        proba = self.predict_proba(X)[:, 1]
        return proba
    
    def get_calibration_curve(self) -> dict:
        """Return calibration data for plotting/diagnostics."""
        return self._calibration_map