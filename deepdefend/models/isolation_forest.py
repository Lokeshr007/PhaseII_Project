"""
Isolation Forest for anomaly detection.
First layer of the cascaded architecture — catches everything suspicious.

Why Isolation Forest for network anomaly detection:
1. Unsupervised — trained on normal traffic only, catches novel attacks
2. Efficient — O(n) training, sublinear scoring
3. Interpretable — anomaly score based on path length
4. Handles high-dimensional network features well
5. No assumptions about attack distribution (critical for zero-day detection)
"""

import numpy as np
from typing import Optional
from sklearn.ensemble import IsolationForest
from deepdefend.models.base import BaseDetector
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class IsolationForestAnomalyDetector(BaseDetector):
    """
    Anomaly detector using Isolation Forest.
    
    Trained on normal traffic patterns. Any significant deviation
    from learned normality triggers the second-stage classifiers.
    
    Configuration tuned for network traffic:
    - contamination: Expected fraction of anomalies. Set low (0.01)
      to avoid false positives from rare-but-normal traffic.
    - n_estimators: Higher than default because network data is noisy.
    - max_samples: Auto to use full dataset during training.
    """
    
    def __init__(self,
                 n_estimators: int = 200,
                 contamination: float = 0.01,  # Expect 1% anomalies
                 max_samples: str = 'auto',
                 max_features: float = 0.8,
                 bootstrap: bool = False,
                 random_state: int = 42,
                 name: str = "isolation_forest"):
        
        super().__init__(name=name, model_type="anomaly_detector")
        
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.model = None
        self._anomaly_threshold: float = 0.0
        self._normal_data_bounds: dict = {}
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None) -> 'IsolationForestAnomalyDetector':
        """
        Train isolation forest on predominantly normal data.
        
        If y is provided, we train on normal samples only.
        This creates a model of "what normal looks like."
        """
        # If labels available, train on normal traffic only
        if y is not None:
            normal_mask = y == 'normal'
            X_train = X[normal_mask]
            logger.logger.info(
                f"Training Isolation Forest on {len(X_train):,} normal samples "
                f"(out of {len(X):,} total — filtering {sum(~normal_mask):,} attacks)"
            )
        else:
            X_train = X
            logger.logger.info(
                f"Training Isolation Forest on {len(X_train):,} samples (unsupervised)"
            )
        
        # Compute normal data bounds for feature validation in production
        self._normal_data_bounds = {
            "min": X_train.min(axis=0).tolist(),
            "max": X_train.max(axis=0).tolist(),
            "mean": X_train.mean(axis=0).tolist(),
            "std": X_train.std(axis=0).tolist(),
        }
        
        # Train the model
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train)
        
        # Must be True before calling score_samples
        self._fitted = True
        
        # Compute anomaly threshold from training data
        # IsolationForest returns -1 for anomalies, 1 for normal
        # We convert to 0 (normal) to 1 (anomalous) scale
        scores = self.score_samples(X_train)
        
        # Threshold at the contamination percentile
        self._anomaly_threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
        logger.logger.info(
            f"Isolation Forest fitted. Anomaly threshold: {self._anomaly_threshold:.4f}. "
            f"Estimators: {self.n_estimators}"
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly: 1 for anomaly, 0 for normal."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        raw_preds = self.model.predict(X)
        # Convert: sklearn returns 1 for inliers (normal), -1 for outliers
        return (raw_preds == -1).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Convert anomaly scores to pseudo-probabilities.
        Uses sigmoid scaling around the threshold.
        """
        scores = self.score_samples(X)
        
        # Sigmoid transformation centered at threshold
        # score > threshold → probability > 0.5
        scaled = (scores - self._anomaly_threshold) * 10  # Sharpness factor
        proba = 1.0 / (1.0 + np.exp(-scaled))
        
        return np.column_stack([1 - proba, proba])  # [normal, anomaly]
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores in [0, 1] range.
        
        Isolation Forest returns negative scores where more negative = more anomalous.
        We transform to: 0 = perfectly normal, 1 = extremely anomalous.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        
        # Raw scores: lower = more anomalous
        raw_scores = self.model.score_samples(X)
        
        # Invert and scale to [0, 1]
        # This is a min-max normalization with clipping
        scores_min = raw_scores.min()
        scores_max = raw_scores.max()
        
        if scores_max > scores_min:
            normalized = (raw_scores - scores_min) / (scores_max - scores_min)
        else:
            normalized = np.zeros_like(raw_scores)
        
        # Invert so 1 = most anomalous
        return 1.0 - normalized
    
    def get_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Alias for score_samples for consistent API."""
        return self.score_samples(X)
    
    def is_anomalous(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Binary anomaly decision with configurable threshold."""
        threshold = threshold or self._anomaly_threshold
        return self.score_samples(X) > threshold
    
    def get_contributing_features(self, X: np.ndarray, n_features: int = 5) -> list:
        """
        Identify which features contribute most to anomaly.
        
        Uses feature perturbation: measures score change when each feature
        is set to its mean (normal) value. Large change = important feature.
        
        This powers the evidence_summary in incident reports.
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        baseline_score = self.score_samples(X)[0]
        feature_impact = []
        
        normal_means = np.array(self._normal_data_bounds["mean"])
        
        for i in range(X.shape[1]):
            perturbed = X.copy()
            perturbed[0, i] = normal_means[i]
            new_score = self.score_samples(perturbed)[0]
            impact = baseline_score - new_score  # Positive = this feature makes it anomalous
            feature_impact.append((i, impact))
        
        # Sort by absolute impact
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
        return feature_impact[:n_features]
    
    def _get_framework_version(self) -> str:
        import sklearn
        return f"scikit-learn-{sklearn.__version__}"