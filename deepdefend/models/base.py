"""
Abstract base class for all detection models.
Enforces consistent interface for training, inference, and serialization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class BaseDetector(ABC):
    """
    Abstract detector interface.
    
    All models in DeepDefend must implement:
    - fit(): Train on data
    - predict(): Raw predictions
    - predict_proba(): Calibrated probabilities
    - score(): Anomaly scores (0=normal, 1=attack)
    - save()/load(): Serialization
    """
    
    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type
        self._fitted = False
        self._metadata = {
            "name": name,
            "type": model_type,
            "created": datetime.now().isoformat(),
            "framework_version": self._get_framework_version()
        }
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
            sample_weight: Optional[np.ndarray] = None) -> 'BaseDetector':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores. Higher = more anomalous."""
        pass
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
    
    def _get_framework_version(self) -> str:
        """Override in subclasses."""
        return "unknown"
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        # Also save metadata
        meta_path = path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(self._metadata, f, indent=2)
        
        logger.logger.info(f"Model saved: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseDetector':
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        logger.logger.info(f"Model loaded: {path}")
        return model
    
    def get_metadata(self) -> Dict:
        return self._metadata
    
    def update_metadata(self, **kwargs) -> None:
        self._metadata.update(kwargs)
        self._metadata["updated"] = datetime.now().isoformat()