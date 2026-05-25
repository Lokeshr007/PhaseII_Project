"""
PyTorch Dataset for network traffic data.
Supports both NSL-KDD training and live inference.
"""

from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class NetworkTrafficDataset(Dataset):
    """
    PyTorch Dataset for network traffic features.
    
    Used by both XGBoost (via numpy arrays) and PyTorch models (autoencoder).
    Supports sample weights for cost-sensitive training.
    """
    
    def __init__(self, 
                 X: np.ndarray, 
                 y: Optional[np.ndarray] = None,
                 sample_weights: Optional[np.ndarray] = None,
                 metadata: Optional[dict] = None):
        """
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) or None for inference
            sample_weights: Per-sample importance weights for training
            metadata: Additional information about the dataset
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        self.sample_weights = torch.FloatTensor(sample_weights) if sample_weights is not None else None
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple:
        if self.y is not None:
            if self.sample_weights is not None:
                return self.X[idx], self.y[idx], self.sample_weights[idx]
            return self.X[idx], self.y[idx]
        return self.X[idx]
    
    @property
    def feature_dim(self) -> int:
        return self.X.shape[1]
    
    @property
    def num_classes(self) -> int:
        if self.y is not None:
            return len(torch.unique(self.y))
        return 0
    
    def numpy(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return as numpy arrays for sklearn/xgboost compatibility."""
        X_np = self.X.numpy()
        y_np = self.y.numpy() if self.y is not None else None
        return X_np, y_np
    
    def class_distribution(self) -> dict:
        """Return class distribution."""
        if self.y is None:
            return {}
        unique, counts = torch.unique(self.y, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}