"""
Feature preprocessing pipeline with strict training/production parity.

Critical constraint: whatever transformation is applied during training MUST
be applied identically during inference. A single scaling difference can make
a U2R attack look like normal traffic or vice versa.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, OneHotEncoder, 
    MinMaxScaler, RobustScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from deepdefend.utils.logging import get_logger
from deepdefend.data.loader import NSL_KDD_COLUMNS

logger = get_logger(__name__)


class FeaturePreprocessor:
    """
    Preprocessor for NSL-KDD features with serialization.
    
    Handles:
    - Categorical encoding (protocol_type, service, flag)
    - Numerical scaling (RobustScaler preferred — handles outliers from attacks)
    - Missing value handling (attacks often have unusual feature values)
    - Feature name preservation for debugging
    - Strict serialization for training→production parity
    
    Uses RobustScaler instead of StandardScaler because:
    - Attack traffic often has extreme values (outliers)
    - RobustScaler uses median/IQR, not affected by extreme port scan byte counts
    - This matters enormously for U2R detection where unusual values ARE the signal
    """
    
    def __init__(self, 
                 scaling_method: str = "robust",
                 handle_missing: str = "median",
                 categorical_encoding: str = "onehot"):
        """
        Args:
            scaling_method: 'standard', 'minmax', or 'robust'
            handle_missing: 'median', 'mean', 'zero', or 'drop'
            categorical_encoding: 'onehot' or 'label'
        """
        self.scaling_method = scaling_method
        self.handle_missing = handle_missing
        self.categorical_encoding = categorical_encoding
        
        self._fitted = False
        self._column_transformer: Optional[ColumnTransformer] = None
        self._categorical_columns = ["protocol_type", "service", "flag"]
        self._numerical_columns = [c for c in NSL_KDD_COLUMNS 
                                   if c not in self._categorical_columns]
        self.feature_names_out: List[str] = []
        
        # Store fit parameters for serialization
        self._scaler_params: Dict = {}
        self._encoder_params: Dict = {}
        self._label_encoders: Dict[str, LabelEncoder] = {}
    
    def fit(self, X: np.ndarray) -> 'FeaturePreprocessor':
        """Fit preprocessor on training data."""
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=NSL_KDD_COLUMNS)
        
        # Separate numerical and categorical
        X_num = X[self._numerical_columns].copy()
        X_cat = X[self._categorical_columns].copy()
        
        # Handle missing values in numerical
        self._num_missing_values = {}
        for col in self._numerical_columns:
            missing_mask = X_num[col].isna()
            if missing_mask.any():
                if self.handle_missing == "median":
                    fill_val = X_num[col].median()
                elif self.handle_missing == "mean":
                    fill_val = X_num[col].mean()
                else:
                    fill_val = 0.0
                self._num_missing_values[col] = fill_val
                X_num[col] = X_num[col].fillna(fill_val)
        
        # Fit scaler
        if self.scaling_method == "robust":
            self._scaler = RobustScaler(quantile_range=(5.0, 95.0))
        elif self.scaling_method == "standard":
            self._scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            self._scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        self._scaler.fit(X_num)
        self._scaler_params = {
            "type": self.scaling_method,
            "center_": self._scaler.center_.tolist() if hasattr(self._scaler, 'center_') else None,
            "scale_": self._scaler.scale_.tolist() if hasattr(self._scaler, 'scale_') else None,
        }
        
        # Fit categorical encoders
        if self.categorical_encoding == "onehot":
            self._encoder = OneHotEncoder(
                sparse_output=False, 
                handle_unknown='ignore'  # Critical: handles new categories in production
            )
            self._encoder.fit(X_cat.astype(str))
            
            # Reconstruct feature names
            cat_features = []
            for i, col in enumerate(self._categorical_columns):
                categories = self._encoder.categories_[i]
                cat_features.extend([f"{col}_{cat}" for cat in categories])
            
            self.feature_names_out = self._numerical_columns + cat_features
            
        elif self.categorical_encoding == "label":
            self._label_encoders = {}
            for col in self._categorical_columns:
                le = LabelEncoder()
                le.fit(X_cat[col].astype(str))
                self._label_encoders[col] = le
            self.feature_names_out = self._numerical_columns + self._categorical_columns
        
        self._fitted = True
        logger.logger.info(
            f"Preprocessor fitted: {len(self.feature_names_out)} features "
            f"({len(self._numerical_columns)} numerical + "
            f"{len(self.feature_names_out) - len(self._numerical_columns)} categorical)"
        )
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=NSL_KDD_COLUMNS)
        
        # Handle missing values
        X_num = X[self._numerical_columns].copy()
        for col, fill_val in self._num_missing_values.items():
            if col in X_num.columns:
                X_num[col] = X_num[col].fillna(fill_val)
        
        X_cat = X[self._categorical_columns].copy()
        
        # Scale numerical
        X_num_scaled = self._scaler.transform(X_num)
        
        # Encode categorical
        if self.categorical_encoding == "onehot":
            X_cat_encoded = self._encoder.transform(X_cat.astype(str))
            X_out = np.hstack([X_num_scaled, X_cat_encoded])
        else:
            X_cat_encoded = np.column_stack([
                self._label_encoders[col].transform(X_cat[col].astype(str))
                for col in self._categorical_columns
            ])
            X_out = np.hstack([X_num_scaled, X_cat_encoded])
        
        return X_out
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform_features(self, X_transformed: np.ndarray) -> pd.DataFrame:
        """Convert transformed features back to interpretable values for forensics."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted.")
        
        n_num = len(self._numerical_columns)
        X_num_scaled = X_transformed[:, :n_num]
        X_num = pd.DataFrame(
            self._scaler.inverse_transform(X_num_scaled),
            columns=self._numerical_columns
        )
        
        # For categorical, find the highest probability/encoding
        if self.categorical_encoding == "onehot":
            X_cat_data = X_transformed[:, n_num:]
            cat_offset = 0
            cat_cols = {}
            for i, col in enumerate(self._categorical_columns):
                n_cats = len(self._encoder.categories_[i])
                cat_slice = X_cat_data[:, cat_offset:cat_offset + n_cats]
                max_idx = np.argmax(cat_slice, axis=1)
                cat_cols[col] = self._encoder.categories_[i][max_idx]
                cat_offset += n_cats
            X_cat = pd.DataFrame(cat_cols)
        else:
            X_cat = pd.DataFrame(X_transformed[:, n_num:], columns=self._categorical_columns)
        
        return pd.concat([X_num, X_cat], axis=1)
    
    def save(self, path: str) -> None:
        """Serialize preprocessor for production deployment."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "scaling_method": self.scaling_method,
            "handle_missing": self.handle_missing,
            "categorical_encoding": self.categorical_encoding,
            "scaler": pickle.dumps(self._scaler),
            "encoder": pickle.dumps(self._encoder) if self.categorical_encoding == "onehot" else None,
            "label_encoders": {k: pickle.dumps(v) for k, v in self._label_encoders.items()},
            "feature_names_out": self.feature_names_out,
            "numerical_columns": self._numerical_columns,
            "categorical_columns": self._categorical_columns,
            "num_missing_values": self._num_missing_values,
            "scaler_params": self._scaler_params,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeaturePreprocessor':
        """Load preprocessor from serialized state."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {path}")
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            scaling_method=state["scaling_method"],
            handle_missing=state["handle_missing"],
            categorical_encoding=state["categorical_encoding"]
        )
        
        preprocessor._scaler = pickle.loads(state["scaler"])
        preprocessor._num_missing_values = state["num_missing_values"]
        preprocessor._categorical_columns = state["categorical_columns"]
        preprocessor._numerical_columns = state["numerical_columns"]
        preprocessor.feature_names_out = state["feature_names_out"]
        preprocessor._scaler_params = state["scaler_params"]
        
        if state["encoder"]:
            preprocessor._encoder = pickle.loads(state["encoder"])
            preprocessor._label_encoders = {}
        else:
            preprocessor._label_encoders = {
                k: pickle.loads(v) for k, v in state["label_encoders"].items()
            }
        
        preprocessor._fitted = True
        logger.logger.info(f"Preprocessor loaded from {path}")
        
        return preprocessor