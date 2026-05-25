"""
Stratified cross-validation for imbalanced network intrusion detection.

Standard k-fold CV can fail when minority classes have fewer samples
than folds. This module implements stratified splitting that guarantees
every fold has at least one sample of each minority class.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class MinorityAwareCV:
    """
    Cross-validation that ensures minority class representation.
    
    Problem: With U2R at 52 samples and 5-fold CV, some folds get 10 U2R,
    some get 11. Standard StratifiedKFold handles this. But with even fewer
    samples (e.g., specific U2R subtypes), standard CV fails.
    
    This wrapper:
    1. Checks if stratification is possible
    2. Falls back to grouped splitting if needed
    3. Reports per-fold class distribution
    4. Aggregates metrics properly across folds
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def validate_split_possible(self, y: np.ndarray) -> bool:
        """Check if stratification is possible with n_splits."""
        counts = Counter(y)
        min_count = min(counts.values())
        
        if min_count < self.n_splits:
            logger.logger.warning(
                f"Smallest class has {min_count} samples, "
                f"but {self.n_splits} folds requested. "
                f"Reducing to {min_count} folds."
            )
            return False
        return True
    
    def split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple]:
        """
        Generate stratified splits.
        
        Returns list of (train_idx, val_idx) tuples.
        """
        n_splits = min(self.n_splits, min(Counter(y).values()))
        
        if n_splits < 2:
            logger.logger.warning(
                "Cannot perform cross-validation with <2 splits. "
                "Using single train/validation split."
            )
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(
                range(len(X)), test_size=0.2, stratify=y,
                random_state=self.random_state
            )
            return [(train_idx, val_idx)]
        
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        splits = []
        for train_idx, val_idx in skf.split(X, y):
            splits.append((train_idx, val_idx))
            
            # Log distribution
            train_dist = Counter(y[train_idx])
            val_dist = Counter(y[val_idx])
            logger.logger.debug(f"Fold distribution — Train: {dict(train_dist)}, Val: {dict(val_dist)}")
        
        return splits
    
    def cross_validate(self, 
                      X: np.ndarray, 
                      y: np.ndarray,
                      train_fn,
                      eval_fn) -> Dict:
        """
        Run cross-validation with custom train/eval functions.
        
        Args:
            X: Feature matrix
            y: Labels
            train_fn: Function(train_X, train_y) -> model
            eval_fn: Function(model, val_X, val_y) -> metrics dict
        
        Returns:
            Aggregated metrics across all folds
        """
        splits = self.split(X, y)
        
        all_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.logger.info(f"Fold {fold_idx + 1}/{len(splits)}")
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            model = train_fn(X_train, y_train)
            metrics = eval_fn(model, X_val, y_val)
            metrics['fold'] = fold_idx
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics)
        aggregated['n_folds'] = len(splits)
        aggregated['fold_metrics'] = all_metrics
        
        return aggregated
    
    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across folds."""
        agg = {}
        
        # Find all numeric metrics
        numeric_keys = set()
        for metrics in all_metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    numeric_keys.add(key)
        
        for key in numeric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                agg[f"{key}_mean"] = float(np.mean(values))
                agg[f"{key}_std"] = float(np.std(values))
        
        return agg