"""
Imbalance handling strategies for minority attack classes.

This is where we fight the 52-sample U2R problem. Multiple strategies,
each appropriate for different attack types.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.utils.class_weight import compute_class_weight
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class ImbalanceHandler:
    """
    Multi-strategy imbalance handler.
    
    Different attack types need different approaches:
    - U2R (52 samples): SMOTE is dangerous — 52 samples may not capture 
      the true distribution. BorderlineSMOTE is more conservative.
    - R2L (995 samples): SMOTE + Tomek links works well.
    - DoS/Probe: Typically don't need oversampling, just class weights.
    
    Strategy selection is per-class, not global.
    """
    
    STRATEGIES = {
        "smote": SMOTE,
        "adasyn": ADASYN,
        "borderline_smote": BorderlineSMOTE,
        "svm_smote": SVMSMOTE,
        "smote_enn": SMOTEENN,  # SMOTE + cleaning
        "smote_tomek": SMOTETomek,  # SMOTE + cleaning
    }
    
    def __init__(self, 
                 strategy: str = "smote_tomek",
                 sampling_strategy: Union[str, Dict] = "auto",
                 random_state: int = 42,
                 k_neighbors: int = 5):
        """
        Args:
            strategy: Resampling strategy name
            sampling_strategy: 
                - 'auto': oversample minority to 50% of majority
                - dict: specific counts per class
                - float: target ratio relative to majority
            random_state: For reproducibility
            k_neighbors: SMOTE k-neighbors
        """
        self.strategy_name = strategy
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        
        self._resampler = None
        self._fitted = False
        
        self.before_counts: Dict = {}
        self.after_counts: Dict = {}
    
    def _get_sampling_strategy(self, y: np.ndarray) -> Dict:
        """
        Compute sensible sampling targets.
        
        Instead of 'auto' which oversamples all minorities to match majority,
        use a tapered approach:
        - DoS: 80% of normal (already close)
        - Probe: 50% of normal
        - R2L: 25% of normal  
        - U2R: 10% of normal (limited by real samples)
        """
        from collections import Counter
        counts = Counter(y)
        majority_count = max(counts.values())
        
        targets = {
            'normal': majority_count,
            'dos': min(counts.get('dos', 0), int(majority_count * 0.8)),
            'probe': min(counts.get('probe', 0), int(majority_count * 0.5)),
            'r2l': min(counts.get('r2l', 0), int(majority_count * 0.2)),
            'u2r': min(counts.get('u2r', 0), int(majority_count * 0.1)),
        }
        
        # Ensure we don't try to oversample beyond what SMOTE can do
        # Cap U2R at 10x its original count (from 42 to max 420)
        if 'u2r' in counts and counts['u2r'] > 0:
            targets['u2r'] = min(targets['u2r'], counts['u2r'] * 10)
        if 'r2l' in counts and counts['r2l'] > 0:
            targets['r2l'] = min(targets['r2l'], counts['r2l'] * 8)
        
        # Remove classes with 0 count
        targets = {k: v for k, v in targets.items() if v > 0 and k in counts}
        
        logger.logger.info(f"Sampling targets: {targets}")
        return targets
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply resampling with sensible targets.
        """
        self.before_counts = dict(Counter(y))
        
        # Validate k_neighbors
        counts = Counter(y)
        min_minority = min(
            count for label, count in counts.items() 
            if label not in ['normal', 'dos']
        )
        max_k = max(1, min_minority - 1)
        k = min(self.k_neighbors, max_k)
        
        if k < 2:
            logger.logger.warning(f"k_neighbors too small ({k}), falling back to class weights only")
            self.after_counts = self.before_counts
            self._fitted = True
            return X, y
        
        # Get sensible sampling targets
        if self.sampling_strategy == "auto":
            sampling_dict = self._get_sampling_strategy(y)
        else:
            sampling_dict = self.sampling_strategy
        
        # Create resampler
        resampler_class = self.STRATEGIES.get(self.strategy_name)
        if resampler_class is None:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
        
        try:
            if self.strategy_name in ["smote", "borderline_smote", "svm_smote"]:
                self._resampler = resampler_class(
                    sampling_strategy=sampling_dict,
                    random_state=self.random_state,
                    k_neighbors=k
                )
            else:
                self._resampler = resampler_class(
                    sampling_strategy=sampling_dict,
                    random_state=self.random_state
                )
            
            X_resampled, y_resampled = self._resampler.fit_resample(X, y)
        except Exception as e:
            logger.logger.warning(f"Resampling failed ({e}), using original data with class weights")
            self.after_counts = self.before_counts
            self._fitted = True
            return X, y
        
        self.after_counts = dict(Counter(y_resampled))
        self._fitted = True
        
        # Log the effect
        logger.logger.info("Resampling complete:")
        for cls in sorted(self.before_counts.keys()):
            before = self.before_counts[cls]
            after = self.after_counts.get(cls, 0)
            change = after - before
            sign = "+" if change > 0 else ""
            logger.logger.info(f"  {cls}: {before:,} → {after:,} ({sign}{change:,})")
        
        return X_resampled, y_resampled
    
    def get_class_weights(self, y: np.ndarray, 
                          custom_weights: Optional[Dict[str, float]] = None) -> Dict[int, float]:
        """
        Compute class weights for cost-sensitive learning.
        
        Uses inverse frequency by default, or custom weights if provided.
        
        XGBoost expects scale_pos_weight for binary classification,
        but for multi-class with sample_weight, we compute per-sample weights.
        """
        unique_classes = np.unique(y)
        
        if custom_weights:
            # Map string class names to integer labels
            return {
                i: custom_weights.get(cls, 1.0)
                for i, cls in enumerate(unique_classes)
            }
        
        # Balanced class weights (inverse frequency)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y
        )
        
        return dict(zip(range(len(unique_classes)), weights))
    
    def get_sample_weights(self, y: np.ndarray, 
                          custom_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Get per-sample weights for training.
        
        Incorporates the cost-sensitive weights from configuration:
        - U2R: 50x weight
        - R2L: 25x weight
        - Probe: 10x weight
        - DoS: 5x weight
        - Normal: 1x weight
        """
        if custom_weights is None:
            custom_weights = {
                "normal": 1.0,
                "dos": 5.0,
                "probe": 10.0,
                "r2l": 25.0,
                "u2r": 50.0
            }
        
        unique_classes = np.unique(y)
        class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        
        weights = np.ones(len(y))
        for cls_name, weight in custom_weights.items():
            if cls_name in class_to_idx:
                mask = y == cls_name
                weights[mask] = weight
        
        return weights
    
    def save(self, path: str) -> None:
        """Save resampling metadata."""
        import json
        metadata = {
            "strategy": self.strategy_name,
            "before_counts": self.before_counts,
            "after_counts": self.after_counts,
            "k_neighbors": self.k_neighbors
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)