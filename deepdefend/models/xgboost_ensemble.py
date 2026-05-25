"""
XGBoost ensemble with per-class specialist classifiers.

Architecture:
- One binary XGBoost classifier per attack class (DoS, Probe, R2L, U2R)
- Each specialist is trained: "Is this {attack_class} or not?"
- Specialists are ONLY invoked on samples flagged by anomaly detector
- This cascaded architecture means specialists see fewer false positives

Why per-class specialists instead of multi-class:
1. Each class has vastly different sample counts and feature patterns
2. U2R classifier gets dedicated hyperparameter tuning (deeper trees, higher weight)
3. A new attack class can be added by training one new specialist
4. Each specialist's threshold can be independently tuned for its class's cost
5. If U2R specialist is uncertain, we can fall back to other specialists

Training strategy per specialist:
- DoS: Large dataset, can use standard parameters
- Probe: Moderate dataset, slightly higher regularization
- R2L: Small dataset, aggressive regularization, higher class weight
- U2R: Tiny dataset (52 samples), deep trees with extreme class weight,
  heavy regularization to prevent overfitting on 52 samples
"""

import numpy as np
import xgboost as xgb
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from deepdefend.models.base import BaseDetector
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class AttackSpecialist:
    """Binary classifier specialized in one attack type."""
    
    def __init__(self, 
                 attack_class: str,
                 params: Dict,
                 class_weight: float = 1.0):
        self.attack_class = attack_class
        self.class_weight = class_weight
        self.model: Optional[xgb.XGBClassifier] = None
        self._params = params.copy()
        self._fitted = False
        self._threshold: float = 0.5
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[list] = None,
            early_stopping_rounds: Optional[int] = None):
        """Train binary classifier: 1 for this attack, 0 for everything else."""
        # Binary labels
        y_binary = (y == self.attack_class).astype(int)
        
        # Check we have both classes
        unique = np.unique(y_binary)
        if len(unique) < 2:
            logger.logger.warning(
                f"No {self.attack_class} samples in training data. "
                f"Specialist will always predict 0."
            )
            self._fitted = True
            return self
        
        # Adjust scale_pos_weight for imbalance
        n_negative = (y_binary == 0).sum()
        n_positive = (y_binary == 1).sum()
        
        if n_positive > 0:
            scale_pos_weight = (n_negative / n_positive) * self.class_weight
        else:
            scale_pos_weight = 1.0
        
        params = self._params.copy()
        params['scale_pos_weight'] = scale_pos_weight
        if early_stopping_rounds is not None:
            params['early_stopping_rounds'] = early_stopping_rounds
        
        self.model = xgb.XGBClassifier(**params)
        
        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs['eval_set'] = eval_set
        fit_kwargs['verbose'] = False

        # Fit with sample weights if provided
        if sample_weight is not None:
            # Adjust sample weights for this binary task
            task_weights = sample_weight.copy()
            task_weights[y_binary == 1] *= self.class_weight
            self.model.fit(X, y_binary, sample_weight=task_weights, **fit_kwargs)
        else:
            self.model.fit(X, y_binary, **fit_kwargs)
        
        self._fitted = True
        
        logger.logger.info(
            f"AttackSpecialist({self.attack_class}) fitted. "
            f"Positives: {n_positive}, Negatives: {n_negative}, "
            f"scale_pos_weight: {scale_pos_weight:.1f}"
        )
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of this attack class."""
        if not self._fitted:
            return np.zeros((len(X), 2))
        if self.model is None:
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])
        return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Binary prediction with configurable threshold."""
        threshold = threshold or self._threshold
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def set_threshold(self, threshold: float):
        self._threshold = threshold


class XGBoostEnsemble(BaseDetector):
    """
    Ensemble of per-class XGBoost specialists.
    
    Each specialist answers: "Is this {attack_class}?"
    Final prediction: class with highest probability, or "normal" if all low.
    
    Key design decisions:
    - Separate specialist per class allows per-class hyperparameter tuning
    - U2R specialist uses deeper trees and higher weight
    - Scale_pos_weight computed per specialist based on its class's imbalance
    - Thresholds independently tuned per class
    """
    
    # Per-class parameter templates
    CLASS_PARAMS = {
        "dos": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "aucpr",
            "use_label_encoder": False,
        },
        "probe": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "aucpr",
            "use_label_encoder": False,
        },
        "r2l": {
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,  # More regularization for small dataset
            "reg_lambda": 2.0,
            "min_child_weight": 3,
            "eval_metric": "aucpr",
            "use_label_encoder": False,
        },
        "u2r": {
            "max_depth": 10,  # Deeper trees — need to capture subtle patterns
            "learning_rate": 0.01,  # Slower learning on tiny dataset
            "n_estimators": 500,  # More trees with early stopping
            "subsample": 0.6,  # Aggressive subsampling to prevent overfitting
            "colsample_bytree": 0.6,
            "reg_alpha": 2.0,  # Heavy regularization
            "reg_lambda": 5.0,
            "min_child_weight": 5,  # Prevent leaf nodes on single samples
            "eval_metric": "aucpr",
            "use_label_encoder": False,
            "early_stopping_rounds": 50,
        },
    }
    
    # Cost weights per class (from config)
    DEFAULT_CLASS_WEIGHTS = {
        "dos": 5.0,
        "probe": 10.0,
        "r2l": 25.0,
        "u2r": 50.0,
    }
    
    def __init__(self,
                 attack_classes: List[str] = None,
                 class_params: Optional[Dict] = None,
                 class_weights: Optional[Dict[str, float]] = None,
                 random_state: int = 42,
                 name: str = "xgboost_ensemble"):
        
        super().__init__(name=name, model_type="attack_classifier")
        
        self.attack_classes = attack_classes or ["dos", "probe", "r2l", "u2r"]
        self.class_params = class_params or self.CLASS_PARAMS
        self.class_weights = class_weights or self.DEFAULT_CLASS_WEIGHTS
        self.random_state = random_state
        
        self.specialists: Dict[str, AttackSpecialist] = {}
        self._class_thresholds: Dict[str, float] = {}
        
        # Normal threshold — if max probability below this, predict normal
        self._normal_threshold: float = 0.5
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'XGBoostEnsemble':
        """
        Train all specialists.
        
        Each specialist gets:
        - Class-specific hyperparameters (U2R gets very different params from DoS)
        - Class weight adjusted for its imbalance ratio
        - Optional eval_set for early stopping
        """
        for attack_class in self.attack_classes:
            params = self.class_params.get(attack_class, self.class_params["dos"]).copy()
            params['random_state'] = self.random_state
            
            eval_set_specialist = None
            early_stopping_rounds = None
            
            # Add early stopping if eval_set provided and params support it
            if eval_set is not None and 'early_stopping_rounds' in params:
                # Convert eval labels to binary for this specialist
                eval_y_binary = (eval_set[1] == attack_class).astype(int)
                early_stopping_rounds = params.pop('early_stopping_rounds')
                eval_set_specialist = [(eval_set[0], eval_y_binary)]
            
            specialist = AttackSpecialist(
                attack_class=attack_class,
                params=params,
                class_weight=self.class_weights.get(attack_class, 1.0)
            )
            
            specialist.fit(X, y, sample_weight, 
                           eval_set=eval_set_specialist,
                           early_stopping_rounds=early_stopping_rounds)
            self.specialists[attack_class] = specialist
            self._class_thresholds[attack_class] = 0.5
        
        self._fitted = True
        logger.logger.info(f"XGBoostEnsemble trained with {len(self.specialists)} specialists")
        
        return self
    
    def predict(self, X: np.ndarray, 
                thresholds: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Predict attack class.
        
        Logic:
        1. Get probability from each specialist
        2. Find class with highest probability
        3. If that probability > threshold, predict that class
        4. Otherwise, predict 'normal'
        """
        if not self._fitted:
            return np.array(['normal'] * len(X))
        
        thresholds = thresholds or self._class_thresholds
        n_samples = len(X)
        
        # Collect probabilities from all specialists
        all_probas = {}
        for attack_class, specialist in self.specialists.items():
            proba = specialist.predict_proba(X)[:, 1]  # Probability of being this attack
            all_probas[attack_class] = proba
        
        # Find best class per sample
        predictions = []
        for i in range(n_samples):
            best_class = 'normal'
            best_prob = self._normal_threshold
            
            for attack_class in self.attack_classes:
                prob = all_probas[attack_class][i]
                threshold = thresholds.get(attack_class, 0.5)
                if prob > threshold and prob > best_prob:
                    best_class = attack_class
                    best_prob = prob
            
            predictions.append(best_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Full probability distribution across all classes.
        
        Returns: (n_samples, n_classes + 1) array
        Columns: [normal, dos, probe, r2l, u2r] (order depends on attack_classes)
        """
        if not self._fitted:
            return np.column_stack([np.ones(len(X)), np.zeros((len(X), len(self.attack_classes)))])
        
        n_samples = len(X)
        n_classes = len(self.attack_classes) + 1  # +1 for normal
        
        proba_matrix = np.zeros((n_samples, n_classes))
        
        # Normal column (index 0): 1 - max(attack probabilities)
        all_attack_probas = []
        for i, attack_class in enumerate(self.attack_classes):
            proba = self.specialists[attack_class].predict_proba(X)[:, 1]
            proba_matrix[:, i + 1] = proba
            all_attack_probas.append(proba)
        
        # Normal probability = 1 - max attack probability
        max_attack = np.max(proba_matrix[:, 1:], axis=1)
        proba_matrix[:, 0] = 1.0 - max_attack
        
        return proba_matrix
        
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score (probability of being any attack)."""
        probas = self.predict_proba(X)
        return 1.0 - probas[:, 0]
        
    def tune_thresholds(self, X_val: np.ndarray, y_val: np.ndarray,
                        target_recall: Dict[str, float] = None,
                        max_fpr: float = 0.01) -> Dict[str, float]:
        """
        Tune per-class thresholds on validation set.
        
        Fix: Use broader search range and fallback to sensible defaults
        when no threshold meets all criteria.
        """
        if target_recall is None:
            target_recall = {cls: 0.8 for cls in self.attack_classes}
        
        tuned_thresholds = {}
        
        for attack_class in self.attack_classes:
            if attack_class not in self.specialists:
                tuned_thresholds[attack_class] = 0.3
                continue
            
            specialist = self.specialists[attack_class]
            probas = specialist.predict_proba(X_val)[:, 1]
            y_binary = (y_val == attack_class).astype(int)
            
            if y_binary.sum() == 0:
                logger.logger.warning(f"No {attack_class} samples in validation set")
                tuned_thresholds[attack_class] = 0.3
                continue
            
            best_threshold = 0.3  # Sensible default
            best_f1 = 0.0
            best_recall = 0.0
            
            # Search wider range with finer granularity
            for threshold in np.arange(0.05, 0.90, 0.02):
                preds = (probas >= threshold).astype(int)
                
                tp = ((preds == 1) & (y_binary == 1)).sum()
                fp = ((preds == 1) & (y_binary == 0)).sum()
                fn = ((preds == 0) & (y_binary == 1)).sum()
                
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                fpr = fp / (y_binary == 0).sum() if (y_binary == 0).sum() > 0 else 1
                
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Prioritize recall for minority classes, balance for majority
                target = target_recall.get(attack_class, 0.7)
                
                # Score: prioritize recall meeting target, then F1
                if recall >= target and fpr <= max_fpr:
                    if f1 > best_f1:
                        best_f1 = f1
                        best_recall = recall
                        best_threshold = threshold
                elif recall > best_recall and fpr <= max_fpr * 3:  # Relaxed FPR
                    if f1 > best_f1 * 0.8:  # Within 80% of best F1
                        best_recall = recall
                        best_threshold = threshold
            
            # If no threshold found meeting criteria, use a low default
            # This is better than missing all attacks
            if best_f1 == 0.0:
                # Find threshold that gives at least 50% recall
                for threshold in np.arange(0.05, 0.50, 0.02):
                    preds = (probas >= threshold).astype(int)
                    tp = ((preds == 1) & (y_binary == 1)).sum()
                    fn = ((preds == 0) & (y_binary == 1)).sum()
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    if recall >= 0.5:
                        best_threshold = threshold
                        break
                else:
                    best_threshold = 0.2  # Hard fallback
            
            tuned_thresholds[attack_class] = float(best_threshold)
            logger.logger.info(
                f"Tuned threshold for {attack_class}: {best_threshold:.4f} "
                f"(best F1: {best_f1:.4f}, recall: {best_recall:.4f})"
            )
        
        self._class_thresholds = tuned_thresholds
        return tuned_thresholds