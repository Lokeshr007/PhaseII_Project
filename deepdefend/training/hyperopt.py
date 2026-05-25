"""
Hyperparameter optimization for imbalanced intrusion detection.

Tunes thresholds and model parameters to maximize minority class recall
while staying within the false positive budget.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from itertools import product
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class ThresholdOptimizer:
    """
    Optimize per-class decision thresholds.
    
    Problem: Default 0.5 threshold fails for imbalanced classes.
    A model might predict U2R with 0.3 probability, which is actually
    high for such a rare class. We need per-class optimal thresholds.
    
    Solution: Grid search over thresholds to maximize F-beta score
    with beta heavily favoring recall.
    """
    
    def __init__(self,
                 beta: float = 2.0,
                 target_recall: Dict[str, float] = None,
                 max_fpr: float = 0.05):
        """
        Args:
            beta: F-beta score beta (2 = recall twice as important as precision)
            target_recall: Minimum recall per class
            max_fpr: Maximum acceptable false positive rate
        """
        self.beta = beta
        self.target_recall = target_recall or {
            "dos": 0.95, "probe": 0.90,
            "r2l": 0.85, "u2r": 0.80
        }
        self.max_fpr = max_fpr
    
    def optimize(self,
                ensemble,
                X_val: np.ndarray,
                y_val: np.ndarray) -> Dict[str, float]:
        """
        Find optimal thresholds per class.
        
        Returns:
            Dictionary of class -> optimal threshold
        """
        optimal_thresholds = {}
        
        for attack_class in ensemble.attack_classes:
            if attack_class not in ensemble.specialists:
                continue
            
            specialist = ensemble.specialists[attack_class]
            probas = specialist.predict_proba(X_val)[:, 1]
            y_binary = (y_val == attack_class).astype(int)
            
            if y_binary.sum() == 0:
                logger.logger.warning(f"No {attack_class} samples in validation set")
                optimal_thresholds[attack_class] = 0.5
                continue
            
            best_threshold = 0.5
            best_score = 0.0
            
            for threshold in np.arange(0.05, 0.95, 0.025):
                preds = (probas >= threshold).astype(int)
                
                tp = ((preds == 1) & (y_binary == 1)).sum()
                fp = ((preds == 1) & (y_binary == 0)).sum()
                fn = ((preds == 0) & (y_binary == 1)).sum()
                
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                fpr = fp / (y_binary == 0).sum() if (y_binary == 0).sum() > 0 else 1
                
                # F-beta score
                if precision + recall > 0:
                    f_beta = (1 + self.beta**2) * (precision * recall) / \
                             (self.beta**2 * precision + recall)
                else:
                    f_beta = 0
                
                target_recall = self.target_recall.get(attack_class, 0.8)
                
                if recall >= target_recall and fpr <= self.max_fpr:
                    if f_beta > best_score:
                        best_score = f_beta
                        best_threshold = threshold
            
            optimal_thresholds[attack_class] = float(best_threshold)
            logger.logger.info(
                f"Optimal threshold for {attack_class}: {best_threshold:.3f} "
                f"(F{self.beta}: {best_score:.3f})"
            )
        
        return optimal_thresholds


class GridSearchCV:
    """
    Simple grid search for XGBoost hyperparameters.
    
    Focuses on parameters that matter for imbalanced learning:
    - scale_pos_weight (handled per-class by ensemble)
    - max_depth
    - min_child_weight
    - reg_alpha, reg_lambda
    """
    
    def __init__(self,
                param_grid: Dict[str, List],
                scoring: Callable = None,
                cv_folds: int = 3):
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv_folds = cv_folds
        
        self.best_params_: Dict = {}
        self.best_score_: float = 0.0
        self.cv_results_: Dict = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs):
        """Run grid search."""
        from sklearn.model_selection import StratifiedKFold
        
        param_combinations = list(self._iter_params())
        logger.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        results = []
        
        for params in param_combinations:
            fold_scores = []
            
            skf = StratifiedKFold(
                n_splits=min(self.cv_folds, min(Counter(y).values())),
                shuffle=True,
                random_state=42
            )
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                score = self._evaluate_params(params, X_train, y_train, X_val, y_val, **fit_kwargs)
                fold_scores.append(score)
            
            mean_score = np.mean(fold_scores)
            results.append({"params": params, "score": mean_score, "std": np.std(fold_scores)})
            
            logger.logger.debug(f"Params: {params} -> Score: {mean_score:.4f}")
        
        # Find best
        best = max(results, key=lambda r: r["score"])
        self.best_params_ = best["params"]
        self.best_score_ = best["score"]
        self.cv_results_ = {"results": results}
        
        logger.logger.info(f"Best params: {self.best_params_}, Score: {self.best_score_:.4f}")
        
        return self
    
    def _iter_params(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for combination in product(*values):
            yield dict(zip(keys, combination))
    
    def _evaluate_params(self, params, X_train, y_train, X_val, y_val, **kwargs):
        """Evaluate one parameter combination."""
        import xgboost as xgb
        
        model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='aucpr')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        
        if self.scoring:
            return self.scoring(y_val, y_pred)
        
        # Default: weighted F2 score
        from sklearn.metrics import fbeta_score
        return fbeta_score(y_val, y_pred, beta=2, average='weighted', zero_division=0)