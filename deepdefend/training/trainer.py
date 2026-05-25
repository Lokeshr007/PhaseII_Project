"""
Training orchestrator for DeepDefend NIDS.

Coordinates the full training pipeline:
1. Data loading with imbalance analysis
2. Preprocessing with serialization
3. Imbalance handling (SMOTE + cost-sensitive weights)
4. Anomaly detector training
5. Per-class specialist training
6. Calibration
7. Evaluation with proper metrics (not accuracy)
8. Model serialization for production deployment
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import numpy as np

from deepdefend.config.settings import get_config
from deepdefend.data.loader import NSLKDLoader
from deepdefend.data.preprocessor import FeaturePreprocessor
from deepdefend.data.augment import ImbalanceHandler
from deepdefend.models.isolation_forest import IsolationForestAnomalyDetector
from deepdefend.models.xgboost_ensemble import XGBoostEnsemble
from deepdefend.models.calibrator import ProbabilityCalibrator
from deepdefend.training.metrics import ImbalanceMetrics
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Orchestrates the complete training pipeline.
    
    Usage:
        trainer = Trainer(config)
        trainer.run()
    """
    
    def __init__(self, config=None, mode: str = "development"):
        self.config = config or get_config()
        self.mode = mode
        
        self.preprocessor: Optional[FeaturePreprocessor] = None
        self.anomaly_detector: Optional[IsolationForestAnomalyDetector] = None
        self.ensemble: Optional[XGBoostEnsemble] = None
        self.calibrator: Optional[ProbabilityCalibrator] = None
        
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
    
    def run(self) -> Dict:
        """Execute full training pipeline."""
        start_time = time.time()
        
        logger.logger.info("=" * 60)
        logger.logger.info("DEEPDEFEND TRAINING PIPELINE STARTING")
        logger.logger.info(f"Mode: {self.mode}")
        logger.logger.info("=" * 60)
        
        # Step 1: Load data
        logger.logger.info("[1/7] Loading NSL-KDD dataset...")
        dataset = self._load_data()
        
        # Step 2: Preprocess
        logger.logger.info("[2/7] Preprocessing features...")
        X_train, X_val, X_test, y_train, y_val, y_test = self._preprocess(dataset)
        
        # Step 3: Handle imbalance
        logger.logger.info("[3/7] Handling class imbalance...")
        X_train_resampled, y_train_resampled, sample_weights = self._handle_imbalance(
            X_train, y_train
        )
        
        # Step 4: Train anomaly detector
        logger.logger.info("[4/7] Training anomaly detector...")
        self._train_anomaly_detector(X_train_resampled, y_train_resampled, X_val, y_val)
        
        # Step 5: Train attack classifiers
        logger.logger.info("[5/7] Training attack specialists...")
        self._train_ensemble(X_train_resampled, y_train_resampled, sample_weights, X_val, y_val)
        
        # Step 6: Calibrate
        logger.logger.info("[6/7] Calibrating probabilities...")
        self._train_calibrator(X_val, y_val)
        
        # Step 7: Evaluate
        logger.logger.info("[7/7] Evaluating on test set...")
        self._evaluate(X_test, y_test)
        
        # Save everything
        self._save_artifacts()
        
        elapsed = time.time() - start_time
        logger.logger.info(f"Training complete in {elapsed:.0f}s")
        
        return self.test_metrics
    
    def _load_data(self):
        loader = NSLKDLoader(data_dir=self.config.data_path)
        return loader.load_all()
    
    def _preprocess(self, dataset):
        self.preprocessor = FeaturePreprocessor(
            scaling_method="robust",
            handle_missing="median",
            categorical_encoding="onehot"
        )
        
        X_train = self.preprocessor.fit_transform(dataset.train.X)
        X_val = self.preprocessor.transform(dataset.val.X)
        X_test = self.preprocessor.transform(dataset.test.X)
        
        logger.logger.info(f"Features: {X_train.shape[1]} after preprocessing")
        logger.logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, dataset.train.y, dataset.val.y, dataset.test.y
    
    def _handle_imbalance(self, X_train, y_train):
        handler = ImbalanceHandler(
            strategy=self.config.imbalance.oversampling_method,
            k_neighbors=3
        )
        
        sample_weights = handler.get_sample_weights(
            y_train,
            custom_weights=self.config.imbalance.class_weights
        )
        
        X_resampled, y_resampled = handler.fit_resample(X_train, y_train)
        
        return X_resampled, y_resampled, sample_weights
    
    def _train_anomaly_detector(self, X_train, y_train, X_val, y_val):
        self.anomaly_detector = IsolationForestAnomalyDetector(
            n_estimators=200,
            contamination=0.01,
            random_state=42
        )
        self.anomaly_detector.fit(X_train, y_train)
        
        # Quick validation
        scores = self.anomaly_detector.score_samples(X_val)
        y_val_binary = (y_val != 'normal').astype(int)
        
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_val_binary, scores)
            logger.logger.info(f"Anomaly detector AUROC: {auc:.4f}")
        except Exception:
            pass
    
    def _train_ensemble(self, X_train, y_train, sample_weights, X_val, y_val):
        self.ensemble = XGBoostEnsemble(
            attack_classes=self.config.model.attack_classifiers,
            class_weights=self.config.imbalance.class_weights
        )
        
        self.ensemble.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=(X_val, y_val)
        )
        
        # Tune thresholds
        thresholds = self.ensemble.tune_thresholds(
            X_val, y_val,
            target_recall={"u2r": 0.85, "r2l": 0.85, "probe": 0.90, "dos": 0.95},
            max_fpr=0.05
        )
        logger.logger.info(f"Tuned thresholds: {thresholds}")
    
    def _train_calibrator(self, X_val, y_val):
        self.calibrator = ProbabilityCalibrator(
            base_detector=self.ensemble,
            method=self.config.model.calibration_method
        )
        self.calibrator.fit(X_val, y_val)
    
    def _evaluate(self, X_test, y_test):
        class_names = ['normal', 'dos', 'probe', 'r2l', 'u2r']
        metrics_calc = ImbalanceMetrics(
            class_names=class_names,
            minority_classes=['u2r', 'r2l']
        )
        
        y_pred = self.ensemble.predict(X_test)
        y_proba = self.ensemble.predict_proba(X_test)
        
        self.test_metrics = metrics_calc.compute_all(y_test, y_pred, y_proba)
        
        logger.logger.info("\n" + metrics_calc.summary(self.test_metrics))
        
        # Specific U2R report
        u2r_mask = y_test == 'u2r'
        if u2r_mask.any():
            u2r_correct = (y_pred[u2r_mask] == 'u2r').sum()
            logger.logger.info(
                f"U2R Detection: {u2r_correct}/{u2r_mask.sum()} "
                f"({u2r_correct/u2r_mask.sum()*100:.1f}%)"
            )
    
    def _save_artifacts(self):
        model_dir = Path(self.config.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor.save(model_dir / "preprocessor.pkl")
        self.anomaly_detector.save(model_dir / "anomaly_detector.pkl")
        self.ensemble.save(model_dir / "attack_ensemble.pkl")
        self.calibrator.save(model_dir / "calibrator.pkl")
        
        artifacts = {
            "training_date": datetime.now().isoformat(),
            "test_metrics": {
                k: v for k, v in self.test_metrics.items()
                if not isinstance(v, (list, dict))
            },
            "per_class_metrics": self.test_metrics.get("per_class", {}),
            "feature_count": len(self.preprocessor.feature_names_out)
                if self.preprocessor else 0,
        }
        
        with open(model_dir / "artifacts.json", 'w') as f:
            json.dump(artifacts, f, indent=2, default=str)
        
        logger.logger.info(f"Artifacts saved to {model_dir}")