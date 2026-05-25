#!/usr/bin/env python3
"""
Training script for DeepDefend NIDS.
Trains the full cascaded architecture: anomaly detector → specialists → calibrator.

Usage:
    python scripts/train.py --config config/production.yaml
    python scripts/train.py --mode development --epochs 100
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime

from deepdefend.config.settings import get_config, ConfigLoader
from deepdefend.data.loader import NSLKDLoader
from deepdefend.data.preprocessor import FeaturePreprocessor
from deepdefend.data.augment import ImbalanceHandler
from deepdefend.models.isolation_forest import IsolationForestAnomalyDetector
from deepdefend.models.xgboost_ensemble import XGBoostEnsemble
from deepdefend.models.calibrator import ProbabilityCalibrator
from deepdefend.training.metrics import ImbalanceMetrics
from deepdefend.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def train(args):
    """Full training pipeline."""
    config = ConfigLoader.load(mode=args.mode)
    setup_logging(level=config.log_level)
    
    logger.logger.info("=" * 60)
    logger.logger.info("DEEPDEFEND TRAINING PIPELINE")
    logger.logger.info(f"Mode: {config.mode}")
    logger.logger.info(f"Model path: {config.model_path}")
    logger.logger.info("=" * 60)
    
    # ─── 1. Load Data ─────────────────────────────────────────
    logger.logger.info("Step 1/6: Loading NSL-KDD dataset...")
    loader = NSLKDLoader(data_dir=config.data_path)
    dataset = loader.load_all()
    
    logger.logger.info(f"Train: {dataset.train.shape}, Val: {dataset.val.shape}, Test: {dataset.test.shape}")
    
    # ─── 2. Preprocess ────────────────────────────────────────
    logger.logger.info("Step 2/6: Preprocessing features (RobustScaler)...")
    preprocessor = FeaturePreprocessor(
        scaling_method="robust",
        handle_missing="median",
        categorical_encoding="onehot"
    )
    
    X_train = preprocessor.fit_transform(dataset.train.X)
    X_val = preprocessor.transform(dataset.val.X)
    X_test = preprocessor.transform(dataset.test.X)
    
    y_train = dataset.train.y
    y_val = dataset.val.y
    y_test = dataset.test.y
    
    logger.logger.info(f"Features: {X_train.shape[1]} after preprocessing")
    
    # ─── 3. Handle Imbalance ─────────────────────────────────
    logger.logger.info("Step 3/6: Handling class imbalance...")
    handler = ImbalanceHandler(
        strategy="smote_tomek",
        sampling_strategy="auto",
        k_neighbors=3  # Conservative — U2R has only ~40 samples in train
    )
    
    # Apply resampling to training data only
    X_train_resampled, y_train_resampled = handler.fit_resample(X_train, y_train)
    
    # Get sample weights for cost-sensitive learning (compute AFTER resampling)
    sample_weights = handler.get_sample_weights(y_train_resampled)
    
    # ─── 4. Train Anomaly Detector ────────────────────────────
    logger.logger.info("Step 4/6: Training anomaly detector (Isolation Forest)...")
    anomaly_detector = IsolationForestAnomalyDetector(
        n_estimators=200,
        contamination=0.01,
        random_state=42
    )
    anomaly_detector.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate anomaly detector
    anomaly_scores = anomaly_detector.score_samples(X_val)
    y_val_binary = (y_val != 'normal').astype(int)
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    try:
        auroc = roc_auc_score(y_val_binary, anomaly_scores)
        auprc = average_precision_score(y_val_binary, anomaly_scores)
        logger.logger.info(f"Anomaly Detector — AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
    except:
        logger.logger.warning("Could not compute anomaly detector metrics")
    
    # ─── 5. Train Attack Specialists ──────────────────────────
    logger.logger.info("Step 5/6: Training per-class attack specialists...")
    ensemble = XGBoostEnsemble(
        attack_classes=["dos", "probe", "r2l", "u2r"],
        class_weights=config.imbalance.class_weights
    )
    
    # Train with eval set for early stopping (critical for U2R's 500 estimators)
    ensemble.fit(
        X_train_resampled, y_train_resampled,
        sample_weight=sample_weights,
        eval_set=(X_val, y_val)
    )
    
    # Tune thresholds on validation set
    logger.logger.info("Tuning per-class thresholds...")
    thresholds = ensemble.tune_thresholds(
        X_val, y_val,
        target_recall={"u2r": 0.85, "r2l": 0.85, "probe": 0.90, "dos": 0.95},
        max_fpr=0.05
    )
    
    # ─── 6. Calibrate ─────────────────────────────────────────
    logger.logger.info("Step 6/6: Calibrating probabilities...")
    calibrator = ProbabilityCalibrator(
        base_detector=ensemble,
        method="platt"
    )
    calibrator.fit(X_val, y_val)
    
    # ─── Evaluate ─────────────────────────────────────────────
    logger.logger.info("\n" + "=" * 60)
    logger.logger.info("EVALUATION ON TEST SET")
    logger.logger.info("=" * 60)
    
    # Predictions
    y_pred = ensemble.predict(X_test, thresholds=thresholds)
    y_proba = ensemble.predict_proba(X_test)
    
    class_names = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    metrics_calculator = ImbalanceMetrics(
        class_names=class_names,
        minority_classes=['u2r', 'r2l']
    )
    
    test_metrics = metrics_calculator.compute_all(y_test, y_pred, y_proba)
    logger.logger.info("\n" + metrics_calculator.summary(test_metrics))
    
    # Specific U2R analysis
    u2r_mask = y_test == 'u2r'
    if u2r_mask.any():
        u2r_pred = y_pred[u2r_mask]
        u2r_correct = (u2r_pred == 'u2r').sum()
        logger.logger.info(f"\nU2R Detection: {u2r_correct}/{u2r_mask.sum()} detected ({u2r_correct/u2r_mask.sum()*100:.1f}%)")
    
    # ─── Save Everything ──────────────────────────────────────
    logger.logger.info("\nSaving models and artifacts...")
    model_dir = Path(config.model_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save preprocessor
    preprocessor.save(model_dir / "preprocessor.pkl")
    
    # Save models
    anomaly_detector.save(model_dir / "anomaly_detector.pkl")
    ensemble.save(model_dir / "attack_ensemble.pkl")
    calibrator.save(model_dir / "calibrator.pkl")
    
    # Save thresholds and metadata
    artifacts = {
        "training_date": datetime.now().isoformat(),
        "config_mode": config.mode,
        "class_names": class_names,
        "thresholds": thresholds,
        "test_metrics": {k: v for k, v in test_metrics.items() 
                        if not isinstance(v, (list, dict)) or k == 'per_class'},
        "feature_count": X_train.shape[1],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "preprocessing": {
            "method": "robust",
            "categorical_encoding": "onehot",
            "feature_names": preprocessor.feature_names_out,
        }
    }
    
    with open(model_dir / "artifacts.json", 'w') as f:
        json.dump(artifacts, f, indent=2, default=str)
    
    logger.logger.info(f"All artifacts saved to {model_dir}")
    
    # ─── Final Summary ────────────────────────────────────────
    logger.logger.info("\n" + "=" * 60)
    logger.logger.info("TRAINING COMPLETE")
    logger.logger.info("=" * 60)
    logger.logger.info(f"U2R Recall: {test_metrics['per_class'].get('u2r', {}).get('recall', 'N/A')}")
    logger.logger.info(f"R2L Recall: {test_metrics['per_class'].get('r2l', {}).get('recall', 'N/A')}")
    logger.logger.info(f"Attack Detection Rate: {test_metrics['attack_detection_rate']:.4f}")
    logger.logger.info(f"False Positive Rate: {test_metrics['false_positive_rate_overall']:.4f}")
    logger.logger.info(f"Models saved to: {model_dir}")
    logger.logger.info("=" * 60)
    
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepDefend NIDS")
    parser.add_argument("--mode", default="development", 
                       choices=["development", "production", "testing"])
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Training epochs (for autoencoder)")
    
    args = parser.parse_args()
    train(args)