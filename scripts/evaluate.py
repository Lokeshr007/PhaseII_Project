#!/usr/bin/env python3
"""
Model evaluation script for DeepDefend NIDS.

Evaluates trained models on test data and generates detailed reports.
Focuses on minority class performance — the metrics that actually matter.
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from deepdefend.config.settings import get_config
from deepdefend.data.loader import NSLKDLoader
from deepdefend.data.preprocessor import FeaturePreprocessor
from deepdefend.models.isolation_forest import IsolationForestAnomalyDetector
from deepdefend.models.xgboost_ensemble import XGBoostEnsemble
from deepdefend.models.calibrator import ProbabilityCalibrator
from deepdefend.training.metrics import ImbalanceMetrics
from deepdefend.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def evaluate(args):
    """Run evaluation pipeline."""
    setup_logging(level="INFO")
    
    config = get_config()
    model_path = Path(args.model_path or config.model_path)
    
    logger.logger.info("=" * 60)
    logger.logger.info("DEEPDEFEND MODEL EVALUATION")
    logger.logger.info("=" * 60)
    
    # ─── Load models ────────────────────────────────────────
    logger.logger.info("Loading models...")
    preprocessor = FeaturePreprocessor.load(model_path / "preprocessor.pkl")
    anomaly_detector = IsolationForestAnomalyDetector.load(model_path / "anomaly_detector.pkl")
    ensemble = XGBoostEnsemble.load(model_path / "attack_ensemble.pkl")
    calibrator = ProbabilityCalibrator.load(model_path / "calibrator.pkl")
    
    # ─── Load test data ──────────────────────────────────────
    logger.logger.info("Loading test data...")
    loader = NSLKDLoader(data_dir=config.data_path)
    dataset = loader.load_all()
    
    X_test = preprocessor.transform(dataset.test.X)
    y_test = dataset.test.y
    
    logger.logger.info(f"Test set: {X_test.shape}")
    
    # ─── Evaluate anomaly detector ───────────────────────────
    logger.logger.info("\n--- Anomaly Detector ---")
    anomaly_scores = anomaly_detector.score_samples(X_test)
    y_test_binary = (y_test != 'normal').astype(int)
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    try:
        auroc = roc_auc_score(y_test_binary, anomaly_scores)
        auprc = average_precision_score(y_test_binary, anomaly_scores)
        logger.logger.info(f"AUROC: {auroc:.4f}")
        logger.logger.info(f"AUPRC: {auprc:.4f}")
    except Exception as e:
        logger.logger.error(f"Anomaly detector evaluation error: {e}")
    
    # ─── Evaluate ensemble ───────────────────────────────────
    logger.logger.info("\n--- Attack Classifiers ---")
    
    class_names = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    metrics_calc = ImbalanceMetrics(
        class_names=class_names,
        minority_classes=['u2r', 'r2l']
    )
    
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)
    
    test_metrics = metrics_calc.compute_all(y_test, y_pred, y_proba)
    
    # Print summary
    print("\n" + metrics_calc.summary(test_metrics))
    
    # ─── Detailed U2R Analysis ───────────────────────────────
    u2r_mask = y_test == 'u2r'
    if u2r_mask.any():
        logger.logger.info("\n--- U2R Detailed Analysis ---")
        u2r_indices = np.where(u2r_mask)[0]
        u2r_predictions = y_pred[u2r_mask]
        
        for idx in u2r_indices[:10]:  # Show first 10
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            probas = y_proba[idx]
            confidence = probas[class_names.index(pred_label)] if pred_label in class_names else 0
            
            status = "✓" if true_label == pred_label else "✗ MISSED"
            logger.logger.info(
                f"  Sample {idx}: True={true_label}, Pred={pred_label}, "
                f"Conf={confidence:.3f} {status}"
            )
    
    # ─── R2L Detailed Analysis ───────────────────────────────
    r2l_mask = y_test == 'r2l'
    if r2l_mask.any():
        r2l_correct = (y_pred[r2l_mask] == 'r2l').sum()
        r2l_missed_as_normal = (y_pred[r2l_mask] == 'normal').sum()
        r2l_missed_as_other = r2l_mask.sum() - r2l_correct - r2l_missed_as_normal
        
        logger.logger.info("\n--- R2L Detailed Analysis ---")
        logger.logger.info(f"Total R2L: {r2l_mask.sum()}")
        logger.logger.info(f"Correct: {r2l_correct}")
        logger.logger.info(f"Missed as normal: {r2l_missed_as_normal}")
        logger.logger.info(f"Missed as other attack: {r2l_missed_as_other}")
    
    # ─── False Positive Analysis ──────────────────────────────
    logger.logger.info("\n--- False Positive Analysis ---")
    fp_mask = (y_test == 'normal') & (y_pred != 'normal')
    fp_count = fp_mask.sum()
    normal_count = (y_test == 'normal').sum()
    
    logger.logger.info(f"False positives: {fp_count} out of {normal_count} normal samples ({fp_count/normal_count*100:.4f}%)")
    
    if fp_count > 0:
        fp_predictions = y_pred[fp_mask]
        from collections import Counter
        fp_dist = Counter(fp_predictions)
        logger.logger.info("False positive distribution:")
        for cls, count in fp_dist.most_common():
            logger.logger.info(f"  Misclassified as {cls}: {count}")
    
    # ─── Save report ─────────────────────────────────────────
    report = {
        "test_metrics": {
            k: v for k, v in test_metrics.items()
            if not isinstance(v, (list, dict)) or k == 'per_class'
        },
        "anomaly_detector_auroc": float(auroc) if 'auroc' in locals() else None,
        "u2r_analysis": {
            "total": int(u2r_mask.sum()) if u2r_mask.any() else 0,
            "correct": int((y_pred[u2r_mask] == 'u2r').sum()) if u2r_mask.any() else 0,
        },
        "false_positives": int(fp_count),
        "false_positive_rate": float(fp_count / normal_count) if normal_count > 0 else 0,
    }
    
    report_path = model_path / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.logger.info(f"\nReport saved to {report_path}")
    
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeepDefend models")
    parser.add_argument("--model-path", help="Path to saved models")
    parser.add_argument("--output", help="Output report path")
    
    args = parser.parse_args()
    evaluate(args)