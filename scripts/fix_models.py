#!/usr/bin/env python3
"""
Quick fix script to re-tune model thresholds after the tune_thresholds bug fix.
This avoids needing a full re-train if the models themselves are already decent.
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepdefend.config.settings import ConfigLoader
from deepdefend.data.loader import NSLKDLoader
from deepdefend.data.preprocessor import FeaturePreprocessor
from deepdefend.models.xgboost_ensemble import XGBoostEnsemble
from deepdefend.training.metrics import ImbalanceMetrics
from deepdefend.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

def fix_models():
    setup_logging(level="INFO")
    logger.logger.info("Applying model fix: Re-tuning decision thresholds...")
    
    # 1. Load configuration and data
    config = ConfigLoader.load(mode="development")
    loader = NSLKDLoader(data_dir=config.data_path)
    dataset = loader.load_all()
    
    model_dir = Path(config.model_path)
    
    # 2. Load artifacts and models
    logger.logger.info("Loading models and preprocessor...")
    preprocessor = FeaturePreprocessor.load(model_dir / "preprocessor.pkl")
    ensemble = XGBoostEnsemble.load(model_dir / "attack_ensemble.pkl")
    
    # 3. Preprocess validation and test data
    X_val = preprocessor.transform(dataset.val.X)
    y_val = dataset.val.y
    X_test = preprocessor.transform(dataset.test.X)
    y_test = dataset.test.y
    
    # 4. Re-tune thresholds using the NEW fixed method
    logger.logger.info("Re-tuning thresholds on validation set...")
    # target_recall prioritize minority classes
    target_recall = {
        'dos': 0.85,
        'probe': 0.85,
        'r2l': 0.50, # More realistic for R2L
        'u2r': 0.50  # More realistic for U2R
    }
    
    new_thresholds = ensemble.tune_thresholds(
        X_val, y_val, 
        target_recall=target_recall,
        max_fpr=0.02
    )
    
    # 5. Evaluate on test set with new thresholds
    logger.logger.info("Evaluating fixed thresholds on test set...")
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)
    
    class_names = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    metrics_calculator = ImbalanceMetrics(class_names=class_names, minority_classes=['u2r', 'r2l'])
    test_metrics = metrics_calculator.compute_all(y_test, y_pred, y_proba)
    
    logger.logger.info("\nNEW TEST PERFORMANCE:")
    logger.logger.info(metrics_calculator.summary(test_metrics))
    
    # 6. Save updated artifacts and model
    logger.logger.info("Saving updated model and artifacts...")
    ensemble.save(model_dir / "attack_ensemble.pkl")
    
    # Update artifacts.json
    artifacts_path = model_dir / "artifacts.json"
    if artifacts_path.exists():
        with open(artifacts_path, 'r') as f:
            artifacts = json.load(f)
        
        artifacts["thresholds"] = new_thresholds
        artifacts["test_metrics_fixed"] = {
            k: v for k, v in test_metrics.items() 
            if not isinstance(v, (list, dict)) or k == 'per_class'
        }
        artifacts["fix_applied_at"] = np.datetime64('now').astype(str)
        
        with open(artifacts_path, 'w') as f:
            json.dump(artifacts, f, indent=2, default=str)
    
    logger.logger.info("Fix applied successfully!")
    logger.logger.info(f"New thresholds: {new_thresholds}")

if __name__ == "__main__":
    fix_models()
