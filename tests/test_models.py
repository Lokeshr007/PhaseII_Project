"""
Tests for ML models.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepdefend.models.isolation_forest import IsolationForestAnomalyDetector
from deepdefend.models.xgboost_ensemble import XGBoostEnsemble
from deepdefend.models.calibrator import ProbabilityCalibrator


class TestIsolationForest(unittest.TestCase):
    def setUp(self):
        # Create synthetic normal + anomaly data
        np.random.seed(42)
        self.X_normal = np.random.randn(100, 10) * 0.5  # Tight cluster
        self.X_anomaly = np.random.randn(20, 10) * 3.0  # Spread out
        
        self.X_train = self.X_normal
        self.y_train = np.array(['normal'] * 100)
        
        self.X_test = np.vstack([self.X_normal[:10], self.X_anomaly[:10]])
        self.y_test = np.array(['normal'] * 10 + ['attack'] * 10)
    
    def test_fit_and_predict(self):
        detector = IsolationForestAnomalyDetector(contamination=0.1)
        detector.fit(self.X_train, self.y_train)
        
        scores = detector.score_samples(self.X_test)
        
        # Normal samples should have lower anomaly scores
        normal_scores = scores[:10]
        anomaly_scores = scores[10:]
        
        self.assertGreater(np.mean(anomaly_scores), np.mean(normal_scores))
    
    def test_anomaly_detection_binary(self):
        detector = IsolationForestAnomalyDetector(contamination=0.1)
        detector.fit(self.X_train, self.y_train)
        
        preds = detector.predict(self.X_test[:15])
        
        # First 10 should be normal
        self.assertEqual(np.mean(preds[:10]), 0.0)
    
    def test_threshold_tuning(self):
        detector = IsolationForestAnomalyDetector(contamination=0.05)
        detector.fit(self.X_train, self.y_train)
        
        self.assertIsNotNone(detector._anomaly_threshold)
        self.assertGreater(detector._anomaly_threshold, 0.0)


class TestXGBoostEnsemble(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        
        # Create synthetic multi-class data
        n_per_class = 100
        self.X = np.vstack([
            np.random.randn(n_per_class, 10) + 0,     # normal
            np.random.randn(n_per_class, 10) + 2,     # dos
            np.random.randn(n_per_class//2, 10) + 4,  # probe
            np.random.randn(n_per_class//5, 10) + 6,  # r2l
            np.random.randn(max(5, n_per_class//20), 10) + 8,  # u2r (very few)
        ])
        
        self.y = np.array(
            ['normal'] * n_per_class +
            ['dos'] * n_per_class +
            ['probe'] * (n_per_class//2) +
            ['r2l'] * (n_per_class//5) +
            ['u2r'] * max(5, n_per_class//20)
        )
    
    def test_all_specialists_trained(self):
        ensemble = XGBoostEnsemble(
            attack_classes=["dos", "probe", "r2l", "u2r"],
            class_weights={"dos": 5.0, "probe": 10.0, "r2l": 25.0, "u2r": 50.0}
        )
        ensemble.fit(self.X, self.y)
        
        for cls in ["dos", "probe", "r2l", "u2r"]:
            self.assertIn(cls, ensemble.specialists)
            self.assertTrue(ensemble.specialists[cls]._fitted)
    
    def test_predict_returns_valid_classes(self):
        ensemble = XGBoostEnsemble()
        ensemble.fit(self.X, self.y)
        
        preds = ensemble.predict(self.X[:10])
        
        valid_classes = set(['normal', 'dos', 'probe', 'r2l', 'u2r'])
        for pred in preds:
            self.assertIn(pred, valid_classes)
    
    def test_predict_proba_sums_to_one(self):
        ensemble = XGBoostEnsemble()
        ensemble.fit(self.X, self.y)
        
        probas = ensemble.predict_proba(self.X[:10])
        
        for row in probas:
            self.assertAlmostEqual(row.sum(), 1.0, places=5)


class TestCalibrator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        
        # Simple detector mock for testing
        from unittest.mock import MagicMock
        self.base_detector = MagicMock()
        self.base_detector.is_fitted = True
        self.base_detector.score_samples.return_value = np.array([0.9, 0.8, 0.2, 0.1])
        
        self.X_val = np.random.randn(100, 10)
        self.y_val = np.array(['attack'] * 50 + ['normal'] * 50)
    
    def test_calibration_improves_confidence(self):
        calibrator = ProbabilityCalibrator(
            base_detector=self.base_detector,
            method='platt'
        )
        calibrator.fit(self.X_val, self.y_val)
        
        scores = calibrator.score_samples(self.X_val[:5])
        
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))


if __name__ == "__main__":
    unittest.main()