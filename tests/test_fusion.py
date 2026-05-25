"""
Tests for the fusion engine.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepdefend.engine.fusion import FusionEngine, FusedAlert
from deepdefend.engine.inference import DetectionResult
from deepdefend.engine.snort import SnortAlert


class TestFusionEngine(unittest.TestCase):
    def setUp(self):
        self.fusion = FusionEngine(
            dedup_window_seconds=60.0,
            correlation_window_seconds=300.0,
            burst_threshold=10
        )
    
    def _make_ml_result(self, src_ip="10.0.0.1", dst_ip="192.168.1.1",
                        attack_class="dos", confidence=0.9):
        return DetectionResult(
            alert_id=f"ml-{src_ip}-{attack_class}",
            timestamp="2024-01-01T00:00:00Z",
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=50000,
            dst_port=80,
            protocol="tcp",
            is_anomalous=True,
            anomaly_score=0.95,
            predicted_class=attack_class,
            confidence=confidence,
            class_probabilities={"dos": 0.9, "probe": 0.05, "r2l": 0.03, "u2r": 0.02},
            contributing_features=[],
            flow_features={},
        )
    
    def _make_snort_alert(self, src_ip="10.0.0.1", dst_ip="192.168.1.1",
                          classification="attempted-dos"):
        return SnortAlert(
            alert_id=f"snort-{src_ip}",
            timestamp="2024-01-01T00:00:00Z",
            rule_id="1:1000001:1",
            rule_name="Test Rule",
            classification=classification,
            priority=2,
            src_ip=src_ip,
            src_port=50000,
            dst_ip=dst_ip,
            dst_port=80,
            protocol="TCP",
            message="Test alert",
            raw_alert=""
        )
    
    def test_dual_confirmation_escalates(self):
        """When both Snort and ML detect, confidence should increase."""
        ml = self._make_ml_result(confidence=0.7)
        snort = self._make_snort_alert()
        
        fused = self.fusion.fuse([ml], [snort])
        
        self.assertEqual(len(fused), 1)
        self.assertIn("snort", fused[0].sources)
        self.assertIn("anomaly_ml", fused[0].sources)
        self.assertGreaterEqual(fused[0].confidence, 0.7)
    
    def test_ml_only_alert(self):
        """ML-only alerts should still be processed."""
        ml = self._make_ml_result(attack_class="u2r", confidence=0.95)
        
        fused = self.fusion.fuse([ml], [])
        
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].attack_class, "u2r")
        self.assertIn("anomaly_ml", fused[0].sources)
    
    def test_snort_only_alert(self):
        """Snort-only alerts should be processed."""
        snort = self._make_snort_alert()
        
        fused = self.fusion.fuse([], [snort])
        
        self.assertEqual(len(fused), 1)
        self.assertIn("snort", fused[0].sources)
        self.assertEqual(fused[0].severity, "medium")
    
    def test_deduplication(self):
        """Duplicate alerts within window should be merged."""
        ml1 = self._make_ml_result(attack_class="dos")
        ml2 = self._make_ml_result(attack_class="dos")  # Same src/dst/class
        
        fused = self.fusion.fuse([ml1, ml2], [])
        
        # Should be deduplicated to one alert with occurrence_count=2
        self.assertLessEqual(len(fused), 1)
        if fused:
            self.assertGreater(fused[0].occurrence_count, 1)
    
    def test_unmatched_alerts_kept_separate(self):
        """Different attack types from same source should be separate."""
        ml1 = self._make_ml_result(attack_class="dos")
        ml2 = self._make_ml_result(attack_class="probe")
        
        fused = self.fusion.fuse([ml1, ml2], [])
        
        # Different attack classes should not be deduplicated
        attack_classes = [f.attack_class for f in fused]
        self.assertIn("dos", attack_classes)
        self.assertIn("probe", attack_classes)
    
    def test_matching_by_network_context(self):
        """ML and Snort alerts with same (src_ip, dst_ip) should be matched."""
        ml = self._make_ml_result(src_ip="10.0.0.1", dst_ip="192.168.1.1")
        snort = self._make_snort_alert(src_ip="10.0.0.1", dst_ip="192.168.1.1")
        
        fused = self.fusion.fuse([ml], [snort])
        
        self.assertEqual(len(fused), 1)
        self.assertEqual(len(fused[0].snort_alerts), 1)
    
    def test_different_sources_not_matched(self):
        """Alerts for different targets should not be matched."""
        ml = self._make_ml_result(src_ip="10.0.0.1", dst_ip="192.168.1.1")
        snort = self._make_snort_alert(src_ip="10.0.0.2", dst_ip="192.168.1.2")
        
        fused = self.fusion.fuse([ml], [snort])
        
        # Should produce two separate alerts
        self.assertEqual(len(fused), 2)


if __name__ == "__main__":
    unittest.main()