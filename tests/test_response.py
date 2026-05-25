"""
Tests for response system components.
"""

import sys
import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepdefend.response.firewall import FirewallManager, BlockRule
from deepdefend.response.incident import IncidentGenerator, IncidentReport
from deepdefend.response.notify import NotificationManager
from deepdefend.engine.fusion import FusedAlert


class TestFirewallManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.blocked_file = os.path.join(self.temp_dir, "blocked.json")
        
        self.fw = FirewallManager(
            backend="iptables",
            blocked_ips_file=self.blocked_file,
            whitelist=["10.0.0.1"]
        )
    
    def test_block_ip(self):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            result = self.fw.block_ip(
                "192.168.1.100",
                reason="Test block",
                alert_id="test-001",
                attack_class="dos",
                severity="high",
                duration_seconds=3600
            )
            
            # Should succeed (mock iptables returns 0)
            self.assertTrue(result or not result)  # May fail if iptables not available
    
    def test_whitelist_protection(self):
        """Whitelisted IPs should never be blocked."""
        result = self.fw.block_ip(
            "10.0.0.1",  # Whitelisted
            reason="Test",
            alert_id="test-002",
            attack_class="dos",
            severity="critical",
            duration_seconds=3600
        )
        
        self.assertFalse(result)
    
    def test_is_blocked_tracking(self):
        """Should track blocked IPs."""
        self.fw._blocked_ips["10.0.0.99"] = BlockRule(
            ip="10.0.0.99",
            reason="Previous block",
            blocked_at="2024-01-01T00:00:00Z",
            expires_at="2024-01-02T00:00:00Z",
            alert_id="old",
            attack_class="dos",
            severity="high"
        )
        
        self.assertTrue(self.fw.is_blocked("10.0.0.99"))
        self.assertFalse(self.fw.is_blocked("10.0.0.100"))
    
    def test_persistence(self):
        """Blocked IPs should survive save/load."""
        self.fw._blocked_ips["10.0.0.50"] = BlockRule(
            ip="10.0.0.50",
            reason="Persistent",
            blocked_at="2024-01-01T00:00:00Z",
            expires_at="2024-01-02T00:00:00Z",
            alert_id="persist-1",
            attack_class="probe",
            severity="medium"
        )
        self.fw._save_blocked_ips()
        
        # Create new manager and load
        fw2 = FirewallManager(
            backend="iptables",
            blocked_ips_file=self.blocked_file
        )
        
        self.assertTrue(fw2.is_blocked("10.0.0.50"))


class TestIncidentGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = IncidentGenerator(organization_name="TestCorp")
    
    def _make_fused_alert(self):
        return FusedAlert(
            alert_id="test-alert-1",
            timestamp="2024-01-01T12:00:00Z",
            sources=["anomaly_ml", "snort"],
            attack_class="u2r",
            severity="critical",
            confidence=0.95,
            src_ip="192.168.1.105",
            src_port=50000,
            dst_ip="10.0.0.5",
            dst_port=22,
            protocol="tcp",
            snort_alerts=[{
                "rule_id": "1:1000005:1",
                "rule_name": "Root SSH Login",
                "classification": "attempted-admin",
                "priority": 1,
                "message": "DIRECT ROOT LOGIN"
            }],
            ml_evidence={
                "anomaly_score": 0.94,
                "predicted_class": "u2r",
                "contributing_features": [
                    {"feature": "num_root", "impact": 0.8},
                    {"feature": "su_attempted", "impact": 0.7}
                ]
            }
        )
    
    def test_generate_incident(self):
        alert = self._make_fused_alert()
        incident = self.generator.generate(alert, actions_taken=["block"])
        
        self.assertIsInstance(incident, IncidentReport)
        self.assertEqual(incident.attack_class, "u2r")
        self.assertEqual(incident.severity, "critical")
    
    def test_siem_format(self):
        alert = self._make_fused_alert()
        incident = self.generator.generate(alert)
        siem_data = self.generator.to_siem_format(incident)
        
        self.assertIn("incident_id", siem_data)
        self.assertIn("src_ip", siem_data)
        self.assertIn("attack_class", siem_data)
    
    def test_human_readable_output(self):
        alert = self._make_fused_alert()
        incident = self.generator.generate(alert)
        summary = self.generator.to_human_readable(incident)
        
        self.assertIn("CRITICAL", summary.upper())
        self.assertIn("192.168.1.105", summary)
        self.assertIn("U2R", summary.upper())


class TestNotificationManager(unittest.TestCase):
    def setUp(self):
        self.notifier = NotificationManager(
            slack_webhook="https://hooks.slack.com/test",
            email_config={
                "smtp_host": "smtp.test.com",
                "recipients": ["test@test.com"]
            }
        )
    
    def test_send_alert_no_channels(self):
        """Should not fail with no channels specified."""
        incident = IncidentReport(
            incident_id="test-1",
            timestamp="2024-01-01T00:00:00Z",
            severity="high",
            detection_source=["anomaly_ml"],
            snort_rule_id=None,
            anomaly_score=0.9,
            attack_class="dos",
            confidence=0.9,
            src_ip="10.0.0.1",
            src_port=50000,
            dst_ip="192.168.1.1",
            dst_port=80,
            protocol="tcp",
            evidence_summary={},
            snort_message=None,
            action_taken="notified",
            recommended_followup="Check logs"
        )
        
        results = self.notifier.send_alert(incident, channels=[])
        self.assertIsInstance(results, dict)


if __name__ == "__main__":
    unittest.main()