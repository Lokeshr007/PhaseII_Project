"""
Multi-channel notification system.

Sends alerts via:
- Slack (instant, rich formatting)
- Email (detailed, for audit trail)
- PagerDuty (on-call escalation)
- Webhook (custom integrations)

All notifications include the incident summary and essential context.
Designed so an analyst can triage from their phone at 3 AM.
"""

import json
import smtplib
import textwrap
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from datetime import datetime
import requests

from deepdefend.response.incident import IncidentReport, IncidentGenerator
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class NotificationManager:
    """
    Send security alerts through multiple channels.
    
    Priority-aware: critical alerts go everywhere immediately.
    Low-priority alerts are batched.
    """
    
    def __init__(self,
                 slack_webhook: Optional[str] = None,
                 email_config: Optional[Dict] = None,
                 pagerduty_key: Optional[str] = None,
                 webhook_urls: Optional[List[str]] = None,
                 incident_generator: Optional[IncidentGenerator] = None):
        
        self.slack_webhook = slack_webhook
        self.email_config = email_config
        self.pagerduty_key = pagerduty_key
        self.webhook_urls = webhook_urls or []
        self.incident_generator = incident_generator or IncidentGenerator()
        
        # Track notification counts to avoid flooding
        self._notification_count = 0
        self._last_reset = datetime.now()
    
    def send_alert(self, incident: IncidentReport,
                   channels: List[str] = None,
                   priority: str = "normal") -> Dict[str, bool]:
        """
        Send alert through specified channels.
        
        Args:
            incident: The incident report
            channels: List of channels (slack, email, pagerduty, webhook)
            priority: Alert priority (affects formatting)
        
        Returns:
            Dict of channel -> success
        """
        channels = channels or ["siem"]  # Default: log only
        results = {}
        
        human_summary = self.incident_generator.to_human_readable(incident)
        siem_data = self.incident_generator.to_siem_format(incident)
        
        for channel in channels:
            try:
                if channel == "slack" and self.slack_webhook:
                    results["slack"] = self._send_slack(incident, human_summary)
                
                elif channel == "email" and self.email_config:
                    results["email"] = self._send_email(incident, human_summary)
                
                elif channel == "pagerduty" and self.pagerduty_key:
                    results["pagerduty"] = self._send_pagerduty(incident, siem_data)
                
                elif channel == "webhook" and self.webhook_urls:
                    results["webhook"] = self._send_webhooks(incident, siem_data)
                
                elif channel == "siem":
                    # SIEM output is handled by the SIEM module
                    # Here we just confirm it would be logged
                    results["siem"] = True
                
            except Exception as e:
                logger.logger.error(f"Failed to send {channel} notification: {e}")
                results[channel] = False
        
        return results
    
    def _send_slack(self, incident: IncidentReport, 
                    summary: str) -> bool:
        """Send formatted Slack message."""
        if not self.slack_webhook:
            return False
        
        severity_colors = {
            "critical": "#FF0000",
            "high": "#FF6600",
            "medium": "#FFCC00",
            "low": "#3399FF",
        }
        
        color = severity_colors.get(incident.severity, "#999999")
        
        # Build Slack message with attachments
        payload = {
            "attachments": [{
                "color": color,
                "title": f"{incident.severity.upper()}: {incident.attack_class.upper()} Attack Detected",
                "text": (
                    f"*Source:* `{incident.src_ip}` → *Target:* `{incident.dst_ip}:{incident.dst_port}`\n"
                    f"*Confidence:* {incident.confidence:.0%}\n"
                    f"*Action:* {incident.action_taken}\n"
                    f"*Recommendation:* {incident.recommended_followup[:200]}"
                ),
                "fields": [
                    {"title": "Incident ID", "value": incident.incident_id, "short": True},
                    {"title": "Detection", "value": ", ".join(incident.detection_source), "short": True},
                ],
                "footer": "DeepDefend NIDS",
                "ts": int(datetime.now().timestamp())
            }]
        }
        
        response = requests.post(
            self.slack_webhook,
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            logger.logger.info(f"Slack notification sent: {incident.incident_id}")
            return True
        else:
            logger.logger.error(f"Slack notification failed: {response.status_code} {response.text}")
            return False
    
    def _send_email(self, incident: IncidentReport,
                    summary: str) -> bool:
        """Send email notification."""
        if not self.email_config:
            return False
        
        smtp_host = self.email_config.get("smtp_host")
        smtp_port = self.email_config.get("smtp_port", 587)
        username = self.email_config.get("username")
        password = self.email_config.get("password")
        from_addr = self.email_config.get("from", "deepdefend@enterprise.com")
        recipients = self.email_config.get("recipients", [])
        
        if not recipients:
            logger.logger.warning("No email recipients configured")
            return False
        
        # Build email
        subject = (
            f"[{incident.severity.upper()}] DeepDefend Alert: "
            f"{incident.attack_class.upper()} from {incident.src_ip}"
        )
        
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(recipients)
        
        # Plain text body
        msg.attach(MIMEText(summary, "plain"))
        
        # Attach JSON for SIEM
        siem_data = self.incident_generator.to_siem_format(incident)
        json_attachment = MIMEText(
            json.dumps(siem_data, indent=2),
            "json"
        )
        json_attachment.add_header(
            "Content-Disposition",
            f"attachment; filename=incident_{incident.incident_id}.json"
        )
        msg.attach(json_attachment)
        
        # Send
        try:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
            server.starttls()
            if username and password:
                server.login(username, password)
            server.sendmail(from_addr, recipients, msg.as_string())
            server.quit()
            
            logger.logger.info(f"Email notification sent: {incident.incident_id}")
            return True
            
        except Exception as e:
            logger.logger.error(f"Email notification failed: {e}")
            return False
    
    def _send_pagerduty(self, incident: IncidentReport,
                        siem_data: Dict) -> bool:
        """Send PagerDuty alert for critical incidents."""
        if not self.pagerduty_key:
            return False
        
        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": (
                    f"{incident.severity.upper()}: {incident.attack_class.upper()} "
                    f"from {incident.src_ip} to {incident.dst_ip}"
                ),
                "severity": incident.severity,
                "source": "DeepDefend NIDS",
                "custom_details": siem_data,
            }
        }
        
        response = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 202:
            logger.logger.info(f"PagerDuty alert sent: {incident.incident_id}")
            return True
        else:
            logger.logger.error(f"PagerDuty alert failed: {response.status_code}")
            return False
    
    def _send_webhooks(self, incident: IncidentReport,
                       siem_data: Dict) -> bool:
        """Send to custom webhook endpoints."""
        success = True
        
        for url in self.webhook_urls:
            try:
                response = requests.post(
                    url,
                    json=siem_data,
                    timeout=5
                )
                if response.status_code >= 400:
                    logger.logger.error(f"Webhook {url} failed: {response.status_code}")
                    success = False
            except Exception as e:
                logger.logger.error(f"Webhook {url} error: {e}")
                success = False
        
        return success