"""
SIEM output formatter for enterprise integration.

Outputs RFC 5424 syslog with structured JSON payload.
Designed for ingestion by Splunk, ELK, QRadar, or any SIEM that consumes JSON syslog.

Every alert, action, and system event is logged in a format that
security analysts can search, dashboard, and alert on within their existing SIEM.
"""

import json
import socket
import sys
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from enum import IntEnum

from deepdefend.response.incident import IncidentReport
from deepdefend.engine.fusion import FusedAlert
from deepdefend.engine.policy import PolicyDecision
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class SyslogSeverity(IntEnum):
    """RFC 5424 syslog severity levels."""
    EMERGENCY = 0
    ALERT = 1
    CRITICAL = 2
    ERROR = 3
    WARNING = 4
    NOTICE = 5
    INFO = 6
    DEBUG = 7


class SyslogFacility(IntEnum):
    """RFC 5424 syslog facility codes."""
    LOCAL0 = 16
    LOCAL1 = 17
    LOCAL2 = 18
    LOCAL3 = 19
    LOCAL4 = 20
    LOCAL5 = 21
    LOCAL6 = 22
    LOCAL7 = 23


class SIEMFormatter:
    """
    Format and output events in SIEM-compatible syslog format.
    
    Produces RFC 5424 compliant messages with structured JSON data.
    Supports both local syslog daemon and direct TCP/UDP syslog.
    """
    
    # Map our severity to syslog severity
    SEVERITY_MAP = {
        "critical": SyslogSeverity.CRITICAL,
        "high": SyslogSeverity.ERROR,
        "medium": SyslogSeverity.WARNING,
        "low": SyslogSeverity.NOTICE,
    }
    
    def __init__(self,
                 facility: str = "local0",
                 hostname: str = "deepdefend",
                 app_name: str = "deepdefend-nids",
                 output: str = "stdout",
                 syslog_server: Optional[str] = None,
                 syslog_port: int = 514,
                 syslog_protocol: str = "udp",
                 include_raw_features: bool = True):
        
        self.facility = getattr(SyslogFacility, facility.upper(), SyslogFacility.LOCAL0)
        self.hostname = hostname
        self.app_name = app_name
        self.output = output
        self.syslog_server = syslog_server
        self.syslog_port = syslog_port
        self.syslog_protocol = syslog_protocol
        self.include_raw_features = include_raw_features
        
        # Initialize syslog socket if needed
        self._socket = None
        if output == "syslog_server" and syslog_server:
            self._init_syslog_socket()
        
        logger.logger.info(
            f"SIEMFormatter initialized: output={output}, "
            f"facility={facility}, hostname={hostname}"
        )
    
    def _init_syslog_socket(self):
        """Initialize syslog network socket."""
        try:
            if self.syslog_protocol == "tcp":
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.connect((self.syslog_server, self.syslog_port))
            else:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except Exception as e:
            logger.logger.error(f"Failed to initialize syslog socket: {e}")
            self._socket = None
    
    def format_alert(self, incident: IncidentReport) -> str:
        """
        Format an incident report as a SIEM event.
        
        Returns RFC 5424 syslog message with structured JSON.
        """
        # Build structured data
        sd_data = {
            "event_type": "intrusion_detection_alert",
            "incident_id": incident.incident_id,
            "severity": incident.severity,
            "attack_class": incident.attack_class,
            "confidence": incident.confidence,
            "detection_source": incident.detection_source,
            "src_ip": incident.src_ip,
            "src_port": incident.src_port,
            "dst_ip": incident.dst_ip,
            "dst_port": incident.dst_port,
            "protocol": incident.protocol,
            "action_taken": incident.action_taken,
            "snort_rule_id": incident.snort_rule_id,
            "anomaly_score": incident.anomaly_score,
            "occurrence_count": incident.occurrence_count,
            "evidence_summary": incident.evidence_summary,
            "recommended_followup": incident.recommended_followup,
            "correlated_alerts": incident.correlated_alerts,
        }
        
        if self.include_raw_features and incident.raw_flow_data:
            sd_data["raw_flow_data"] = incident.raw_flow_data
        
        return self._format_syslog(
            severity=self.SEVERITY_MAP.get(incident.severity, SyslogSeverity.WARNING),
            msg_id="ALERT",
            structured_data=sd_data
        )
    
    def format_action(self, decision: PolicyDecision, 
                      success: bool, detail: str = "") -> str:
        """Format a response action as a SIEM event."""
        sd_data = {
            "event_type": "response_action",
            "alert_id": decision.alert.alert_id,
            "actions": [a.value for a in decision.actions],
            "success": success,
            "detail": detail,
            "block_duration": decision.block_duration_seconds,
            "rule_triggered": decision.metadata.get("rule_name", "unknown"),
        }
        
        severity = SyslogSeverity.WARNING if success else SyslogSeverity.ERROR
        
        return self._format_syslog(
            severity=severity,
            msg_id="ACTION",
            structured_data=sd_data
        )
    
    def format_system_event(self, event_type: str, 
                           data: Dict[str, Any],
                           severity: str = "info") -> str:
        """Format a system event as a SIEM event."""
        sd_data = {
            "event_type": event_type,
            **data
        }
        
        sev_map = {
            "info": SyslogSeverity.INFO,
            "warning": SyslogSeverity.WARNING,
            "error": SyslogSeverity.ERROR,
            "critical": SyslogSeverity.CRITICAL,
        }
        
        return self._format_syslog(
            severity=sev_map.get(severity, SyslogSeverity.INFO),
            msg_id="SYSTEM",
            structured_data=sd_data
        )
    
    def format_health_check(self, health_data: Dict) -> str:
        """Format a health check result as SIEM event."""
        return self.format_system_event(
            "health_check", 
            health_data,
            severity="info"
        )
    
    def format_metrics(self, metrics: Dict) -> str:
        """Format metrics snapshot as SIEM event."""
        return self.format_system_event(
            "metrics_snapshot",
            metrics,
            severity="info"
        )
    
    def _format_syslog(self, severity: SyslogSeverity, 
                       msg_id: str, 
                       structured_data: Dict) -> str:
        """
        Format message in RFC 5424 syslog format.
        
        RFC 5424 format:
        <PRI>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID [SD-ID] MESSAGE
        """
        priority = (self.facility.value * 8) + severity.value
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        # Build structured data element
        sd_elements = []
        for key, value in structured_data.items():
            # Format value as string
            value_str = json.dumps(value) if not isinstance(value, str) else str(value)
            # RFC 5424 requires escaping \ , " and ]
            value_str = value_str.replace('\\', '\\\\').replace('"', '\\"').replace(']', '\\]')
            sd_elements.append(f'{key}="{value_str}"')
        
        sd_id = f"[deepdefend@1 {' '.join(sd_elements)}]"
        
        # Message (empty for structured logging — all data in SD-ELEMENT)
        message = "-"
        
        syslog_msg = (
            f"<{priority}>1 {timestamp} {self.hostname} "
            f"{self.app_name} - {msg_id} {sd_id} {message}"
        )
        
        return syslog_msg
    
    def send(self, message: str):
        """
        Send a formatted SIEM message to the configured output.
        """
        if self.output == "stdout":
            print(message, flush=True)
            
        elif self.output == "syslog_server" and self._socket:
            try:
                if self.syslog_protocol == "tcp":
                    self._socket.send((message + "\n").encode('utf-8'))
                else:
                    self._socket.send(message.encode('utf-8'))
            except Exception as e:
                logger.logger.error(f"Failed to send syslog message: {e}")
                # Try to reconnect
                self._init_syslog_socket()
                
        elif self.output == "syslog_local":
            # Send to local syslog daemon
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                sock.sendto(message.encode('utf-8'), '/dev/log')
                sock.close()
            except Exception:
                # Fallback to stdout
                print(message, flush=True)
        
        else:
            # Unknown output, fallback to stdout
            print(message, flush=True)