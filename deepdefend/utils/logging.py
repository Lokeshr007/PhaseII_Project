"""
Structured logging for enterprise deployment.
Outputs JSON-formatted logs suitable for SIEM ingestion.
"""

import logging
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured log output."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Include extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Include exception info if present
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }
        
        return json.dumps(log_entry)


class SecurityLogger:
    """Logger with security-specific context."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"deepdefend.{name}")
    
    def alert_detected(self, alert_id: str, severity: str, 
                       attack_class: str, src_ip: str, 
                       confidence: float, extra: Optional[Dict] = None):
        """Log a detected security event."""
        fields = {
            "event_type": "alert",
            "alert_id": alert_id,
            "severity": severity,
            "attack_class": attack_class,
            "src_ip": src_ip,
            "confidence": confidence,
            **(extra or {})
        }
        self.logger.warning(
            f"Alert {alert_id}: {severity} {attack_class} from {src_ip} "
            f"(confidence: {confidence:.2f})",
            extra={"extra_fields": fields}
        )
    
    def action_taken(self, alert_id: str, action: str, target: str, 
                     success: bool, detail: Optional[str] = None):
        """Log an automated response action."""
        fields = {
            "event_type": "action",
            "alert_id": alert_id,
            "action": action,
            "target": target,
            "success": success,
            "detail": detail
        }
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"Action {action} on {target}: {'SUCCESS' if success else 'FAILED'}",
            extra={"extra_fields": fields}
        )
    
    def model_inference(self, model_name: str, latency_ms: float, 
                        batch_size: int, prediction_count: int):
        """Log model inference metrics."""
        fields = {
            "event_type": "inference",
            "model": model_name,
            "latency_ms": latency_ms,
            "batch_size": batch_size,
            "predictions": prediction_count
        }
        self.logger.debug(
            f"Inference: {model_name} ({latency_ms:.1f}ms, {prediction_count} preds)",
            extra={"extra_fields": fields}
        )


def setup_logging(level: str = "INFO", output: str = "stdout"):
    """Configure logging for the entire application."""
    root_logger = logging.getLogger("deepdefend")
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create handler
    if output == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    elif output == "syslog":
        from logging.handlers import SysLogHandler
        handler = SysLogHandler(address='/dev/log')
    else:
        handler = logging.FileHandler(output)
    
    handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(handler)
    
    return root_logger


def get_logger(name: str) -> SecurityLogger:
    """Get a security logger instance."""
    return SecurityLogger(name)