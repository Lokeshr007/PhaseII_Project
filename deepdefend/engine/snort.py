"""
Snort integration manager.

Manages Snort as a parallel detection path:
- Starts/stops Snort process
- Parses Snort alerts in real-time via UNIX socket
- Normalizes Snort output into our DetectionResult format
- Feeds alerts to fusion engine

Why Snort stays separate:
- Deterministic rules catch known attacks with zero false positives
- ML catches novel/behavioral attacks Snort misses
- Fusion engine combines both signals for higher confidence
- If ML engine fails, Snort still provides baseline protection
"""

import os
import subprocess
import socket
import threading
import time
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import uuid

from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SnortAlert:
    """Normalized Snort alert."""
    alert_id: str
    timestamp: str
    rule_id: str
    rule_name: str
    classification: str
    priority: int
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: str
    message: str
    raw_alert: str
    
    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "source": "snort",
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "classification": self.classification,
            "priority": self.priority,
            "src_ip": self.src_ip,
            "src_port": self.src_port,
            "dst_ip": self.dst_ip,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "message": self.message,
        }


class SnortManager:
    """
    Manages Snort IDS process and alert parsing.
    
    Handles:
    - Snort process lifecycle (start, stop, restart)
    - Real-time alert parsing via UNIX socket
    - Alert normalization to common format
    - Health monitoring (is Snort running?)
    """
    
    # Snort alert patterns for parsing
    ALERT_PATTERN = re.compile(
        r'\[\*\*\] \[(?P<gid>\d+):(?P<sid>\d+):(?P<rev>\d+)\] '
        r'(?P<message>.*?) \[\*\*\]\n'
        r'\[Classification: (?P<classification>.*?)\] '
        r'\[Priority: (?P<priority>\d+)\]\n'
        r'\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+ '
        r'(?P<src_ip>\d+\.\d+\.\d+\.\d+):(?P<src_port>\d+) -> '
        r'(?P<dst_ip>\d+\.\d+\.\d+\.\d+):(?P<dst_port>\d+)\n'
        r'(?P<protocol>\w+)'
    )
    
    # Backward compatibility pattern (older Snort formats)
    ALERT_PATTERN_LEGACY = re.compile(
        r'\[\*\*\] \[(?P<gid>\d+):(?P<sid>\d+):(?P<rev>\d+)\] '
        r'(?P<message>.*?) \[\*\*\]\n'
        r'\[Priority: (?P<priority>\d+)\]\n'
        r'\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+ '
        r'(?P<src_ip>\d+\.\d+\.\d+\.\d+):(?P<src_port>\d+) -> '
        r'(?P<dst_ip>\d+\.\d+\.\d+\.\d+):(?P<dst_port>\d+)\n'
    )
    
    def __init__(self,
                 binary_path: str = "/usr/sbin/snort",
                 config_path: str = "/etc/snort/snort.conf",
                 interface: str = "eth0",
                 alert_socket_path: str = "/tmp/snort_alert",
                 enabled: bool = True):
        
        self.binary_path = binary_path
        self.config_path = config_path
        self.interface = interface
        self.alert_socket_path = alert_socket_path
        self.enabled = enabled
        
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._alert_thread: Optional[threading.Thread] = None
        self._alert_socket: Optional[socket.socket] = None
        
        # Alert queue
        self.alert_queue: List[SnortAlert] = []
        self._alert_lock = threading.Lock()
        self.max_queue_size = 1000
        
        # Callbacks
        self.on_snort_alert: Optional[Callable] = None
        
        # Check if Snort is available
        self._available = self._check_availability()
        
        if not self._available and self.enabled:
            logger.logger.warning(
                f"Snort not found at {binary_path}. "
                f"Running without signature-based detection."
            )
    
    def _check_availability(self) -> bool:
        """Check if Snort binary exists and is executable."""
        if not self.enabled:
            return False
        
        if os.path.exists(self.binary_path):
            return os.access(self.binary_path, os.X_OK)
        
        # Check PATH
        import shutil
        snort_path = shutil.which("snort")
        if snort_path:
            self.binary_path = snort_path
            return True
        
        return False
    
    def start(self) -> bool:
        """Start Snort process and alert listener."""
        if not self._available:
            logger.logger.warning("Snort not available. Skipping.")
            return False
        
        if self._running:
            logger.logger.warning("Snort already running")
            return True
        
        # Clean up old socket
        if os.path.exists(self.alert_socket_path):
            os.unlink(self.alert_socket_path)
        
        # Build Snort command
        cmd = [
            self.binary_path,
            "-c", self.config_path,
            "-i", self.interface,
            "-A", "unixsock",  # Output to UNIX socket
            "-N",  # No logging (we handle alerts)
            "-q",  # Quiet
            "--alert-unixsock"  # Use UNIX socket for alerts
        ]
        
        logger.logger.info(f"Starting Snort: {' '.join(cmd)}")
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give Snort a moment to start
            time.sleep(2)
            
            if self._process.poll() is not None:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                logger.logger.error(f"Snort failed to start: {stderr}")
                return False
            
            self._running = True
            
            # Start alert listener thread
            self._alert_thread = threading.Thread(
                target=self._alert_listener,
                name="snort-alert-listener",
                daemon=True
            )
            self._alert_thread.start()
            
            logger.logger.info("Snort started successfully")
            return True
            
        except Exception as e:
            logger.logger.error(f"Failed to start Snort: {e}")
            return False
    
    def stop(self):
        """Stop Snort process gracefully."""
        self._running = False
        
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            except Exception:
                pass
            
            self._process = None
        
        if os.path.exists(self.alert_socket_path):
            os.unlink(self.alert_socket_path)
        
        logger.logger.info("Snort stopped")
    
    def restart(self) -> bool:
        """Restart Snort process."""
        self.stop()
        time.sleep(1)
        return self.start()
    
    def _alert_listener(self):
        """Listen for Snort alerts on UNIX socket."""
        import os
        
        # UNIX sockets are not supported on standard Windows Python versions
        if os.name == 'nt' or not hasattr(socket, 'AF_UNIX'):
            logger.logger.warning("UNIX sockets not supported on this platform. Snort alert listener disabled.")
            return
            
        logger.logger.info("Snort alert listener started")
        
        # Create UNIX socket
        try:
            self._alert_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._alert_socket.bind(self.alert_socket_path)
            self._alert_socket.settimeout(1.0)  # Check _running every second
        except Exception as e:
            logger.logger.error(f"Failed to bind alert socket: {e}")
            return
        
        buffer = ""
        
        while self._running:
            try:
                data = self._alert_socket.recv(4096)
                if data:
                    buffer += data.decode('utf-8', errors='ignore')
                    
                    # Process complete alerts (separated by blank lines)
                    while "\n\n" in buffer:
                        alert_text, buffer = buffer.split("\n\n", 1)
                        try:
                            alert = self._parse_alert(alert_text.strip())
                            if alert:
                                self._handle_alert(alert)
                        except Exception as e:
                            logger.logger.error(f"Failed to parse alert: {e}")
                            
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.logger.error(f"Alert socket error: {e}")
                break
        
        try:
            self._alert_socket.close()
        except Exception:
            pass
        
        logger.logger.info("Snort alert listener stopped")
    
    def _parse_alert(self, alert_text: str) -> Optional[SnortAlert]:
        """Parse raw Snort alert text into structured format."""
        # Try primary pattern first
        match = self.ALERT_PATTERN.search(alert_text)
        if not match:
            match = self.ALERT_PATTERN_LEGACY.search(alert_text)
        
        if not match:
            return None
        
        groups = match.groupdict()
        
        return SnortAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            rule_id=f"{groups.get('gid', '1')}:{groups.get('sid', '0')}:{groups.get('rev', '0')}",
            rule_name=groups.get("message", "Unknown rule"),
            classification=groups.get("classification", "Unknown"),
            priority=int(groups.get("priority", 3)),
            src_ip=groups["src_ip"],
            src_port=int(groups["src_port"]),
            dst_ip=groups["dst_ip"],
            dst_port=int(groups["dst_port"]),
            protocol=groups.get("protocol", "TCP"),
            message=groups.get("message", ""),
            raw_alert=alert_text
        )
    
    def _handle_alert(self, alert: SnortAlert):
        """Process a parsed Snort alert."""
        with self._alert_lock:
            if len(self.alert_queue) < self.max_queue_size:
                self.alert_queue.append(alert)
            else:
                logger.logger.warning("Snort alert queue full, dropping oldest")
                self.alert_queue.pop(0)
                self.alert_queue.append(alert)
        
        if self.on_snort_alert:
            try:
                self.on_snort_alert(alert)
            except Exception as e:
                logger.logger.error(f"Snort alert callback failed: {e}")
    
    def get_alerts(self) -> List[SnortAlert]:
        """Get queued Snort alerts."""
        with self._alert_lock:
            alerts = list(self.alert_queue)
            self.alert_queue.clear()
        return alerts
    
    def get_status(self) -> Dict:
        """Get Snort health status."""
        return {
            "enabled": self.enabled,
            "available": self._available,
            "running": self._running,
            "process_alive": self._process is not None and self._process.poll() is None,
            "alerts_queued": len(self.alert_queue),
        }
    
    def is_healthy(self) -> bool:
        """Check if Snort is healthy."""
        if not self.enabled:
            return True  # Not required
        return self._running and self._process is not None and self._process.poll() is None
    
    def reload_rules(self) -> bool:
        """Reload Snort rules without restarting."""
        if self._process and self._process.poll() is None:
            try:
                self._process.send_signal(subprocess.signal.SIGHUP)
                logger.logger.info("Sent SIGHUP to Snort for rule reload")
                return True
            except Exception as e:
                logger.logger.error(f"Failed to reload Snort rules: {e}")
        return False