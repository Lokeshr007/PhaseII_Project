"""
Network quarantine manager for compromised devices.

Handles isolation of compromised internal hosts:
- VLAN reassignment via SNMP or API
- Port shutdown on managed switches
- ACL-based isolation

This is an advanced feature requiring network infrastructure integration.
Defaults to logging recommendations when infrastructure API is unavailable.
"""

import subprocess
import json
from typing import Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path

from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QuarantineRule:
    """A quarantine entry."""
    target_ip: str
    target_mac: Optional[str]
    reason: str
    alert_id: str
    quarantined_at: str
    expires_at: Optional[str]
    method: str  # vlan, acl, port_shutdown
    status: str  # active, expired, failed


class QuarantineManager:
    """
    Manage network quarantine for compromised devices.
    
    Supports multiple isolation methods:
    1. VLAN reassignment: Move device to isolated VLAN
    2. ACL-based: Apply restrictive ACL to device traffic  
    3. Port shutdown: Disable switch port
    
    Requires appropriate credentials and API access to network
    infrastructure. Falls back to recommendations when unavailable.
    """
    
    def __init__(self,
                 method: str = "recommend_only",
                 quarantine_vlan: int = 999,
                 switch_api_url: Optional[str] = None,
                 switch_api_token: Optional[str] = None,
                 quarantine_file: str = "/var/lib/deepdefend/quarantine.json",
                 enabled: bool = False):
        
        self.method = method
        self.quarantine_vlan = quarantine_vlan
        self.switch_api_url = switch_api_url
        self.switch_api_token = switch_api_token
        self.quarantine_file = Path(quarantine_file)
        self.enabled = enabled
        
        self._quarantined: Dict[str, QuarantineRule] = {}
        self._load_state()
        
        logger.logger.info(
            f"QuarantineManager initialized: method={method}, enabled={enabled}"
        )
    
    def quarantine_device(self, 
                         ip: str,
                         mac: Optional[str] = None,
                         reason: str = "",
                         alert_id: str = "",
                         duration_hours: int = 4) -> bool:
        """
        Quarantine a device from the network.
        
        Args:
            ip: IP address to quarantine
            mac: MAC address (for port-level quarantine)
            reason: Reason for quarantine
            alert_id: Triggering alert ID
            duration_hours: Quarantine duration
        
        Returns:
            True if quarantine was applied or recommended
        """
        if not self.enabled:
            logger.logger.info(f"Quarantine disabled. Would quarantine {ip}: {reason}")
            return False
        
        # Check if already quarantined
        if ip in self._quarantined:
            logger.logger.info(f"Device {ip} already quarantined")
            return True
        
        rule = QuarantineRule(
            target_ip=ip,
            target_mac=mac,
            reason=reason,
            alert_id=alert_id,
            quarantined_at=datetime.now(timezone.utc).isoformat(),
            expires_at=None if duration_hours == 0 else 
                     (datetime.now(timezone.utc) + 
                      __import__('datetime').timedelta(hours=duration_hours)).isoformat(),
            method=self.method,
            status="active"
        )
        
        success = False
        
        if self.method == "vlan":
            success = self._quarantine_vlan(ip, mac)
        elif self.method == "acl":
            success = self._quarantine_acl(ip)
        elif self.method == "port_shutdown":
            success = self._quarantine_port(mac) if mac else False
        elif self.method == "recommend_only":
            logger.logger.warning(
                f"QUARANTINE RECOMMENDATION: Isolate {ip} ({mac or 'unknown MAC'}) "
                f"— {reason}"
            )
            success = True  # Recommendation is always "successful"
        
        if success or self.method == "recommend_only":
            self._quarantined[ip] = rule
            self._save_state()
            logger.logger.warning(
                f"Device quarantined: {ip} (method: {self.method}, reason: {reason})"
            )
        
        return success
    
    def release_device(self, ip: str) -> bool:
        """Release a device from quarantine."""
        if ip not in self._quarantined:
            return True
        
        rule = self._quarantined[ip]
        success = False
        
        if self.method == "vlan":
            success = self._release_vlan(ip, rule.target_mac)
        elif self.method == "acl":
            success = self._release_acl(ip)
        elif self.method == "port_shutdown":
            success = self._release_port(rule.target_mac) if rule.target_mac else False
        elif self.method == "recommend_only":
            logger.logger.info(f"QUARANTINE RELEASE: Restore {ip} to network")
            success = True
        
        if success or self.method == "recommend_only":
            del self._quarantined[ip]
            self._save_state()
            logger.logger.info(f"Device released from quarantine: {ip}")
        
        return success
    
    def get_quarantined_devices(self) -> List[Dict]:
        """Get list of quarantined devices."""
        return [asdict(rule) for rule in self._quarantined.values()]
    
    def is_quarantined(self, ip: str) -> bool:
        """Check if a device is quarantined."""
        return ip in self._quarantined
    
    def _quarantine_vlan(self, ip: str, mac: Optional[str]) -> bool:
        """Move device to quarantine VLAN via switch API."""
        if not self.switch_api_url:
            logger.logger.warning("No switch API configured for VLAN quarantine")
            return False
        
        # This would integrate with actual switch API
        logger.logger.info(f"Would move {ip} ({mac}) to VLAN {self.quarantine_vlan}")
        return False  # Not implemented without actual infrastructure
    
    def _quarantine_acl(self, ip: str) -> bool:
        """Apply restrictive ACL to device IP."""
        try:
            cmd = [
                "iptables", "-A", "QUARANTINE",
                "-s", ip, "-j", "DROP"
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            cmd = [
                "iptables", "-A", "QUARANTINE",
                "-d", ip, "-j", "DROP"
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            return True
        except Exception as e:
            logger.logger.error(f"ACL quarantine failed for {ip}: {e}")
            return False
    
    def _quarantine_port(self, mac: str) -> bool:
        """Shutdown switch port by MAC address."""
        if not self.switch_api_url:
            logger.logger.warning("No switch API for port shutdown")
            return False
        
        logger.logger.info(f"Would shut down port for MAC {mac}")
        return False
    
    def _release_vlan(self, ip: str, mac: Optional[str]) -> bool:
        logger.logger.info(f"Would restore {ip} to production VLAN")
        return False
    
    def _release_acl(self, ip: str) -> bool:
        try:
            subprocess.run(
                ["iptables", "-D", "QUARANTINE", "-s", ip, "-j", "DROP"],
                capture_output=True, check=False
            )
            subprocess.run(
                ["iptables", "-D", "QUARANTINE", "-d", ip, "-j", "DROP"],
                capture_output=True, check=False
            )
            return True
        except Exception:
            return False
    
    def _release_port(self, mac: str) -> bool:
        logger.logger.info(f"Would re-enable port for MAC {mac}")
        return False
    
    def _save_state(self):
        self.quarantine_file.parent.mkdir(parents=True, exist_ok=True)
        data = {ip: asdict(rule) for ip, rule in self._quarantined.items()}
        with open(self.quarantine_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_state(self):
        if self.quarantine_file.exists():
            try:
                with open(self.quarantine_file) as f:
                    data = json.load(f)
                self._quarantined = {
                    ip: QuarantineRule(**rule_data)
                    for ip, rule_data in data.items()
                }
            except Exception as e:
                logger.logger.error(f"Failed to load quarantine state: {e}")