"""
Firewall management for automated IP blocking.

Supports:
- iptables (legacy)
- nftables (modern)
- API-based firewalls (extensible)

All operations are logged for audit trail.
Blocked IPs are tracked in a persistent file for survivability across restarts.
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import threading

from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BlockRule:
    """Represents a blocked IP."""
    ip: str
    reason: str
    blocked_at: str
    expires_at: str
    alert_id: str
    attack_class: str
    severity: str
    chain: str = "INPUT"


class FirewallManager:
    """
    Manage firewall rules for blocking malicious IPs.
    
    Features:
    - Add/remove block rules
    - Time-based auto-expiry
    - Persistence across restarts
    - Audit trail logging
    - Rate limiting to prevent blocking entire network
    - Whitelist for critical IPs that must never be blocked
    """
    
    def __init__(self,
                 backend: str = "iptables",
                 blocked_ips_file: str = "./data/blocked_ips.json",
                 chain: str = "DEEPDEFEND_BLOCK",
                 max_blocks: int = 1000,
                 whitelist: List[str] = None):
        
        import os
        if os.name == 'nt' or backend == 'dummy':
            logger.logger.warning("Windows detected or dummy backend requested. Using log-only firewall backend.")
            self.backend = "dummy"
        else:
            self.backend = backend
        self.blocked_ips_file = Path(blocked_ips_file)
        self.chain = chain
        self.max_blocks = max_blocks
        self.whitelist = set(whitelist or [
            "127.0.0.1", "::1",
            # Add management IPs, load balancers, etc.
        ])
        
        self._lock = threading.Lock()
        self._blocked_ips: Dict[str, BlockRule] = {}
        
        # Initialize
        self._ensure_chain()
        self._load_blocked_ips()
        self._cleanup_expired()
        
        logger.logger.info(
            f"FirewallManager initialized: backend={backend}, "
            f"chain={chain}, currently blocked: {len(self._blocked_ips)}"
        )
    
    def block_ip(self, ip: str, reason: str, alert_id: str,
                 attack_class: str, severity: str,
                 duration_seconds: int = 3600) -> bool:
        """
        Block an IP address.
        
        Args:
            ip: IP to block
            reason: Human-readable reason for block
            alert_id: Alert that triggered this block
            attack_class: Attack classification
            severity: Alert severity
            duration_seconds: How long to block (0 = permanent)
        
        Returns:
            True if blocked successfully
        """
        # Validate IP
        if not self._is_valid_ip(ip):
            logger.logger.error(f"Invalid IP: {ip}")
            return False
        
        # Check whitelist
        if ip in self.whitelist:
            logger.logger.warning(f"Attempted to block whitelisted IP: {ip}. Skipping.")
            return False
        
        # Check if already blocked
        with self._lock:
            if ip in self._blocked_ips:
                # Update existing block
                logger.logger.info(f"IP {ip} already blocked, updating expiry")
                existing = self._blocked_ips[ip]
                existing.reason = f"{existing.reason}; {reason}"
                existing.expires_at = (datetime.now(timezone.utc) + 
                                      timedelta(seconds=duration_seconds)).isoformat()
                self._save_blocked_ips()
                return True
            
            # Check block limit
            if len(self._blocked_ips) >= self.max_blocks:
                logger.logger.error(
                    f"Block limit reached ({self.max_blocks}). "
                    f"Cannot block {ip}. Consider raising max_blocks or reviewing blocked IPs."
                )
                return False
            
            # Create block rule
            now = datetime.now(timezone.utc)
            expires = now + timedelta(seconds=duration_seconds) if duration_seconds > 0 else None
            
            rule = BlockRule(
                ip=ip,
                reason=reason,
                blocked_at=now.isoformat(),
                expires_at=expires.isoformat() if expires else "permanent",
                alert_id=alert_id,
                attack_class=attack_class,
                severity=severity,
                chain=self.chain
            )
            
            # Execute block
            success = False
            if self.backend == "iptables":
                success = self._iptables_block(ip)
            elif self.backend == "nftables":
                success = self._nftables_block(ip)
            elif self.backend == "dummy":
                logger.logger.info(f"[FIREWALL-DUMMY] Would block IP {ip} (Reason: {reason})")
                success = True
            else:
                logger.logger.error(f"Unknown firewall backend: {self.backend}")
                return False
            
            if success:
                self._blocked_ips[ip] = rule
                self._save_blocked_ips()
                
                logger.logger.warning(
                    f"BLOCKED {ip} — {attack_class} ({severity}) — "
                    f"duration: {duration_seconds}s — reason: {reason} — "
                    f"alert: {alert_id}"
                )
            else:
                logger.logger.error(f"Failed to block {ip}")
            
            return success
    
    def unblock_ip(self, ip: str) -> bool:
        """Remove block for an IP."""
        with self._lock:
            if ip not in self._blocked_ips:
                logger.logger.info(f"IP {ip} is not blocked")
                return True
            
            success = False
            if self.backend == "iptables":
                success = self._iptables_unblock(ip)
            elif self.backend == "nftables":
                success = self._nftables_unblock(ip)
            elif self.backend == "dummy":
                logger.logger.info(f"[FIREWALL-DUMMY] Would unblock IP {ip}")
                success = True
            
            if success:
                del self._blocked_ips[ip]
                self._save_blocked_ips()
                logger.logger.info(f"Unblocked {ip}")
            
            return success
    
    def is_blocked(self, ip: str) -> bool:
        """Check if an IP is currently blocked."""
        with self._lock:
            return ip in self._blocked_ips
    
    def get_blocked_ips(self) -> List[Dict]:
        """Get list of all blocked IPs with their details."""
        with self._lock:
            return [asdict(rule) for rule in self._blocked_ips.values()]
    
    def get_block_count(self) -> int:
        """Get number of currently blocked IPs."""
        with self._lock:
            return len(self._blocked_ips)
    
    def _iptables_block(self, ip: str) -> bool:
        """Block IP using iptables."""
        cmd = [
            "iptables", "-A", self.chain,
            "-s", ip,
            "-j", "DROP",
            "-m", "comment", "--comment", f"deepdefend-block-{int(time.time())}"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logger.logger.error(f"iptables block failed: {result.stderr}")
                return False
            return True
        except FileNotFoundError:
            logger.logger.error("iptables not found. Is it installed?")
            return False
        except Exception as e:
            logger.logger.error(f"iptables error: {e}")
            return False
    
    def _iptables_unblock(self, ip: str) -> bool:
        """Remove iptables block for IP."""
        cmd = [
            "iptables", "-D", self.chain,
            "-s", ip,
            "-j", "DROP"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            # Return code may be non-zero if rule doesn't exist, that's OK
            return True
        except Exception as e:
            logger.logger.error(f"iptables unblock error: {e}")
            return False
    
    def _nftables_block(self, ip: str) -> bool:
        """Block IP using nftables."""
        cmd = [
            "nft", "add", "rule", "inet", "filter", self.chain,
            "ip", "saddr", ip, "drop",
            "comment", f"\"deepdefend-block-{int(time.time())}\""
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logger.logger.error(f"nftables block failed: {result.stderr}")
                return False
            return True
        except FileNotFoundError:
            logger.logger.error("nftables not found. Is it installed?")
            return False
        except Exception as e:
            logger.logger.error(f"nftables error: {e}")
            return False
    
    def _nftables_unblock(self, ip: str) -> bool:
        """Remove nftables block for IP."""
        # nftables uses handles for deletion, this is approximate
        cmd = [
            "nft", "delete", "rule", "inet", "filter", self.chain,
            "ip", "saddr", ip, "drop"
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=False)
            return True
        except Exception as e:
            logger.logger.error(f"nftables unblock error: {e}")
            return False
    
    def _ensure_chain(self):
        """Ensure the block chain exists."""
        if self.backend == "iptables":
            # Check if chain exists
            check = subprocess.run(
                ["iptables", "-L", self.chain],
                capture_output=True, text=True
            )
            
            if check.returncode != 0:
                # Create chain
                subprocess.run(
                    ["iptables", "-N", self.chain],
                    capture_output=True, check=False
                )
                
                # Add jump rule from INPUT to our chain
                subprocess.run(
                    ["iptables", "-I", "INPUT", "-j", self.chain],
                    capture_output=True, check=False
                )
                
                logger.logger.info(f"Created iptables chain: {self.chain}")
        
        elif self.backend == "nftables":
            # nftables chains are created in the table definition
            # Assume the chain exists in the filter table
            pass
        elif self.backend == "dummy":
            logger.logger.info("Firewall chain initialization skipped (dummy backend)")
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format."""
        import ipaddress
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    def _save_blocked_ips(self):
        """Save blocked IPs to persistent storage."""
        self.blocked_ips_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "updated": datetime.now(timezone.utc).isoformat(),
            "backend": self.backend,
            "chain": self.chain,
            "blocked_ips": {
                ip: asdict(rule) for ip, rule in self._blocked_ips.items()
            }
        }
        
        # Write atomically
        temp_path = self.blocked_ips_file.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        temp_path.replace(self.blocked_ips_file)
    
    def _load_blocked_ips(self):
        """Load blocked IPs from persistent storage."""
        if not self.blocked_ips_file.exists():
            return
        
        try:
            with open(self.blocked_ips_file) as f:
                data = json.load(f)
            
            for ip, rule_data in data.get("blocked_ips", {}).items():
                self._blocked_ips[ip] = BlockRule(**rule_data)
            
            logger.logger.info(f"Loaded {len(self._blocked_ips)} blocked IPs from disk")
        except Exception as e:
            logger.logger.error(f"Failed to load blocked IPs: {e}")
    
    def _cleanup_expired(self):
        """Remove expired blocks."""
        now = datetime.now(timezone.utc)
        expired = []
        
        for ip, rule in self._blocked_ips.items():
            if rule.expires_at != "permanent":
                try:
                    expires = datetime.fromisoformat(rule.expires_at)
                    if expires < now:
                        expired.append(ip)
                except ValueError:
                    pass
        
        for ip in expired:
            self.unblock_ip(ip)
            logger.logger.info(f"Removed expired block for {ip}")
        
        if expired:
            logger.logger.info(f"Cleaned up {len(expired)} expired blocks")
    
    def cleanup_all(self):
        """Remove all blocks (used for shutdown or testing)."""
        ips = list(self._blocked_ips.keys())
        for ip in ips:
            self.unblock_ip(ip)
        logger.logger.info(f"Removed all {len(ips)} blocks")