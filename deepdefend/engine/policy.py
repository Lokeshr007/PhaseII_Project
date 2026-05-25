"""
Action policy engine — decides what to do based on alert severity and rules.

This is the decision layer between detection and response.
Not every alert gets blocked. Not every block is permanent.
The policy engine applies configurable rules to determine:
- Should we auto-block this IP?
- Should we quarantine this device?
- Who needs to be notified, and how urgently?
- What evidence needs to be included?

Policy is defined as a set of rules evaluated in order.
First matching rule wins. This allows both broad defaults and specific overrides.
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from deepdefend.engine.fusion import FusedAlert
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class ActionType(Enum):
    """Possible response actions."""
    LOG_ONLY = "log_only"
    NOTIFY = "notify"
    RECOMMEND_BLOCK = "recommend_block"
    AUTO_BLOCK = "auto_block"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"


@dataclass
class PolicyDecision:
    """Result of policy evaluation for an alert."""
    alert: FusedAlert
    actions: List[ActionType]
    block_duration_seconds: int = 3600  # Default 1 hour
    notification_channels: List[str] = field(default_factory=list)
    notification_priority: str = "normal"
    requires_approval: bool = False
    reason: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class PolicyRule:
    """A single policy rule."""
    name: str
    conditions: Dict  # Conditions that must match
    actions: List[ActionType]
    priority: int = 100  # Lower = higher priority
    enabled: bool = True


class ActionPolicyEngine:
    """
    Policy engine that determines response actions based on alert characteristics.
    
    Policy rules are evaluated in priority order (lower number = higher priority).
    First rule that matches all conditions determines the actions.
    
    Conditions can check:
    - alert.severity (critical, high, medium, low)
    - alert.attack_class (dos, probe, r2l, u2r)
    - alert.confidence (0.0 to 1.0)
    - alert.sources (snort, anomaly_ml, specialist_ml)
    - alert.src_ip (specific IP, CIDR range, or "internal"/"external")
    - Time of day, day of week
    - Occurrence count (how many times seen)
    """
    
    def __init__(self, 
                 auto_block_enabled: bool = False,
                 quarantine_enabled: bool = False,
                 internal_networks: List[str] = None,
                 critical_assets: List[str] = None,
                 blocked_ips_file: str = "/var/lib/deepdefend/blocked_ips.json",
                 notification_severity_threshold: str = "high"):
        
        self.auto_block_enabled = auto_block_enabled
        self.quarantine_enabled = quarantine_enabled
        self.internal_networks = internal_networks or ["192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12"]
        self.critical_assets = critical_assets or []
        self.blocked_ips_file = Path(blocked_ips_file)
        self.notification_severity_threshold = notification_severity_threshold
        
        # Action history for rate limiting
        self._action_history: List[tuple] = []
        self._action_limit_window = 300  # 5 minutes
        self._max_actions_per_window = 50
        
        # Initialize default rules
        self.rules: List[PolicyRule] = self._default_rules()
        
        # Custom rules from config
        self.custom_rules: List[PolicyRule] = []
        
        logger.logger.info(
            f"ActionPolicyEngine initialized. "
            f"Auto-block: {auto_block_enabled}, "
            f"Quarantine: {quarantine_enabled}"
        )
    
    def evaluate(self, alert: FusedAlert) -> PolicyDecision:
        """
        Evaluate alert against policy rules and determine actions.
        
        Returns:
            PolicyDecision with recommended actions
        """
        # Combine and sort rules by priority
        all_rules = sorted(
            self.rules + self.custom_rules,
            key=lambda r: r.priority
        )
        
        # Find first matching rule
        for rule in all_rules:
            if not rule.enabled:
                continue
            
            if self._rule_matches(rule, alert):
                actions = rule.actions.copy()
                
                # Override based on global settings
                if ActionType.AUTO_BLOCK in actions and not self.auto_block_enabled:
                    actions.remove(ActionType.AUTO_BLOCK)
                    if ActionType.RECOMMEND_BLOCK not in actions:
                        actions.append(ActionType.RECOMMEND_BLOCK)
                
                if ActionType.QUARANTINE in actions and not self.quarantine_enabled:
                    actions.remove(ActionType.QUARANTINE)
                
                # Rate limit actions
                if not self._can_execute_action():
                    if ActionType.AUTO_BLOCK in actions:
                        actions.remove(ActionType.AUTO_BLOCK)
                        actions.append(ActionType.RECOMMEND_BLOCK)
                
                decision = PolicyDecision(
                    alert=alert,
                    actions=actions,
                    block_duration_seconds=self._block_duration(alert),
                    notification_channels=self._notification_channels(alert),
                    notification_priority=alert.severity,
                    reason=f"Matched rule: {rule.name}",
                    metadata={
                        "rule_name": rule.name,
                        "rule_priority": rule.priority,
                        "global_auto_block": self.auto_block_enabled,
                        "global_quarantine": self.quarantine_enabled,
                    }
                )
                
                logger.logger.info(
                    f"Policy decision for {alert.alert_id}: "
                    f"{[a.value for a in actions]} (rule: {rule.name})"
                )
                
                return decision
        
        # Default: log only
        logger.logger.debug(f"No policy rule matched for {alert.alert_id}, defaulting to LOG_ONLY")
        
        return PolicyDecision(
            alert=alert,
            actions=[ActionType.LOG_ONLY],
            reason="No matching rule, default action"
        )
    
    def _rule_matches(self, rule: PolicyRule, alert: FusedAlert) -> bool:
        """Check if all conditions in the rule match the alert."""
        conditions = rule.conditions
        
        # Check severity
        if "severity" in conditions:
            if isinstance(conditions["severity"], list):
                if alert.severity not in conditions["severity"]:
                    return False
            elif alert.severity != conditions["severity"]:
                return False
        
        # Check attack class
        if "attack_class" in conditions:
            if isinstance(conditions["attack_class"], list):
                if alert.attack_class not in conditions["attack_class"]:
                    return False
            elif alert.attack_class != conditions["attack_class"]:
                return False
        
        # Check minimum confidence
        if "min_confidence" in conditions:
            if alert.confidence < conditions["min_confidence"]:
                return False
        
        # Check detection sources
        if "has_source" in conditions:
            required = conditions["has_source"]
            if isinstance(required, str):
                required = [required]
            if not all(src in alert.sources for src in required):
                return False
        
        # Check if internal source
        if "is_internal" in conditions:
            is_internal = self._is_internal_ip(alert.src_ip)
            if is_internal != conditions["is_internal"]:
                return False
        
        # Check if target is critical asset
        if "target_is_critical" in conditions:
            is_critical = alert.dst_ip in self.critical_assets
            if is_critical != conditions["target_is_critical"]:
                return False
        
        # Check occurrence count
        if "min_occurrences" in conditions:
            if alert.occurrence_count < conditions["min_occurrences"]:
                return False
        
        # Check time of day (format: "HH:MM-HH:MM")
        if "time_range" in conditions:
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            time_range = conditions["time_range"]
            start, end = time_range.split("-")
            if not (start <= current_time <= end):
                return False
        
        return True
    
    def _default_rules(self) -> List[PolicyRule]:
        """Generate default policy rules."""
        return [
            # CRITICAL: U2R on critical asset → full response
            PolicyRule(
                name="u2r_on_critical_asset",
                conditions={
                    "severity": "critical",
                    "attack_class": "u2r",
                    "target_is_critical": True,
                },
                actions=[
                    ActionType.AUTO_BLOCK,
                    ActionType.NOTIFY,
                    ActionType.ESCALATE,
                ],
                priority=10
            ),
            
            # CRITICAL: Any critical severity with both detectors → block
            PolicyRule(
                name="critical_dual_confirmation",
                conditions={
                    "severity": "critical",
                    "has_source": ["snort", "anomaly_ml"],
                },
                actions=[
                    ActionType.AUTO_BLOCK,
                    ActionType.NOTIFY,
                ],
                priority=20
            ),
            
            # HIGH: Attack chain detected → block
            PolicyRule(
                name="attack_chain_detected",
                conditions={
                    "severity": ["critical", "high"],
                    "min_occurrences": 3,
                },
                actions=[
                    ActionType.AUTO_BLOCK,
                    ActionType.NOTIFY,
                ],
                priority=30
            ),
            
            # HIGH: U2R or R2L with high confidence → recommend block
            PolicyRule(
                name="high_privilege_attack",
                conditions={
                    "severity": "high",
                    "attack_class": ["u2r", "r2l"],
                    "min_confidence": 0.8,
                },
                actions=[
                    ActionType.RECOMMEND_BLOCK,
                    ActionType.NOTIFY,
                ],
                priority=40
            ),
            
            # MEDIUM-HIGH: DoS from external source → notify
            PolicyRule(
                name="external_dos",
                conditions={
                    "severity": ["high", "medium"],
                    "attack_class": "dos",
                    "is_internal": False,
                },
                actions=[
                    ActionType.NOTIFY,
                ],
                priority=50
            ),
            
            # PROBE: Reconnaissance activity → notify if repeated
            PolicyRule(
                name="repeated_probe",
                conditions={
                    "attack_class": "probe",
                    "min_occurrences": 5,
                },
                actions=[
                    ActionType.NOTIFY,
                ],
                priority=60
            ),
            
            # LOW: Everything else → log only
            PolicyRule(
                name="default_log",
                conditions={},
                actions=[
                    ActionType.LOG_ONLY,
                ],
                priority=100
            ),
        ]
    
    def add_rule(self, rule: PolicyRule):
        """Add a custom policy rule."""
        self.custom_rules.append(rule)
        logger.logger.info(f"Added custom policy rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a custom policy rule by name."""
        self.custom_rules = [
            r for r in self.custom_rules if r.name != rule_name
        ]
    
    def _block_duration(self, alert: FusedAlert) -> int:
        """Determine block duration based on severity."""
        durations = {
            "critical": 86400,   # 24 hours
            "high": 3600,        # 1 hour
            "medium": 1800,      # 30 minutes
            "low": 600,          # 10 minutes
        }
        return durations.get(alert.severity, 3600)
    
    def _notification_channels(self, alert: FusedAlert) -> List[str]:
        """Determine which notification channels to use."""
        channels = ["siem"]  # Always log to SIEM
        
        if alert.severity in ["critical", "high"]:
            channels.append("slack")
            channels.append("email")
        
        if alert.severity == "critical":
            channels.append("pagerduty")
        
        return channels
    
    def _is_internal_ip(self, ip: str) -> bool:
        """Check if IP is in internal network ranges."""
        import ipaddress
        
        try:
            ip_addr = ipaddress.ip_address(ip)
            
            for network_str in self.internal_networks:
                network = ipaddress.ip_network(network_str, strict=False)
                if ip_addr in network:
                    return True
            
            return False
        except ValueError:
            return False
    
    def _can_execute_action(self) -> bool:
        """Rate limit actions to prevent storm from false positives."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._action_limit_window)
        
        # Clean old history
        self._action_history = [
            (t, a) for t, a in self._action_history
            if t > cutoff
        ]
        
        return len(self._action_history) < self._max_actions_per_window
    
    def record_action(self, action_type: ActionType):
        """Record that an action was executed for rate limiting."""
        self._action_history.append((datetime.now(timezone.utc), action_type))
    
    def get_stats(self) -> Dict:
        """Get policy engine statistics."""
        recent_actions = len(self._action_history)
        return {
            "auto_block_enabled": self.auto_block_enabled,
            "quarantine_enabled": self.quarantine_enabled,
            "rules_count": len(self.rules) + len(self.custom_rules),
            "actions_in_last_5min": recent_actions,
            "action_limit": self._max_actions_per_window,
            "rate_limited": recent_actions >= self._max_actions_per_window,
        }