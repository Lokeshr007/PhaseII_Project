"""
Fusion engine — combines Snort and ML alerts into unified threat assessment.

Core logic:
- Snort alert + High ML anomaly score → Critical (two detection methods agree)
- Snort alert + Low ML anomaly score → Medium (known signature, benign-looking)
- No Snort + High ML anomaly score → High (possible novel/zero-day attack)
- No Snort + Medium ML score → Low (behavioral anomaly only)
- No Snort + Low ML score → Suppressed (noise)

Also handles:
- Alert deduplication (same src_ip, same attack_class, same window)
- Alert correlation (multiple alerts from same source → aggregated)
- Attack chain detection (port scan → exploit → privilege escalation)
"""

import time
import uuid
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib

from deepdefend.engine.inference import DetectionResult
from deepdefend.engine.snort import SnortAlert
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FusedAlert:
    """Unified alert from combined detection sources."""
    alert_id: str
    timestamp: str
    
    # Detection sources
    sources: List[str]  # ["snort", "anomaly_ml", "specialist_ml"]
    
    # Attack classification
    attack_class: str  # Normalized across Snort and ML
    severity: str  # critical, high, medium, low
    confidence: float  # 0.0 to 1.0
    
    # Network context
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: str
    
    # Evidence
    snort_alerts: List[Dict] = field(default_factory=list)
    ml_evidence: Dict = field(default_factory=dict)
    
    # Correlation
    correlated_alerts: List[str] = field(default_factory=list)  # Related alert IDs
    occurrence_count: int = 1  # How many times this pattern was seen
    
    # Action recommendation
    recommended_action: str = "log"  # log, notify, block, quarantine
    
    # Metadata
    fusion_rules_triggered: List[str] = field(default_factory=list)
    raw_data: Dict = field(default_factory=dict)


class FusionEngine:
    """
    Correlates and fuses alerts from Snort and ML anomaly engine.
    
    Implements the fusion rules from our architecture design:
    - Two independent detection methods agreeing = high confidence
    - Snort-only = medium confidence (known attack, but behaved normally)
    - ML-only = variable confidence (novel attack or false positive)
    
    Also tracks alert history for:
    - Deduplication (same source + same class in short window)
    - Burst detection (many alerts from same source = more serious)
    - Attack chain correlation (sequential attack stages)
    """
    
    def __init__(self,
                 dedup_window_seconds: float = 60.0,
                 correlation_window_seconds: float = 300.0,
                 burst_threshold: int = 10):
        
        self.dedup_window_seconds = dedup_window_seconds
        self.correlation_window_seconds = correlation_window_seconds
        self.burst_threshold = burst_threshold
        
        # Alert history for dedup and correlation
        self._alert_history: deque = deque(maxlen=10000)
        self._source_tracker: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )  # src_ip -> recent alerts
        
        # Fusion rules
        self.fusion_rules = [
            self._rule_dual_confirmation,
            self._rule_snort_only,
            self._rule_ml_high_confidence,
            self._rule_burst_detection,
            self._rule_attack_chain,
        ]
        
        # Callback
        self.on_fused_alert: Optional[callable] = None
        
        logger.logger.info(
            f"FusionEngine initialized. "
            f"Dedup window: {dedup_window_seconds}s, "
            f"Correlation window: {correlation_window_seconds}s"
        )
    
    def fuse(self, 
             ml_alerts: List[DetectionResult],
             snort_alerts: List[SnortAlert]) -> List[FusedAlert]:
        """
        Combine ML and Snort alerts into unified threat assessment.
        
        Args:
            ml_alerts: Detection results from ML engine
            snort_alerts: Alerts from Snort
        
        Returns:
            List of fused, deduplicated alerts
        """
        fused_alerts = []
        
        # ─── Step 1: Match Snort alerts to ML alerts by (src_ip, dst_ip) ────
        matched_pairs = self._match_alerts(ml_alerts, snort_alerts)
        
        # ─── Step 2: Create fused alerts for matched pairs ────────────────
        for ml_alert, snort_alert in matched_pairs:
            fused = self._create_fused_alert(
                ml_alert=ml_alert,
                snort_alert=snort_alert,
                sources=["snort", "anomaly_ml", "specialist_ml"]
            )
            fused_alerts.append(fused)
        
        # ─── Step 3: Handle unmatched ML alerts ────────────────────────────
        matched_ml_ids = {ml.alert_id for ml, _ in matched_pairs}
        for ml_alert in ml_alerts:
            if ml_alert.alert_id not in matched_ml_ids:
                fused = self._create_fused_alert(
                    ml_alert=ml_alert,
                    snort_alert=None,
                    sources=["anomaly_ml", "specialist_ml"]
                )
                fused_alerts.append(fused)
        
        # ─── Step 4: Handle unmatched Snort alerts ─────────────────────────
        matched_snort_ids = {s.alert_id for _, s in matched_pairs}
        for snort_alert in snort_alerts:
            if snort_alert.alert_id not in matched_snort_ids:
                fused = self._create_fused_alert(
                    ml_alert=None,
                    snort_alert=snort_alert,
                    sources=["snort"]
                )
                fused_alerts.append(fused)
        
        # ─── Step 5: Apply fusion rules ────────────────────────────────────
        for alert in fused_alerts:
            for rule in self.fusion_rules:
                try:
                    rule(alert)
                except Exception as e:
                    logger.logger.error(f"Fusion rule error: {e}")
        
        # ─── Step 6: Deduplicate ───────────────────────────────────────────
        fused_alerts = self._deduplicate(fused_alerts)
        
        # ─── Step 7: Update history ────────────────────────────────────────
        now = time.time()
        for alert in fused_alerts:
            self._alert_history.append((now, alert))
            self._source_tracker[alert.src_ip].append((now, alert))
        
        # ─── Step 8: Fire callbacks ────────────────────────────────────────
        for alert in fused_alerts:
            if self.on_fused_alert:
                try:
                    self.on_fused_alert(alert)
                except Exception as e:
                    logger.logger.error(f"Fused alert callback failed: {e}")
        
        return fused_alerts
    
    def _match_alerts(self, 
                      ml_alerts: List[DetectionResult],
                      snort_alerts: List[SnortAlert]) -> List[Tuple]:
        """Match ML alerts to Snort alerts by network context."""
        matched = []
        
        # Build lookup of Snort alerts by key
        snort_by_key = defaultdict(list)
        for sa in snort_alerts:
            key = (sa.src_ip, sa.dst_ip)
            snort_by_key[key].append(sa)
        
        # Match ML alerts
        used_snort_ids = set()
        
        for ml_alert in ml_alerts:
            key = (ml_alert.src_ip, ml_alert.dst_ip)
            candidates = snort_by_key.get(key, [])
            
            # Find best matching Snort alert (same or nearby port)
            best_match = None
            best_score = 0
            
            for sa in candidates:
                if sa.alert_id in used_snort_ids:
                    continue
                
                # Score match quality
                score = 1.0  # Same IPs
                if sa.dst_port == ml_alert.dst_port:
                    score += 0.5  # Same destination port
                if sa.src_port == ml_alert.src_port:
                    score += 0.3  # Same source port
                
                if score > best_score:
                    best_score = score
                    best_match = sa
            
            if best_match and best_score > 0.8:  # Threshold for matching
                matched.append((ml_alert, best_match))
                used_snort_ids.add(best_match.alert_id)
        
        return matched
    
    def _create_fused_alert(self,
                           ml_alert: Optional[DetectionResult],
                           snort_alert: Optional[SnortAlert],
                           sources: List[str]) -> FusedAlert:
        """Create a FusedAlert from available sources."""
        
        # Determine attack class
        if ml_alert and ml_alert.predicted_class != "normal":
            attack_class = ml_alert.predicted_class
        elif snort_alert:
            # Map Snort classification to our classes
            attack_class = self._map_snort_classification(snort_alert.classification)
        else:
            attack_class = "unknown"
        
        # Determine confidence
        confidence = 0.0
        confidence_sources = 0
        
        if ml_alert:
            confidence += ml_alert.confidence
            confidence_sources += 1
        if snort_alert:
            # Snort has implicit high confidence for known signatures
            confidence += 0.85
            confidence_sources += 1
        
        if confidence_sources > 0:
            confidence /= confidence_sources
        
        # Network context (prefer ML if available)
        src_ip = (ml_alert.src_ip if ml_alert else snort_alert.src_ip)
        dst_ip = (ml_alert.dst_ip if ml_alert else snort_alert.dst_ip)
        src_port = (ml_alert.src_port if ml_alert else snort_alert.src_port)
        dst_port = (ml_alert.dst_port if ml_alert else snort_alert.dst_port)
        protocol = (ml_alert.protocol if ml_alert else snort_alert.protocol)
        
        # Severity based on confidence
        severity = "low"
        if confidence >= 0.9:
            severity = "critical"
        elif confidence >= 0.75:
            severity = "high"
        elif confidence >= 0.5:
            severity = "medium"
        
        # Build evidence
        snort_data = []
        if snort_alert:
            snort_data = [{
                "rule_id": snort_alert.rule_id,
                "rule_name": snort_alert.rule_name,
                "classification": snort_alert.classification,
                "priority": snort_alert.priority,
                "message": snort_alert.message,
            }]
        
        ml_data = {}
        if ml_alert:
            ml_data = {
                "anomaly_score": ml_alert.anomaly_score,
                "predicted_class": ml_alert.predicted_class,
                "confidence": ml_alert.confidence,
                "contributing_features": ml_alert.contributing_features,
                "class_probabilities": ml_alert.class_probabilities,
            }
        
        return FusedAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            sources=sources,
            attack_class=attack_class,
            severity=severity,
            confidence=confidence,
            src_ip=src_ip,
            src_port=src_port,
            dst_ip=dst_ip,
            dst_port=dst_port,
            protocol=protocol,
            snort_alerts=snort_data,
            ml_evidence=ml_data,
            raw_data={
                "ml_alert": ml_alert.__dict__ if ml_alert else None,
                "snort_alert": snort_alert.to_dict() if snort_alert else None,
            }
        )
    
    # ─── Fusion Rules ─────────────────────────────────────────────
    
    def _rule_dual_confirmation(self, alert: FusedAlert):
        """Both Snort and ML agree → escalate severity."""
        if "snort" in alert.sources and "anomaly_ml" in alert.sources:
            if alert.severity == "medium":
                alert.severity = "high"
            elif alert.severity == "high":
                alert.severity = "critical"
            alert.confidence = min(1.0, alert.confidence * 1.2)
            alert.fusion_rules_triggered.append("dual_confirmation")
            alert.recommended_action = "block" if alert.severity == "critical" else "notify"
    
    def _rule_snort_only(self, alert: FusedAlert):
        """Snort detected but ML didn't → known attack, moderate confidence."""
        if "snort" in alert.sources and "anomaly_ml" not in alert.sources:
            alert.severity = "medium"
            alert.confidence = 0.7
            alert.fusion_rules_triggered.append("snort_only")
            alert.recommended_action = "notify"
    
    def _rule_ml_high_confidence(self, alert: FusedAlert):
        """ML high confidence without Snort → possible zero-day."""
        if "anomaly_ml" in alert.sources and "snort" not in alert.sources:
            if alert.confidence > 0.85:
                alert.severity = "high"
                alert.attack_class = alert.attack_class or "unknown_anomaly"
                alert.fusion_rules_triggered.append("ml_high_confidence_zero_day")
                alert.recommended_action = "notify"
    
    def _rule_burst_detection(self, alert: FusedAlert):
        """Many alerts from same source → may be automated attack."""
        recent = self._source_tracker.get(alert.src_ip, deque())
        # Count alerts in correlation window
        cutoff = time.time() - self.correlation_window_seconds
        recent_count = sum(1 for t, a in recent if t > cutoff)
        
        if recent_count >= self.burst_threshold:
            alert.severity = "critical"
            alert.occurrence_count = recent_count
            alert.fusion_rules_triggered.append("burst_detection")
            alert.recommended_action = "block"
            alert.ml_evidence["burst_detected"] = True
            alert.ml_evidence["recent_alerts_from_source"] = recent_count
    
    def _rule_attack_chain(self, alert: FusedAlert):
        """Detect sequential attack stages from same source."""
        recent = self._source_tracker.get(alert.src_ip, deque())
        cutoff = time.time() - self.correlation_window_seconds
        
        recent_classes = [
            a.attack_class for t, a in recent 
            if t > cutoff and a.alert_id != alert.alert_id
        ]
        
        # Known attack chains
        chains = [
            (["probe", "r2l"], "recon_to_access"),
            (["r2l", "u2r"], "access_to_privilege_escalation"),
            (["probe", "r2l", "u2r"], "full_attack_chain"),
        ]
        
        for chain_stages, chain_name in chains:
            if all(stage in recent_classes for stage in chain_stages):
                alert.severity = "critical"
                alert.fusion_rules_triggered.append(f"attack_chain_{chain_name}")
                alert.recommended_action = "block"
                alert.ml_evidence["attack_chain"] = chain_name
                alert.ml_evidence["previous_stages"] = recent_classes
                break
    
    # ─── Deduplication ──────────────────────────────────────────
    
    def _deduplicate(self, alerts: List[FusedAlert]) -> List[FusedAlert]:
        """
        Remove duplicate alerts within the dedup window.
        
        Two alerts are duplicates if:
        - Same source IP
        - Same attack class
        - Within dedup_window_seconds
        """
        if not alerts:
            return []
        
        deduped = []
        now = time.time()
        
        for alert in sorted(alerts, key=lambda a: a.confidence, reverse=True):
            is_dup = False
            
            for existing in deduped:
                if (existing.src_ip == alert.src_ip and
                    existing.attack_class == alert.attack_class):
                    # Increment occurrence counter on existing alert
                    existing.occurrence_count += 1
                    is_dup = True
                    break
            
            # Also check history
            if not is_dup:
                for t, hist_alert in self._alert_history:
                    if now - t > self.dedup_window_seconds:
                        continue
                    if (hist_alert.src_ip == alert.src_ip and
                        hist_alert.attack_class == alert.attack_class):
                        alert.occurrence_count += 1
                        is_dup = True
                        break
            
            if not is_dup:
                deduped.append(alert)
        
        return deduped
    
    # ─── Helpers ────────────────────────────────────────────────
    
    def _map_snort_classification(self, classification: str) -> str:
        """Map Snort classification to our attack categories."""
        classification = classification.lower()
        
        if any(word in classification for word in ["denial", "dos", "flood"]):
            return "dos"
        if any(word in classification for word in ["scan", "probe", "recon"]):
            return "probe"
        if any(word in classification for word in ["access", "login", "auth", "r2l"]):
            return "r2l"
        if any(word in classification for word in ["privilege", "escalation", "root", "u2r"]):
            return "u2r"
        
        return "unknown"


# ─── Alert class mapping ───────────────────────────────────

def map_snort_to_attack_class(snort_classification: str) -> str:
    """Static method for mapping Snort classifications."""
    mapping = {
        "attempted-dos": "dos",
        "successful-dos": "dos",
        "attempted-recon": "probe",
        "successful-recon": "probe",
        "attempted-user": "r2l",
        "successful-user": "r2l",
        "attempted-admin": "u2r",
        "successful-admin": "u2r",
    }
    return mapping.get(snort_classification.lower(), "unknown")