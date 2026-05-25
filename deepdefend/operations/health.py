"""
Health check system for DeepDefend NIDS.

Monitors all components and provides status for:
- Kubernetes/container health probes (liveness/readiness)
- Operational dashboards
- Automated alerting on component failures
"""

import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: str = ""
    metrics: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class HealthChecker:
    """
    Aggregate health checker for all DeepDefend components.
    
    Checks:
    - Flow aggregator (is it processing packets?)
    - Inference engine (is it running? queue depth ok?)
    - Snort process (is it alive?)
    - Fusion engine (is it producing alerts?)
    - Firewall manager (is iptables accessible?)
    - System resources (CPU, memory within limits?)
    
    Exposes health status for container orchestration.
    """
    
    def __init__(self,
                 cpu_warning_threshold: float = 80.0,
                 cpu_critical_threshold: float = 95.0,
                 memory_warning_threshold_mb: float = 1024,
                 memory_critical_threshold_mb: float = 2048,
                 max_queue_depth: int = 10000,
                 max_inference_gap_seconds: float = 60.0,
                 max_snort_gap_seconds: float = 300.0):
        
        self.cpu_warning = cpu_warning_threshold
        self.cpu_critical = cpu_critical_threshold
        self.memory_warning = memory_warning_threshold_mb
        self.memory_critical = memory_critical_threshold_mb
        self.max_queue_depth = max_queue_depth
        self.max_inference_gap = max_inference_gap_seconds
        self.max_snort_gap = max_snort_gap_seconds
        
        # Component references (set by engine runner)
        self.flow_aggregator = None
        self.inference_engine = None
        self.snort_manager = None
        self.fusion_engine = None
        self.firewall_manager = None
        self.metrics_collector = None
        
        # Tracking
        self._last_packet_time: float = 0.0
        self._last_inference_time: float = 0.0
        self._last_snort_alert_time: float = 0.0
        self._last_fusion_time: float = 0.0
        
        # Callbacks
        self.on_status_change: Optional[Callable] = None
        self._previous_overall: HealthStatus = HealthStatus.UNKNOWN
        
        logger.logger.info("HealthChecker initialized")
    
    def check_all(self) -> Dict[str, ComponentHealth]:
        """Check health of all components."""
        now = datetime.now(timezone.utc).isoformat()
        results = {}
        
        # Check flow aggregator
        flow_health = self._check_flow_aggregator(now)
        results["flow_aggregator"] = flow_health
        
        # Check inference engine
        inference_health = self._check_inference_engine(now)
        results["inference_engine"] = inference_health
        
        # Check Snort
        snort_health = self._check_snort(now)
        results["snort"] = snort_health
        
        # Check fusion engine
        fusion_health = self._check_fusion_engine(now)
        results["fusion"] = fusion_health
        
        # Check firewall
        firewall_health = self._check_firewall(now)
        results["firewall"] = firewall_health
        
        # Check system resources
        system_health = self._check_system_resources(now)
        results["system"] = system_health
        
        # Determine overall status
        overall = self._overall_status(results.values())
        
        if overall != self._previous_overall:
            logger.logger.warning(f"Health status changed: {self._previous_overall.value} → {overall.value}")
            self._previous_overall = overall
            if self.on_status_change:
                self.on_status_change(overall, results)
        
        return results
    
    def is_ready(self) -> bool:
        """Check if system is ready to serve traffic."""
        results = self.check_all()
        overall = self._overall_status(results.values())
        return overall in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    def is_alive(self) -> bool:
        """Check if system is alive (basic liveness)."""
        return True  # If this code runs, we're alive
    
    def _check_flow_aggregator(self, now: str) -> ComponentHealth:
        """Check flow aggregator health."""
        errors = []
        
        if self.flow_aggregator is None:
            return ComponentHealth(
                name="flow_aggregator",
                status=HealthStatus.UNKNOWN,
                message="Not configured",
                last_check=now
            )
        
        # Check if packets are flowing
        if self._last_packet_time > 0:
            gap = time.time() - self._last_packet_time
            if gap > 30:
                errors.append(f"No packets for {gap:.0f}s")
        
        status = HealthStatus.UNHEALTHY if errors else HealthStatus.HEALTHY
        
        return ComponentHealth(
            name="flow_aggregator",
            status=status,
            message="Processing packets" if not errors else "; ".join(errors),
            last_check=now,
            errors=errors,
            metrics={
                "last_packet_seconds_ago": time.time() - self._last_packet_time if self._last_packet_time else -1
            }
        )
    
    def _check_inference_engine(self, now: str) -> ComponentHealth:
        """Check inference engine health."""
        errors = []
        
        if self.inference_engine is None:
            return ComponentHealth(
                name="inference_engine",
                status=HealthStatus.UNKNOWN,
                message="Not configured",
                last_check=now
            )
        
        if not self.inference_engine._running:
            errors.append("Inference engine not running")
        
        # Check inference gap
        if self._last_inference_time > 0:
            gap = time.time() - self._last_inference_time
            if gap > self.max_inference_gap:
                errors.append(f"No inference for {gap:.0f}s")
        
        # Check queue depth
        if self.metrics_collector:
            queue_depth = self.inference_engine.input_queue.qsize() if hasattr(self.inference_engine.input_queue, 'qsize') else 0
            if queue_depth > self.max_queue_depth:
                errors.append(f"Input queue depth critical: {queue_depth}")
        
        if not self.inference_engine._running:
            status = HealthStatus.UNHEALTHY
        elif errors:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return ComponentHealth(
            name="inference_engine",
            status=status,
            message="Running" if not errors else "; ".join(errors),
            last_check=now,
            errors=errors
        )
    
    def _check_snort(self, now: str) -> ComponentHealth:
        """Check Snort health."""
        errors = []
        
        if self.snort_manager is None:
            return ComponentHealth(
                name="snort",
                status=HealthStatus.UNKNOWN,
                message="Not configured",
                last_check=now
            )
        
        if not self.snort_manager.enabled:
            return ComponentHealth(
                name="snort",
                status=HealthStatus.HEALTHY,
                message="Disabled by configuration",
                last_check=now
            )
        
        snort_status = self.snort_manager.get_status()
        
        if not snort_status.get("available", False):
            errors.append("Snort binary not available")
        elif not snort_status.get("running", False):
            errors.append("Snort process not running")
        
        # Check alert gap
        if self._last_snort_alert_time > 0:
            gap = time.time() - self._last_snort_alert_time
            # Only warn if Snort is supposed to be running
            if gap > self.max_snort_gap and snort_status.get("running", False):
                errors.append(f"No Snort alerts for {gap:.0f}s (may be normal)")
        
        if not snort_status.get("available"):
            status = HealthStatus.UNHEALTHY
        elif not snort_status.get("running"):
            status = HealthStatus.UNHEALTHY
        elif errors:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return ComponentHealth(
            name="snort",
            status=status,
            message="Running" if not errors else "; ".join(errors),
            last_check=now,
            metrics=snort_status,
            errors=errors
        )
    
    def _check_fusion_engine(self, now: str) -> ComponentHealth:
        """Check fusion engine health."""
        if self.fusion_engine is None:
            return ComponentHealth(
                name="fusion",
                status=HealthStatus.UNKNOWN,
                message="Not configured",
                last_check=now
            )
        
        return ComponentHealth(
            name="fusion",
            status=HealthStatus.HEALTHY,
            message="Ready",
            last_check=now
        )
    
    def _check_firewall(self, now: str) -> ComponentHealth:
        """Check firewall manager health."""
        if self.firewall_manager is None:
            return ComponentHealth(
                name="firewall",
                status=HealthStatus.UNKNOWN,
                message="Not configured",
                last_check=now
            )
        
        errors = []
        
        # Check if iptables is accessible
        import subprocess
        try:
            result = subprocess.run(
                ["iptables", "-L", "-n"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                errors.append(f"iptables not accessible: {result.stderr[:100]}")
        except FileNotFoundError:
            errors.append("iptables command not found")
        except Exception as e:
            errors.append(f"iptables check failed: {e}")
        
        block_count = self.firewall_manager.get_block_count()
        
        return ComponentHealth(
            name="firewall",
            status=HealthStatus.UNHEALTHY if errors else HealthStatus.HEALTHY,
            message="Ready" if not errors else "; ".join(errors),
            last_check=now,
            metrics={"active_blocks": block_count},
            errors=errors
        )
    
    def _check_system_resources(self, now: str) -> ComponentHealth:
        """Check system resource usage."""
        import psutil
        import os
        
        errors = []
        
        try:
            process = psutil.Process(os.getpid())
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
            }
            
            if cpu_percent > self.cpu_critical:
                errors.append(f"CPU critical: {cpu_percent:.1f}%")
            elif cpu_percent > self.cpu_warning:
                errors.append(f"CPU warning: {cpu_percent:.1f}%")
            
            if memory_mb > self.memory_critical:
                errors.append(f"Memory critical: {memory_mb:.0f}MB")
            elif memory_mb > self.memory_warning:
                errors.append(f"Memory warning: {memory_mb:.0f}MB")
            
            if cpu_percent > self.cpu_critical or memory_mb > self.memory_critical:
                status = HealthStatus.UNHEALTHY
            elif errors:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return ComponentHealth(
                name="system",
                status=status,
                message="OK" if not errors else "; ".join(errors),
                last_check=now,
                metrics=metrics,
                errors=errors
            )
            
        except Exception as e:
            return ComponentHealth(
                name="system",
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {e}",
                last_check=now,
                errors=[str(e)]
            )
    
    def _overall_status(self, component_healths) -> HealthStatus:
        """Determine overall health from component statuses."""
        statuses = [c.status for c in component_healths]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if HealthStatus.UNKNOWN in statuses:
            return HealthStatus.DEGRADED  # Unknown = degraded until confirmed
        return HealthStatus.HEALTHY
    
    def record_packet_processed(self):
        self._last_packet_time = time.time()
    
    def record_inference_complete(self):
        self._last_inference_time = time.time()
    
    def record_snort_alert_seen(self):
        self._last_snort_alert_time = time.time()
    
    def get_liveness_response(self) -> tuple:
        """Return HTTP response for liveness probe."""
        return ({"status": "alive"}, 200)
    
    def get_readiness_response(self) -> tuple:
        """Return HTTP response for readiness probe."""
        if self.is_ready():
            return ({"status": "ready"}, 200)
        else:
            return ({"status": "not ready", "checks": self.check_all()}, 503)