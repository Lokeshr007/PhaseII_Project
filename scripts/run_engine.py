#!/usr/bin/env python3
"""
DeepDefend NIDS — Main Engine Runner

This is the entry point for production deployment.
Starts all components and runs the detection pipeline continuously.

Architecture:
    Packet Capture → Flow Aggregator → ML Inference → Fusion ← Snort
                                              ↓
                                         Policy Engine
                                              ↓
                                    Response (Block/Notify/Log)
"""

import sys
import signal
import time
import threading
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepdefend.config.settings import get_config, ConfigLoader
from deepdefend.data.preprocessor import FeaturePreprocessor
from deepdefend.engine.aggregator import FlowAggregator, PacketParser
from deepdefend.engine.inference import InferenceEngine, create_inference_engine
from deepdefend.engine.snort import SnortManager
from deepdefend.engine.fusion import FusionEngine
from deepdefend.engine.policy import ActionPolicyEngine, ActionType
from deepdefend.response.firewall import FirewallManager
from deepdefend.response.incident import IncidentGenerator
from deepdefend.response.notify import NotificationManager
from deepdefend.operations.siem import SIEMFormatter
from deepdefend.operations.monitor import MetricsCollector
from deepdefend.operations.health import HealthChecker
from deepdefend.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


class DeepDefendEngine:
    """
    Main engine orchestrator for DeepDefend NIDS.
    
    Manages the lifecycle of all components:
    - Start/stop order
    - Thread coordination
    - Graceful shutdown
    - Health monitoring
    """
    
    def __init__(self, config_path: str = None, mode: str = "production"):
        self.config = ConfigLoader.load(config_path, mode)
        setup_logging(level=self.config.log_level)
        
        logger.logger.info("=" * 60)
        logger.logger.info("DEEPDEFEND NIDS STARTING")
        logger.logger.info(f"Mode: {self.config.mode}")
        logger.logger.info("=" * 60)
        
        # Dashboard settings
        self.dashboard_url = "http://localhost:5000"
        self.dashboard_enabled = True
        
        # Initialize components
        self._init_components()
        
        # State
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Threads
        self._threads = []
    
    def _init_components(self):
        """Initialize all system components."""
        cfg = self.config
        
        # ─── Operations Layer ────────────────────────────
        self.metrics = MetricsCollector()
        self.health = HealthChecker(
            cpu_warning_threshold=80.0,
            cpu_critical_threshold=95.0,
            memory_warning_threshold_mb=1024,
            memory_critical_threshold_mb=2048,
        )
        self.siem = SIEMFormatter(
            facility=cfg.siem.syslog_facility,
            include_raw_features=cfg.siem.include_raw_features,
            output="stdout" if cfg.mode == "development" else "syslog_local"
        )
        self.incident_generator = IncidentGenerator()
        
        # ─── Detection Layer ──────────────────────────────
        # Models
        self.inference_engine = create_inference_engine(
            model_path=cfg.model_path,
            batch_size=256
        )
        
        # Flow aggregator
        self.flow_aggregator = FlowAggregator(
            window_seconds=cfg.flow.window_seconds,
            slide_seconds=cfg.flow.slide_seconds
        )
        
        # Snort
        self.snort_manager = SnortManager(
            binary_path=cfg.snort.binary_path,
            config_path=cfg.snort.config_path,
            interface=cfg.snort.interface,
            enabled=cfg.snort.enabled
        )
        
        # Fusion
        self.fusion_engine = FusionEngine(
            dedup_window_seconds=60.0,
            correlation_window_seconds=300.0,
            burst_threshold=10
        )
        
        # ─── Response Layer ───────────────────────────────
        self.policy_engine = ActionPolicyEngine(
            auto_block_enabled=cfg.response.auto_block,
            quarantine_enabled=cfg.response.quarantine_enabled,
        )
        
        self.firewall = FirewallManager(
            backend=cfg.response.firewall_backend,
            blocked_ips_file=cfg.response.blocked_ips_file,
        )
        
        self.notifier = NotificationManager(
            slack_webhook=cfg.notification.slack_webhook,
            email_config={
                "smtp_host": cfg.notification.email_smtp_host,
                "recipients": cfg.notification.email_recipients,
            } if cfg.notification.email_smtp_host else None,
            pagerduty_key=cfg.notification.pagerduty_key,
            incident_generator=self.incident_generator,
        )
        
        # ─── Wire up health checker references ────────────
        self.health.flow_aggregator = self.flow_aggregator
        self.health.inference_engine = self.inference_engine
        self.health.snort_manager = self.snort_manager
        self.health.fusion_engine = self.fusion_engine
        self.health.firewall_manager = self.firewall
        self.health.metrics_collector = self.metrics
        
        # ─── Wire up callbacks ────────────────────────────
        self.flow_aggregator.on_window_complete = self._on_flow_window
        self.snort_manager.on_snort_alert = self._on_snort_alert
        self.inference_engine.on_alert = self._on_ml_alert
        
        logger.logger.info("All components initialized")
    
    def start(self):
        """Start the engine and all sub-components."""
        logger.logger.info("Starting DeepDefend engine...")
        
        # Start inference engine
        self.inference_engine.start()
        self.metrics.set_inference_status(True)
        self._update_dashboard_status("engine_running", True)
        self._update_dashboard_status("models_loaded", True)
        
        # Start Snort
        if self.config.snort.enabled:
            snort_started = self.snort_manager.start()
            self.metrics.set_snort_status(snort_started)
        
        # Start system monitoring
        self.metrics.start_system_monitoring(interval=15.0)
        
        # Start health check server (runs in background)
        self._start_health_server()
        
        # Start metrics server (runs in background)
        self._start_metrics_server()
        
        # In development mode, start a simulator to feed data
        if self.config.mode == "development":
            self._start_simulator()
        
        self._running = True
        
        # Main loop — process flow windows
        self._main_loop()
    
    def _start_simulator(self):
        """Starts a background thread to simulate network traffic."""
        def simulator():
            from deepdefend.engine.aggregator import Packet, Protocol
            import random
            import time
            
            logger.logger.info("Simulation started: Generating synthetic network traffic")
            
            attack_profiles = [
                {"name": "normal", "src": "192.168.1.50", "dst": "8.8.8.8", "port": 443, "freq": 10},
                {"name": "dos", "src": "10.0.0.1", "dst": "192.168.1.10", "port": 80, "freq": 100},
                {"name": "probe", "src": "172.16.0.5", "dst": "192.168.1.20", "port": random.randint(1000, 5000), "freq": 5},
                {"name": "r2l", "src": "10.0.0.170", "dst": "192.168.1.123", "port": 21, "freq": 3},
                {"name": "u2r", "src": "10.0.0.18", "dst": "192.168.1.6", "port": 22, "freq": 1},
            ]
            
            while self._running:
                now = time.time()
                profile = random.choice(attack_profiles)
                
                # Update port for probe simulation
                if profile["name"] == "probe":
                    profile["port"] = random.randint(1000, 10000)
                
                for _ in range(profile["freq"]):
                    pkt = Packet(
                        timestamp=now,
                        src_ip=profile["src"],
                        dst_ip=profile["dst"],
                        src_port=random.randint(49152, 65535),
                        dst_port=profile["port"],
                        protocol=Protocol.TCP,
                        tcp_flags=0x02,  # SYN
                        payload_size=random.randint(64, 1500)
                    )
                    self.flow_aggregator.process_packet(pkt)
                
                time.sleep(0.5)
                
        thread = threading.Thread(target=simulator, name="network-simulator", daemon=True)
        thread.start()
        self._threads.append(thread)

    def stop(self):
        """Gracefully stop all components."""
        logger.logger.info("Shutting down DeepDefend engine...")
        
        self._running = False
        self._shutdown_event.set()
        
        # Stop in reverse order
        self.inference_engine.stop()
        self.snort_manager.stop()
        self.metrics.stop_system_monitoring()
        
        # Cleanup firewall (optional — we could leave blocks in place)
        # self.firewall.cleanup_all()
        
        logger.logger.info("DeepDefend engine stopped")
    
    def _main_loop(self):
        """
        Main processing loop.
        
        In production, this would receive packets from capture.
        For now, it processes flow windows from the aggregator.
        """
        logger.logger.info("Main processing loop started")
        
        # Batch buffers for alerts
        ml_alert_buffer = []
        snort_alert_buffer = []
        last_fusion_time = time.time()
        fusion_interval = 1.0  # Run fusion every second
        
        while self._running:
            try:
                # Get ML alerts from inference engine
                ml_alerts = self.inference_engine.get_alerts(timeout=0.1)
                ml_alert_buffer.extend(ml_alerts)
                
                # Get Snort alerts
                snort_alerts = self.snort_manager.get_alerts()
                snort_alert_buffer.extend(snort_alerts)
                
                # Process alert buffers periodically
                now = time.time()
                if now - last_fusion_time >= fusion_interval:
                    if ml_alert_buffer or snort_alert_buffer:
                        self._process_alerts(ml_alert_buffer, snort_alert_buffer)
                        ml_alert_buffer.clear()
                        snort_alert_buffer.clear()
                    last_fusion_time = now
                
                # Update metrics
                if self.metrics:
                    inf_metrics = self.inference_engine.get_metrics()
                    self.metrics.record_queue_depth(
                        "inference_input",
                        inf_metrics.get("total_processed", 0)
                    )
                
                # Health recording
                self.health.record_packet_processed()
                self.health.record_inference_complete()
                
                time.sleep(0.001)  # Yield CPU
                
            except KeyboardInterrupt:
                logger.logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.logger.error(f"Main loop error: {e}", exc_info=True)
    
    def _process_alerts(self, ml_alerts: list, snort_alerts: list):
        """Process batched alerts through fusion and response pipeline."""
        
        # ─── Step 1: Fusion ────────────────────────────────
        fused_alerts = self.fusion_engine.fuse(ml_alerts, snort_alerts)
        
        if not fused_alerts:
            return
        
        logger.logger.info(f"Processing {len(fused_alerts)} fused alerts")
        
        # ─── Step 2: Policy evaluation ─────────────────────
        for alert in fused_alerts:
            # Evaluate policy
            decision = self.policy_engine.evaluate(alert)
            
            # Generate incident report
            actions_taken = [a.value for a in decision.actions]
            if ActionType.AUTO_BLOCK in decision.actions:
                actions_taken.append("block")
            
            incident = self.incident_generator.generate(
                alert,
                actions_taken=actions_taken,
                block_duration=decision.block_duration_seconds
            )
            
            # ─── Step 3: Execute actions ───────────────────
            for action in decision.actions:
                if action == ActionType.AUTO_BLOCK:
                    success = self.firewall.block_ip(
                        ip=alert.src_ip,
                        reason=f"{alert.attack_class} attack (confidence: {alert.confidence:.0%})",
                        alert_id=alert.alert_id,
                        attack_class=alert.attack_class,
                        severity=alert.severity,
                        duration_seconds=decision.block_duration_seconds
                    )
                    if success:
                        self.metrics.record_block(alert.severity)
                        self.policy_engine.record_action(action)
                    
                    # Log action
                    self.siem.send(
                        self.siem.format_action(decision, success)
                    )
                
                elif action == ActionType.NOTIFY:
                    self.notifier.send_alert(
                        incident,
                        channels=decision.notification_channels,
                        priority=decision.notification_priority
                    )
                    self.metrics.record_notification(decision.notification_channels[0] if decision.notification_channels else "unknown")
                
                elif action == ActionType.LOG_ONLY:
                    pass  # Handled below
            
            # ─── Step 4: SIEM output ───────────────────────
            self.siem.send(self.siem.format_alert(incident))
            
            # ─── Step 5: Record metrics ────────────────────
            self.metrics.record_alert(
                severity=alert.severity,
                attack_class=alert.attack_class,
                confidence=alert.confidence,
                sources=alert.sources
            )
            self.metrics.record_incident(alert.severity)
            self.metrics.set_active_blocks(self.firewall.get_block_count())
            
            # Log human-readable summary
            logger.logger.info(
                f"ALERT [{alert.severity.upper()}] {alert.attack_class} "
                f"from {alert.src_ip} → {alert.dst_ip}:{alert.dst_port} "
                f"(confidence: {alert.confidence:.0%})"
            )
            
            # ─── Step 6: Push to Dashboard ──────────────────
            self._push_alert_to_dashboard(incident)
    
    def _push_alert_to_dashboard(self, incident_data: dict):
        """Push alert to the web dashboard via HTTP."""
        if not self.dashboard_enabled:
            return
            
        def push_worker():
            try:
                import requests
                # Extract from IncidentReport object
                alert_payload = {
                    "incident_id": incident_data.incident_id,
                    "severity": incident_data.severity,
                    "attack_class": incident_data.attack_class,
                    "confidence": incident_data.confidence,
                    "src_ip": incident_data.src_ip,
                    "src_port": incident_data.src_port,
                    "dst_ip": incident_data.dst_ip,
                    "dst_port": incident_data.dst_port,
                    "protocol": incident_data.protocol,
                    "timestamp": incident_data.timestamp,
                    "anomaly_score": incident_data.anomaly_score,
                    "action_taken": incident_data.action_taken,
                    "evidence_summary": incident_data.evidence_summary
                }
                requests.post(f"{self.dashboard_url}/api/alerts/collect", 
                             json=alert_payload, timeout=0.5)
            except Exception:
                pass # Silent fail if dashboard is down
                
        threading.Thread(target=push_worker, daemon=True).start()

    def _update_dashboard_status(self, key, value):
        """Update a status flag on the dashboard."""
        if not self.dashboard_enabled:
            return
            
        def push_worker():
            try:
                import requests
                requests.post(f"{self.dashboard_url}/api/status/update", 
                             json={key: value}, timeout=0.5)
            except Exception:
                pass
                
        threading.Thread(target=push_worker, daemon=True).start()
    
    def _on_flow_window(self, features: list):
        """Callback when flow aggregator completes a window."""
        if self.inference_engine:
            self.inference_engine.submit({"features": features})
        if self.metrics:
            self.metrics.record_window_complete(len(features))
    
    def _on_snort_alert(self, snort_alert):
        """Callback for Snort alerts."""
        if self.metrics:
            self.metrics.record_snort_alert(snort_alert.classification)
        self.health.record_snort_alert_seen()
    
    def _on_ml_alert(self, ml_result):
        """Callback for ML detection results."""
        if self.metrics:
            self.metrics.record_anomaly()
    
    def _start_health_server(self):
        """Start a minimal HTTP server for health checks."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        health_checker = self.health
        
        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/healthz' or self.path == '/live':
                    status, code = health_checker.get_liveness_response()
                elif self.path == '/ready':
                    status, code = health_checker.get_readiness_response()
                elif self.path == '/health':
                    status, code = health_checker.check_all(), 200
                else:
                    self.send_response(404)
                    return
                
                import json
                import dataclasses
                
                def encode_health(obj):
                    if dataclasses.is_dataclass(obj):
                        return dataclasses.asdict(obj)
                    if hasattr(obj, 'value'): # Handle Enum
                        return obj.value
                    return str(obj)
                    
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(status, default=encode_health).encode())
            
            def log_message(self, format, *args):
                pass  # Suppress HTTP access logs
        
        def run_server():
            server = HTTPServer(('0.0.0.0', 8080), HealthHandler)
            server.serve_forever()
        
        thread = threading.Thread(target=run_server, name="health-server", daemon=True)
        thread.start()
        logger.logger.info("Health check server started on port 8080")
    
    def _start_metrics_server(self):
        """Start a minimal HTTP server for Prometheus metrics."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        metrics_collector = self.metrics
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/metrics':
                    metrics_data = metrics_collector.get_metrics()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(metrics_data)
                else:
                    self.send_response(404)
            
            def log_message(self, format, *args):
                pass
        
        def run_server():
            server = HTTPServer(('0.0.0.0', 9090), MetricsHandler)
            server.serve_forever()
        
        thread = threading.Thread(target=run_server, name="metrics-server", daemon=True)
        thread.start()
        logger.logger.info("Metrics server started on port 9090")


def main():
    parser = argparse.ArgumentParser(description="DeepDefend NIDS Engine")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--mode", default="production", 
                       choices=["development", "production", "testing"])
    
    args = parser.parse_args()
    
    engine = DeepDefendEngine(config_path=args.config, mode=args.mode)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        engine.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        engine.start()
    except KeyboardInterrupt:
        engine.stop()
    except Exception as e:
        logger.logger.error(f"Fatal error: {e}", exc_info=True)
        engine.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()