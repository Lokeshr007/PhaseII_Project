"""
Real-time inference server for cascaded anomaly + specialist detection.

Architecture:
┌─────────────────────────────────────────────────────────┐
│                  InferenceEngine                         │
│                                                          │
│  Flow Features ──→ Anomaly Detector ──→ Normal? ──→ OK  │
│                         │                                │
│                         │ Anomalous                      │
│                         ▼                                │
│                  Attack Specialists                      │
│                    (per-class XGBoost)                   │
│                         │                                │
│                         ▼                                │
│                  Probability Calibrator                  │
│                         │                                │
│                         ▼                                │
│                  Structured Alert                        │
└─────────────────────────────────────────────────────────┘

Key design decisions:
- Thread-safe for concurrent producer (capture) and consumer (alert processing)
- Batched inference for throughput
- Graceful degradation under load (skip non-critical processing)
- Structured output that feeds directly into fusion engine
"""

import time
import uuid
import threading
from typing import Dict, Optional, List, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from deepdefend.data.loader import NSL_KDD_COLUMNS
from deepdefend.models.base import BaseDetector
from deepdefend.models.isolation_forest import IsolationForestAnomalyDetector
from deepdefend.models.xgboost_ensemble import XGBoostEnsemble
from deepdefend.models.calibrator import ProbabilityCalibrator
from deepdefend.data.preprocessor import FeaturePreprocessor
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    """Structured result from the inference pipeline."""
    alert_id: str
    timestamp: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    
    # Detection results
    is_anomalous: bool
    anomaly_score: float
    predicted_class: str
    confidence: float
    
    # Per-class probabilities
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Evidence
    contributing_features: List[Dict] = field(default_factory=list)
    
    # Raw data for forensics
    flow_features: Dict = field(default_factory=dict)
    
    # Metadata
    inference_time_ms: float = 0.0
    model_version: str = ""

    def is_attack(self) -> bool:
        return self.predicted_class != "normal"
    
    def severity(self, thresholds: Dict[str, float]) -> str:
        """Map confidence to severity level."""
        if self.confidence >= thresholds.get("critical", 0.9):
            return "critical"
        elif self.confidence >= thresholds.get("high", 0.75):
            return "high"
        elif self.confidence >= thresholds.get("medium", 0.6):
            return "medium"
        return "low"


class InferenceEngine:
    """
    Real-time inference engine for network intrusion detection.
    
    Runs the three-layer cascaded architecture:
    1. Anomaly detection (Isolation Forest) → filter normal traffic
    2. Attack classification (XGBoost ensemble) → identify attack type
    3. Probability calibration → trustworthy confidence scores
    
    Performance targets:
    - Per-sample inference: < 5ms
    - Batch throughput: > 10,000 flows/second on modern CPU
    - Memory: < 500MB for all loaded models
    """
    
    def __init__(self,
                 anomaly_detector: BaseDetector,
                 attack_ensemble: XGBoostEnsemble,
                 calibrator: ProbabilityCalibrator,
                 preprocessor: FeaturePreprocessor,
                 thresholds: Optional[Dict[str, float]] = None,
                 batch_size: int = 256,
                 inference_queue_size: int = 10000):
        """
        Args:
            anomaly_detector: Trained anomaly detector
            attack_ensemble: Trained XGBoost ensemble
            calibrator: Fitted probability calibrator
            preprocessor: Fitted feature preprocessor
            thresholds: Per-class confidence thresholds
            batch_size: Inference batch size
            inference_queue_size: Max queued feature batches
        """
        self.anomaly_detector = anomaly_detector
        self.attack_ensemble = attack_ensemble
        self.calibrator = calibrator
        self.preprocessor = preprocessor
        self.thresholds = thresholds or {
            "critical": 0.9, "high": 0.75, 
            "medium": 0.6, "low": 0.4
        }
        self.batch_size = batch_size
        self.inference_queue_size = inference_queue_size
        
        # Queues
        self.input_queue: deque = deque(maxlen=inference_queue_size)
        self.output_queue: deque = deque(maxlen=inference_queue_size)
        
        # State
        self._running = False
        self._inference_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            "total_processed": 0,
            "anomalies_detected": 0,
            "attacks_detected": 0,
            "inference_time_total_ms": 0.0,
            "queue_dropped": 0,
            "last_inference_time_ms": 0.0,
            "processed_in_last_minute": 0,
            "alerts_in_last_minute": 0,
        }
        self._metrics_lock = threading.Lock()
        
        # Callbacks
        self.on_alert: Optional[callable] = None  # Called when attack detected
        self.on_error: Optional[callable] = None  # Called on inference errors
        
        # Warmup
        self._warmup()
        
        logger.logger.info(
            f"InferenceEngine initialized. "
            f"Anomaly detector: {anomaly_detector.name}, "
            f"Ensemble: {attack_ensemble.name}, "
            f"Batch size: {batch_size}"
        )
    
    def _warmup(self):
        """Warm up models with dummy data to avoid cold start latency."""
        dummy_data = np.random.randn(10, self.preprocessor.feature_names_out.__len__())
        dummy_data = np.abs(dummy_data)  # Network features are non-negative
        
        start = time.perf_counter()
        _ = self.anomaly_detector.score_samples(dummy_data)
        _ = self.attack_ensemble.predict_proba(dummy_data)
        warmup_time = (time.perf_counter() - start) * 1000
        
        logger.logger.info(f"Warmup complete: {warmup_time:.1f}ms")
    
    def start(self):
        """Start inference thread."""
        if self._running:
            logger.logger.warning("InferenceEngine already running")
            return
        
        self._running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="inference-engine",
            daemon=True
        )
        self._inference_thread.start()
        logger.logger.info("InferenceEngine started")
    
    def stop(self):
        """Stop inference thread gracefully."""
        self._running = False
        if self._inference_thread:
            self._inference_thread.join(timeout=5.0)
        logger.logger.info("InferenceEngine stopped")
    
    def submit(self, flow_batch: Dict) -> bool:
        """
        Submit a batch of flow features for inference.
        
        Called by flow aggregator when a window completes.
        Non-blocking — drops if queue is full (graceful degradation).
        
        Returns:
            True if queued successfully, False if dropped
        """
        if len(self.input_queue) >= self.inference_queue_size:
            with self._metrics_lock:
                self.metrics["queue_dropped"] += 1
            return False
        
        self.input_queue.append(flow_batch)
        return True
    
    def get_alerts(self, timeout: float = 1.0) -> List[DetectionResult]:
        """Consumer interface: get completed detection results."""
        alerts = []
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                while self.output_queue:
                    alerts.append(self.output_queue.popleft())
                if alerts:
                    break
            except IndexError:
                pass
            time.sleep(0.01)
        
        return alerts
    
    def _inference_loop(self):
        """Main inference loop running in background thread."""
        logger.logger.info("Inference loop started")
        
        batch_buffer = []
        
        while self._running:
            try:
                # Collect items from input queue
                while self.input_queue and len(batch_buffer) < self.batch_size:
                    batch_buffer.append(self.input_queue.popleft())
                
                if not batch_buffer:
                    time.sleep(0.001)  # No data, yield CPU
                    continue
                
                # Process batch
                self._process_batch(batch_buffer)
                batch_buffer.clear()
                
            except Exception as e:
                logger.logger.error(f"Inference error: {e}", exc_info=True)
                if self.on_error:
                    self.on_error(e)
        
        logger.logger.info("Inference loop ended")
    
    def _process_batch(self, batch: List[Dict]):
        """Process a batch of flow windows through the detection pipeline."""
        batch_start = time.perf_counter()
        
        # Extract all flow records from the batch
        all_flows = []
        all_metadata = []
        
        for window in batch:
            for flow_features in window.get("features", []):
                all_flows.append(flow_features)
                all_metadata.append({
                    "window_end": window.get("window_end"),
                    "window_duration": window.get("window_duration"),
                })
        
        if not all_flows:
            return
        
        n_flows = len(all_flows)
        
        # Extract metadata (IPs, ports) before preprocessing
        flow_metadata = []
        for flow in all_flows:
            flow_metadata.append({
                "src_ip": flow.get("src_ip", "unknown"),
                "dst_ip": flow.get("dst_ip", "unknown"),
                "src_port": flow.get("src_port", 0),
                "dst_port": flow.get("dst_port", 0),
                "protocol": flow.get("protocol_type", "unknown"),
            })
        
        # ─── Step 1: Preprocess ───────────────────────────────
        try:
            # Convert to DataFrame and ensure correct column order/count (41 features)
            df_flows = pd.DataFrame(all_flows)
            df_flows_filtered = df_flows[NSL_KDD_COLUMNS]
            
            features_array = self.preprocessor.transform(df_flows_filtered)
        except Exception as e:
            logger.logger.error(f"Preprocessing failed: {e}")
            return
        
        # ─── Step 2: Anomaly Detection ────────────────────────
        anomaly_scores = self.anomaly_detector.score_samples(features_array)
        is_anomalous = self.anomaly_detector.is_anomalous(features_array)
        
        # ─── Step 3: Attack Classification (anomalies only) ────
        anomalous_indices = np.where(is_anomalous)[0]
        
        # Initialize results for all flows
        all_class_probas = np.zeros((n_flows, len(self.attack_ensemble.attack_classes) + 1))
        all_class_probas[:, 0] = 1.0  # Default: normal
        all_predictions = np.array(["normal"] * n_flows)
        all_confidences = np.zeros(n_flows)
        
        if len(anomalous_indices) > 0:
            # Only run expensive specialist classification on anomalies
            anomalous_features = features_array[anomalous_indices]
            
            try:
                class_probas = self.attack_ensemble.predict_proba(anomalous_features)
                predictions = self.attack_ensemble.predict(
                    anomalous_features, 
                    thresholds=self.attack_ensemble._class_thresholds
                )
                
                # Get calibrated confidence
                calibrated_scores = self.calibrator.score_samples(anomalous_features)
                
                # Update results for anomalous flows
                for i, idx in enumerate(anomalous_indices):
                    all_class_probas[idx] = class_probas[i]
                    all_predictions[idx] = predictions[i]
                    all_confidences[idx] = calibrated_scores[i] if i < len(calibrated_scores) else class_probas[i].max()
                
            except Exception as e:
                logger.logger.error(f"Attack classification failed: {e}")
        
        # ─── Step 4: Generate Detection Results ────────────────
        inference_time = (time.perf_counter() - batch_start) * 1000
        
        for i in range(n_flows):
            is_attack = all_predictions[i] != "normal"
            
            if not is_attack:
                # Skip normal traffic unless we want to log it
                continue
            
            # Get contributing features for evidence
            contributing = []
            try:
                feature_impacts = self.anomaly_detector.get_contributing_features(
                    features_array[i:i+1], n_features=5
                )
                feature_names = self.preprocessor.feature_names_out
                contributing = [
                    {
                        "feature": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                        "impact": float(impact)
                    }
                    for idx, impact in feature_impacts
                ]
            except Exception:
                pass
            
            result = DetectionResult(
                alert_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                src_ip=flow_metadata[i]["src_ip"],
                dst_ip=flow_metadata[i]["dst_ip"],
                src_port=flow_metadata[i]["src_port"],
                dst_port=flow_metadata[i]["dst_port"],
                protocol=flow_metadata[i]["protocol"],
                is_anomalous=bool(is_anomalous[i]),
                anomaly_score=float(anomaly_scores[i]),
                predicted_class=str(all_predictions[i]),
                confidence=float(all_confidences[i]),
                class_probabilities={
                    cls: float(all_class_probas[i][j])
                    for j, cls in enumerate(self.attack_ensemble.attack_classes)
                },
                contributing_features=contributing,
                flow_features=all_flows[i],
                inference_time_ms=inference_time / n_flows,
                model_version="1.0.0",
            )
            
            # Push to output queue
            if len(self.output_queue) < self.inference_queue_size:
                self.output_queue.append(result)
            
            # Fire callback
            if self.on_alert:
                try:
                    self.on_alert(result)
                except Exception as e:
                    logger.logger.error(f"Alert callback failed: {e}")
        
        # ─── Update Metrics ────────────────────────────────────
        with self._metrics_lock:
            self.metrics["total_processed"] += n_flows
            self.metrics["anomalies_detected"] += len(anomalous_indices)
            self.metrics["attacks_detected"] += sum(1 for p in all_predictions if p != "normal")
            self.metrics["inference_time_total_ms"] += inference_time
            self.metrics["last_inference_time_ms"] = inference_time / max(n_flows, 1)
            self.metrics["processed_in_last_minute"] += n_flows
            self.metrics["alerts_in_last_minute"] += sum(1 for p in all_predictions if p != "normal")
    
    def get_metrics(self) -> Dict:
        """Return current inference metrics."""
        with self._metrics_lock:
            return dict(self.metrics)
    
    def reset_minute_metrics(self):
        """Reset per-minute counters (called by monitoring)."""
        with self._metrics_lock:
            self.metrics["processed_in_last_minute"] = 0
            self.metrics["alerts_in_last_minute"] = 0


# ─── Inference Engine Factory ───────────────────────────────────

def create_inference_engine(model_path: str, 
                           thresholds: Optional[Dict] = None,
                           batch_size: int = 256) -> InferenceEngine:
    """
    Factory function to create and configure an InferenceEngine
    from saved model artifacts.
    
    Args:
        model_path: Path to saved model directory
        thresholds: Override confidence thresholds
        batch_size: Inference batch size
    
    Returns:
        Configured InferenceEngine ready to start()
    """
    from pathlib import Path
    import json
    
    model_dir = Path(model_path)
    
    # Load artifacts metadata
    with open(model_dir / "artifacts.json") as f:
        artifacts = json.load(f)
    
    # Load models
    preprocessor = FeaturePreprocessor.load(model_dir / "preprocessor.pkl")
    anomaly_detector = IsolationForestAnomalyDetector.load(model_dir / "anomaly_detector.pkl")
    ensemble = XGBoostEnsemble.load(model_dir / "attack_ensemble.pkl")
    calibrator = ProbabilityCalibrator.load(model_dir / "calibrator.pkl")
    
    # Use stored thresholds or override
    if thresholds is None:
        thresholds = artifacts.get("thresholds", {
            "critical": 0.9, "high": 0.75,
            "medium": 0.6, "low": 0.4
        })
    
    # Update ensemble thresholds
    ensemble._class_thresholds = thresholds
    
    engine = InferenceEngine(
        anomaly_detector=anomaly_detector,
        attack_ensemble=ensemble,
        calibrator=calibrator,
        preprocessor=preprocessor,
        thresholds=thresholds,
        batch_size=batch_size
    )
    
    logger.logger.info(f"InferenceEngine created from {model_path}")
    return engine