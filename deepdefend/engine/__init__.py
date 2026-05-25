"""
Real-time inference engine for DeepDefend NIDS.
Consumes flow features, runs cascaded detection, outputs structured alerts.
"""

from deepdefend.engine.aggregator import FlowAggregator, PacketParser
from deepdefend.engine.inference import InferenceEngine
from deepdefend.engine.snort import SnortManager
from deepdefend.engine.fusion import FusionEngine
from deepdefend.engine.policy import ActionPolicyEngine

__all__ = [
    "FlowAggregator",
    "PacketParser", 
    "InferenceEngine",
    "SnortManager",
    "FusionEngine",
    "ActionPolicyEngine",
]