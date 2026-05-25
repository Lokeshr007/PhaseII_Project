"""
Configuration management with YAML + environment overrides.
Supports development, production, and custom deployment profiles.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, validator
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class ImbalanceConfig(BaseModel):
    """Configuration for handling class imbalance."""
    minority_classes: list = Field(
        default=["u2r", "r2l"],
        description="Classes with extreme imbalance"
    )
    class_weights: Dict[str, float] = Field(
        default={
            "normal": 1.0,
            "dos": 5.0,
            "probe": 10.0,
            "r2l": 25.0,
            "u2r": 50.0
        },
        description="Cost-sensitive weights per class"
    )
    oversampling_method: str = Field(
        default="smote",
        description="SMOTE, ADASYN, or border_smote"
    )
    minority_oversampling_ratio: float = Field(
        default=0.3,
        description="Target ratio of minority to majority after oversampling"
    )
    false_positive_budget_per_day: int = Field(
        default=20,
        description="Maximum acceptable false positives per day for analyst sanity"
    )


class ModelConfig(BaseModel):
    """ML model configuration."""
    anomaly_detector: str = Field(
        default="isolation_forest",
        description="isolation_forest or autoencoder"
    )
    attack_classifiers: list = Field(
        default=["dos", "probe", "r2l", "u2r"],
        description="Per-class specialist classifiers"
    )
    classifier_algorithm: str = Field(
        default="xgboost",
        description="xgboost, lightgbm, or random_forest"
    )
    xgboost_params: Dict[str, Any] = Field(
        default={
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1,
            "eval_metric": "aucpr",  # Precision-Recall AUC, not accuracy
            "tree_method": "hist",
            "random_state": 42
        }
    )
    calibration_method: str = Field(
        default="platt",
        description="Probability calibration: platt or isotonic"
    )
    confidence_thresholds: Dict[str, float] = Field(
        default={
            "critical": 0.9,
            "high": 0.75,
            "medium": 0.6,
            "low": 0.4
        }
    )


class FlowConfig(BaseModel):
    """Flow aggregation configuration."""
    window_seconds: float = Field(default=2.0, ge=0.5, le=10.0)
    slide_seconds: float = Field(default=1.0, ge=0.5, le=10.0)
    max_flows_per_window: int = Field(default=100000)
    ring_buffer_size: int = Field(default=10000)
    feature_set: str = Field(
        default="nsl_kdd_extended",
        description="nsl_kdd, nsl_kdd_extended, or custom"
    )


class SnortConfig(BaseModel):
    """Snort integration configuration."""
    enabled: bool = Field(default=True)
    binary_path: str = Field(default="/usr/sbin/snort")
    config_path: str = Field(default="/etc/snort/snort.conf")
    rules_path: str = Field(default="/etc/snort/rules")
    custom_rules_path: str = Field(default="./config/snort/custom.rules")
    interface: str = Field(default="eth0")
    alert_socket_path: str = Field(default="/tmp/snort_alert")


class ResponseConfig(BaseModel):
    """Automated response configuration."""
    auto_block: bool = Field(
        default=False,
        description="Enable automatic IP blocking (DANGEROUS - verify first)"
    )
    auto_block_threshold: str = Field(
        default="critical",
        description="Minimum severity for auto-block"
    )
    quarantine_enabled: bool = Field(default=False)
    firewall_backend: str = Field(
        default="iptables",
        description="iptables or nftables"
    )
    blocked_ips_file: str = Field(
        default="/var/lib/deepdefend/blocked_ips.json"
    )


class NotificationConfig(BaseModel):
    """Alert notification configuration."""
    slack_webhook: Optional[str] = None
    email_smtp_host: Optional[str] = None
    email_recipients: list = Field(default_factory=list)
    pagerduty_key: Optional[str] = None
    notification_severity: str = Field(default="high")


class SIEMConfig(BaseModel):
    """SIEM integration configuration."""
    output_format: str = Field(default="json_syslog")
    syslog_facility: str = Field(default="local0")
    syslog_severity_map: Dict[str, str] = Field(
        default={
            "critical": "critical",
            "high": "error",
            "medium": "warning",
            "low": "info"
        }
    )
    include_raw_features: bool = Field(
        default=True,
        description="Include feature vectors in SIEM output for forensics"
    )


class DeepDefendConfig(BaseModel):
    """Master configuration."""
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    flow: FlowConfig = Field(default_factory=FlowConfig)
    snort: SnortConfig = Field(default_factory=SnortConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)
    notification: NotificationConfig = Field(default_factory=NotificationConfig)
    siem: SIEMConfig = Field(default_factory=SIEMConfig)
    
    # Deployment
    mode: str = Field(default="development")
    log_level: str = Field(default="INFO")
    model_path: str = Field(default="./models/saved")
    data_path: str = Field(default="./data")
    
    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['development', 'production', 'testing']:
            raise ValueError(f"Mode must be development/production/testing, got {v}")
        return v


class ConfigLoader:
    """Load configuration from YAML with environment variable overrides."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load(cls, config_path: Optional[str] = None, mode: Optional[str] = None) -> DeepDefendConfig:
        """Load and validate configuration."""
        if cls._config is not None:
            return cls._config
        
        # Determine config path
        if config_path is None:
            mode = mode or os.environ.get("DEEPDEFEND_MODE", "development")
            config_dir = Path(__file__).parent.parent.parent / "config"
            config_path = config_dir / f"{mode}.yaml"
        
        # Load YAML
        config_data = {}
        if Path(config_path).exists():
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        
        # Override from environment variables
        config_data = cls._apply_env_overrides(config_data)
        
        # Validate and create config object
        cls._config = DeepDefendConfig(**config_data)
        logger.logger.info(f"Configuration loaded: mode={cls._config.mode}")
        
        return cls._config
    
    @staticmethod
    def _apply_env_overrides(config: dict) -> dict:
        """Override config values from environment variables."""
        env_mappings = {
            "DEEPDEFEND_MODE": "mode",
            "DEEPDEFEND_MODEL_PATH": "model_path",
            "DEEPDEFEND_SLACK_WEBHOOK": "notification.slack_webhook",
            "DEEPDEFEND_AUTO_BLOCK": "response.auto_block",
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ and os.environ[env_var]:
                # Simple dot-notation setter
                keys = config_path.split(".")
                d = config
                for key in keys[:-1]:
                    if key not in d:
                        d[key] = {}
                    d = d[key]
                
                # Type conversion
                value = os.environ[env_var]
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                
                d[keys[-1]] = value
        
        return config


def get_config() -> DeepDefendConfig:
    """Get the current configuration singleton."""
    return ConfigLoader.load()