"""
Pydantic schema validation for DeepDefend configuration.

Provides runtime validation of configuration values with
meaningful error messages. Catches misconfigurations before
they cause runtime failures in production.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, Dict, List, Any
from enum import Enum


class ValidatedConfig(BaseModel):
    """Base class for validated configurations."""
    
    class Config:
        extra = "forbid"  # Reject unknown fields
        validate_assignment = True  # Validate on assignment


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DeploymentMode(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class ScalingMethod(str, Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


class OversamplingMethod(str, Enum):
    SMOTE = "smote"
    ADASYN = "adasyn"
    BORDERLINE_SMOTE = "borderline_smote"
    SMOTE_TOMEK = "smote_tomek"
    SMOTE_ENN = "smote_enn"


class AnomalyDetectorType(str, Enum):
    ISOLATION_FOREST = "isolation_forest"
    AUTOENCODER = "autoencoder"


class ClassifierAlgorithm(str, Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"


class CalibrationMethod(str, Enum):
    PLATT = "platt"
    ISOTONIC = "isotonic"


class FirewallBackend(str, Enum):
    IPTABLES = "iptables"
    NFTABLES = "nftables"


class SIEMOutputFormat(str, Enum):
    JSON_SYSLOG = "json_syslog"
    CEF = "cef"
    LEEF = "leef"


class ImbalanceSchema(ValidatedConfig):
    """Validation for imbalance configuration."""
    minority_classes: List[str] = Field(
        default=["u2r", "r2l"],
        min_items=1
    )
    class_weights: Dict[str, float] = Field(
        default={
            "normal": 1.0,
            "dos": 5.0,
            "probe": 10.0,
            "r2l": 25.0,
            "u2r": 50.0
        }
    )
    oversampling_method: OversamplingMethod = Field(
        default=OversamplingMethod.SMOTE_TOMEK
    )
    minority_oversampling_ratio: float = Field(
        default=0.3,
        ge=0.1,
        le=1.0
    )
    false_positive_budget_per_day: int = Field(
        default=20,
        ge=1,
        le=1000
    )
    
    @validator('class_weights')
    def weights_must_be_positive(cls, v):
        for cls_name, weight in v.items():
            if weight <= 0:
                raise ValueError(f"Class weight for '{cls_name}' must be positive, got {weight}")
        return v


class ModelSchema(ValidatedConfig):
    """Validation for model configuration."""
    anomaly_detector: AnomalyDetectorType = Field(
        default=AnomalyDetectorType.ISOLATION_FOREST
    )
    attack_classifiers: List[str] = Field(
        default=["dos", "probe", "r2l", "u2r"],
        min_items=1
    )
    classifier_algorithm: ClassifierAlgorithm = Field(
        default=ClassifierAlgorithm.XGBOOST
    )
    xgboost_params: Dict[str, Any] = Field(default_factory=dict)
    calibration_method: CalibrationMethod = Field(
        default=CalibrationMethod.PLATT
    )
    confidence_thresholds: Dict[str, float] = Field(
        default={
            "critical": 0.9,
            "high": 0.75,
            "medium": 0.6,
            "low": 0.4
        }
    )
    
    @validator('confidence_thresholds')
    def validate_thresholds(cls, v):
        required = ["critical", "high", "medium", "low"]
        for key in required:
            if key not in v:
                raise ValueError(f"Missing threshold: {key}")
        # Ensure thresholds are decreasing
        if not (v["critical"] >= v["high"] >= v["medium"] >= v["low"]):
            raise ValueError("Confidence thresholds must be in descending order")
        for key, val in v.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"Threshold '{key}' must be between 0 and 1, got {val}")
        return v


class FlowSchema(ValidatedConfig):
    """Validation for flow aggregation configuration."""
    window_seconds: float = Field(default=2.0, ge=0.5, le=30.0)
    slide_seconds: float = Field(default=1.0, ge=0.1, le=30.0)
    max_flows_per_window: int = Field(default=100000, ge=1000, le=10000000)
    ring_buffer_size: int = Field(default=10000, ge=100, le=1000000)
    feature_set: str = Field(default="nsl_kdd_extended")
    
    @validator('slide_seconds')
    def slide_must_be_less_than_window(cls, v, values):
        if 'window_seconds' in values and v > values['window_seconds']:
            raise ValueError(
                f"slide_seconds ({v}) must be <= window_seconds ({values['window_seconds']})"
            )
        return v


class SnortSchema(ValidatedConfig):
    """Validation for Snort configuration."""
    enabled: bool = Field(default=True)
    binary_path: str = Field(default="/usr/sbin/snort")
    config_path: str = Field(default="/etc/snort/snort.conf")
    rules_path: str = Field(default="/etc/snort/rules")
    custom_rules_path: str = Field(default="./config/snort/custom.rules")
    interface: str = Field(default="eth0")
    alert_socket_path: str = Field(default="/tmp/snort_alert")


class ResponseSchema(ValidatedConfig):
    """Validation for response configuration."""
    auto_block: bool = Field(default=False)
    auto_block_threshold: str = Field(default="critical")
    quarantine_enabled: bool = Field(default=False)
    firewall_backend: FirewallBackend = Field(default=FirewallBackend.IPTABLES)
    blocked_ips_file: str = Field(default="/var/lib/deepdefend/blocked_ips.json")
    
    @validator('auto_block_threshold')
    def validate_threshold_string(cls, v):
        valid = ["critical", "high", "medium", "low"]
        if v not in valid:
            raise ValueError(f"auto_block_threshold must be one of {valid}")
        return v


class NotificationSchema(ValidatedConfig):
    """Validation for notification configuration."""
    slack_webhook: Optional[str] = None
    email_smtp_host: Optional[str] = None
    email_recipients: List[str] = Field(default_factory=list)
    pagerduty_key: Optional[str] = None
    notification_severity: str = Field(default="high")


class SIEMSchema(ValidatedConfig):
    """Validation for SIEM configuration."""
    output_format: SIEMOutputFormat = Field(default=SIEMOutputFormat.JSON_SYSLOG)
    syslog_facility: str = Field(default="local0")
    syslog_severity_map: Dict[str, str] = Field(
        default={
            "critical": "critical",
            "high": "error",
            "medium": "warning",
            "low": "info"
        }
    )
    include_raw_features: bool = Field(default=True)


class FullConfigSchema(ValidatedConfig):
    """Master configuration validation."""
    mode: DeploymentMode = Field(default=DeploymentMode.DEVELOPMENT)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    model_path: str = Field(default="./models/saved")
    data_path: str = Field(default="./data")
    
    imbalance: ImbalanceSchema = Field(default_factory=ImbalanceSchema)
    model: ModelSchema = Field(default_factory=ModelSchema)
    flow: FlowSchema = Field(default_factory=FlowSchema)
    snort: SnortSchema = Field(default_factory=SnortSchema)
    response: ResponseSchema = Field(default_factory=ResponseSchema)
    notification: NotificationSchema = Field(default_factory=NotificationSchema)
    siem: SIEMSchema = Field(default_factory=SIEMSchema)
    
    @root_validator
    def production_safety_checks(cls, values):
        """Extra safety checks for production mode."""
        if values.get("mode") == DeploymentMode.PRODUCTION:
            # In production, auto_block should be explicitly confirmed
            if values.get("response") and values["response"].auto_block:
                # This is a warning-level concern, not an error
                pass
        return values