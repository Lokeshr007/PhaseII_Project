"""
NSL-KDD data loader with comprehensive imbalance analysis.

The NSL-KDD dataset contains:
- KDDTrain+: 125,973 samples (full training set)
- KDDTrain+_20Percent: 25,192 samples (20% subset)
- KDDTest+: 22,544 samples (test set)
- KDDTest-21: 11,850 samples (test set without difficulty level 21)

Attack categories:
- DoS: denial-of-service (e.g., syn flood)
- Probe: surveillance and probing (e.g., port scanning)
- R2L: remote-to-local unauthorized access (e.g., password guessing)
- U2R: user-to-root privilege escalation (e.g., buffer overflow)

The imbalance in NSL-KDD (KDDTrain+):
- Normal: 67,343 (53.5%)
- DoS: 45,927 (36.5%) 
- Probe: 11,656 (9.3%)
- R2L: 995 (0.79%)
- U2R: 52 (0.04%)  ← This is what we're fighting for

A naive model predicting "normal" on everything gets 53.5% accuracy on training.
A model predicting "normal or DoS" gets 90% accuracy while missing all Probe/R2L/U2R.
Accuracy is worse than useless here — it's actively misleading.
"""

import os
import zipfile
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from collections import Counter
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


# ─── Column Definitions ─────────────────────────────────────────

# NSL-KDD feature names in order (41 features)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

# Which columns are categorical (need encoding)
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# Which columns are numerical
NUMERICAL_COLUMNS = [c for c in NSL_KDD_COLUMNS if c not in CATEGORICAL_COLUMNS]
# Attack category mapping (from specific attack to category)
ATTACK_CATEGORIES = {
    # DoS
    "apache2": "dos", "back": "dos", "land": "dos", "neptune": "dos",
    "mailbomb": "dos", "pod": "dos", "processtable": "dos", "smurf": "dos",
    "teardrop": "dos", "udpstorm": "dos",
    # Probe
    "ipsweep": "probe", "mscan": "probe", "nmap": "probe", "portsweep": "probe",
    "saint": "probe", "satan": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "httptunnel": "r2l",
    "imap": "r2l", "multihop": "r2l", "named": "r2l", "phf": "r2l",
    "sendmail": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "xlock": "r2l", "xsnoop": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "ps": "u2r", "rootkit": "u2r", "sqlattack": "u2r", "xterm": "u2r",
    # Normal
    "normal": "normal"
}


@dataclass
class ImbalanceReport:
    """Detailed class imbalance analysis."""
    total_samples: int
    class_counts: Dict[str, int]
    class_percentages: Dict[str, float]
    minority_classes: List[str]  # Classes below 1% representation
    extreme_minority_classes: List[str]  # Classes below 0.1%
    imbalance_ratio: float  # Majority count / rarest minority count
    recommended_metrics: List[str]  # Don't use accuracy
    
    def summary(self) -> str:
        lines = ["=" * 60]
        lines.append("CLASS IMBALANCE ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"Total samples: {self.total_samples:,}")
        lines.append(f"Imbalance ratio (majority/rarest): {self.imbalance_ratio:.0f}:1")
        lines.append("")
        lines.append(f"{'Class':<12} {'Count':>10} {'Percentage':>12}")
        lines.append("-" * 40)
        
        for cls in sorted(self.class_counts.keys(), 
                         key=lambda c: self.class_counts[c], reverse=True):
            count = self.class_counts[cls]
            pct = self.class_percentages[cls]
            warning = " ← EXTREME IMBALANCE" if cls in self.extreme_minority_classes else ""
            warning = warning or (" ← MINORITY" if cls in self.minority_classes else "")
            lines.append(f"{cls:<12} {count:>10,} {pct:>11.2f}%{warning}")
        
        lines.append("")
        lines.append("⚠️  DO NOT USE ACCURACY AS A METRIC")
        lines.append(f"   A model predicting 'normal' on everything gets "
                    f"{self.class_percentages.get('normal', 0):.1f}% accuracy")
        lines.append(f"   Recommended metrics: {', '.join(self.recommended_metrics)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


class DatasetSplit(NamedTuple):
    """Container for a dataset split."""
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    
    @property
    def shape(self) -> Tuple:
        return self.X.shape
    
    def class_distribution(self) -> Dict[str, int]:
        return dict(Counter(self.y))


@dataclass
class FullDataset:
    """Complete dataset with train/val/test splits."""
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    feature_names: List[str]
    class_names: List[str]
    imbalance_report: ImbalanceReport
    preprocessor: Optional['FeaturePreprocessor'] = None


class NSLKDLoader:
    """
    Load NSL-KDD dataset with comprehensive imbalance analysis.
    
    Handles:
    - Download from UCI repository if not present
    - Parse ARFF/TXT format with correct column mapping
    - Map specific attacks to categories
    - Generate detailed imbalance report
    - Stratified splitting preserving minority class distribution
    """
    
    # UCI NSL-KDD download URLs
    TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00349/KDDTrain+.zip"
    TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00349/KDDTest+.zip"
    TRAIN_20_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00349/KDDTrain+_20Percent.zip"
    
    def __init__(self, data_dir: str = "./data/nsl_kdd"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._train_data: Optional[pd.DataFrame] = None
        self._test_data: Optional[pd.DataFrame] = None
    
    def download_if_needed(self) -> None:
        """Download NSL-KDD datasets if not present."""
        files_to_check = {
            "KDDTrain+.txt": self.TRAIN_URL,
            "KDDTest+.txt": self.TEST_URL,
        }
        
        for filename, url in files_to_check.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                continue
            
            zipname = filename.replace(".txt", ".zip")
            zippath = self.data_dir / zipname
            
            if not zippath.exists():
                logger.logger.info(f"Downloading {zipname} from UCI repository...")
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(zippath, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True,
                             desc=zipname) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Extract
            logger.logger.info(f"Extracting {zipname}...")
            with zipfile.ZipFile(zippath, 'r') as zf:
                zf.extractall(self.data_dir)
            
            # The zip might extract to a subdirectory, find the file
            extracted = list(self.data_dir.rglob(filename))
            if extracted and extracted[0] != filepath:
                extracted[0].rename(filepath)
    
    def load_all(self) -> FullDataset:
        """Load complete dataset with all splits."""
        self.download_if_needed()
        
        # Load raw data
        train_df = self._load_file("KDDTrain+.txt")
        test_df = self._load_file("KDDTest+.txt")
        
        # Analyze imbalance on training data
        imbalance_report = self._analyze_imbalance(train_df)
        logger.logger.info(f"\n{imbalance_report.summary()}")
        
        # Prepare features and labels
        X_train, y_train = self._prepare_data(train_df)
        X_test, y_test = self._prepare_data(test_df)
        
        # Create stratified validation split from training
        from sklearn.model_selection import train_test_split
        
        # Stratify on the rare classes — this is critical
        # Regular stratification can fail if a class has < 2 samples in a fold
        # For U2R with 52 samples, we need to ensure it appears in both splits
        X_train_sp, X_val, y_train_sp, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=42
        )
        
        logger.logger.info(
            f"Split sizes — Train: {len(X_train_sp)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )
        logger.logger.info(f"Train distribution: {dict(Counter(y_train_sp))}")
        logger.logger.info(f"Val distribution: {dict(Counter(y_val))}")
        
        return FullDataset(
            train=DatasetSplit(X_train_sp, y_train_sp, 
                             train_df.iloc[:len(X_train_sp)]),
            val=DatasetSplit(X_val, y_val,
                           train_df.iloc[len(X_train_sp):]),
            test=DatasetSplit(X_test, y_test, test_df),
            feature_names=NSL_KDD_COLUMNS,
            class_names=sorted(set(y_train)),
            imbalance_report=imbalance_report
        )
    
    def _load_file(self, filename: str) -> pd.DataFrame:
        """Load a single NSL-KDD data file."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"NSL-KDD file not found: {filepath}. "
                f"Run download_if_needed() first."
            )
        
        # NSL-KDD is CSV with no header
        df = pd.read_csv(filepath, header=None)
        
        # Some files have a difficulty column at the end
        if df.shape[1] == 43:
            df = df.drop(columns=[42])
        
        # Assign names now that shape is correct
        df.columns = NSL_KDD_COLUMNS + ["label"]
        
        # Map specific attack labels to categories
        df["label"] = df["label"].str.rstrip('.').str.lower()
        df["attack_category"] = df["label"].map(ATTACK_CATEGORIES)
        
        # Any unmapped attack -> flag as potential issue
        unmapped = df[df["attack_category"].isna()]["label"].unique()
        if len(unmapped) > 0:
            logger.logger.warning(f"Unmapped attack labels: {unmapped}. Treating as 'unknown'.")
            df["attack_category"] = df["attack_category"].fillna("unknown")
        
        return df
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels."""
        X = df[NSL_KDD_COLUMNS].copy()
        y = df["attack_category"].values
        
        # Keep metadata about which columns are categorical
        self._categorical_cols = CATEGORICAL_COLUMNS
        self._numerical_cols = NUMERICAL_COLUMNS
        
        return X.values, y
    
    def _analyze_imbalance(self, df: pd.DataFrame) -> ImbalanceReport:
        """Generate comprehensive imbalance analysis."""
        class_counts = Counter(df["attack_category"])
        total = sum(class_counts.values())
        
        class_pcts = {
            cls: (count / total) * 100 
            for cls, count in class_counts.items()
        }
        
        minority = [cls for cls, pct in class_pcts.items() if pct < 1.0]
        extreme = [cls for cls, pct in class_pcts.items() if pct < 0.1]
        
        majority_count = max(class_counts.values())
        rarest_count = min(class_counts.values())
        imbalance_ratio = majority_count / rarest_count if rarest_count > 0 else float('inf')
        
        return ImbalanceReport(
            total_samples=total,
            class_counts=dict(class_counts),
            class_percentages=class_pcts,
            minority_classes=minority,
            extreme_minority_classes=extreme,
            imbalance_ratio=imbalance_ratio,
            recommended_metrics=[
                "Per-class Recall (especially U2R/R2L)",
                "Precision-Recall AUC",
                "F2-score (recall-weighted)",
                "Geometric Mean Score",
                "False Positive Rate (target < 0.1%)"
            ]
        )
    
    def get_attack_specific_samples(self, attack_type: str) -> pd.DataFrame:
        """Get all samples of a specific attack type for analysis."""
        if self._train_data is None:
            self.load_all()
        return self._train_data[self._train_data["attack_category"] == attack_type]


# ─── Custom Data Loader for Live Capture ─────────────────────────

class LiveTrafficLoader:
    """
    Load and preprocess live captured traffic features.
    Bridges the gap between NSL-KDD training and real-time inference.
    Ensures feature semantics match exactly between training and production.
    """
    
    def __init__(self, preprocessor: 'FeaturePreprocessor'):
        self.preprocessor = preprocessor
        self._feature_names = preprocessor.feature_names_out
    
    def process_flow_features(self, features: dict) -> np.ndarray:
        """
        Convert flow aggregator output to model-ready features.
        Ensures identical preprocessing to training data.
        """
        # Convert to DataFrame for preprocessing consistency
        df = pd.DataFrame([features])
        
        # Ensure all expected columns are present
        for col in self._feature_names:
            if col not in df.columns:
                df[col] = 0.0
        
        # Apply same preprocessing as training
        return self.preprocessor.transform(df[self._feature_names])
