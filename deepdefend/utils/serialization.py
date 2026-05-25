"""
Model and artifact serialization utilities.

Handles saving/loading of models, preprocessors, and metadata
with version tracking for production deployment.
"""

import pickle
import json
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime, timezone
import joblib

from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class ModelSerializer:
    """
    Serialize and deserialize ML models with versioning.
    
    Supports:
    - pickle (standard)
    - joblib (for large numpy arrays)
    - JSON (for metadata)
    
    Always saves with version info for rollback capability.
    """
    
    def __init__(self, base_path: str = "./models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: Any, name: str, 
                   version: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> str:
        """
        Save a model with versioning.
        
        Args:
            model: Model object to save
            name: Model name
            version: Version string (auto-generated if None)
            metadata: Additional metadata
        
        Returns:
            Path to saved model
        """
        if version is None:
            version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        model_dir = self.base_path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "model.pkl"
        
        # Save model
        try:
            joblib.dump(model, model_path)
        except Exception:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Compute checksum
        checksum = self._compute_checksum(model_path)
        
        # Save metadata
        meta = {
            "name": name,
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checksum": checksum,
            "model_type": type(model).__name__,
            **(metadata or {})
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        
        # Update current pointer
        current_link = self.base_path / name / "current"
        if current_link.is_symlink():
            current_link.unlink()
        try:
            current_link.symlink_to(model_dir.absolute(), target_is_directory=True)
        except OSError:
            # Symlink not supported, write pointer file
            with open(self.base_path / name / "current_version.txt", 'w') as f:
                f.write(version)
        
        logger.logger.info(f"Model saved: {name} v{version} ({model_path})")
        
        return str(model_path)
    
    def load_model(self, name: str, version: Optional[str] = None) -> Any:
        """
        Load a model.
        
        Args:
            name: Model name
            version: Specific version or 'latest' (default)
        
        Returns:
            Loaded model object
        """
        if version is None or version == "latest":
            version = self._get_latest_version(name)
        
        if version is None:
            raise FileNotFoundError(f"No saved models found for '{name}'")
        
        model_path = self.base_path / name / version / "model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Verify checksum
        expected_checksum = None
        meta_path = self.base_path / name / version / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                expected_checksum = meta.get("checksum")
        
        if expected_checksum:
            actual = self._compute_checksum(model_path)
            if actual != expected_checksum:
                logger.logger.warning(
                    f"Checksum mismatch for {name} v{version}! "
                    f"Expected: {expected_checksum[:8]}, Got: {actual[:8]}"
                )
        
        # Load model
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        logger.logger.info(f"Model loaded: {name} v{version}")
        
        return model
    
    def list_versions(self, name: str) -> list:
        """List all versions of a model."""
        model_dir = self.base_path / name
        if not model_dir.exists():
            return []
        
        versions = []
        for d in model_dir.iterdir():
            if d.is_dir() and (d / "model.pkl").exists():
                meta_path = d / "metadata.json"
                meta = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                versions.append({
                    "version": d.name,
                    "timestamp": meta.get("timestamp", "unknown"),
                    "checksum": meta.get("checksum", "unknown"),
                })
        
        return sorted(versions, key=lambda v: v["timestamp"], reverse=True)
    
    def _get_latest_version(self, name: str) -> Optional[str]:
        """Get latest version of a model."""
        current_link = self.base_path / name / "current"
        if current_link.is_symlink():
            return current_link.resolve().name
        
        pointer_file = self.base_path / name / "current_version.txt"
        if pointer_file.exists():
            with open(pointer_file) as f:
                return f.read().strip()
        
        versions = self.list_versions(name)
        if versions:
            return versions[0]["version"]
        
        return None
    
    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def export_for_production(self, name: str, 
                              version: Optional[str] = None,
                              output_dir: Optional[str] = None) -> Path:
        """
        Export model for production deployment.
        
        Creates a minimal package with just the model file and metadata,
        without version history.
        """
        if version is None:
            version = self._get_latest_version(name)
        
        source_dir = self.base_path / name / version
        if not source_dir.exists():
            raise FileNotFoundError(f"Model version not found: {source_dir}")
        
        output_dir = Path(output_dir or f"./export/{name}_{version}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy(source_dir / "model.pkl", output_dir / "model.pkl")
        shutil.copy(source_dir / "metadata.json", output_dir / "metadata.json")
        
        logger.logger.info(f"Model exported for production: {output_dir}")
        
        return output_dir