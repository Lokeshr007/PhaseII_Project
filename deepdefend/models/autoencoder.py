"""
Deep autoencoder for anomaly detection.
Alternative to Isolation Forest — better for complex, non-linear patterns.

Architecture: encoder → bottleneck → decoder
Anomaly score: reconstruction error
Normal traffic has low reconstruction error (we've seen it before).
Attack traffic has high reconstruction error (novel patterns).

Why autoencoder for U2R detection:
- U2R attacks manipulate legitimate features (login counts, shell commands)
- These look like normal traffic at feature level but in unusual COMBINATIONS
- Isolation Forest may miss subtle combinatorial anomalies
- Autoencoder learns normal feature correlations and flags violations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from deepdefend.models.base import BaseDetector
from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class AutoencoderNetwork(nn.Module):
    """
    Autoencoder with bottleneck architecture.
    
    Encoder: input_dim → 128 → 64 → 32 → latent_dim
    Decoder: latent_dim → 32 → 64 → 128 → input_dim
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 16,
                 hidden_dims: list = [128, 64, 32]):
        super().__init__()
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        # Bottleneck
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (reverse)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get latent representation."""
        return self.encoder(x)


class AutoencoderAnomalyDetector(BaseDetector):
    """
    Deep autoencoder for anomaly detection.
    
    Training:
    - Trained on normal traffic only (or predominantly normal)
    - Minimizes reconstruction error on normal patterns
    - Attack traffic yields high reconstruction error
    
    Inference:
    - Anomaly score = reconstruction error (MSE per sample)
    - Threshold calibrated on validation normal traffic
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 16,
                 hidden_dims: list = None,
                 learning_rate: float = 1e-3,
                 batch_size: int = 256,
                 epochs: int = 50,
                 early_stopping_patience: int = 10,
                 device: str = 'cpu',
                 random_state: int = 42,
                 name: str = "autoencoder"):
        
        super().__init__(name=name, model_type="anomaly_detector")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.random_state = random_state
        
        torch.manual_seed(random_state)
        
        self.network: Optional[AutoencoderNetwork] = None
        self._anomaly_threshold: float = 0.0
        self._training_losses: list = []
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None) -> 'AutoencoderAnomalyDetector':
        """
        Train autoencoder on normal traffic.
        """
        # Filter to normal traffic if labels provided
        if y is not None:
            normal_mask = y == 'normal'
            X_train = X[normal_mask]
            logger.logger.info(
                f"Training Autoencoder on {len(X_train):,} normal samples"
            )
        else:
            X_train = X
        
        # Initialize network
        actual_input_dim = X_train.shape[1]
        self.network = AutoencoderNetwork(
            input_dim=actual_input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Prepare data
        dataset = TensorDataset(torch.FloatTensor(X_train))
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=False
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        best_loss = float('inf')
        patience_counter = 0
        
        logger.logger.info(f"Starting autoencoder training ({self.epochs} epochs)...")
        
        for epoch in range(self.epochs):
            self.network.train()
            epoch_loss = 0.0
            
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.network(x)
                loss = criterion(reconstructed, x)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(x)
            
            epoch_loss /= len(X_train)
            self._training_losses.append(epoch_loss)
            scheduler.step(epoch_loss)
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.logger.info(f"Epoch {epoch}: loss={epoch_loss:.6f}")
            
            if patience_counter >= self.early_stopping_patience:
                logger.logger.info(f"Early stopping at epoch {epoch}")
                self.network.load_state_dict(best_state)
                break
        
        self.network.eval()
        
        # Must be True before computing errors
        self._fitted = True
        
        # Compute anomaly threshold from training reconstruction errors
        train_errors = self._compute_reconstruction_error(X_train)
        self._anomaly_threshold = np.percentile(train_errors, 99)  # Top 1% = anomalous
        
        logger.logger.info(
            f"Autoencoder trained. Best loss: {best_loss:.6f}. "
            f"Anomaly threshold: {self._anomaly_threshold:.6f}"
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly: 1 for anomaly, 0 for normal."""
        errors = self._compute_reconstruction_error(X)
        return (errors > self._anomaly_threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convert reconstruction errors to probabilities."""
        errors = self._compute_reconstruction_error(X)
        
        # Sigmoid around threshold
        scaled = (errors - self._anomaly_threshold) / (self._anomaly_threshold + 1e-8)
        proba = 1.0 / (1.0 + np.exp(-scaled * 5))
        
        return np.column_stack([1 - proba, proba])
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score from reconstruction error."""
        errors = self._compute_reconstruction_error(X)
        # Normalize to [0, 1]
        max_error = errors.max() if errors.max() > 0 else 1.0
        return errors / max_error
    
    def _compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction error."""
        if not self._fitted or self.network is None:
            raise RuntimeError("Model not fitted.")
        
        self.network.eval()
        
        dataset = TensorDataset(torch.FloatTensor(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        errors = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                reconstructed = self.network(x)
                # MSE per sample
                sample_errors = torch.mean((reconstructed - x) ** 2, dim=1)
                errors.append(sample_errors.cpu().numpy())
        
        return np.concatenate(errors)
    
    def get_contributing_features(self, X: np.ndarray, n_features: int = 5) -> list:
        """Identify features with largest reconstruction error contribution."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.network(X_tensor)
            feature_errors = (reconstructed - X_tensor) ** 2
        
        feature_errors = feature_errors[0].cpu().numpy()
        top_features = np.argsort(feature_errors)[-n_features:][::-1]
        
        return [(int(i), float(feature_errors[i])) for i in top_features]
    
    def _get_framework_version(self) -> str:
        return f"pytorch-{torch.__version__}"
    
    def save(self, path: str) -> None:
        """Save model with PyTorch state dict."""
        super().save(path)
        
        # Also save torch state dict
        if self.network is not None:
            torch_path = Path(path).with_suffix('.pth')
            torch.save({
                'state_dict': self.network.state_dict(),
                'training_losses': self._training_losses,
                'anomaly_threshold': self._anomaly_threshold,
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'hidden_dims': self.hidden_dims,
            }, torch_path)
    
    @classmethod
    def load(cls, path: str) -> 'AutoencoderAnomalyDetector':
        model = super().load(path)
        
        torch_path = Path(path).with_suffix('.pth')
        if torch_path.exists():
            checkpoint = torch.load(torch_path, map_location='cpu')
            model.network = AutoencoderNetwork(
                input_dim=checkpoint['input_dim'],
                latent_dim=checkpoint['latent_dim'],
                hidden_dims=checkpoint['hidden_dims']
            )
            model.network.load_state_dict(checkpoint['state_dict'])
            model._training_losses = checkpoint['training_losses']
            model._anomaly_threshold = checkpoint['anomaly_threshold']
        
        return model