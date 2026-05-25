"""
Tests for feature preprocessor.
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepdefend.data.preprocessor import FeaturePreprocessor


class TestFeaturePreprocessor(unittest.TestCase):
    def setUp(self):
        # Create synthetic data mimicking NSL-KDD features
        np.random.seed(42)
        n_samples = 100
        
        self.numerical_cols = [
            "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment",
            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
            "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count",
            "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate"
        ]
        
        self.categorical_cols = ["protocol_type", "service", "flag"]
        
        # Create DataFrame
        data = {}
        for col in self.numerical_cols:
            data[col] = np.random.randn(n_samples) * 10 + 50
        
        data["protocol_type"] = np.random.choice(["tcp", "udp", "icmp"], n_samples)
        data["service"] = np.random.choice(["http", "ssh", "ftp", "smtp"], n_samples)
        data["flag"] = np.random.choice(["SYN", "SYN,ACK", "ACK", "RST"], n_samples)
        
        self.df = pd.DataFrame(data)
        self.X = self.df.values
        self.column_names = self.numerical_cols + self.categorical_cols
    
    def test_fit_transform_shape(self):
        preprocessor = FeaturePreprocessor(
            scaling_method="robust",
            categorical_encoding="onehot"
        )
        
        X_transformed = preprocessor.fit_transform(self.X)
        
        # Should have more columns after one-hot encoding
        self.assertGreater(X_transformed.shape[1], len(self.numerical_cols))
        self.assertEqual(X_transformed.shape[0], len(self.X))
    
    def test_transform_after_fit(self):
        preprocessor = FeaturePreprocessor(scaling_method="robust")
        preprocessor.fit(self.X)
        
        X_transformed = preprocessor.transform(self.X)
        
        self.assertEqual(X_transformed.shape[0], len(self.X))
        self.assertFalse(np.any(np.isnan(X_transformed)))
    
    def test_handle_unknown_categories(self):
        """Preprocessor should handle categories not seen during fit."""
        preprocessor = FeaturePreprocessor(
            scaling_method="robust",
            categorical_encoding="onehot"
        )
        
        # Fit on limited categories
        fit_data = self.df.copy()
        fit_data["service"] = np.random.choice(["http", "ssh"], len(fit_data))
        preprocessor.fit(fit_data.values)
        
        # Transform with unseen category
        transform_data = self.df.copy()  # Contains "ftp", "smtp"
        X_transformed = preprocessor.transform(transform_data.values)
        
        self.assertEqual(X_transformed.shape[0], len(self.X))
    
    def test_missing_value_handling(self):
        """Preprocessor should handle missing values gracefully."""
        preprocessor = FeaturePreprocessor(
            scaling_method="robust",
            handle_missing="median"
        )
        
        # Create data with NaN
        data_with_nan = self.df.copy()
        data_with_nan.iloc[0:5, 0] = np.nan  # NaN in first numerical column
        
        preprocessor.fit(data_with_nan.values)
        X_transformed = preprocessor.transform(data_with_nan.values)
        
        self.assertFalse(np.any(np.isnan(X_transformed)))
    
    def test_serialization_roundtrip(self):
        """Preprocessor should survive save/load cycle."""
        import tempfile
        import os
        
        preprocessor = FeaturePreprocessor(scaling_method="robust")
        preprocessor.fit(self.X)
        
        original_output = preprocessor.transform(self.X[:10])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "preprocessor.pkl")
            preprocessor.save(path)
            
            loaded = FeaturePreprocessor.load(path)
            loaded_output = loaded.transform(self.X[:10])
        
        np.testing.assert_array_almost_equal(original_output, loaded_output)
    
    def test_label_encoding(self):
        """Test label encoding instead of one-hot."""
        preprocessor = FeaturePreprocessor(
            categorical_encoding="label"
        )
        
        X_transformed = preprocessor.fit_transform(self.X)
        
        # With label encoding, column count = numerical + categorical
        expected_cols = len(self.numerical_cols) + len(self.categorical_cols)
        self.assertEqual(X_transformed.shape[1], expected_cols)


if __name__ == "__main__":
    unittest.main()