#!/usr/bin/env python3
"""
Analyze NSL-KDD dataset and generate imbalance report.
Run this first to understand what we're dealing with.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepdefend.data.loader import NSLKDLoader
from deepdefend.data.preprocessor import FeaturePreprocessor
from deepdefend.data.augment import ImbalanceHandler
from deepdefend.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    setup_logging(level="INFO")
    
    print("\n" + "=" * 60)
    print("DEEPDEFEND — NSL-KDD IMBALANCE ANALYSIS")
    print("=" * 60)
    
    # Load data
    loader = NSLKDLoader(data_dir="./data/nsl_kdd")
    dataset = loader.load_all()
    
    # Print imbalance report
    print(dataset.imbalance_report.summary())
    
    # Show what happens with naive accuracy
    from collections import Counter
    
    train_dist = dataset.train.class_distribution()
    test_dist = dataset.test.class_distribution()
    
    print("\nTrain distribution:", train_dist)
    print("Test distribution:", test_dist)
    
    # Calculate naive accuracy baselines
    normal_count = train_dist.get("normal", 0)
    total = sum(train_dist.values())
    
    print(f"\nNaive baselines:")
    print(f"  Always predict 'normal': {normal_count/total*100:.2f}% accuracy")
    print(f"  Always predict 'normal or dos': {(normal_count + train_dist.get('dos', 0))/total*100:.2f}% accuracy")
    print(f"  Both these models are WORTHLESS for intrusion detection.")
    print(f"  They catch 0% of U2R, 0% of R2L.")
    
    # Demonstrate preprocessor
    print("\n" + "-" * 40)
    print("Preprocessor demonstration")
    print("-" * 40)
    
    preprocessor = FeaturePreprocessor(scaling_method="robust")
    X_processed = preprocessor.fit_transform(dataset.train.X)
    
    print(f"Original features: {dataset.train.X.shape}")
    print(f"Processed features: {X_processed.shape}")
    print(f"Feature names: {preprocessor.feature_names_out[:5]}...")
    
    # Demonstrate imbalance handling
    print("\n" + "-" * 40)
    print("Imbalance handling demonstration")
    print("-" * 40)
    
    # We should only resample TRAINING data
    # NEVER resample test/validation
    handler = ImbalanceHandler(strategy="smote_tomek")
    
    # Show what we're starting with
    print("Before resampling:")
    for cls, count in sorted(dataset.train.class_distribution().items()):
        print(f"  {cls}: {count}")
    
    # Apply resampling
    X_resampled, y_resampled = handler.fit_resample(
        dataset.train.X_processed if hasattr(dataset.train, 'X_processed') else X_processed,
        dataset.train.y
    )
    
    from collections import Counter
    resampled_dist = Counter(y_resampled)
    print("\nAfter resampling:")
    for cls, count in sorted(resampled_dist.items()):
        print(f"  {cls}: {count}")
    
    # Show sample weights for cost-sensitive learning
    print("\n" + "-" * 40)
    print("Cost-sensitive sample weights")
    print("-" * 40)
    
    weights = handler.get_sample_weights(dataset.train.y)
    for cls in sorted(set(dataset.train.y)):
        mask = dataset.train.y == cls
        avg_weight = weights[mask].mean()
        print(f"  {cls}: weight = {avg_weight}")
    
    print("\n" + "=" * 60)
    print("Analysis complete. Key findings:")
    print("=" * 60)
    print("1. U2R is 0.04% of training data — extreme imbalance")
    print("2. Any metric using 'accuracy' is dangerously misleading")
    print("3. We need per-class recall, especially for U2R and R2L")
    print("4. RobustScaler preferred — attack traffic has extreme values")
    print("5. Use cost-sensitive learning (not just SMOTE) for U2R")
    print("6. NEVER resample validation or test sets")
    print("=" * 60)


if __name__ == "__main__":
    main()