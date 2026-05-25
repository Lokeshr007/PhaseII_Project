"""
Custom metrics for imbalanced network intrusion detection.

These are the metrics that MATTER. Not accuracy.
Accuracy is reported only to show how misleading it is.
"""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    fbeta_score,
    recall_score,
    precision_score
)
from typing import Dict, Optional
from collections import defaultdict


class ImbalanceMetrics:
    """
    Metrics suite designed for extreme class imbalance.
    
    Tracks per-class performance with emphasis on minority classes.
    """
    
    def __init__(self, class_names: list, minority_classes: list = None):
        self.class_names = class_names
        self.minority_classes = minority_classes or []
        
    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: Optional[np.ndarray] = None) -> Dict:
        """Compute comprehensive metrics."""
        
        metrics = {}
        
        # Standard classification report
        report = classification_report(
            y_true, y_pred,
            labels=self.class_names,
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Accuracy (reported only to show its uselessness)
        accuracy = (y_true == y_pred).mean()
        metrics['accuracy'] = float(accuracy)
        metrics['accuracy_warning'] = (
            f"Accuracy is {accuracy:.4f}. A model predicting 'normal' on everything "
            f"gets {max(np.mean(y_true == cls) for cls in self.class_names):.4f}. "
            f"Do not use accuracy to evaluate this model."
        )
        
        # Per-class metrics
        per_class = defaultdict(dict)
        for cls in self.class_names:
            y_true_cls = (y_true == cls)
            y_pred_cls = (y_pred == cls)
            
            tp = ((y_pred_cls) & (y_true_cls)).sum()
            fp = ((y_pred_cls) & (~y_true_cls)).sum()
            fn = ((~y_pred_cls) & (y_true_cls)).sum()
            tn = ((~y_pred_cls) & (~y_true_cls)).sum()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            per_class[cls] = {
                'recall': float(recall),
                'precision': float(precision),
                'f1': float(f1),
                'fpr': float(fpr),
                'support': int((y_true == cls).sum()),
                'detected': int(tp),
                'missed': int(fn),
                'false_positives': int(fp),
            }
        
        metrics['per_class'] = dict(per_class)
        
        # Aggregate metrics for minority classes
        minority_recalls = []
        for cls in self.minority_classes:
            if cls in per_class:
                minority_recalls.append(per_class[cls]['recall'])
        
        if minority_recalls:
            metrics['minority_mean_recall'] = float(np.mean(minority_recalls))
            metrics['minority_min_recall'] = float(np.min(minority_recalls))
        
        # F-beta scores (beta > 1 weights recall higher)
        for beta in [1, 2, 3]:
            f_beta = fbeta_score(
                y_true, y_pred,
                beta=beta,
                average='weighted',
                labels=self.class_names,
                zero_division=0
            )
            metrics[f'f{beta}_score_weighted'] = float(f_beta)
        
        # If probabilities available
        if y_proba is not None:
            # One-vs-rest AUPRC for each class
            for i, cls in enumerate(self.class_names):
                if i < y_proba.shape[1]:
                    y_true_binary = (y_true == cls).astype(int)
                    try:
                        auprc = average_precision_score(y_true_binary, y_proba[:, i])
                        metrics[f'auprc_{cls}'] = float(auprc)
                    except:
                        metrics[f'auprc_{cls}'] = 0.0
        
        # Alert fatigue metric
        total_fp = sum(pc['false_positives'] for pc in per_class.values())
        total_samples = len(y_true)
        metrics['false_positive_rate_overall'] = float(total_fp / total_samples) if total_samples > 0 else 0.0
        metrics['alerts_per_100k'] = float((total_fp / total_samples) * 100000) if total_samples > 0 else 0.0
        
        # Detection effectiveness
        total_attacks = sum(
            pc['support'] for cls, pc in per_class.items() 
            if cls != 'normal'
        )
        total_detected = sum(
            pc['detected'] for cls, pc in per_class.items() 
            if cls != 'normal'
        )
        metrics['attack_detection_rate'] = float(total_detected / total_attacks) if total_attacks > 0 else 0.0
        metrics['total_attacks'] = int(total_attacks)
        metrics['total_detected'] = int(total_detected)
        metrics['total_missed'] = int(total_attacks - total_detected)
        
        return metrics
    
    def summary(self, metrics: Dict) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("INTRUSION DETECTION METRICS (Imbalance-Aware)")
        lines.append("=" * 60)
        lines.append(f"Accuracy: {metrics['accuracy']:.4f} (IGNORE THIS)")
        lines.append(f"Attack Detection Rate: {metrics['attack_detection_rate']:.4f}")
        lines.append(f"Total Attacks: {metrics['total_attacks']} | Detected: {metrics['total_detected']} | Missed: {metrics['total_missed']}")
        lines.append(f"False Positives per 100k samples: {metrics['alerts_per_100k']:.1f}")
        lines.append("")
        lines.append(f"{'Class':<10} {'Recall':>8} {'Precision':>10} {'F1':>8} {'FPR':>8} {'Support':>8}")
        lines.append("-" * 55)
        
        for cls in self.class_names:
            pc = metrics['per_class'].get(cls, {})
            lines.append(
                f"{cls:<10} {pc.get('recall', 0):>8.4f} "
                f"{pc.get('precision', 0):>10.4f} {pc.get('f1', 0):>8.4f} "
                f"{pc.get('fpr', 0):>8.4f} {pc.get('support', 0):>8}"
            )
        
        if 'minority_mean_recall' in metrics:
            lines.append(f"\nMinority Class (U2R/R2L) Mean Recall: {metrics['minority_mean_recall']:.4f}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)