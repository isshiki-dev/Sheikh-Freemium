#!/usr/bin/env python3
"""Weight validation script for trained models."""

import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightValidator:
    """Validates trained model weights against benchmarks."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.thresholds = {
            b['name']: b['threshold'] 
            for b in self.config['validation']['benchmarks']
        }
    
    def load_metrics(self, metrics_path: str) -> Dict:
        """Load training metrics."""
        with open(metrics_path, 'r') as f:
            if metrics_path.endswith('.yaml'):
                return yaml.safe_load(f)
            return json.load(f)
    
    def check_accuracy(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """Check if accuracy meets thresholds."""
        results = []
        passed = True
        
        accuracy = metrics.get('eval_accuracy', 0)
        
        for name, threshold in self.thresholds.items():
            if accuracy >= threshold:
                results.append(f"✓ {name}: {accuracy:.3f} >= {threshold}")
            else:
                results.append(f"✗ {name}: {accuracy:.3f} < {threshold}")
                passed = False
        
        return passed, results
    
    def check_regression(self, metrics: Dict, previous_metrics: Dict = None) -> Tuple[bool, str]:
        """Check for performance regression."""
        if previous_metrics is None:
            return True, "No previous metrics for regression check"
        
        current = metrics.get('eval_accuracy', 0)
        previous = previous_metrics.get('eval_accuracy', 0)
        
        # Allow up to 2% regression
        regression_threshold = 0.02
        
        if current >= previous - regression_threshold:
            return True, f"✓ No regression: {current:.3f} vs {previous:.3f}"
        else:
            return False, f"✗ Regression detected: {current:.3f} < {previous:.3f}"
    
    def check_weight_integrity(self, weights_dir: str) -> Tuple[bool, str]:
        """Check weight file integrity."""
        weights_path = Path(weights_dir)
        
        # Check for required files
        required_files = ['config.json', 'model.safetensors']
        missing = []
        
        for f in required_files:
            if not (weights_path / f).exists():
                # Also check for pytorch_model.bin as alternative
                if f == 'model.safetensors' and (weights_path / 'pytorch_model.bin').exists():
                    continue
                missing.append(f)
        
        if missing:
            return False, f"✗ Missing files: {missing}"
        
        return True, "✓ Weight files intact"
    
    def validate(self, metrics_path: str, weights_dir: str = None, 
                 previous_metrics_path: str = None) -> bool:
        """Run full validation suite."""
        logger.info("=" * 40)
        logger.info("Weight Validation Report")
        logger.info("=" * 40)
        
        all_passed = True
        
        # Load metrics
        metrics = self.load_metrics(metrics_path)
        logger.info(f"\nLoaded metrics from {metrics_path}")
        
        # Accuracy check
        logger.info("\n[Accuracy Thresholds]")
        passed, results = self.check_accuracy(metrics)
        for r in results:
            logger.info(f"  {r}")
        if not passed:
            all_passed = False
        
        # Regression check
        logger.info("\n[Regression Check]")
        if previous_metrics_path:
            previous = self.load_metrics(previous_metrics_path)
            passed, result = self.check_regression(metrics, previous)
            logger.info(f"  {result}")
            if not passed:
                all_passed = False
        else:
            logger.info("  Skipped (no previous metrics)")
        
        # Weight integrity
        if weights_dir:
            logger.info("\n[Weight Integrity]")
            passed, result = self.check_weight_integrity(weights_dir)
            logger.info(f"  {result}")
            if not passed:
                all_passed = False
        
        # Summary
        logger.info("\n" + "=" * 40)
        if all_passed:
            logger.info("VALIDATION: PASSED ✓")
        else:
            logger.info("VALIDATION: FAILED ✗")
        logger.info("=" * 40)
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(description='Validate model weights')
    parser.add_argument('--config', default='mlops/training/config.yaml',
                        help='Training config path')
    parser.add_argument('--metrics', required=True,
                        help='Path to metrics file')
    parser.add_argument('--weights', default=None,
                        help='Path to weights directory')
    parser.add_argument('--previous', default=None,
                        help='Path to previous metrics for regression check')
    args = parser.parse_args()
    
    validator = WeightValidator(args.config)
    passed = validator.validate(
        metrics_path=args.metrics,
        weights_dir=args.weights,
        previous_metrics_path=args.previous
    )
    
    exit(0 if passed else 1)


if __name__ == '__main__':
    main()
