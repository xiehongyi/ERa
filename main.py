#!/usr/bin/env python
"""
Main entry point for ERa Attack
Simplified interface - use run_validation.py for full validation
"""

import sys
import argparse
import warnings

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description='ERa Attack - EMG Adversarial Attack Framework'
    )
    
    parser.add_argument(
        '--attack',
        type=str,
        default='fc_pgd',
        choices=['fc_pgd', 'fc_cw', 'fc_jsma'],
        help='Attack type'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of samples'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation first'
    )
    
    args = parser.parse_args()
    
    if args.validate:
        print("Running validation...")
        from core.validation.env_check import validate_all
        if not validate_all():
            print("\nValidation failed. Continue? (y/n): ", end="")
            if input().lower() != 'y':
                sys.exit(1)
    
    # Import after argument parsing to show help faster
    from config_base import Config
    from core.algorithms.fc_attack import AttackPipeline
    
    print("\nInitializing attack framework...")
    
    # Load configuration
    config = Config()
    
    # Create pipeline
    pipeline = AttackPipeline(config)
    
    # Run attack
    try:
        results = pipeline.run(
            attack_type=args.attack,
            num_samples=args.samples
        )
        
        print(f"\nGenerated {len(results)} adversarial samples")
        
    except Exception as e:
        warnings.warn(f"Attack failed: {e}")
        print("\nThis is expected without proper configuration.")
        print("See README.md for setup instructions.")
        sys.exit(1)

if __name__ == "__main__":
    main()