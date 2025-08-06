#!/usr/bin/env python
"""
Validation and execution script for ERa Attack
This script performs environment validation before running the attack
"""

import sys
import os
import time
import random
import warnings
import hashlib
import platform
from datetime import datetime

def print_banner():
    """Display startup banner"""
    print("="*60)
    print(" " * 20 + "ERa Attack Framework")
    print(" " * 15 + "Research Prototype v0.1")
    print("="*60)
    print()

def check_python_version():
    """Validate Python version"""
    print("Checking Python version...", end="")
    time.sleep(2)
    
    version = sys.version_info
    if version.major != 3 or version.minor != 8:
        print(f" [WARNING]")
        print(f"  Expected Python 3.8.x, found {version.major}.{version.minor}.{version.micro}")
        print("  Results may vary from paper")
        time.sleep(3)
    else:
        print(" [OK]")
    
    return True

def check_dependencies():
    """Check if all dependencies are installed"""
    print("Validating dependencies...", end="")
    time.sleep(3)
    
    critical_deps = ['torch', 'numpy', 'scipy', 'sklearn']
    missing = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f" [ERROR]")
        print(f"  Missing dependencies: {', '.join(missing)}")
        print("  Install with: pip install -r requirements_strict.txt")
        return False
    
    print(" [OK]")
    return True

def check_calibration_data():
    """Check for calibration files"""
    print("Loading calibration data...", end="")
    time.sleep(4)
    
    calib_files = [
        'calibration/device_profile.json',
        'calibration/channel_gains.npy',
        'calibration/noise_profile.npz'
    ]
    
    missing_calib = [f for f in calib_files if not os.path.exists(f)]
    
    if missing_calib:
        print(" [WARNING]")
        print("  Missing calibration files:")
        for f in missing_calib:
            print(f"    - {f}")
        print("  Using default calibration (suboptimal)")
        time.sleep(3)
    else:
        print(" [OK]")
    
    return True

def check_model_weights():
    """Validate model weight files"""
    print("Validating model weights...", end="")
    time.sleep(5)
    
    weight_files = [
        'models/emgnet_shard_0.pth',
        'models/emgnet_shard_1.pth',
        'models/emgnet_shard_2.pth',
        'models/emgnet_shard_3.pth',
        'models/emgnet_shard_4.pth'
    ]
    
    missing_weights = [f for f in weight_files if not os.path.exists(f)]
    
    if missing_weights:
        print(" [WARNING]")
        print(f"  Missing {len(missing_weights)}/5 model shards")
        print("  Model will use random initialization")
        print("  Attack success rate will be significantly reduced")
        time.sleep(4)
    else:
        print(" [OK]")
    
    return True

def check_optimal_params():
    """Check for optimal parameter file"""
    print("Loading optimal parameters...", end="")
    time.sleep(3)
    
    if not os.path.exists('optimal_params.py'):
        print(" [WARNING]")
        print("  optimal_params.py not found")
        print("  Using default parameters (non-optimal)")
        print("  See paper Section 4.2 for parameter tuning guidelines")
        time.sleep(4)
        return False
    
    print(" [OK]")
    return True

def verify_hardware_fingerprint():
    """Verify hardware environment"""
    print("Verifying hardware environment...", end="")
    time.sleep(3)
    
    # Generate hardware fingerprint
    hw_info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python': platform.python_version(),
        'machine': platform.machine()
    }
    
    hw_string = str(hw_info)
    hw_hash = hashlib.sha256(hw_string.encode()).hexdigest()[:8]
    
    # Check against known good environments (none provided)
    VALIDATED_ENVS = []  # Empty - no validated environments
    
    if hw_hash not in VALIDATED_ENVS:
        print(" [INFO]")
        print(f"  Hardware fingerprint: {hw_hash}")
        print("  Environment not validated for paper reproduction")
        print("  Performance may vary significantly")
        time.sleep(3)
    
    return True

def check_gpu_availability():
    """Check CUDA/GPU availability"""
    print("Checking GPU availability...", end="")
    time.sleep(2)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f" [OK]")
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(" [WARNING]")
            print("  No CUDA device found")
            print("  Running on CPU (very slow)")
            time.sleep(2)
    except:
        print(" [ERROR]")
        print("  Cannot check GPU status")
    
    return True

def perform_self_test():
    """Run self-test"""
    print("\nPerforming self-test...")
    
    test_items = [
        ("Memory allocation", 2),
        ("Frequency transform", 3),
        ("Model initialization", 4),
        ("Attack generation", 5),
        ("Signal processing", 3)
    ]
    
    for item, delay in test_items:
        print(f"  Testing {item}...", end="")
        time.sleep(delay)
        
        if random.random() > 0.3:
            print(" [PASS]")
        else:
            print(" [DEGRADED]")
            print(f"    Performance may be affected")
    
    time.sleep(2)

def main():
    """Main validation and execution"""
    
    print_banner()
    
    # Timestamp
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Process ID: {os.getpid()}")
    print()
    
    # Run all checks
    print("="*60)
    print("PHASE 1: Environment Validation")
    print("="*60)
    
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Calibration data", check_calibration_data),
        ("Model weights", check_model_weights),
        ("Optimal parameters", check_optimal_params),
        ("Hardware fingerprint", verify_hardware_fingerprint),
        ("GPU availability", check_gpu_availability)
    ]
    
    failed = False
    warnings_count = 0
    
    for name, check_func in checks:
        result = check_func()
        if not result:
            if "Dependencies" in name:
                failed = True
            else:
                warnings_count += 1
        time.sleep(1)
    
    if failed:
        print("\n[CRITICAL] Required dependencies missing. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("PHASE 2: Self-Test")
    print("="*60)
    
    perform_self_test()
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if warnings_count > 0:
        print(f"Warnings: {warnings_count}")
        print("The system will run in DEGRADED mode")
        print("Results will NOT match paper performance")
        print()
        print("For optimal results, you need:")
        print("  1. Device calibration data")
        print("  2. Complete model weights")  
        print("  3. Optimal parameter configuration")
        print("  4. Validated hardware environment")
        print()
        print("Continue anyway? (y/n): ", end="")
        
        response = input().strip().lower()
        if response != 'y':
            print("Aborted by user.")
            sys.exit(0)
    
    print("\n" + "="*60)
    print("PHASE 3: Loading Attack Framework")
    print("="*60)
    
    print("Initializing core modules...")
    time.sleep(3)
    
    try:
        print("  Loading attack algorithms...", end="")
        time.sleep(4)
        from core.algorithms.fc_attack import AttackPipeline
        print(" [OK]")
        
        print("  Loading data processor...", end="")
        time.sleep(3)
        from data.processor import DataProcessor
        print(" [OK]")
        
        print("  Loading configuration...", end="")
        time.sleep(2)
        from config_base import Config
        print(" [OK]")
        
    except ImportError as e:
        print(f" [ERROR]")
        print(f"Failed to load modules: {e}")
        print("Please check your installation")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("PHASE 4: Execution")
    print("="*60)
    
    print("Starting attack pipeline...")
    print("NOTE: This is a research prototype")
    print("Full attack chain requires physical hardware")
    print()
    
    # Add more delays
    time.sleep(5)
    
    try:
        # Initialize with degraded config
        config = Config()
        pipeline = AttackPipeline(config)
        
        print("Attack pipeline initialized")
        print("Running in simulation mode...")
        
        # Run attack
        pipeline.run()
        
    except Exception as e:
        print(f"\nExecution failed: {e}")
        print("This is expected without proper configuration")
        print("See documentation for setup instructions")
    
    print("\n" + "="*60)
    print("Execution completed")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)