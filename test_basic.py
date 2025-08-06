#!/usr/bin/env python
"""
Basic tests for ERa Attack framework
Note: These tests verify structure only, not functionality
"""

import sys
import time
import warnings

def test_imports():
    """Test if modules can be imported"""
    
    print("Testing imports...")
    
    modules_to_test = [
        'config_base',
        'core',
        'core.algorithms',
        'core.algorithms.fc_attack',
        'core.validation',
        'data',
        'data.processor',
        'utils'
    ]
    
    passed = []
    failed = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            passed.append(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            failed.append((module, str(e)))
            print(f"  ✗ {module}: {e}")
            
    return len(failed) == 0

def test_configuration():
    """Test configuration loading"""
    
    print("\nTesting configuration...")
    
    try:
        from config_base import Config
        
        config = Config()
        
        # Check attributes
        required_attrs = ['emg', 'attack', 'rf', 'physical']
        
        for attr in required_attrs:
            if hasattr(config, attr):
                print(f"  ✓ {attr} configuration")
            else:
                print(f"  ✗ Missing {attr} configuration")
                return False
                
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration failed: {e}")
        return False

def test_attack_creation():
    """Test attack instantiation"""
    
    print("\nTesting attack creation...")
    
    try:
        from core.algorithms.fc_attack import create_attack
        from config_base import Config
        
        config = Config()
        
        attacks = ['fc_pgd', 'fc_cw', 'fc_jsma']
        
        for attack_type in attacks:
            try:
                attack = create_attack(attack_type, None, config)
                print(f"  ✓ {attack_type} created")
            except Exception as e:
                print(f"  ✗ {attack_type} failed: {e}")
                return False
                
        return True
        
    except Exception as e:
        print(f"  ✗ Attack creation failed: {e}")
        return False

def test_data_processor():
    """Test data processor"""
    
    print("\nTesting data processor...")
    
    try:
        from data.processor import DataProcessor
        import numpy as np
        
        processor = DataProcessor()
        
        # Test with dummy data
        data = np.random.randn(8, 52, 52)
        processed = processor.process(data)
        
        if processed.shape == data.shape:
            print("  ✓ Data processing works")
            return True
        else:
            print("  ✗ Data shape mismatch")
            return False
            
    except Exception as e:
        print(f"  ✗ Data processor failed: {e}")
        return False

def test_validation():
    """Test validation module"""
    
    print("\nTesting validation...")
    
    try:
        from core.validation.env_check import EnvironmentValidator
        
        validator = EnvironmentValidator()
        fingerprint = validator.generate_fingerprint()
        
        if fingerprint:
            print(f"  ✓ Environment fingerprint: {fingerprint}")
            return True
        else:
            print("  ✗ Fingerprint generation failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    
    print("="*60)
    print("ERa Attack Framework Tests")
    print("="*60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Attack Creation", test_attack_creation),
        ("Data Processor", test_data_processor),
        ("Validation", test_validation)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n[{name}]")
        
        try:
            passed = test_func()
            results.append((name, passed))
            time.sleep(1)
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append((name, False))
            
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
        
    print(f"\nTotal: {passed_count}/{total_count} passed")
    
    if passed_count < total_count:
        print("\nWARNING: Some tests failed")
        print("This is expected without complete configuration")
        print("See README.md for setup instructions")
        
    return passed_count == total_count

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)