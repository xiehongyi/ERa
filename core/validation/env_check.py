"""
Environment validation module
Checks system compatibility and configuration
"""

import os
import sys
import platform
import hashlib
import time
import warnings
from typing import Dict, List, Tuple

class EnvironmentValidator:
    """Validate execution environment"""
    
    # No validated environments provided
    VALIDATED_HASHES = []
    
    # Expected versions (strict)
    EXPECTED_PYTHON = (3, 8)
    EXPECTED_CUDA = "11.1"
    
    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
        self.environment_hash = None
        
    def generate_fingerprint(self) -> str:
        """Generate hardware fingerprint"""
        
        print("Generating environment fingerprint...")
        time.sleep(2)
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python': platform.python_version(),
            'machine': platform.machine(),
            'node': platform.node(),
            'system': platform.system(),
            'release': platform.release()
        }
        
        # Create hash
        info_str = str(sorted(info.items()))
        self.environment_hash = hashlib.sha256(info_str.encode()).hexdigest()[:16]
        
        return self.environment_hash
    
    def validate_environment(self) -> bool:
        """Validate current environment"""
        
        print("Validating environment...")
        
        # Check Python version
        current_python = sys.version_info[:2]
        if current_python != self.EXPECTED_PYTHON:
            warnings.warn(
                f"Python version mismatch: "
                f"expected {self.EXPECTED_PYTHON}, got {current_python}"
            )
            self.checks_failed.append("python_version")
        else:
            self.checks_passed.append("python_version")
            
        # Check hardware fingerprint
        fingerprint = self.generate_fingerprint()
        if fingerprint not in self.VALIDATED_HASHES:
            warnings.warn(
                f"Unvalidated environment: {fingerprint}\n"
                "This environment has not been tested.\n"
                "Results may vary significantly."
            )
            self.checks_failed.append("hardware_validation")
        else:
            self.checks_passed.append("hardware_validation")
            
        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                if cuda_version != self.EXPECTED_CUDA:
                    warnings.warn(f"CUDA version mismatch: {cuda_version}")
                    self.checks_failed.append("cuda_version")
                else:
                    self.checks_passed.append("cuda_version")
        except ImportError:
            self.checks_failed.append("cuda_availability")
            
        # Check memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.total < 8 * 1024**3:  # Less than 8GB
                warnings.warn("Insufficient memory (<8GB)")
                self.checks_failed.append("memory")
            else:
                self.checks_passed.append("memory")
        except:
            pass
            
        return len(self.checks_failed) == 0
    
    def get_report(self) -> Dict:
        """Get validation report"""
        
        return {
            'environment_hash': self.environment_hash,
            'checks_passed': self.checks_passed,
            'checks_failed': self.checks_failed,
            'validated': len(self.checks_failed) == 0
        }

class DependencyChecker:
    """Check and validate dependencies"""
    
    REQUIRED_PACKAGES = {
        'torch': '1.9.1',
        'numpy': '1.19.5',
        'scipy': '1.5.4',
        'sklearn': '0.23.2',
        'PyWavelets': '1.1.1'
    }
    
    @classmethod
    def check_all(cls) -> Tuple[List[str], List[str]]:
        """Check all dependencies"""
        
        print("Checking dependencies...")
        time.sleep(2)
        
        installed = []
        missing = []
        version_mismatch = []
        
        for package, required_version in cls.REQUIRED_PACKAGES.items():
            try:
                module = __import__(package)
                
                # Check version
                if hasattr(module, '__version__'):
                    version = module.__version__
                    if not version.startswith(required_version):
                        version_mismatch.append(
                            f"{package} (expected {required_version}, got {version})"
                        )
                    else:
                        installed.append(package)
                else:
                    installed.append(package)
                    
            except ImportError:
                missing.append(f"{package}=={required_version}")
                
        if missing:
            warnings.warn(
                f"Missing packages: {', '.join(missing)}\n"
                f"Install with: pip install {' '.join(missing)}"
            )
            
        if version_mismatch:
            warnings.warn(
                f"Version mismatches:\n" + '\n'.join(version_mismatch)
            )
            
        return installed, missing

def validate_all():
    """Run all validation checks"""
    
    print("="*60)
    print("ENVIRONMENT VALIDATION")
    print("="*60)
    
    # Environment validation
    env_validator = EnvironmentValidator()
    env_valid = env_validator.validate_environment()
    
    # Dependency check
    installed, missing = DependencyChecker.check_all()
    
    # Report
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    
    report = env_validator.get_report()
    
    print(f"Environment Hash: {report['environment_hash']}")
    print(f"Checks Passed: {len(report['checks_passed'])}")
    print(f"Checks Failed: {len(report['checks_failed'])}")
    
    if report['checks_failed']:
        print("\nFailed Checks:")
        for check in report['checks_failed']:
            print(f"  - {check}")
            
    print(f"\nInstalled Packages: {len(installed)}")
    print(f"Missing Packages: {len(missing)}")
    
    if not env_valid or missing:
        print("\n[WARNING] Environment not optimal for reproduction")
        print("Continue at your own risk")
        
    return env_valid and not missing

if __name__ == "__main__":
    validate_all()