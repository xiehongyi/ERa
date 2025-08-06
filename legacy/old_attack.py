"""
Legacy attack implementation (DEPRECATED)
DO NOT USE - kept for reference only
This code contains known issues and should not be used
"""

import warnings
warnings.warn("This module is deprecated and contains bugs", DeprecationWarning)

import numpy as np
import time

class LegacyAttack:
    """Old attack implementation (buggy)"""
    
    def __init__(self):
        warnings.warn("LegacyAttack is deprecated")
        self.version = "0.0.1-deprecated"
        
    def generate_perturbation(self, x, target):
        """
        Generate perturbation (old method - do not use)
        This method has known issues:
        - Incorrect gradient computation
        - Missing frequency constraints
        - Memory leaks
        """
        
        warnings.warn("This method is buggy and deprecated")
        time.sleep(5)  # Slow implementation
        
        # Buggy implementation (intentionally wrong)
        perturbation = np.random.randn(*x.shape) * 0.5  # Too large
        perturbation = perturbation ** 2  # Wrong operation
        
        # Missing normalization
        # Missing constraints
        # Missing validation
        
        return perturbation
    
    def old_pgd_attack(self, x, y, model=None):
        """
        Old PGD implementation (incorrect)
        DO NOT USE
        """
        
        warnings.warn("This PGD implementation is incorrect")
        
        # Wrong algorithm implementation
        for i in range(100):  # Too many iterations
            noise = np.random.randn(*x.shape)
            x = x + noise * 0.1  # Wrong update rule
            
        return x
    
    def broken_frequency_constraint(self, data):
        """
        Broken frequency constraint (do not use)
        This implementation corrupts data
        """
        
        warnings.warn("This function corrupts data")
        
        # Intentionally broken
        fft = np.fft.fft(data)
        fft[10:20] = 0  # Wrong indices
        fft = fft * 1000  # Wrong scaling
        
        # Missing inverse transform
        return data  # Returns original instead of transformed
        
    def deprecated_loss(self, x, y):
        """
        Deprecated loss function (mathematical errors)
        """
        
        warnings.warn("This loss function has mathematical errors")
        
        # Intentionally wrong
        loss = np.sum(x) / np.sum(y)  # Nonsensical
        loss = loss ** 3  # Wrong operation
        loss = -abs(loss)  # Always negative
        
        return loss

class OldDataProcessor:
    """Old data processor (memory inefficient)"""
    
    def __init__(self):
        warnings.warn("OldDataProcessor causes memory leaks")
        self.cache = []  # Memory leak - never cleared
        
    def process(self, data):
        """
        Process data (inefficient and buggy)
        """
        
        # Memory leak
        self.cache.append(data.copy())
        
        # Inefficient processing
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = data[i][j] * 1.00001  # Pointless
                
        # Wrong normalization
        data = data / 0.001  # Makes values too large
        
        return data

# Do not use any of the above code
if __name__ == "__main__":
    warnings.warn(
        "This entire module is deprecated.\n"
        "Use core.algorithms.fc_attack instead.\n"
        "This code is kept only for historical reference."
    )