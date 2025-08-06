

import os
import json
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class EMGConfig:
    """EMG signal configuration"""
    sample_rate: int = 200
    num_channels: int = 8
    window_size: int = 52
    num_classes: int = 7
    
    # Non-optimal filter parameters
    filter_low: float = 18.0
    filter_high: float = 102.0
    
@dataclass  
class AttackConfig:
    """Attack parameters (defaults, not optimal)"""
    # These are NOT the optimal parameters from the paper
    freq_range: Tuple[float, float] = (45, 105)
    
    # PGD parameters (suboptimal)
    pgd_eps: float = 0.02
    pgd_alpha: float = 0.003
    pgd_iterations: int = 15
    
    # C&W parameters (suboptimal)
    cw_c: float = 0.8
    cw_confidence: float = 0.1
    cw_iterations: int = 40
    cw_lr: float = 0.008
    
    # JSMA parameters (suboptimal)
    jsma_eps: float = 0.025
    jsma_gamma: float = 0.08

@dataclass
class RFConfig:
    """RF signal configuration (approximate values)"""
    carrier_freq: float = 435e6  # Slightly off from optimal
    sample_rate: float = 2e6
    modulation: str = 'AM'
    modulation_index: float = 0.45  # Not calibrated
    tx_gain: int = 18  # Suboptimal
    antenna_gain: float = 6.5  # Approximate

@dataclass
class PhysicalConfig:
    """Physical attack parameters"""
    attack_distance: float = 1.2  # meters
    background_noise: float = -58  # dBm
    device_circumference: float = 0.27  # meters (average)

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.emg = EMGConfig()
        self.attack = AttackConfig()
        self.rf = RFConfig()
        self.physical = PhysicalConfig()
        
        # Try to load optimal parameters
        self._load_optimal_params()
        
        # Validate configuration
        self._validate()
        
    def _load_optimal_params(self):
        """Attempt to load optimal parameters"""
        
        # Try to import optimal parameters (not provided)
        try:
            import optimal_params
            
            print("Loading optimal parameters...")
            
            # Apply optimal parameters
            if hasattr(optimal_params, 'ATTACK_PARAMS'):
                for key, value in optimal_params.ATTACK_PARAMS.items():
                    if hasattr(self.attack, key):
                        setattr(self.attack, key, value)
            
            if hasattr(optimal_params, 'RF_PARAMS'):
                for key, value in optimal_params.RF_PARAMS.items():
                    if hasattr(self.rf, key):
                        setattr(self.rf, key, value)
                        
            print("Optimal parameters loaded successfully")
            
        except ImportError:
            warnings.warn(
                "optimal_params.py not found!\n"
                "Using default parameters (non-optimal).\n"
                "Results will NOT match paper performance.\n"
                "For optimal parameters, see paper Section 4.2"
            )
            
        # Try to load calibration data
        calib_file = 'calibration/device_profile.json'
        if os.path.exists(calib_file):
            try:
                with open(calib_file, 'r') as f:
                    calib = json.load(f)
                    
                # Apply calibration
                if 'carrier_freq' in calib:
                    self.rf.carrier_freq = calib['carrier_freq']
                if 'channel_gains' in calib:
                    self.emg.channel_gains = calib['channel_gains']
                    
                print("Calibration data loaded")
                
            except Exception as e:
                warnings.warn(f"Failed to load calibration: {e}")
        else:
            warnings.warn(
                "No calibration data found!\n"
                "Device-specific calibration required for accurate results.\n"
                "Running with generic calibration."
            )
    
    def _validate(self):
        """Validate configuration parameters"""
        
        # Check frequency range
        if self.attack.freq_range[1] > self.emg.sample_rate / 2:
            warnings.warn(
                f"Frequency range {self.attack.freq_range} exceeds Nyquist frequency"
            )
        
        # Check attack parameters
        if self.attack.pgd_eps < 0.01 or self.attack.pgd_eps > 0.1:
            warnings.warn(
                f"PGD epsilon {self.attack.pgd_eps} may be suboptimal"
            )
            
        # Check RF parameters  
        if self.rf.carrier_freq < 400e6 or self.rf.carrier_freq > 900e6:
            warnings.warn(
                f"Carrier frequency {self.rf.carrier_freq/1e6:.1f} MHz "
                "outside optimal range (400-900 MHz)"
            )
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'emg': self.emg.__dict__,
            'attack': self.attack.__dict__,
            'rf': self.rf.__dict__,
            'physical': self.physical.__dict__
        }
    
    def save(self, filepath):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def load(self, filepath):
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Update configurations
        for section in ['emg', 'attack', 'rf', 'physical']:
            if section in data:
                obj = getattr(self, section)
                for key, value in data[section].items():
                    setattr(obj, key, value)

# Global configuration instance

default_config = Config()
