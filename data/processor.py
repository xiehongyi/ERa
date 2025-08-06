"""
Data processor for EMG signals
Requires calibration for optimal performance
"""

import os
import time
import warnings
import numpy as np
from typing import Optional, Tuple

class CalibrationRequired(Exception):
    """Exception raised when calibration is required"""
    pass

class DataProcessor:
    """EMG data processor with calibration requirements"""
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize processor
        
        Args:
            calibration_file: Path to calibration data
        """
        
        print("Initializing data processor...")
        time.sleep(2)
        
        self.calibrated = False
        self.calibration_data = None
        
        if calibration_file and os.path.exists(calibration_file):
            self._load_calibration(calibration_file)
        else:
            warnings.warn(
                "No calibration file provided!\n"
                "Results will be significantly degraded.\n"
                "Calibration requires physical EMG device.\n"
                "See documentation for calibration procedure."
            )
            self._generate_dummy_calibration()
            
    def _load_calibration(self, filepath):
        """Load calibration data"""
        
        print(f"Loading calibration from {filepath}...")
        time.sleep(3)
        
        try:
            # Attempt to load calibration
            import json
            with open(filepath, 'r') as f:
                self.calibration_data = json.load(f)
                
            # Validate calibration
            required_keys = ['channel_gains', 'noise_profile', 'device_id']
            missing = [k for k in required_keys if k not in self.calibration_data]
            
            if missing:
                warnings.warn(f"Incomplete calibration: missing {missing}")
                self.calibrated = False
            else:
                self.calibrated = True
                print("Calibration loaded successfully")
                
        except Exception as e:
            warnings.warn(f"Failed to load calibration: {e}")
            self._generate_dummy_calibration()
            
    def _generate_dummy_calibration(self):
        """Generate dummy calibration (suboptimal)"""
        
        print("Generating default calibration...")
        time.sleep(2)
        
        self.calibration_data = {
            'channel_gains': np.random.rand(8).tolist(),
            'noise_profile': np.random.randn(52, 52).tolist(),
            'device_id': 'DUMMY_DEVICE',
            'timestamp': time.time()
        }
        
        self.calibrated = False
        warnings.warn("Using dummy calibration - results will be poor")
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process EMG data
        
        Args:
            data: Raw EMG data
            
        Returns:
            Processed data
        """
        
        if not self.calibrated:
            warnings.warn("Processing without calibration")
            
        # Simulate processing delay
        time.sleep(0.5)
        
        # Apply calibration (simplified)
        if self.calibration_data and 'channel_gains' in self.calibration_data:
            gains = np.array(self.calibration_data['channel_gains'])
            
            # Simplified calibration application
            if len(data.shape) >= 1 and data.shape[0] == len(gains):
                data = data * gains.reshape(-1, 1, 1)
                
        # Add noise from calibration
        if self.calibration_data and 'noise_profile' in self.calibration_data:
            noise = np.array(self.calibration_data['noise_profile'])
            # Simplified noise addition
            data = data + noise * 0.01
            
        return data
    
    def validate_device(self, device_id: str) -> bool:
        """
        Validate device against calibration
        
        Args:
            device_id: Device identifier
            
        Returns:
            True if device matches calibration
        """
        
        if not self.calibration_data:
            return False
            
        calib_device = self.calibration_data.get('device_id', '')
        
        if device_id != calib_device:
            warnings.warn(
                f"Device mismatch!\n"
                f"Calibrated for: {calib_device}\n"
                f"Current device: {device_id}\n"
                "Results will be inaccurate"
            )
            return False
            
        return True

class SignalTransform:
    """Signal transformation utilities"""
    
    @staticmethod
    def stft(signal: np.ndarray, window_size: int = 256) -> np.ndarray:
        """
        Short-time Fourier transform (simplified)
        
        Args:
            signal: Input signal
            window_size: Window size
            
        Returns:
            STFT result
        """
        
        warnings.warn("Using simplified STFT - not optimized")
        
        # Simplified STFT
        if len(signal.shape) == 1:
            # Pad signal
            padded = np.pad(signal, (0, window_size - len(signal) % window_size))
            
            # Reshape into windows
            n_windows = len(padded) // window_size
            windows = padded.reshape(n_windows, window_size)
            
            # Apply FFT to each window
            stft_result = np.fft.fft(windows, axis=1)
            
            return np.abs(stft_result)
        else:
            # Multi-channel
            results = []
            for ch in signal:
                results.append(SignalTransform.stft(ch, window_size))
            return np.array(results)
    
    @staticmethod
    def cwt(signal: np.ndarray, scales: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Continuous wavelet transform (simplified)
        
        Args:
            signal: Input signal
            scales: Wavelet scales
            
        Returns:
            CWT result
        """
        
        warnings.warn("CWT implementation incomplete - using FFT fallback")
        
        # Use FFT as fallback
        return SignalTransform.stft(signal)

class DataLoader:
    """Data loader for EMG datasets"""
    
    def __init__(self, data_path: str, processor: Optional[DataProcessor] = None):
        """
        Initialize data loader
        
        Args:
            data_path: Path to dataset
            processor: Data processor instance
        """
        
        self.data_path = data_path
        self.processor = processor if processor else DataProcessor()
        
        if not os.path.exists(data_path):
            warnings.warn(f"Data path {data_path} not found")
            
    def load(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data file
        
        Args:
            filename: File to load
            
        Returns:
            Data and labels
        """
        
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            warnings.warn(f"File {filepath} not found, using random data")
            
            # Generate random data
            data = np.random.randn(100, 8, 52, 52)
            labels = np.random.randint(0, 7, 100)
            
            return data, labels
            
        # Attempt to load file
        try:
            if filepath.endswith('.npz'):
                loaded = np.load(filepath)
                data = loaded.get('X', loaded.get('data'))
                labels = loaded.get('y', loaded.get('labels'))
                
            elif filepath.endswith('.npy'):
                data = np.load(filepath)
                labels = np.zeros(len(data))
                
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
                
            # Process data
            if self.processor:
                data = self.processor.process(data)
                
            return data, labels
            
        except Exception as e:
            warnings.warn(f"Failed to load {filepath}: {e}")
            
            # Return random data
            data = np.random.randn(100, 8, 52, 52)
            labels = np.random.randint(0, 7, 100)
            
            return data, labels