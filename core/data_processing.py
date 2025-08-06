"""
EMG信号数据处理模块
包含信号预处理、时频变换、数据加载等功能
"""

import numpy as np
import torch
import pywt
from scipy import signal
from scipy.io import loadmat
import os
from typing import Tuple, List, Optional, Union


class EMGDataProcessor:
    """EMG数据处理器"""
    
    def __init__(self, sample_rate=200, window_size=52, num_channels=8):
        """
        参数:
            sample_rate: 采样率 (Hz)
            window_size: 窗口大小（样本数）
            num_channels: EMG通道数
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.num_channels = num_channels
        self.nyquist_freq = sample_rate / 2
        
    def continuous_wavelet_transform(self, data, scales=None, wavelet='mexh'):
        """
        连续小波变换 (CWT)
        
        参数:
            data: EMG信号 [channels, time]
            scales: 小波尺度
            wavelet: 小波基函数
        
        返回:
            cwt_data: CWT系数 [channels, scales, time]
        """
        if scales is None:
            # 默认尺度，对应20-100 Hz频率范围
            freqs = np.arange(20, 101, 2)
            scales = self.sample_rate / (2 * freqs)
            
        cwt_data = []
        for ch in range(data.shape[0]):
            coeffs, _ = pywt.cwt(data[ch], scales, wavelet)
            cwt_data.append(coeffs)
            
        return np.array(cwt_data)
    
    def stft_transform(self, data, nperseg=None, noverlap=None):
        """
        短时傅里叶变换 (STFT)
        
        参数:
            data: EMG信号 [channels, time]
            nperseg: 每段长度
            noverlap: 重叠长度
        
        返回:
            stft_data: STFT系数 [channels, frequency, time]
        """
        if nperseg is None:
            nperseg = min(256, data.shape[1])
        if noverlap is None:
            noverlap = nperseg // 2
            
        stft_data = []
        for ch in range(data.shape[0]):
            f, t, Zxx = signal.stft(data[ch], fs=self.sample_rate, 
                                   nperseg=nperseg, noverlap=noverlap)
            stft_data.append(np.abs(Zxx))
            
        return np.array(stft_data), f, t
    
    def bandpass_filter(self, data, low_freq=20, high_freq=100, order=4):
        """
        带通滤波器
        
        参数:
            data: EMG信号
            low_freq: 低频截止频率 (Hz)
            high_freq: 高频截止频率 (Hz)
            order: 滤波器阶数
        
        返回:
            filtered_data: 滤波后的信号
        """
        sos = signal.butter(order, [low_freq, high_freq], 
                          btype='band', fs=self.sample_rate, output='sos')
        
        if len(data.shape) == 1:
            return signal.sosfilt(sos, data)
        else:
            filtered_data = []
            for ch in range(data.shape[0]):
                filtered_data.append(signal.sosfilt(sos, data[ch]))
            return np.array(filtered_data)
    
    def extract_features(self, data, feature_type='stft'):
        """
        提取EMG特征
        
        参数:
            data: EMG信号 [channels, time]
            feature_type: 特征类型 ('stft', 'cwt', 'raw')
        
        返回:
            features: 提取的特征
        """
        if feature_type == 'stft':
            features, _, _ = self.stft_transform(data)
        elif feature_type == 'cwt':
            features = self.continuous_wavelet_transform(data)
        elif feature_type == 'raw':
            features = data
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
        return features
    
    def normalize(self, data, method='zscore'):
        """
        数据归一化
        
        参数:
            data: 输入数据
            method: 归一化方法 ('zscore', 'minmax', 'none')
        
        返回:
            normalized_data: 归一化后的数据
        """
        if method == 'zscore':
            mean = np.mean(data, axis=-1, keepdims=True)
            std = np.std(data, axis=-1, keepdims=True) + 1e-8
            return (data - mean) / std
        elif method == 'minmax':
            min_val = np.min(data, axis=-1, keepdims=True)
            max_val = np.max(data, axis=-1, keepdims=True)
            return (data - min_val) / (max_val - min_val + 1e-8)
        elif method == 'none':
            return data
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class MyoArmbandDataLoader:
    """Myo Armband数据加载器"""
    
    def __init__(self, data_path, processor=None):
        """
        参数:
            data_path: 数据文件路径
            processor: EMGDataProcessor实例
        """
        self.data_path = data_path
        self.processor = processor if processor else EMGDataProcessor()
        
    def load_preprocessed_data(self, file_path):
        """
        加载预处理的数据文件 (.npz格式)
        
        参数:
            file_path: 数据文件路径
        
        返回:
            X: 特征数据
            y: 标签
        """
        data = np.load(file_path)
        return data['X'], data['y']
    
    def load_raw_data(self, subject_id, session_id):
        """
        加载原始EMG数据
        
        参数:
            subject_id: 被试ID
            session_id: 会话ID
        
        返回:
            emg_data: EMG信号
            labels: 动作标签
        """
        # 这里应该根据实际数据格式进行调整
        file_name = f"S{subject_id}_E{session_id}_A1.mat"
        file_path = os.path.join(self.data_path, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        mat_data = loadmat(file_path)
        emg_data = mat_data['emg']  # 假设数据存储在'emg'字段
        labels = mat_data['restimulus']  # 假设标签存储在'restimulus'字段
        
        return emg_data, labels
    
    def segment_data(self, emg_data, labels, window_size=None, overlap=0):
        """
        将连续EMG信号分割成窗口
        
        参数:
            emg_data: EMG信号 [time, channels]
            labels: 标签序列
            window_size: 窗口大小
            overlap: 重叠率 (0-1)
        
        返回:
            segments: 分割后的数据 [num_windows, channels, window_size]
            segment_labels: 每个窗口的标签
        """
        if window_size is None:
            window_size = self.processor.window_size
            
        step_size = int(window_size * (1 - overlap))
        segments = []
        segment_labels = []
        
        for i in range(0, len(emg_data) - window_size + 1, step_size):
            window = emg_data[i:i+window_size].T  # 转置为 [channels, time]
            label = labels[i + window_size // 2]  # 使用窗口中点的标签
            
            if label > 0:  # 排除休息状态（标签0）
                segments.append(window)
                segment_labels.append(label - 1)  # 将标签从1-based转为0-based
                
        return np.array(segments), np.array(segment_labels)
    
    def create_train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        创建训练集和测试集
        
        参数:
            X: 特征数据
            y: 标签
            test_size: 测试集比例
            random_state: 随机种子
        
        返回:
            X_train, X_test, y_train, y_test
        """
        np.random.seed(random_state)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return (X[train_indices], X[test_indices], 
                y[train_indices], y[test_indices])
    
    def to_torch_tensors(self, X, y, device='cpu'):
        """
        转换为PyTorch张量
        
        参数:
            X: 特征数据
            y: 标签
            device: 设备 ('cpu' 或 'cuda')
        
        返回:
            X_tensor, y_tensor
        """
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        return X_tensor, y_tensor


class AdversarialDataGenerator:
    """对抗样本数据生成器"""
    
    def __init__(self, model, attack_method, processor=None):
        """
        参数:
            model: 目标模型
            attack_method: 攻击方法实例
            processor: EMGDataProcessor实例
        """
        self.model = model
        self.attack = attack_method
        self.processor = processor if processor else EMGDataProcessor()
        
    def generate_adversarial_batch(self, X, y, target_labels=None, **attack_params):
        """
        批量生成对抗样本
        
        参数:
            X: 原始样本
            y: 原始标签
            target_labels: 目标标签（用于定向攻击）
            **attack_params: 攻击参数
        
        返回:
            X_adv: 对抗样本
            perturbations: 扰动
        """
        X_adv = []
        perturbations = []
        
        for i in range(len(X)):
            # 生成目标标签
            if target_labels is None:
                # 非定向攻击：随机选择不同于原标签的类别
                num_classes = self.model(X[i:i+1]).shape[1]
                target = np.random.choice([j for j in range(num_classes) if j != y[i]])
            else:
                target = target_labels[i]
            
            # 生成对抗扰动
            delta = self.attack.generate(X[i], target, **attack_params)
            
            # 创建对抗样本
            x_adv = X[i] + delta.squeeze()
            
            X_adv.append(x_adv)
            perturbations.append(delta.squeeze())
            
        return torch.stack(X_adv), torch.stack(perturbations)
    
    def save_adversarial_data(self, X_adv, perturbations, labels, save_path):
        """
        保存对抗样本数据
        
        参数:
            X_adv: 对抗样本
            perturbations: 扰动
            labels: 标签
            save_path: 保存路径
        """
        np.savez(save_path,
                 X_adv=X_adv.cpu().numpy() if torch.is_tensor(X_adv) else X_adv,
                 perturbations=perturbations.cpu().numpy() if torch.is_tensor(perturbations) else perturbations,
                 labels=labels.cpu().numpy() if torch.is_tensor(labels) else labels)
        print(f"Adversarial data saved to {save_path}")
    
    def load_adversarial_data(self, load_path):
        """
        加载对抗样本数据
        
        参数:
            load_path: 加载路径
        
        返回:
            X_adv, perturbations, labels
        """
        data = np.load(load_path)
        return data['X_adv'], data['perturbations'], data['labels']