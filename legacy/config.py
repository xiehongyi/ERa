"""
ERa Attack 配置文件
统一管理所有实验参数
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class EMGConfig:
    """EMG信号相关配置"""
    sample_rate: int = 200  # 采样率 (Hz)
    num_channels: int = 8  # EMG通道数
    window_size: int = 52  # 分析窗口大小
    num_classes: int = 7  # 手势类别数
    
    # 信号处理参数
    filter_low_freq: float = 20.0  # 带通滤波器低频截止 (Hz)
    filter_high_freq: float = 100.0  # 带通滤波器高频截止 (Hz)
    filter_order: int = 4  # 滤波器阶数
    
    # 特征提取
    feature_type: str = 'stft'  # 特征类型: 'stft', 'cwt', 'raw'
    normalization: str = 'zscore'  # 归一化方法: 'zscore', 'minmax', 'none'


@dataclass
class AttackConfig:
    """攻击参数配置"""
    # 频域约束
    freq_range: Tuple[float, float] = (50, 100)  # 有效频率范围 (Hz)
    
    # FC-PGD参数
    pgd_eps: float = 8/255  # L_inf扰动边界
    pgd_alpha: float = 2/255  # 步长
    pgd_num_iter: int = 20  # 迭代次数
    
    # FC-C&W参数
    cw_c: float = 1.0  # 正则化权重
    cw_kappa: float = 0  # 置信度参数
    cw_num_iter: int = 50  # 迭代次数
    cw_lr: float = 0.005  # 学习率
    
    # FC-JSMA参数
    jsma_eps: float = 8/255  # 扰动幅度
    jsma_gamma: float = 0.05  # 特征选择比例
    
    # 攻击类型
    attack_type: str = 'fc_pgd'  # 默认攻击类型
    targeted: bool = True  # 是否为定向攻击
    
    # 批处理
    batch_size: int = 32  # 批大小


@dataclass
class RFConfig:
    """射频信号配置"""
    # 载波参数
    carrier_freq: float = 433e6  # 载波频率 (Hz)
    carrier_freq_range: Tuple[float, float] = (430e6, 440e6)  # 搜索范围
    
    # 调制参数
    modulation_type: str = 'AM'  # 调制类型
    modulation_index: float = 0.5  # 调制指数
    
    # SDR参数
    sdr_type: str = 'hackrf'  # SDR类型
    tx_gain: int = 20  # 发射增益 (dB)
    sample_rate: float = 2e6  # SDR采样率 (Hz)
    
    # 天线参数
    antenna_type: str = 'log_periodic'  # 天线类型
    antenna_gain: float = 7.0  # 天线增益 (dBi)
    
    # GNU Radio参数
    gnuradio_enabled: bool = True  # 是否使用GNU Radio
    output_power: float = 10  # 输出功率 (dBm)


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 随机种子
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])
    
    # 数据相关
    data_path: str = './MyoArmbandDataset'  # 数据路径
    preprocessed_data_file: str = 'preprocessed_data_myo_armband.npz'  # 预处理数据文件
    
    # 模型相关
    model_path: str = 'emgnet_model.pth'  # 模型文件路径
    device: str = 'cuda'  # 计算设备: 'cuda' 或 'cpu'
    
    # 实验参数
    num_subjects: int = 10  # 被试数量
    test_size: float = 0.2  # 测试集比例
    cross_subject: bool = True  # 是否跨被试验证
    
    # 输出路径
    output_dir: str = './results'  # 结果保存目录
    save_adversarial: bool = True  # 是否保存对抗样本
    save_visualizations: bool = True  # 是否保存可视化结果
    
    # 日志
    log_level: str = 'INFO'  # 日志级别
    log_file: str = 'era_attack.log'  # 日志文件


@dataclass
class PhysicalConfig:
    """物理攻击参数"""
    # 攻击距离
    attack_distance: float = 1.0  # 攻击距离 (米)
    
    # 环境参数
    background_noise: float = -60  # 背景噪声水平 (dBm)
    temperature: float = 25  # 环境温度 (°C)
    humidity: float = 50  # 相对湿度 (%)
    
    # 受害设备参数
    victim_device: str = 'Myo Armband'  # 受害设备名称
    armband_circumference: float = 0.26  # 臂环周长 (米)
    
    # 物理约束
    max_power: float = 20  # 最大发射功率 (dBm)
    min_power: float = -10  # 最小发射功率 (dBm)
    
    # 时间同步
    sync_required: bool = False  # 是否需要时间同步
    sync_tolerance: float = 0.001  # 同步容差 (秒)


class Config:
    """主配置类"""
    
    def __init__(self):
        self.emg = EMGConfig()
        self.attack = AttackConfig()
        self.rf = RFConfig()
        self.experiment = ExperimentConfig()
        self.physical = PhysicalConfig()
        
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        os.makedirs(self.experiment.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment.output_dir, 'adversarial_samples'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment.output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment.output_dir, 'logs'), exist_ok=True)
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if '.' in key:
                # 处理嵌套属性，如 'attack.pgd_eps'
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self, key, value)
    
    def to_dict(self):
        """转换为字典"""
        return {
            'emg': self.emg.__dict__,
            'attack': self.attack.__dict__,
            'rf': self.rf.__dict__,
            'experiment': self.experiment.__dict__,
            'physical': self.physical.__dict__
        }
    
    def save(self, file_path):
        """保存配置到文件"""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    def load(self, file_path):
        """从文件加载配置"""
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for section_name, section_data in data.items():
            section = getattr(self, section_name)
            for key, value in section_data.items():
                setattr(section, key, value)
    
    def validate(self):
        """验证配置参数的合理性"""
        # 验证频率范围
        assert self.attack.freq_range[0] < self.attack.freq_range[1], \
            "Invalid frequency range"
        assert self.attack.freq_range[1] <= self.emg.sample_rate / 2, \
            "Frequency range exceeds Nyquist frequency"
        
        # 验证功率范围
        assert self.physical.min_power < self.physical.max_power, \
            "Invalid power range"
        assert self.rf.tx_gain <= 30, \
            "TX gain too high (max 30 dB for HackRF)"
        
        # 验证文件路径
        if not os.path.exists(self.experiment.data_path):
            print(f"Warning: Data path {self.experiment.data_path} does not exist")
        
        return True
    
    def __str__(self):
        """字符串表示"""
        import json
        return json.dumps(self.to_dict(), indent=2)


# 创建默认配置实例
default_config = Config()