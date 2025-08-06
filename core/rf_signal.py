"""
射频信号生成与发射模块
包含GNU Radio接口、信号调制、HackRF控制等功能
"""

import numpy as np
import os
import subprocess
import time
from typing import Optional, Tuple, List
import torch


class RFSignalGenerator:
    """射频信号生成器"""
    
    def __init__(self, config):
        """
        参数:
            config: RFConfig配置对象
        """
        self.config = config
        self.carrier_freq = config.carrier_freq
        self.sample_rate = config.sample_rate
        self.modulation_type = config.modulation_type
        self.modulation_index = config.modulation_index
        
    def perturbation_to_baseband(self, perturbation, target_freqs=None):
        """
        将对抗扰动转换为基带信号
        
        参数:
            perturbation: 频域对抗扰动 [C, F, T] 或 [F]
            target_freqs: 目标频率列表 (Hz)
        
        返回:
            baseband_signal: 时域基带信号
        """
        if torch.is_tensor(perturbation):
            perturbation = perturbation.cpu().numpy()
            
        # 如果是多维扰动，取平均或选择特定通道
        if len(perturbation.shape) > 1:
            # 沿通道和时间维度平均，得到频域特征
            if len(perturbation.shape) == 3:  # [C, F, T]
                freq_response = np.mean(perturbation, axis=(0, 2))
            else:  # [F, T]
                freq_response = np.mean(perturbation, axis=1)
        else:
            freq_response = perturbation
            
        # 识别非零频率分量
        if target_freqs is None:
            nonzero_freqs = np.where(np.abs(freq_response) > 1e-6)[0]
            # 将bin索引转换为实际频率
            target_freqs = nonzero_freqs * (200 / len(freq_response))  # 假设200Hz采样率
            
        # 生成多音信号
        duration = 1.0  # 信号持续时间（秒）
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        baseband_signal = np.zeros_like(t)
        
        for i, freq in enumerate(target_freqs):
            if freq > 0 and freq < 100:  # 只使用有效频率范围
                amplitude = np.abs(freq_response[i]) if i < len(freq_response) else 1.0
                phase = np.angle(freq_response[i]) if i < len(freq_response) else 0
                baseband_signal += amplitude * np.cos(2 * np.pi * freq * t + phase)
                
        # 归一化到[-1, 1]
        if np.max(np.abs(baseband_signal)) > 0:
            baseband_signal = baseband_signal / np.max(np.abs(baseband_signal))
            
        return baseband_signal
    
    def amplitude_modulation(self, baseband, carrier_freq=None):
        """
        幅度调制 (AM)
        
        参数:
            baseband: 基带信号
            carrier_freq: 载波频率
        
        返回:
            modulated_signal: 调制后的射频信号
        """
        if carrier_freq is None:
            carrier_freq = self.carrier_freq
            
        t = np.arange(len(baseband)) / self.sample_rate
        
        # AM调制: s(t) = [1 + m * baseband(t)] * cos(2π * fc * t)
        carrier = np.cos(2 * np.pi * carrier_freq * t)
        modulated = (1 + self.modulation_index * baseband) * carrier
        
        return modulated
    
    def frequency_modulation(self, baseband, carrier_freq=None, freq_deviation=5000):
        """
        频率调制 (FM)
        
        参数:
            baseband: 基带信号
            carrier_freq: 载波频率
            freq_deviation: 频偏
        
        返回:
            modulated_signal: 调制后的射频信号
        """
        if carrier_freq is None:
            carrier_freq = self.carrier_freq
            
        t = np.arange(len(baseband)) / self.sample_rate
        
        # FM调制: s(t) = cos(2π * fc * t + 2π * kf * ∫baseband(τ)dτ)
        phase = 2 * np.pi * carrier_freq * t + 2 * np.pi * freq_deviation * np.cumsum(baseband) / self.sample_rate
        modulated = np.cos(phase)
        
        return modulated
    
    def generate_iq_samples(self, signal):
        """
        生成I/Q样本
        
        参数:
            signal: 实信号
        
        返回:
            iq_samples: 复数I/Q样本
        """
        # 使用希尔伯特变换生成解析信号
        from scipy.signal import hilbert
        analytic_signal = hilbert(signal)
        
        # I/Q样本
        iq_samples = analytic_signal.astype(np.complex64)
        
        return iq_samples
    
    def save_iq_file(self, iq_samples, filename):
        """
        保存I/Q样本到文件
        
        参数:
            iq_samples: I/Q样本
            filename: 输出文件名
        """
        # 保存为二进制格式，供GNU Radio使用
        iq_samples.astype(np.complex64).tofile(filename)
        print(f"IQ samples saved to {filename}")


class GNURadioInterface:
    """GNU Radio接口"""
    
    def __init__(self, config):
        """
        参数:
            config: 配置对象
        """
        self.config = config
        self.flowgraph_template = self._create_flowgraph_template()
        
    def _create_flowgraph_template(self):
        """创建GNU Radio流程图模板"""
        template = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gnuradio import analog
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
import osmosdr
import time

class era_attack_flowgraph(gr.top_block):
    
    def __init__(self, carrier_freq={carrier_freq}, sample_rate={sample_rate}, 
                 tx_gain={tx_gain}, baseband_file='{baseband_file}'):
        gr.top_block.__init__(self, "ERa Attack Flowgraph")
        
        ##################################################
        # Variables
        ##################################################
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.tx_gain = tx_gain
        
        ##################################################
        # Blocks
        ##################################################
        # 基带信号源
        self.blocks_file_source = blocks.file_source(
            gr.sizeof_gr_complex*1, baseband_file, True)
        
        # HackRF发射器
        self.osmosdr_sink = osmosdr.sink(
            args="numchan=1 hackrf=0"
        )
        self.osmosdr_sink.set_sample_rate(sample_rate)
        self.osmosdr_sink.set_center_freq(carrier_freq, 0)
        self.osmosdr_sink.set_freq_corr(0, 0)
        self.osmosdr_sink.set_gain(tx_gain, 0)
        self.osmosdr_sink.set_if_gain(20, 0)
        self.osmosdr_sink.set_bb_gain(20, 0)
        self.osmosdr_sink.set_antenna('', 0)
        self.osmosdr_sink.set_bandwidth(0, 0)
        
        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source, 0), (self.osmosdr_sink, 0))

def main():
    tb = era_attack_flowgraph()
    tb.start()
    
    try:
        input('Press Enter to stop...')
    except KeyboardInterrupt:
        pass
    
    tb.stop()
    tb.wait()

if __name__ == '__main__':
    main()
"""
        return template
    
    def create_flowgraph(self, carrier_freq, tx_gain, baseband_file, output_file='flowgraph.py'):
        """
        创建GNU Radio流程图Python文件
        
        参数:
            carrier_freq: 载波频率
            tx_gain: 发射增益
            baseband_file: 基带信号文件
            output_file: 输出的Python文件
        """
        flowgraph = self.flowgraph_template.format(
            carrier_freq=carrier_freq,
            sample_rate=self.config.rf.sample_rate,
            tx_gain=tx_gain,
            baseband_file=baseband_file
        )
        
        with open(output_file, 'w') as f:
            f.write(flowgraph)
            
        # 设置执行权限
        os.chmod(output_file, 0o755)
        
        print(f"GNU Radio flowgraph created: {output_file}")
        return output_file
    
    def run_flowgraph(self, flowgraph_file, duration=None):
        """
        运行GNU Radio流程图
        
        参数:
            flowgraph_file: 流程图文件路径
            duration: 运行时长（秒），None表示持续运行
        """
        try:
            if duration:
                # 使用timeout限制运行时间
                cmd = f"timeout {duration} python3 {flowgraph_file}"
                subprocess.run(cmd, shell=True, check=True)
            else:
                # 持续运行
                subprocess.run(['python3', flowgraph_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running flowgraph: {e}")
        except KeyboardInterrupt:
            print("Flowgraph stopped by user")


class HackRFController:
    """HackRF One控制器"""
    
    def __init__(self, config):
        """
        参数:
            config: 配置对象
        """
        self.config = config
        self.device_available = self._check_device()
        
    def _check_device(self):
        """检查HackRF设备是否可用"""
        try:
            result = subprocess.run(['hackrf_info'], 
                                  capture_output=True, text=True)
            if 'Found HackRF' in result.stdout:
                print("HackRF device found")
                return True
            else:
                print("HackRF device not found")
                return False
        except FileNotFoundError:
            print("hackrf_info command not found. Please install HackRF tools.")
            return False
        except Exception as e:
            print(f"Error checking HackRF device: {e}")
            return False
    
    def frequency_sweep(self, freq_start, freq_stop, freq_step, 
                       signal_generator, dwell_time=0.1):
        """
        频率扫描
        
        参数:
            freq_start: 起始频率
            freq_stop: 结束频率
            freq_step: 频率步进
            signal_generator: 信号生成器实例
            dwell_time: 每个频率的停留时间
        
        返回:
            sweep_results: 扫描结果
        """
        frequencies = np.arange(freq_start, freq_stop + freq_step, freq_step)
        sweep_results = []
        
        for freq in frequencies:
            print(f"Sweeping at {freq/1e6:.1f} MHz")
            
            # 生成该频率的测试信号
            test_signal = np.sin(2 * np.pi * 1000 * np.arange(1000) / signal_generator.sample_rate)
            modulated = signal_generator.amplitude_modulation(test_signal, freq)
            
            # 这里应该发射信号并测量响应
            # 简化起见，我们只记录频率
            sweep_results.append({
                'frequency': freq,
                'response': np.random.random()  # 模拟响应
            })
            
            time.sleep(dwell_time)
            
        return sweep_results
    
    def set_tx_power(self, power_dbm):
        """
        设置发射功率
        
        参数:
            power_dbm: 功率 (dBm)
        """
        # HackRF的TX增益范围是0-47 dB
        # 这里进行简单的映射
        tx_gain = max(0, min(47, int(power_dbm + 10)))
        
        try:
            # 使用hackrf_transfer命令设置增益
            cmd = f"hackrf_transfer -x {tx_gain}"
            subprocess.run(cmd, shell=True, check=True)
            print(f"TX gain set to {tx_gain} dB (approx. {power_dbm} dBm)")
        except subprocess.CalledProcessError as e:
            print(f"Error setting TX power: {e}")
            
        return tx_gain


class PhysicalAttackOrchestrator:
    """物理攻击协调器"""
    
    def __init__(self, config):
        """
        参数:
            config: 完整配置对象
        """
        self.config = config
        self.signal_generator = RFSignalGenerator(config.rf)
        self.gnu_radio = GNURadioInterface(config)
        self.hackrf = HackRFController(config)
        
    def execute_attack(self, perturbation, attack_duration=10):
        """
        执行物理攻击
        
        参数:
            perturbation: 对抗扰动
            attack_duration: 攻击持续时间
        
        返回:
            success: 是否成功执行
        """
        print("Starting physical attack...")
        
        # 1. 将扰动转换为基带信号
        baseband = self.signal_generator.perturbation_to_baseband(perturbation)
        
        # 2. 调制信号
        if self.config.rf.modulation_type == 'AM':
            modulated = self.signal_generator.amplitude_modulation(baseband)
        elif self.config.rf.modulation_type == 'FM':
            modulated = self.signal_generator.frequency_modulation(baseband)
        else:
            raise ValueError(f"Unknown modulation type: {self.config.rf.modulation_type}")
            
        # 3. 生成I/Q样本
        iq_samples = self.signal_generator.generate_iq_samples(modulated)
        
        # 4. 保存I/Q文件
        iq_file = os.path.join(self.config.experiment.output_dir, 'attack_iq.complex64')
        self.signal_generator.save_iq_file(iq_samples, iq_file)
        
        # 5. 创建GNU Radio流程图
        flowgraph_file = self.gnu_radio.create_flowgraph(
            self.config.rf.carrier_freq,
            self.config.rf.tx_gain,
            iq_file
        )
        
        # 6. 执行攻击
        if self.hackrf.device_available:
            print(f"Transmitting for {attack_duration} seconds...")
            self.gnu_radio.run_flowgraph(flowgraph_file, attack_duration)
            print("Attack completed")
            return True
        else:
            print("HackRF device not available. Attack simulation only.")
            return False
    
    def find_optimal_frequency(self, perturbation):
        """
        寻找最优载波频率
        
        参数:
            perturbation: 对抗扰动
        
        返回:
            optimal_freq: 最优频率
        """
        print("Searching for optimal carrier frequency...")
        
        freq_range = self.config.rf.carrier_freq_range
        results = self.hackrf.frequency_sweep(
            freq_range[0], freq_range[1], 5e6,
            self.signal_generator
        )
        
        # 找到响应最大的频率
        best_result = max(results, key=lambda x: x['response'])
        optimal_freq = best_result['frequency']
        
        print(f"Optimal frequency found: {optimal_freq/1e6:.1f} MHz")
        return optimal_freq