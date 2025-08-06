"""
可视化工具模块
用于结果分析、图表生成和实验数据可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import torch
from typing import Optional, List, Tuple
import os


class AttackVisualizer:
    """攻击结果可视化器"""
    
    def __init__(self, save_dir='./results/visualizations'):
        """
        参数:
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_perturbation_analysis(self, original, perturbation, adversarial, 
                                  save_name='perturbation_analysis.png'):
        """
        绘制扰动分析图
        
        参数:
            original: 原始信号
            perturbation: 扰动
            adversarial: 对抗样本
            save_name: 保存文件名
        """
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 3, figure=fig)
        
        # 转换为numpy数组
        if torch.is_tensor(original):
            original = original.cpu().numpy()
        if torch.is_tensor(perturbation):
            perturbation = perturbation.cpu().numpy()
        if torch.is_tensor(adversarial):
            adversarial = adversarial.cpu().numpy()
        
        # 1. 原始信号时频图
        ax1 = fig.add_subplot(gs[0, :])
        if len(original.shape) == 3:  # [C, F, T]
            im1 = ax1.imshow(np.mean(original, axis=0), aspect='auto', cmap='viridis')
        else:
            im1 = ax1.imshow(original, aspect='auto', cmap='viridis')
        ax1.set_title('原始EMG信号时频图', fontsize=12, fontweight='bold')
        ax1.set_xlabel('时间 (样本)')
        ax1.set_ylabel('频率 (Hz)')
        plt.colorbar(im1, ax=ax1)
        
        # 2. 扰动时频图
        ax2 = fig.add_subplot(gs[1, :])
        if len(perturbation.shape) == 3:
            im2 = ax2.imshow(np.mean(perturbation, axis=0), aspect='auto', cmap='coolwarm')
        else:
            im2 = ax2.imshow(perturbation, aspect='auto', cmap='coolwarm')
        ax2.set_title('对抗扰动时频图', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间 (样本)')
        ax2.set_ylabel('频率 (Hz)')
        plt.colorbar(im2, ax=ax2)
        
        # 3. 对抗样本时频图
        ax3 = fig.add_subplot(gs[2, :])
        if len(adversarial.shape) == 3:
            im3 = ax3.imshow(np.mean(adversarial, axis=0), aspect='auto', cmap='viridis')
        else:
            im3 = ax3.imshow(adversarial, aspect='auto', cmap='viridis')
        ax3.set_title('对抗样本时频图', fontsize=12, fontweight='bold')
        ax3.set_xlabel('时间 (样本)')
        ax3.set_ylabel('频率 (Hz)')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Perturbation analysis saved to {save_path}")
    
    def plot_frequency_spectrum(self, signals, labels, frequencies=None,
                               save_name='frequency_spectrum.png'):
        """
        绘制频谱图
        
        参数:
            signals: 信号列表
            labels: 标签列表
            frequencies: 频率轴
            save_name: 保存文件名
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        for signal, label in zip(signals, labels):
            if torch.is_tensor(signal):
                signal = signal.cpu().numpy()
            
            # 计算频谱
            if len(signal.shape) > 1:
                signal = signal.flatten()
            
            fft = np.fft.fft(signal)
            freq = np.fft.fftfreq(len(signal), 1/200)  # 假设200Hz采样率
            
            # 幅度谱
            magnitude = np.abs(fft)[:len(fft)//2]
            freq_positive = freq[:len(freq)//2]
            
            ax1.plot(freq_positive, magnitude, label=label, linewidth=2)
            
            # 相位谱
            phase = np.angle(fft)[:len(fft)//2]
            ax2.plot(freq_positive, phase, label=label, linewidth=2)
        
        ax1.set_xlabel('频率 (Hz)')
        ax1.set_ylabel('幅度')
        ax1.set_title('频率幅度谱', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 100])
        
        ax2.set_xlabel('频率 (Hz)')
        ax2.set_ylabel('相位 (rad)')
        ax2.set_title('频率相位谱', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 100])
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Frequency spectrum saved to {save_path}")
    
    def plot_attack_success_rate(self, attack_types, success_rates, 
                                conditions=None, save_name='attack_success_rate.png'):
        """
        绘制攻击成功率对比图
        
        参数:
            attack_types: 攻击类型列表
            success_rates: 成功率数据
            conditions: 实验条件
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(attack_types))
        width = 0.35
        
        if conditions is None:
            bars = ax.bar(x, success_rates, width, label='成功率')
        else:
            # 多条件对比
            n_conditions = len(conditions)
            width = 0.8 / n_conditions
            for i, (condition, rates) in enumerate(zip(conditions, success_rates)):
                offset = (i - n_conditions/2) * width + width/2
                ax.bar(x + offset, rates, width, label=condition)
        
        ax.set_xlabel('攻击方法', fontsize=12)
        ax.set_ylabel('成功率 (%)', fontsize=12)
        ax.set_title('不同攻击方法成功率对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(attack_types)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Attack success rate plot saved to {save_path}")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None,
                            save_name='confusion_matrix.png'):
        """
        绘制混淆矩阵
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            save_name: 保存文件名
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   square=True, ax=ax)
        
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        ax.set_title('攻击后的混淆矩阵', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_power_vs_success(self, power_levels, success_rates,
                            save_name='power_vs_success.png'):
        """
        绘制功率与成功率关系图
        
        参数:
            power_levels: 功率水平 (dBm)
            success_rates: 对应的成功率
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(power_levels, success_rates, 'o-', linewidth=2, markersize=8)
        ax.fill_between(power_levels, success_rates, alpha=0.3)
        
        ax.set_xlabel('发射功率 (dBm)', fontsize=12)
        ax.set_ylabel('攻击成功率 (%)', fontsize=12)
        ax.set_title('发射功率与攻击成功率关系', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标记关键点
        threshold_idx = np.where(np.array(success_rates) > 50)[0]
        if len(threshold_idx) > 0:
            threshold_power = power_levels[threshold_idx[0]]
            ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label=f'50%成功率')
            ax.axvline(x=threshold_power, color='r', linestyle='--', alpha=0.5)
            ax.annotate(f'{threshold_power} dBm', 
                       xy=(threshold_power, 50),
                       xytext=(threshold_power + 2, 40),
                       arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.legend()
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Power vs success plot saved to {save_path}")
    
    def plot_frequency_sweep_results(self, frequencies, responses,
                                    save_name='frequency_sweep.png'):
        """
        绘制频率扫描结果
        
        参数:
            frequencies: 频率列表 (Hz)
            responses: 响应值
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 转换为MHz
        freq_mhz = np.array(frequencies) / 1e6
        
        ax.plot(freq_mhz, responses, 'b-', linewidth=2)
        ax.fill_between(freq_mhz, responses, alpha=0.3)
        
        # 标记峰值
        peak_idx = np.argmax(responses)
        peak_freq = freq_mhz[peak_idx]
        peak_response = responses[peak_idx]
        
        ax.plot(peak_freq, peak_response, 'ro', markersize=10)
        ax.annotate(f'峰值: {peak_freq:.1f} MHz', 
                   xy=(peak_freq, peak_response),
                   xytext=(peak_freq + 10, peak_response * 0.9),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.set_xlabel('频率 (MHz)', fontsize=12)
        ax.set_ylabel('响应强度 (归一化)', fontsize=12)
        ax.set_title('Myo臂环频率响应扫描结果', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标记430-440 MHz范围
        ax.axvspan(430, 440, alpha=0.2, color='green', label='最优频段')
        ax.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Frequency sweep results saved to {save_path}")
    
    def plot_temporal_pattern(self, signals, timestamps=None, 
                            save_name='temporal_pattern.png'):
        """
        绘制时域模式图
        
        参数:
            signals: 信号字典 {label: signal}
            timestamps: 时间戳
            save_name: 保存文件名
        """
        n_signals = len(signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=(12, 3*n_signals), sharex=True)
        
        if n_signals == 1:
            axes = [axes]
        
        for idx, (label, signal) in enumerate(signals.items()):
            if torch.is_tensor(signal):
                signal = signal.cpu().numpy()
            
            if timestamps is None:
                timestamps = np.arange(len(signal.flatten()))
            
            axes[idx].plot(timestamps, signal.flatten(), linewidth=1)
            axes[idx].set_ylabel('幅度', fontsize=10)
            axes[idx].set_title(label, fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('时间 (ms)', fontsize=12)
        
        plt.suptitle('EMG信号时域模式', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal pattern saved to {save_path}")


class ExperimentAnalyzer:
    """实验结果分析器"""
    
    def __init__(self, results_dir='./results'):
        """
        参数:
            results_dir: 结果目录
        """
        self.results_dir = results_dir
        self.visualizer = AttackVisualizer(os.path.join(results_dir, 'visualizations'))
    
    def analyze_attack_performance(self, results):
        """
        分析攻击性能
        
        参数:
            results: 实验结果字典
        
        返回:
            analysis: 分析报告
        """
        analysis = {}
        
        # 计算整体成功率
        total_attacks = results.get('total_attacks', 0)
        successful_attacks = results.get('successful_attacks', 0)
        
        if total_attacks > 0:
            analysis['overall_success_rate'] = (successful_attacks / total_attacks) * 100
        else:
            analysis['overall_success_rate'] = 0
        
        # 按攻击类型分析
        attack_types = results.get('attack_types', {})
        type_analysis = {}
        
        for attack_type, type_results in attack_types.items():
            type_total = type_results.get('total', 0)
            type_success = type_results.get('success', 0)
            
            if type_total > 0:
                type_analysis[attack_type] = {
                    'success_rate': (type_success / type_total) * 100,
                    'avg_perturbation_norm': type_results.get('avg_perturbation_norm', 0),
                    'avg_time': type_results.get('avg_time', 0)
                }
        
        analysis['by_attack_type'] = type_analysis
        
        # 按目标类别分析
        class_results = results.get('class_results', {})
        class_analysis = {}
        
        for class_id, class_data in class_results.items():
            class_total = class_data.get('total', 0)
            class_success = class_data.get('success', 0)
            
            if class_total > 0:
                class_analysis[class_id] = (class_success / class_total) * 100
        
        analysis['by_target_class'] = class_analysis
        
        # 物理参数分析
        physical_params = results.get('physical_params', {})
        analysis['optimal_carrier_freq'] = physical_params.get('optimal_freq', 433e6)
        analysis['optimal_tx_power'] = physical_params.get('optimal_power', 10)
        
        return analysis
    
    def generate_report(self, analysis, save_path='experiment_report.txt'):
        """
        生成实验报告
        
        参数:
            analysis: 分析结果
            save_path: 保存路径
        """
        report_path = os.path.join(self.results_dir, save_path)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ERa Attack 实验报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 整体性能
            f.write("1. 整体攻击性能\n")
            f.write("-" * 40 + "\n")
            f.write(f"总体成功率: {analysis['overall_success_rate']:.2f}%\n\n")
            
            # 按攻击类型
            f.write("2. 按攻击类型分析\n")
            f.write("-" * 40 + "\n")
            for attack_type, metrics in analysis['by_attack_type'].items():
                f.write(f"\n{attack_type.upper()}:\n")
                f.write(f"  - 成功率: {metrics['success_rate']:.2f}%\n")
                f.write(f"  - 平均扰动范数: {metrics['avg_perturbation_norm']:.4f}\n")
                f.write(f"  - 平均耗时: {metrics['avg_time']:.3f}秒\n")
            
            # 按目标类别
            f.write("\n3. 按目标类别分析\n")
            f.write("-" * 40 + "\n")
            for class_id, success_rate in analysis['by_target_class'].items():
                f.write(f"类别 {class_id}: {success_rate:.2f}%\n")
            
            # 物理参数
            f.write("\n4. 最优物理参数\n")
            f.write("-" * 40 + "\n")
            f.write(f"最优载波频率: {analysis['optimal_carrier_freq']/1e6:.1f} MHz\n")
            f.write(f"最优发射功率: {analysis['optimal_tx_power']} dBm\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"Experiment report saved to {report_path}")
        return report_path