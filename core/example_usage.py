"""
ERa Attack 使用示例
演示如何使用各个模块进行攻击
"""

import numpy as np
import torch
import torch.nn as nn
from config import Config
from fc_attacks import create_attack
from data_processing import EMGDataProcessor, AdversarialDataGenerator
from visualization import AttackVisualizer
import matplotlib.pyplot as plt


def example_1_basic_attack():
    """示例1: 基础数字域攻击"""
    print("=" * 60)
    print("示例1: 基础FC-PGD攻击")
    print("=" * 60)
    
    # 创建配置
    config = Config()
    
    # 创建简单的示例模型
    class SimpleEMGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, 7)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = SimpleEMGNet()
    model.eval()
    
    # 生成示例EMG数据 [batch, channels, freq, time]
    sample_data = torch.randn(1, 8, 52, 52)
    target_label = 3
    
    # 创建FC-PGD攻击
    attack = create_attack('fc_pgd', model, freq_range=(50, 100))
    
    # 生成对抗扰动
    print("生成对抗扰动...")
    perturbation = attack.generate(
        sample_data, 
        target_label,
        eps=8/255,
        alpha=2/255,
        num_iter=20
    )
    
    # 创建对抗样本
    adversarial_sample = sample_data + perturbation
    
    # 评估攻击效果
    with torch.no_grad():
        clean_pred = torch.argmax(model(sample_data), dim=1)
        adv_pred = torch.argmax(model(adversarial_sample), dim=1)
    
    print(f"原始预测: {clean_pred.item()}")
    print(f"对抗预测: {adv_pred.item()}")
    print(f"目标标签: {target_label}")
    print(f"攻击成功: {adv_pred.item() == target_label}")
    print(f"扰动L2范数: {torch.norm(perturbation).item():.4f}")
    print()


def example_2_compare_attacks():
    """示例2: 比较三种攻击算法"""
    print("=" * 60)
    print("示例2: 比较FC-PGD, FC-C&W, FC-JSMA")
    print("=" * 60)
    
    # 创建模型
    class SimpleEMGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, 7)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = SimpleEMGNet()
    model.eval()
    
    # 生成测试数据
    num_samples = 10
    test_data = torch.randn(num_samples, 8, 52, 52)
    
    results = {}
    
    # 测试每种攻击
    for attack_type in ['fc_pgd', 'fc_cw', 'fc_jsma']:
        print(f"\n测试 {attack_type.upper()} 攻击...")
        
        attack = create_attack(attack_type, model, freq_range=(50, 100))
        
        success_count = 0
        total_norm = 0
        
        for i in range(num_samples):
            sample = test_data[i:i+1]
            original_pred = torch.argmax(model(sample), dim=1).item()
            target = (original_pred + 1) % 7  # 选择不同的目标类别
            
            # 生成扰动
            if attack_type == 'fc_pgd':
                delta = attack.generate(sample, target, eps=8/255, alpha=2/255, num_iter=20)
            elif attack_type == 'fc_cw':
                delta = attack.generate(sample, target, c=1.0, kappa=0, num_iter=50)
            else:  # fc_jsma
                delta = attack.generate(sample, target, eps=8/255, gamma=0.05)
            
            # 评估
            adv_sample = sample + delta
            adv_pred = torch.argmax(model(adv_sample), dim=1).item()
            
            if adv_pred == target:
                success_count += 1
            
            total_norm += torch.norm(delta).item()
        
        # 记录结果
        results[attack_type] = {
            'success_rate': success_count / num_samples * 100,
            'avg_norm': total_norm / num_samples
        }
        
        print(f"  成功率: {results[attack_type]['success_rate']:.1f}%")
        print(f"  平均扰动范数: {results[attack_type]['avg_norm']:.4f}")
    
    # 可视化对比
    print("\n攻击算法对比总结:")
    print("-" * 40)
    for attack_type, metrics in results.items():
        print(f"{attack_type.upper():8} - 成功率: {metrics['success_rate']:5.1f}%, "
              f"平均范数: {metrics['avg_norm']:.4f}")
    print()


def example_3_frequency_analysis():
    """示例3: 频域分析"""
    print("=" * 60)
    print("示例3: 对抗扰动的频域分析")
    print("=" * 60)
    
    # 创建数据处理器
    processor = EMGDataProcessor(sample_rate=200)
    
    # 生成示例信号
    duration = 1.0  # 1秒
    t = np.linspace(0, duration, int(200 * duration))
    
    # 创建包含多个频率成分的EMG信号
    clean_signal = (
        0.5 * np.sin(2 * np.pi * 30 * t) +  # 30 Hz
        0.3 * np.sin(2 * np.pi * 60 * t) +  # 60 Hz
        0.2 * np.sin(2 * np.pi * 90 * t)    # 90 Hz
    )
    
    # 创建频域约束的扰动（只在50-100Hz范围）
    perturbation_freqs = [55, 75, 85]  # Hz
    perturbation = np.zeros_like(t)
    for freq in perturbation_freqs:
        perturbation += 0.1 * np.sin(2 * np.pi * freq * t)
    
    # 创建对抗样本
    adversarial_signal = clean_signal + perturbation
    
    # 计算频谱
    def compute_spectrum(signal, fs=200):
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/fs)
        magnitude = np.abs(fft)
        
        # 只取正频率部分
        pos_mask = freqs > 0
        return freqs[pos_mask], magnitude[pos_mask]
    
    # 分析三个信号的频谱
    freqs_clean, mag_clean = compute_spectrum(clean_signal)
    freqs_pert, mag_pert = compute_spectrum(perturbation)
    freqs_adv, mag_adv = compute_spectrum(adversarial_signal)
    
    # 打印关键频率成分
    print("原始信号主要频率成分:")
    for i in np.argsort(mag_clean)[-3:]:
        if freqs_clean[i] < 100:
            print(f"  {freqs_clean[i]:.1f} Hz: 幅度 {mag_clean[i]:.2f}")
    
    print("\n扰动信号频率成分 (50-100 Hz范围):")
    for i in np.argsort(mag_pert)[-3:]:
        if 50 <= freqs_pert[i] <= 100:
            print(f"  {freqs_pert[i]:.1f} Hz: 幅度 {mag_pert[i]:.2f}")
    
    print("\n对抗样本的频谱变化:")
    print("  原始能量: {:.2f}".format(np.sum(mag_clean)))
    print("  扰动能量: {:.2f}".format(np.sum(mag_pert)))
    print("  对抗样本能量: {:.2f}".format(np.sum(mag_adv)))
    print("  信噪比 (SNR): {:.2f} dB".format(
        20 * np.log10(np.sum(mag_clean) / np.sum(mag_pert))
    ))
    print()


def example_4_visualization():
    """示例4: 可视化攻击结果"""
    print("=" * 60)
    print("示例4: 生成可视化结果")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = AttackVisualizer(save_dir='./results/demo')
    
    # 生成示例数据
    original = np.random.randn(8, 52, 52)  # [channels, freq, time]
    perturbation = np.random.randn(8, 52, 52) * 0.1
    adversarial = original + perturbation
    
    # 绘制扰动分析图
    print("生成扰动分析图...")
    visualizer.plot_perturbation_analysis(
        original, perturbation, adversarial,
        save_name='demo_perturbation_analysis.png'
    )
    
    # 绘制频谱对比
    print("生成频谱对比图...")
    visualizer.plot_frequency_spectrum(
        [original.flatten()[:1000], 
         perturbation.flatten()[:1000],
         adversarial.flatten()[:1000]],
        ['原始信号', '扰动', '对抗样本'],
        save_name='demo_frequency_spectrum.png'
    )
    
    # 绘制攻击成功率对比
    print("生成攻击成功率对比图...")
    attack_types = ['FC-PGD', 'FC-C&W', 'FC-JSMA']
    success_rates = [85.2, 92.1, 78.5]  # 示例数据
    visualizer.plot_attack_success_rate(
        attack_types, success_rates,
        save_name='demo_attack_comparison.png'
    )
    
    # 绘制功率-成功率曲线
    print("生成功率-成功率关系图...")
    power_levels = np.arange(-10, 21, 2)  # dBm
    success_rates = 100 / (1 + np.exp(-(power_levels - 5) / 3))  # Sigmoid曲线
    visualizer.plot_power_vs_success(
        power_levels, success_rates,
        save_name='demo_power_vs_success.png'
    )
    
    print("可视化结果已保存到 ./results/demo/")
    print()


def example_5_physical_parameters():
    """示例5: 物理参数配置"""
    print("=" * 60)
    print("示例5: 射频攻击参数配置")
    print("=" * 60)
    
    from rf_signal import RFSignalGenerator
    
    # 创建RF配置
    class RFConfig:
        carrier_freq = 433e6
        sample_rate = 2e6
        modulation_type = 'AM'
        modulation_index = 0.5
    
    rf_config = RFConfig()
    signal_gen = RFSignalGenerator(rf_config)
    
    # 创建一个简单的频域扰动
    # 假设扰动在 55Hz, 75Hz, 95Hz
    freq_components = {
        55: 0.3,   # 频率: 幅度
        75: 0.5,
        95: 0.2
    }
    
    # 生成基带信号
    duration = 0.01  # 10ms
    t = np.linspace(0, duration, int(rf_config.sample_rate * duration))
    baseband = np.zeros_like(t)
    
    for freq, amplitude in freq_components.items():
        baseband += amplitude * np.sin(2 * np.pi * freq * t)
    
    # 幅度调制
    modulated = signal_gen.amplitude_modulation(baseband)
    
    print("射频信号参数:")
    print(f"  载波频率: {rf_config.carrier_freq/1e6:.1f} MHz")
    print(f"  采样率: {rf_config.sample_rate/1e6:.1f} MHz")
    print(f"  调制方式: {rf_config.modulation_type}")
    print(f"  调制指数: {rf_config.modulation_index}")
    
    print("\n基带信号特征:")
    print(f"  频率成分: {list(freq_components.keys())} Hz")
    print(f"  信号长度: {len(baseband)} 样本")
    print(f"  持续时间: {duration*1000:.1f} ms")
    
    print("\n调制信号特征:")
    print(f"  峰值功率: {20*np.log10(np.max(np.abs(modulated))):.1f} dB")
    print(f"  平均功率: {20*np.log10(np.mean(np.abs(modulated))):.1f} dB")
    
    # 生成I/Q样本
    iq_samples = signal_gen.generate_iq_samples(modulated)
    print(f"\nI/Q样本:")
    print(f"  样本数: {len(iq_samples)}")
    print(f"  数据类型: complex64")
    print(f"  文件大小: ~{len(iq_samples) * 8 / 1024:.1f} KB")
    print()


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("ERa Attack 使用示例")
    print("="*60 + "\n")
    
    # 运行各个示例
    example_1_basic_attack()
    example_2_compare_attacks()
    example_3_frequency_analysis()
    example_4_visualization()
    example_5_physical_parameters()
    
    print("="*60)
    print("所有示例运行完成!")
    print("="*60)


if __name__ == '__main__':
    main()