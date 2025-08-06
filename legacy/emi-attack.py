import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from modelo import EMGNet
import os


"""
EMI对抗样本攻击实现
目标：模拟物理EMI攻击，针对EMGNet生成所有通道具有相同扰动的对抗样本
特点：8个通道的时域与频域扰动保持一致，更符合物理EMI攻击的实际情况
"""

# 定义超参数
epsilon = 0.1           # FGSM扰动强度
device = torch.device("cuda" )

def emi_fgsm_attack(model, X, y, epsilon):
    """
    实现模拟EMI攻击的FGSM对抗样本生成算法
    
    参数:
    - model: 待攻击的EMGNet模型
    - X: 输入数据，形状为(batch_size, 8, height, width)
    - y: 真实标签
    - epsilon: 扰动强度
    
    返回:
    - X_adv: 生成的对抗样本
    """
    # 确保模型在评估模式
    model.eval()
    
    # 将输入设置为可求梯度
    X_adv = X.clone().detach().to(device)
    X_adv.requires_grad = True
    
    y = y.to(device)
    
    # 前向传播
    outputs = model(X_adv)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, y)
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 获取梯度
    grad = X_adv.grad.data  # 形状为(batch_size, 8, height, width)
    
    # 重要：计算所有通道的平均梯度，保证所有通道扰动一致
    # 在通道维度上求平均，得到形状为(batch_size, 1, height, width)的平均梯度
    avg_grad = torch.mean(grad, dim=1, keepdim=True)
    
    # 将平均梯度扩展到所有通道，确保所有通道扰动一致
    # 复制到所有8个通道，形状变回(batch_size, 8, height, width)
    channel_consistent_grad = avg_grad.repeat(1, 8, 1, 1)
    
    # 生成对抗样本
    # 使用sign函数获取梯度方向，乘以epsilon作为扰动大小
    X_adv = X_adv.detach() + epsilon * torch.sign(channel_consistent_grad)
    
    # 确保扰动后的样本仍在有效范围内（假设数据范围为[0,1]或已标准化）
    X_adv = torch.clamp(X_adv, X.min().item(), X.max().item())
    
    return X_adv

def untargeted_emi_attack(model, X, y, epsilon, steps=10, alpha=0.01):
    """
    实现迭代版的EMI对抗样本攻击（PGD变种）
    
    参数:
    - model: 待攻击的EMGNet模型
    - X: 输入数据，形状为(batch_size, 8, height, width)
    - y: 真实标签
    - epsilon: 总扰动强度上限
    - steps: 迭代步数
    - alpha: 每步扰动大小
    
    返回:
    - X_adv: 生成的对抗样本
    """
    # 确保模型在评估模式
    model.eval()
    
    # 将输入移动到正确的设备上
    X = X.clone().detach().to(device)
    
    # 将输入拷贝并初始化
    X_adv = X.clone().detach()
    
    # 添加小的随机初始扰动
    X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon/2, epsilon/2)
    X_adv = torch.clamp(X_adv, X.min().item(), X.max().item())
    
    y = y.to(device)
    
    for i in range(steps):
        X_adv.requires_grad = True
        
        # 前向传播
        outputs = model(X_adv)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, y)
        
        # 反向传播
        model.zero_grad()
        loss.backward()
        
        # 获取梯度
        grad = X_adv.grad.data
        
        # 计算所有通道的平均梯度，保证所有通道扰动一致
        avg_grad = torch.mean(grad, dim=1, keepdim=True)
        channel_consistent_grad = avg_grad.repeat(1, 8, 1, 1)
        
        # 更新对抗样本
        X_adv = X_adv.detach() + alpha * torch.sign(channel_consistent_grad)  # 正号表示梯度上升，使损失增大
        
        # 投影回epsilon球内
        delta = torch.clamp(X_adv - X, -epsilon, epsilon)
        X_adv = X + delta
        
        # 确保样本在有效范围内
        X_adv = torch.clamp(X_adv, X.min().item(), X.max().item())
    
    return X_adv

def visualize_perturbation(original, adversarial, idx=0):
    """
    可视化对抗扰动
    
    参数:
    - original: 原始样本
    - adversarial: 对抗样本
    - idx: 要可视化的样本索引
    """
    # 计算扰动
    perturbation = adversarial[idx] - original[idx]
    
    # 绘制原始样本、对抗样本和扰动的前两个通道
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # 原始样本的两个通道
    axes[0, 0].imshow(original[idx, 0].cpu().numpy())
    axes[0, 0].set_title("原始样本 - 通道1")
    axes[0, 1].imshow(original[idx, 1].cpu().numpy())
    axes[0, 1].set_title("原始样本 - 通道2")
    
    # 对抗样本的两个通道
    axes[1, 0].imshow(adversarial[idx, 0].cpu().numpy())
    axes[1, 0].set_title("对抗样本 - 通道1")
    axes[1, 1].imshow(adversarial[idx, 1].cpu().numpy())
    axes[1, 1].set_title("对抗样本 - 通道2")
    
    # 扰动的两个通道
    axes[2, 0].imshow(perturbation[0].cpu().numpy())
    axes[2, 0].set_title("扰动 - 通道1")
    axes[2, 1].imshow(perturbation[1].cpu().numpy())
    axes[2, 1].set_title("扰动 - 通道2")
    
    plt.tight_layout()
    plt.savefig('emi_perturbation.png')
    plt.close()
    
    # 验证所有通道扰动是否一致
    is_consistent = True
    for i in range(1, 8):
        if not torch.allclose(perturbation[0], perturbation[i], atol=1e-6):
            is_consistent = False
            break
    print(f"所有通道扰动是否一致: {is_consistent}")

def main():
    data = np.load('preprocessed_data_myo_armband.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    
    # 归一化
    data_mean = X_test.mean()
    data_std = X_test.std()
    X_test = (X_test - data_mean) / (data_std + 1e-8)  # 加上小值避免除零
    
    # 将数据转换为Tensor
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # 加载训练好的EMGNet模型
    model = EMGNet(num_classes=7).to(device)
    model.load_state_dict(torch.load('emgnet_model-15.pth', map_location=device))
    model.eval()
    
    # 设置批量大小
    batch_size = 4096  # 根据您的GPU内存调整这个值
    
    # 计算原始数据准确率（分批处理）
    num_samples = len(X_test)
    num_batches = (num_samples + batch_size - 1) // batch_size
    correct = 0
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_X = X_test[start_idx:end_idx].to(device)
            batch_y = y_test[start_idx:end_idx]
            
            batch_outputs = model(batch_X)
            batch_preds = torch.argmax(batch_outputs, dim=1)
            correct += (batch_preds.cpu() == batch_y).sum().item()
    
    orig_acc = correct / num_samples
    print(f"原始数据准确率: {orig_acc*100:.2f}%")
    
    # 测试不同epsilon值
    epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fgsm_accs = []
    pgd_accs = []
    fgsm_success_rates = []
    pgd_success_rates = []
    
    for eps in epsilon_values:
        print(f"\n测试 ε = {eps} 的情况:")
        
        # 分批生成EMI-FGSM对抗样本并评估
        print(f"生成EMI-FGSM对抗样本 (ε={eps})...")
        fgsm_correct = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_X = X_test[start_idx:end_idx]
            batch_y = y_test[start_idx:end_idx]
            
            # 生成当前批次的对抗样本
            batch_X_adv_fgsm = emi_fgsm_attack(model, batch_X, batch_y, eps)
            
            # 评估当前批次
            with torch.no_grad():
                batch_outputs = model(batch_X_adv_fgsm)
                batch_preds = torch.argmax(batch_outputs, dim=1)
                fgsm_correct += (batch_preds.cpu() == batch_y).sum().item()
            
            # 清除缓存，释放显存
            torch.cuda.empty_cache()
        
        fgsm_adv_acc = fgsm_correct / num_samples
        
        # 分批生成EMI-PGD对抗样本并评估
        print(f"生成EMI-PGD对抗样本 (ε={eps})...")
        pgd_correct = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_X = X_test[start_idx:end_idx]
            batch_y = y_test[start_idx:end_idx]
            
            # 生成当前批次的对抗样本
            batch_X_adv_pgd = untargeted_emi_attack(model, batch_X, batch_y, eps, steps=10, alpha=0.01)
            
            # 评估当前批次
            with torch.no_grad():
                batch_outputs = model(batch_X_adv_pgd)
                batch_preds = torch.argmax(batch_outputs, dim=1)
                pgd_correct += (batch_preds.cpu() == batch_y).sum().item()
            
            # 清除缓存，释放显存
            torch.cuda.empty_cache()
        
        pgd_adv_acc = pgd_correct / num_samples
        
        # 计算攻击成功率
        fgsm_attack_success_rate = 1 - fgsm_adv_acc / orig_acc
        pgd_attack_success_rate = 1 - pgd_adv_acc / orig_acc
        
        # 存储结果
        fgsm_accs.append(fgsm_adv_acc)
        pgd_accs.append(pgd_adv_acc)
        fgsm_success_rates.append(fgsm_attack_success_rate)
        pgd_success_rates.append(pgd_attack_success_rate)
        
        print(f"EMI-FGSM对抗样本准确率: {fgsm_adv_acc*100:.2f}% (ε={eps})")
        print(f"EMI-PGD对抗样本准确率: {pgd_adv_acc*100:.2f}% (ε={eps})")
        print(f"EMI-FGSM攻击成功率: {fgsm_attack_success_rate*100:.2f}%")
        print(f"EMI-PGD攻击成功率: {pgd_attack_success_rate*100:.2f}%")
    
    # 可视化不同epsilon值下的攻击成功率
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, [rate * 100 for rate in fgsm_success_rates], 'b-o', label='EMI-FGSM')
    plt.plot(epsilon_values, [rate * 100 for rate in pgd_success_rates], 'r-o', label='EMI-PGD')
    plt.xlabel('扰动强度 ε')
    plt.ylabel('攻击成功率 (%)')
    plt.title('不同扰动强度下的EMI攻击成功率')
    plt.legend()
    plt.grid(True)
    plt.savefig('emi_attack_success_rates.png')
    plt.close()
    
    # 打印汇总数据
    print("\n攻击成功率汇总:")
    print("扰动强度(ε) | EMI-FGSM成功率(%) | EMI-PGD成功率(%)")
    print("---------------------------------------------")
    for i, eps in enumerate(epsilon_values):
        print(f"{eps:.1f}        | {fgsm_success_rates[i]*100:.2f}%          | {pgd_success_rates[i]*100:.2f}%")
if __name__ == "__main__":
    main()
