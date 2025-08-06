#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attack.py

本代码实现了针对EMGNet模型的FGSM对抗样本攻击，要求扰动仅作用在原始8通道EMG数据
的频域部分中某一指定频率行（对应连续小波变换结果中的行索引，范围0-14）。

【算法说明】
1. 为了使扰动只集中在某个频率行（例如第 target_idx 行），直接对梯度进行“掩膜”（masking）
   虽然能将非目标频率的梯度置零，但这样会忽略非目标频率中包含的有用信息，导致攻击效果减弱。
2. 本算法的核心思想是在构造损失函数时，通过“频率掩码”引导梯度主要来自目标频率行，
   同时允许非目标频率保留极少的梯度（通过设置一个较小的比例因子alpha），
   这样可以保留部分全局信息，进而对最终分类决策产生较大影响。
3. 具体来说：
   - 我们先计算模型对原始输入的分类损失（交叉熵损失），并对输入求梯度。
   - 接着构造一个与输入同形状的频率掩码：在目标频率行（索引target_idx）上赋值1，
     而在其他频率行上赋值一个较小的比例因子（例如0或0.1）。
   - 然后将梯度与该掩码逐元素相乘，相当于在反向传播时“内置”了对特定频率的依赖，
     最后用FGSM公式（扰动 = ε * sign(梯度)）生成扰动，并加到原始输入上。
4. 这样生成的对抗样本，其扰动主要集中在目标频率行上，对分类决策产生攻击效果。

【参数说明】
- ε（epsilon）：FGSM扰动强度，本例中设置为0.1。
- target_idx：目标频率行索引，取值范围0~14（默认设为7，可根据需求调整）。
- alpha：非目标频率行的梯度保留比例，默认0.0表示完全屏蔽，也可设为0.1等小值以保留少量梯度信息。

注意：本代码中为了方便演示，使用了随机生成的样本数据；在实际攻击实验中，
请将预处理后的数据加载进来，替换示例中的x与y。

"""

import torch
import torch.nn as nn
import numpy as np
from model import EMGNet  # 假设model.py和本文件在同一目录下
import argparse

def generate_frequency_mask(x, target_idx, alpha=0.0):
    """
    生成频率掩码，与输入x形状一致，要求在目标频率行上（target_idx）赋值为1，
    非目标频率行赋值为alpha（通常设为0或一个很小的正数）。
    
    参数:
        x: 输入张量，形状为 (batch_size, channels, freq, time)
        target_idx: 目标频率行索引（0-14）
        alpha: 非目标频率行的梯度比例因子
        
    返回:
        mask: 频率掩码张量
    """
    mask = torch.full_like(x, alpha)
    # 假设频率维度在第3个维度（shape: [batch, channels, freq, time]）
    mask[:, :, target_idx, :] = 1.0
    return mask

def fgsm_attack(model, x, y, epsilon, target_idx, alpha=0.0):
    """
    对输入样本x施加FGSM对抗攻击，扰动仅集中在目标频率行上（以及非目标频率保留极小梯度）。
    
    参数:
        model: 预训练的EMGNet模型
        x: 原始输入样本，形状 (batch_size, 8, 15, 25)
        y: 样本真实标签，形状 (batch_size,)
        epsilon: FGSM扰动强度
        target_idx: 目标频率行索引（0-14）
        alpha: 非目标频率行的梯度比例因子
        
    返回:
        x_adv: 生成的对抗样本
    """
    model.eval()  # 设置为评估模式
    
    # 复制输入，并开启梯度追踪
    x_adv = x.clone().detach().requires_grad_(True)
    
    # 正向传播计算输出和交叉熵损失
    outputs = model(x_adv)
    loss = nn.CrossEntropyLoss()(outputs, y)
    
    # 反向传播，计算输入x的梯度
    model.zero_grad()
    loss.backward()
    grad = x_adv.grad.data
    
    # 生成频率掩码，目标频率行保持全梯度，其他频率按alpha比例保留
    mask = generate_frequency_mask(x_adv, target_idx, alpha)
    
    # 将梯度按频率掩码加权
    weighted_grad = mask * grad
    
    # FGSM扰动公式：扰动 = epsilon * sign(加权梯度)
    perturbation = epsilon * weighted_grad.sign()
    
    # 生成对抗样本（可在此添加投影步骤确保扰动不超出预定义范围）
    x_adv = x + perturbation
    
    return x_adv

def main(target_idx=7):
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="针对EMGNet的FGSM对抗样本攻击")
    parser.add_argument("--target_idx", type=int, default=target_idx, help="目标频率行索引，范围0-14")
    parser.add_argument("--epsilon", type=float, default=0.1, help="FGSM扰动强度")
    parser.add_argument("--alpha", type=float, default=0.0, help="非目标频率行梯度比例因子")
    parser.add_argument("--use_cuda", action="store_true", help="是否使用GPU进行运算")
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    
    # 加载预训练的EMGNet模型
    num_classes = 7  # 根据实际分类类别数设置
    model = EMGNet(num_classes=num_classes).to(device)
    model_path = "emgnet_model.pth"  # 假设预训练模型保存在此文件中
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载预训练模型参数：{model_path}")
    except Exception as e:
        print(f"加载预训练模型失败：{e}")
        return
    

    batch_size = 4

    #加载预处理数据
    data = np.load('preprocessed_data_myo_armband.npz')
    X_test = data['X_test']
    y_test = data['y_test']

    #归一化
    data_mean = X_test.mean()
    data_std = X_test.std()
    X_test = (X_test - data_mean) / (data_std + 1e-8)  # 加上小值避免除零

    

    x = torch.from_numpy(X_test).to(device)
    y = torch.from_numpy(y_test).to(device)
    
    # 显示原始样本预测结果
    with torch.no_grad():
        outputs = model(x)
        pred = outputs.argmax(dim=1)
        print("原始样本预测标签:", pred.cpu().numpy())
    
    # 生成对抗样本
    x_adv = fgsm_attack(model, x, y, args.epsilon, args.target_idx, args.alpha)
    
    # 显示对抗样本预测结果
    with torch.no_grad():
        outputs_adv = model(x_adv)
        pred_adv = outputs_adv.argmax(dim=1)
        print("对抗样本预测标签:", pred_adv.cpu().numpy())

    #计算原始和对抗样本的准确率
    acc_ori = (pred == y).sum().item() / batch_size
    acc_adv = (pred_adv == y).sum().item() / batch_size
    print(f"原始样本准确率: {acc_ori:.2f}%")
    print(f"对抗样本准确率: {acc_adv:.2f}%")
    
    # 将对抗样本保存为npz文件，便于后续分析或可视化
    #_adv_np = x_adv.cpu().numpy()
    #p.savez("adversarial_examples.npz", x_adv=x_adv_np, y=y.cpu().numpy())
    #rint("对抗样本已保存至 adversarial_examples.npz")

if __name__ == "__main__":
    for i in range(15):
        main(i)
