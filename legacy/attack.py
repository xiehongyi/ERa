
# attack.py
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from model import EMGNet

"""
说明：
- 这是一个单一频率FGSM对抗攻击的示例代码。
- 目标：只在指定的target_freq附近添加扰动。
- 思路：在回传梯度的时候，在频域对梯度进行约束/加窗，只保留 target_freq 附近的能量。
- 该方法比直接把除target_freq外的梯度清零更灵活，可以保留更多梯度信息，并将其集中到指定频率附近。

"""

# 1. 定义一些超参数
epsilon = 0.2           # FGSM扰动强度
target_freq = 25       # 目标频率(0~100Hz之间)，可自行修改
sample_rate = 200       # EMG采样率（根据实际情况设置）

device = torch.device("cpu")

def single_freq_fgsm(model, X, y, epsilon, target_freq, sample_rate):
    """
    对输入数据X在单一频率target_freq上生成FGSM扰动。
    X, y都是Tensor格式，X形状(batch_size, 8, 15, 25)，
    因为EMGNet的输入是(8, 15, 25)。
    
    但我们仍需要一个"时域"的概念，所以需要把(8, 15, 25)展开成时域信号再进行FFT。
    这里的实现只是一个示例，并没有完全匹配你实际预处理的细节。
    
    实际中，你需要在此处逆向地把(8, 15, 25)对应回原时域的长度，比如原始窗口是52个采样点。
    然后做FFT，在target_freq上加窗，再做iFFT还原到时域，然后映射回(8, 15, 25)。
    
    简化起见，这里只演示对(8, 15, 25)的一个小型FFT操作。
    """
    
    # 确保模型在推理模式
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
    grad = X_adv.grad.data  # 形状与X_adv相同 (batch_size, 8, 15, 25)
    
    # -------------------------------------------------------------
    # 下面是将梯度转到频域，只保留 target_freq 附近的能量，然后再转回时域
    # -------------------------------------------------------------
    # 1. 对 grad 在batch和通道维度上做循环(或flatten)——简单示例如下
    grad_freq_domain = torch.fft.rfft(grad, dim=-1)  # 仅在最后一维(25)做FFT
    
    # 我们只示例对最后一维(25点)做FFT，得到频率分辨率大约 = sample_rate/(2*25)
    # 这并不对应真实的0~100Hz映射，纯粹是教学示例
    
    # 计算目标频率对应的下标
    freq_res = (sample_rate / 2) / grad_freq_domain.shape[-1]  # 注意rfft后shape[-1]是N/2+1
    freq_index = int(target_freq // freq_res)
    freq_index = max(0, min(freq_index, grad_freq_domain.shape[-1]-1))  # 越界处理
    
    # 在这里，你可以定义一个带宽，比如只保留 freq_index±1之类
    bandwidth = 1
    # 构建一个掩码，只有在 freq_index±bandwidth 范围内的保留
    mask = torch.zeros_like(grad_freq_domain, dtype=torch.bool)
    low_idx = max(0, freq_index - bandwidth)
    high_idx = min(freq_index + bandwidth, grad_freq_domain.shape[-1]-1)
    mask[..., low_idx:high_idx+1] = True
    
    # 将不在掩码范围内的频率分量置为0
    grad_freq_domain_masked = grad_freq_domain * mask
    
    # 做逆FFT回到时域
    grad_time_domain_masked = torch.fft.irfft(grad_freq_domain_masked, n=grad.shape[-1], dim=-1)
    
    # -------------------------------------------------------------
    # FGSM更新
    # -------------------------------------------------------------
    # sign(grad_time_domain_masked) -> 同号操作
    update = epsilon * torch.sign(grad_time_domain_masked)
    X_adv_updated = X_adv + update
    
    # 返回对抗样本
    return X_adv_updated.detach()

def main():
    # 加载预处理好的测试数据
    data = np.load('preprocessed_data_myo_armband.npz')
    X_test = data['X_test']
    #归一化
    data_mean = X_test.mean()
    data_std = X_test.std()
    X_test = (X_test - data_mean) / (data_std + 1e-8)  # 加上小值避免除零

    y_test = data['y_test']

    # 将数据转换为 Tensor（但 ART 需要用 numpy 数组进行预测）
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)


    
    
    # 加载训练好的EMGNet模型
    model = EMGNet(num_classes=7).to(device)
    model.load_state_dict(torch.load('emgnet_model.pth', map_location=device))
    model.eval()
    
    # 取一个batch做示例，也可以分多批量
    batch_size = 64
    X_batch = X_test[:batch_size]
    y_batch = y_test[:batch_size]
    
    # 生成对抗样本
    X_batch_adv = single_freq_fgsm(model, X_batch, y_batch, epsilon, target_freq, sample_rate)
    
    # 比较模型在原始数据和对抗数据上的准确率
    with torch.no_grad():
        orig_preds = model(X_batch.to(device))
        orig_preds = torch.argmax(orig_preds, dim=1)
        orig_acc = (orig_preds.cpu() == y_batch).float().mean().item()
        
        adv_preds = model(X_batch_adv.to(device))
        adv_preds = torch.argmax(adv_preds, dim=1)
        adv_acc = (adv_preds.cpu() == y_batch).float().mean().item()
    
    print(f"原始数据准确率: {orig_acc*100:.2f}%")
    print(f"对抗样本准确率: {adv_acc*100:.2f}% (单一频率: {target_freq}Hz, ε={epsilon})")

if __name__ == "__main__":
    main()
