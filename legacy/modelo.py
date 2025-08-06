import torch
import torch.nn as nn

class EMGNet(nn.Module):
    def __init__(self, num_classes=8):
        """
        参数:
        -------
        num_classes: 分类的类别数，这里默认为 8
        """
        super(EMGNet, self).__init__()
        
        # 特征提取部分(四层卷积)，在需要减半特征图时设置 stride=2
        # 每层卷积后均进行 BN (批归一化) 和 ReLU 激活
        self.features = nn.Sequential(
            # 第1层: BN -> ReLU -> Conv(输入通道=8, 输出通道=16, 卷积核=3x3, stride=1, padding=1)
            nn.BatchNorm2d(8),           # 对输入的 8 个通道做批归一化
            nn.ReLU(inplace=True),       
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            
            # 第2层: BN -> ReLU -> Conv(16->32, 3x3, stride=2, padding=1)，
            # 这里将特征图尺寸减半
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            
            # 第3层: BN -> ReLU -> Conv(32->32, 3x3, stride=1, padding=1)，
            # 这里不缩放特征图尺寸
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            # 第4层: BN -> ReLU -> Conv(32->64, 3x3, stride=2, padding=1)，
            # 再次将特征图尺寸减半
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )
        
        # 自适应平均池化，将不同大小的特征图统一输出到 (64, 1, 1)
        # (此时批大小不变，通道数为 64，高宽均变为 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 最后一层卷积(相当于 1×1 卷积做分类)，输出通道数为类别数 num_classes
        # 注意：这里不再使用全连接层
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x):
        """
        前向传播的计算流程:
        1. 特征提取(4 层卷积)
        2. 自适应平均池化 (将 H, W 变为 1, 1)
        3. 1×1 卷积分类层
        4. 去掉最后两个维度(1,1)，得到 (batch_size, num_classes)
        """
        x = self.features(x)         # 经过四层卷积
        x = self.avg_pool(x)         # 自适应平均池化
        x = self.classifier(x)       # 1×1 卷积分类
        
        # 输出张量现在形状是 [batch_size, num_classes, 1, 1]
        # 我们需要变成 [batch_size, num_classes] 便于计算交叉熵损失
        x = x.view(x.size(0), -1)    # 压缩空间维度
        
        return x

if __name__ == "__main__":
    model = EMGNet(num_classes=7)
    print(model)
