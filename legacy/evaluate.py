import torch
import numpy as np
import matplotlib.pyplot as plt
from model import EMGNet
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 加载预处理好的数据
data = np.load('preprocessed_data_all_subjects.npz')
X_test = data['X_test']
y_test = data['y_test']

# 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EMGNet(num_classes=8)
model.load_state_dict(torch.load('emgnet_model.pth'))
model = model.to(device)
model.eval()  # 设置为评估模式

# 将测试数据转换为 Tensor
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 准备预测结果容器
y_pred = []
y_true = []

# 批量处理测试数据，避免内存溢出
batch_size = 64
with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        inputs = X_test[i:i+batch_size].to(device)
        labels = y_test[i:i+batch_size].to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# 转换为numpy数组
y_pred = np.array(y_pred)
y_true = np.array(y_true)

# 计算准确率
accuracy = accuracy_score(y_true, y_pred)
print(f"\n测试集准确率: {accuracy:.4f}")

# 打印分类报告
print("\n分类报告:")
# 映射原始标签
selected_gestures = [1, 2, 3, 4, 5, 6, 13, 14]
target_names = [f"手势{gesture}" for gesture in selected_gestures]
print(classification_report(y_true, y_pred, target_names=target_names))

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 绘制混淆矩阵热图
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 计算每个类别的准确率
class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, acc in enumerate(class_accuracy):
    print(f"手势 {selected_gestures[i]} 准确率: {acc:.4f}")

# 绘制每个类别的准确率条形图
plt.figure(figsize=(12, 6))
bars = plt.bar(target_names, class_accuracy, color='skyblue')
plt.ylim(0, 1.0)
plt.xlabel('手势类别')
plt.ylabel('准确率')
plt.title('各手势类别准确率')

# 在条形上标注具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom')

plt.savefig('class_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# 可视化一些错误分类的样本
def plot_sample(sample, true_label, pred_label, idx):
    """绘制样本的8个通道小波图"""
    plt.figure(figsize=(15, 8))
    plt.suptitle(f"样本 #{idx}: 真实标签={selected_gestures[true_label]} 预测标签={selected_gestures[pred_label]}", 
                 fontsize=16)
    
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(sample[i], aspect='auto', cmap='viridis')
        plt.title(f"通道 {i+1}")
        plt.colorbar()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'error_sample_{idx}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 找出一些错误分类的样本
misclassified = np.where(y_pred != y_true)[0]
num_errors = min(5, len(misclassified))  # 最多显示5个错误样本

if num_errors > 0:
    print(f"\n显示 {num_errors} 个错误分类的样本:")
    for i in range(num_errors):
        idx = misclassified[i]
        plot_sample(X_test[idx].numpy(), y_true[idx], y_pred[idx], idx) 