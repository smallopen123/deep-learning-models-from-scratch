# 深度学习术语表

- **Tensor**：多维数组；模型输入、参数和输出的统一表示。
- **Batch**：一次并行送入模型的一组样本。
- **Epoch**：完整遍历一次训练集。
- **Parameter**：训练中由优化器更新的权重与偏置。
- **Logit**：进入 sigmoid/softmax 前的原始分数。
- **Loss**：可求导的训练目标，不等同于最终业务指标。
- **Gradient**：损失对参数的偏导，表示参数微小变化对损失的影响。
- **Backpropagation**：沿计算图反向应用链式法则计算梯度。
- **Optimizer**：根据梯度更新参数的算法，例如 SGD、Adam。
- **Learning rate**：每次参数更新的步长。
- **Overfitting**：训练表现好但验证/测试差。
- **Latent vector**：模型学习的压缩内部表示。
- **Mask**：告诉注意力忽略 padding 或禁止查看未来位置的布尔/数值矩阵。
