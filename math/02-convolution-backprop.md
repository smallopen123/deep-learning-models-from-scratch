# 02｜卷积与池化的反向传播

## 1. 二维卷积前向

为简化先忽略 batch。输入 `X[c,i,j]`、卷积核 `W[k,c,u,v]`、输出通道 k：

```text
Y[k,i,j] = b[k] + Σ_c Σ_u Σ_v W[k,c,u,v] X[c,i+u,j+v]
```

深度学习库实际执行的是互相关（核不翻转），但通常仍称卷积。输出尺寸：`H_out=floor((H+2P-K)/S)+1`。

## 2. 卷积核梯度

设上游梯度 `G[k,i,j]=∂L/∂Y[k,i,j]`。某个核元素在所有输出位置复用，所以梯度要累加所有位置贡献：

```text
∂L/∂W[k,c,u,v] = Σ_i Σ_j G[k,i,j] X[c,i+u,j+v]
∂L/∂b[k] = Σ_i Σ_j G[k,i,j]
```

“权重共享”既减少参数，也意味着同一核从整张图收集梯度。

## 3. 输入梯度

输入像素会影响覆盖它的所有输出窗口：

```text
∂L/∂X[c,p,q] = Σ_k Σ_i Σ_j G[k,i,j] W[k,c,p-i,q-j]
```

有效求和范围要求对应核索引合法。实现上可理解为上游梯度与翻转卷积核的 full convolution。

## 4. stride 与 padding

stride>1 时，前向只采样部分位置；反向可想象在上游梯度位置间插零再卷积。padding 的补零不是参数，因此只保留裁剪回原输入区域的梯度。

## 5. 最大池化反向

`y=max(window)` 的梯度只传给前向最大值位置：

```text
∂L/∂x_r = g,  若 r=argmax(window)
          0,  其他位置
```

平均池化则把梯度平均分给窗口内每个位置。最大值并列时具体分配由实现定义。

## 6. BatchNorm 与 Dropout 提醒

BatchNorm 训练时依赖批次均值方差，完整推导包含中心化与方差路径；推理时使用运行统计。Dropout 训练时乘随机 mask 并缩放，反向使用相同 mask；推理时关闭。

## 7. 数据验证

用 FashionMNIST 检验 CNN 的空间归纳偏置：`python experiments/train_vision.py --model cnn --dataset fashion_mnist --epochs 5 --limit 12000`。对照 [CNN 代码](../dl_models/cnn.py)。
