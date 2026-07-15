# 01｜MLP、Softmax 与交叉熵的完整推导

## 1. 前向定义

对批量输入 `X∈R^(B×D)`：

```text
Z₁ = XW₁ + b₁            W₁:[D,H], b₁:[H]
H  = ReLU(Z₁)            H:[B,H]
Z₂ = HW₂ + b₂            W₂:[H,C], b₂:[C]
P_ik = exp(Z₂_ik)/Σ_j exp(Z₂_ij)
L = -(1/B)Σ_i log P_i,y_i
```

实际计算先对每行 logits 减去最大值，避免 `exp` 溢出；PyTorch `CrossEntropyLoss` 已稳定合并 LogSoftmax 与 NLL。

## 2. Softmax Jacobian

对一个样本，`p_k=exp(z_k)/Σ_j exp(z_j)`：

```text
∂p_k/∂z_l = p_k(δ_kl-p_l)
```

其中 `δ_kl` 在 k=l 时为 1，否则为 0。

## 3. 与交叉熵合并后的简洁结果

令 one-hot 标签 `Y`。将 `∂L/∂p` 与 softmax Jacobian 相乘后大量项抵消：

```text
∂L/∂Z₂ = (P-Y)/B
```

这解释了“预测概率减真实标签”。如果正确类别概率偏小，该类别 logit 获得负梯度，SGD 更新会把它提高。

## 4. 输出层梯度

由 `Z₂=HW₂+b₂`：

```text
∂L/∂W₂ = Hᵀ(∂L/∂Z₂)       [H,B]@[B,C] -> [H,C]
∂L/∂b₂ = Σ_batch ∂L/∂Z₂    [C]
∂L/∂H  = (∂L/∂Z₂)W₂ᵀ      [B,C]@[C,H] -> [B,H]
```

偏置在每个样本上广播，所以其梯度要沿 batch 求和。

## 5. ReLU 与第一层

```text
∂L/∂Z₁ = (∂L/∂H) ⊙ 1[Z₁>0]
∂L/∂W₁ = Xᵀ(∂L/∂Z₁)
∂L/∂b₁ = Σ_batch ∂L/∂Z₁
∂L/∂X  = (∂L/∂Z₁)W₁ᵀ
```

`⊙` 是逐元素乘法。`∂L/∂X` 通常不用于更新输入，但继续传给更早层。

## 6. 加 L2 正则化

若 `L_total=L_data+(λ/2)||W||²`，则 `∂L_total/∂W=∂L_data/∂W+λW`。通常不正则化偏置与归一化参数。

## 7. 对照代码与数据验证

- 模型：[MLP 实现](../dl_models/mlp.py)
- 自动求导验证：[反向传播测试](../tests/test_models.py)
- FashionMNIST/MNIST 实验：[视觉数据实验](../experiments/train_vision.py)

运行 `python experiments/train_vision.py --model mlp --dataset fashion_mnist --epochs 5 --limit 12000`。练习：为一个 `B=2,D=2,H=3,C=2` 的网络逐项写出 shape。
