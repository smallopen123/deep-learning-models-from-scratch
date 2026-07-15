# 05｜自注意力与反向传播

## 1. 前向

输入 `X∈R^(B×L×D)`：

```text
Q=XW_Q, K=XW_K, V=XW_V
S=QKᵀ/√d_k
A=softmax(S)
O=AV
```

每个注意力头独立计算后拼接并线性映射。

## 2. 从输出到 V 与 A

令上游梯度 `G=∂L/∂O`：

```text
∂L/∂V = AᵀG
∂L/∂A = GVᵀ
```

矩阵乘法梯度规则：若 Y=AB，则 `∂L/∂A=GBᵀ`、`∂L/∂B=AᵀG`。

## 3. Softmax 反向

对每个 query 行，若 `g_A=∂L/∂a`：

```text
∂L/∂s = a ⊙ (g_A - Σ_j g_Aj a_j)
```

这是 softmax Jacobian `diag(a)-aaᵀ` 与上游梯度相乘的高效形式。被 mask 的位置在 softmax 前设为极小值，概率近零，梯度也应被屏蔽。

## 4. 从分数到 Q/K

```text
∂L/∂Q = (∂L/∂S)K/√d_k
∂L/∂K = (∂L/∂S)ᵀQ/√d_k
```

`1/√d_k` 防止维度大时点积方差过大、softmax 过度饱和导致梯度很小。

## 5. 投影参数与输入

```text
∂L/∂W_Q = Xᵀ(∂L/∂Q),  ∂L/∂X_Q=(∂L/∂Q)W_Qᵀ
```

K、V 同理。因为同一个 X 同时产生 Q/K/V，输入总梯度是三条路径梯度之和，再加残差路径梯度。

## 6. LayerNorm 与残差

Encoder 通常含 `x + Attention(LN(x))` 或 `LN(x+Attention(x))`。残差的梯度结构与 ResNet 类似；LayerNorm 在每个样本/位置的特征维度归一化，其缩放 γ、偏移 β 也学习。

## 7. 位置编码

无位置编码时，自注意力对输入排列等变。固定正弦位置编码作为常量与 embedding 相加，不接收参数梯度；可学习位置 embedding 则像普通参数一样更新。

## 8. 数据验证

`python experiments/train_text.py --model transformer --epochs 30`。观察 padding mask 后再尝试删除它。代码见 [transformer.py](../dl_models/transformer.py)。
