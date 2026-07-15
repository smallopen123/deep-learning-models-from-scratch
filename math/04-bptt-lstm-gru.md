# 04｜BPTT、LSTM 与 GRU 的梯度

## 1. RNN 前向

```text
a_t = W_x x_t + W_h h_{t-1} + b
h_t = tanh(a_t)
```

参数在所有时间步共享。若损失只依赖最后状态 `L=L(h_T)`，仍需沿时间展开计算图反传，这叫 Backpropagation Through Time（BPTT）。

## 2. 隐藏状态递推梯度

```text
δ_t = ∂L/∂a_t
δ_t = (∂L/∂h_t + δ_{t+1}W_h) ⊙ (1-h_t²)
```

若每步都有损失，`∂L_t/∂h_t` 也加入上式。参数梯度汇总所有时间步：

```text
∂L/∂W_x = Σ_t δ_t x_tᵀ
∂L/∂W_h = Σ_t δ_t h_{t-1}ᵀ
∂L/∂b   = Σ_t δ_t
```

## 3. 梯度消失/爆炸

从 T 传到早期 t 的梯度含多次 `W_hᵀ diag(1-h²)` 相乘。若乘积谱范数长期小于 1，梯度指数衰减；大于 1 则可能爆炸。梯度裁剪处理爆炸，LSTM/GRU 门控改善长期信息通路。

## 4. LSTM 方程

```text
i_t=σ(W_i[x_t,h_{t-1}]+b_i)   输入门
f_t=σ(W_f[x_t,h_{t-1}]+b_f)   遗忘门
o_t=σ(W_o[x_t,h_{t-1}]+b_o)   输出门
g_t=tanh(W_g[x_t,h_{t-1}]+b_g)
c_t=f_t⊙c_{t-1}+i_t⊙g_t
h_t=o_t⊙tanh(c_t)
```

关键梯度：`∂c_t/∂c_{t-1}=f_t`。当遗忘门接近 1 时，记忆梯度可跨时间较稳定传播；但门饱和和长序列仍会带来困难。

## 5. LSTM 门的反向

令 `q_t=∂L/∂c_t`，还要加入从 `h_t` 进入 c_t 的路径：

```text
q_t += (∂L/∂h_t)⊙o_t⊙(1-tanh²(c_t))
∂L/∂f_t = q_t⊙c_{t-1}
∂L/∂i_t = q_t⊙g_t
∂L/∂g_t = q_t⊙i_t
∂L/∂c_{t-1} = q_t⊙f_t
```

再分别乘 sigmoid/tanh 局部导数，传入对应仿射层。

## 6. GRU

GRU 使用更新门 z、重置门 r，将 cell 与 hidden 合并。常见形式 `h_t=(1-z_t)⊙h_{t-1}+z_t⊙h~_t`，直接加法同样提供梯度通路，参数少于 LSTM。

## 7. 数据验证

使用仓库内文本情感数据：`python experiments/train_text.py --model lstm --epochs 30`，再切换 rnn/gru 比较。模型代码见 [sequence.py](../dl_models/sequence.py)。
