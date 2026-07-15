# 06｜Autoencoder 反传与 VAE ELBO 推导

## 1. Autoencoder

编码器 `z=f_φ(x)`，解码器 `x̂=g_θ(z)`，均方重建损失：

```text
L_rec=(1/B)Σ_i ||x̂_i-x_i||²
∂L/∂x̂ = 2(x̂-x)/B
```

梯度先通过解码器更新 θ，再通过 z 继续进入编码器更新 φ。普通 AE 的潜变量是确定性的，不能保证潜空间连续或便于采样。

## 2. VAE 的概率模型

假设先验 `p(z)=N(0,I)`、生成模型 `p_θ(x|z)`，编码器近似难求后验 `q_φ(z|x)`。我们希望最大化 `log p_θ(x)`。

## 3. ELBO

乘除 `q(z|x)` 并使用 Jensen 不等式：

```text
log p(x)=log ∫ q(z|x) p(x,z)/q(z|x) dz
       ≥ E_q[log p_θ(x|z)] - KL(q_φ(z|x)||p(z))
       = ELBO
```

训练最小化负 ELBO：

```text
L = reconstruction_loss + KL
```

第一项要求重建，第二项让近似后验接近先验。

## 4. 对角高斯 KL 闭式

若 `q=N(μ,diag(σ²))`、`p=N(0,I)`：

```text
KL(q||p)= -1/2 Σ_j [1+log σ_j²-μ_j²-σ_j²]
```

代码输出 `log_variance` 而不是 σ，既稳定又无需约束为正。

## 5. 为什么需要重参数化

直接 `z~N(μ,σ²)` 的随机采样节点不便对 μ、σ 反传。改写：

```text
ε~N(0,I)          （随机性移到与参数无关的 ε）
z=μ+σ⊙ε
σ=exp(0.5 logσ²)
```

于是 `∂z/∂μ=1`，`∂z/∂logσ²=0.5σ⊙ε`，梯度可进入编码器。

## 6. β-VAE 与 posterior collapse

`L=L_rec+βKL`。β 大强调规则潜空间但可能损害重建；强解码器可能忽略 z，使 `q≈p`，称 posterior collapse。可用 KL warm-up、减弱解码器等方法。

## 7. MNIST 验证

`python experiments/train_mnist_autoencoder.py --model ae --epochs 5` 或 `--model vae`。脚本报告训练/测试重建损失并保存重建图。代码见 [autoencoders.py](../dl_models/autoencoders.py)。
