# 07｜GAN 目标函数与两次反向传播

## 1. Minimax 目标

```text
min_G max_D V(D,G)
V = E_{x~pdata}[log D(x)] + E_{z~pz}[log(1-D(G(z)))]
```

D 提高真实样本得分并降低假样本得分；G 改变假样本使 D 难以区分。

## 2. 判别器更新

用 logits 与 `BCEWithLogitsLoss`：

```text
L_D = BCE(D_logit(x),1) + BCE(D_logit(G(z).detach()),0)
```

对单个 logit s，sigmoid+BCE 的导数是 `σ(s)-y`。真实样本 y=1 时梯度推动 s 增大，假样本 y=0 时推动 s 减小。

`detach()` 切断 D 更新阶段通向 G 的计算图。否则会计算无用 G 梯度，若处理错误还可能污染下一步。

## 3. 生成器更新

原 minimax 生成器最小化 `log(1-D(G(z)))`，当 D 很强时 sigmoid 饱和、梯度弱。常用非饱和目标：

```text
L_G = BCE(D_logit(G(z)),1) = -log D(G(z))
```

梯度路径：`L_G → D输出 → D对输入的梯度 → G输出 → G参数`。这一步需要经过 D 的运算，但优化器只更新 G；D 的参数可暂时冻结以减少计算。

## 4. 生成器梯度形式

记假样本 `x_fake=G(z;θ_G)`：

```text
∂L_G/∂θ_G = ∂L_G/∂D · ∂D/∂x_fake · ∂G/∂θ_G
```

对抗训练不稳定，是因为优化目标随对手参数持续变化，而不是固定损失面。

## 5. Mode collapse

G 可能把许多 z 映射到少数模式，虽然能骗过 D，却未覆盖数据分布。只看损失无法发现；必须可视化样本、多样性和覆盖率。

## 6. 常见稳定技巧

合理初始化、Adam `lr≈2e-4, β1≈0.5`、标签平滑、平衡 G/D 更新、归一化输入。更现代方法包括 Wasserstein 距离和 gradient penalty。

## 7. MNIST 验证

`python experiments/train_mnist_gan.py --epochs 10 --limit 20000`。每轮保存生成网格；先确认输出从噪声变成笔画，再讨论质量。基础二维版本见 [train_gan.py](../examples/train_gan.py)。
