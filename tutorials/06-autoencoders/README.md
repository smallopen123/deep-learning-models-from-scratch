# 06｜Autoencoder 与 VAE：学习潜在表示

> 数学与反向传播详解：[AE 反传与 VAE ELBO](../../math/06-autoencoder-vae-elbo.md)

## Autoencoder

编码器将输入 x 压缩为 z，解码器从 z 重建 x；最小化重建损失。瓶颈迫使模型保留重要信息，但容量太大时可能只学恒等映射。它可用于降维、去噪和异常检测，但重建误差阈值必须验证。

## VAE

VAE 不把样本编码为单点，而是输出 `μ` 和 `log σ²`。用重参数化 `z=μ+σ⊙ε` 保持可微。损失 = 重建损失 + KL 散度；KL 让潜空间接近标准正态，便于采样和插值。

## Shape 与代码

`x:[B,64] → hidden:[B,32] → μ/logvar:[B,8] → z:[B,8] → reconstruction:[B,64]`。查看 [`autoencoders.py`](../../dl_models/autoencoders.py)，注意 VAE 的四个返回值和 `vae_loss`。

## 运行

```powershell
python examples/train_autoencoder.py --model ae --epochs 10
python examples/train_autoencoder.py --model vae --epochs 10
```

## 常见错误

- 输入范围与输出激活/损失不匹配。
- VAE 直接采样导致梯度断开。
- KL 权重过大造成 posterior collapse。
- 只看平均重建损失，不观察样本。

## 练习

改变 latent_dim；加入输入噪声训练去噪 AE；画潜向量；给 KL 项加入 β 并比较。
