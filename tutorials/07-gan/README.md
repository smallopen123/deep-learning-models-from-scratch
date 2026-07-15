# 07｜GAN：生成器与判别器的对抗训练

> 数学与反向传播详解：[GAN 目标与两次反向传播](../../math/07-gan-objective-gradients.md)

## 原理

生成器 G 将噪声 z 映射为假样本；判别器 D 区分真实与生成样本。D 最小化真假分类损失，G 让 D 把假样本判断为真。两者不是同时对同一个损失做普通最小化，而是交替优化。

教学代码使用更稳定的非饱和生成器目标，并让判别器输出 logits 配合 `BCEWithLogitsLoss`。训练 D 时必须对假样本 `detach()`，否则会无意计算/累积 G 的梯度。

## Shape

`noise:[B,8] → G → fake:[B,2]`；`real/fake:[B,2] → D → logits:[B,1]`。真实数据是四个二维高斯簇，CPU 可快速运行。

## 代码释义

模型在 [`gan.py`](../../dl_models/gan.py)，交替训练在 [`train_gan.py`](../../examples/train_gan.py)。先更新 D：真实目标 1、假目标 0；再重新生成样本更新 G：希望 D 输出目标 1。

## 运行

```powershell
python examples/train_gan.py
```

## 常见错误

- 更新 D 时不 detach 假样本。
- 生成器和判别器能力严重失衡。
- 只看损失判断生成质量。
- mode collapse：生成器只覆盖少数模式。

## 练习

打印生成样本均值/方差；画真实与生成散点图；修改噪声维度；尝试标签平滑；再阅读官方 DCGAN 教程。
