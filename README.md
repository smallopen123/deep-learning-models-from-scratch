# 深度学习模型：从原理到 PyTorch 代码

面向零基础学习者的中文深度学习仓库。每个模型都按 **直觉 → 数学原理 → 张量形状 → 代码逐段释义 → 运行方法 → 常见错误 → 练习** 组织。所有默认示例使用程序生成的小数据，可在 CPU 上运行，不需要先下载数据集。

## 立即开始

1. [安装与环境检查](SETUP.md)
2. [第一个模型：MLP](tutorials/01-mlp/README.md)
3. 运行：`python examples/train_classifier.py --model mlp --epochs 8`
4. 按 [8 周学习计划](ROADMAP.md) 和下表顺序学习，不建议直接跳到 GAN 或 Transformer。

## 模型学习路线

| 顺序 | 模型 | 核心问题 | 教程 | 运行命令 |
|---:|---|---|---|---|
| 1 | MLP | 全连接、激活、反向传播 | [学习](tutorials/01-mlp/README.md) | `--model mlp` |
| 2 | CNN | 局部连接、共享权重、池化 | [学习](tutorials/02-cnn/README.md) | `--model cnn` |
| 3 | ResNet | 残差连接与深层优化 | [学习](tutorials/03-resnet/README.md) | `--model resnet` |
| 4 | RNN/LSTM/GRU | 顺序状态与长期依赖 | [学习](tutorials/04-sequence/README.md) | `--model lstm` |
| 5 | Transformer | 自注意力、位置编码、mask | [学习](tutorials/05-transformer/README.md) | `--model transformer` |
| 6 | Autoencoder/VAE | 表示学习与概率潜变量 | [学习](tutorials/06-autoencoders/README.md) | `train_autoencoder.py` |
| 7 | GAN | 生成器与判别器的对抗训练 | [学习](tutorials/07-gan/README.md) | `train_gan.py` |

统一分类命令：

```powershell
python examples/train_classifier.py --model mlp
python examples/train_classifier.py --model cnn
python examples/train_classifier.py --model resnet
python examples/train_classifier.py --model rnn
python examples/train_classifier.py --model lstm
python examples/train_classifier.py --model gru
python examples/train_classifier.py --model transformer
```

## 代码导航

- [`dl_models/`](dl_models)：模型结构，每个 `forward` 都标注输入输出。
- [`examples/`](examples)：完整训练循环，包括数据、损失、优化、验证。
- [`tests/`](tests)：模型输出形状和一次反向传播测试。
- [`GLOSSARY.md`](GLOSSARY.md)：logits、batch、epoch、梯度等术语。

## 如何真正学会

每个模型至少完成四遍：先画数据流和 shape；再逐行运行；第三遍关掉教程自己重写；最后修改一个结构并记录验证结果。能运行不等于理解，至少要能回答：输入输出 shape 是什么、损失为什么匹配、梯度从哪里流回、训练/推理模式有何不同。

## 覆盖范围

“每个模型”在本仓库中指上述十个基础模型/家族，而不是穷举所有论文架构。它们构成计算机视觉、序列建模和生成模型的核心积木，学会后再扩展到 U-Net、Diffusion、BERT、ViT 等模型。

## 参考

API 用法以 [PyTorch 官方教程](https://docs.pytorch.org/tutorials/) 为准。原创教程与代码采用 MIT License。
