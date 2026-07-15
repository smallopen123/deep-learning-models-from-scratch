# 数据集与验证设计

## 为什么同时保留合成数据与真实数据

合成数据用于先验证代码和 shape，几秒内运行且没有网络依赖；真实数据用于检验模型能否在标准任务上泛化。学习顺序是：先合成数据跑通，再用真实数据训练，不能只报告训练集结果。

| 模型 | 快速数据 | 真实/固定数据 | 主要指标 |
|---|---|---|---|
| MLP | 二维非线性点 | FashionMNIST | test accuracy/loss |
| CNN | 几何图案 | FashionMNIST | test accuracy/loss |
| ResNet | 几何图案 | CIFAR-10 | test accuracy/loss |
| RNN/LSTM/GRU | 整数序列 | `tiny_sentiment.tsv` | held-out accuracy |
| Transformer | 整数序列 | `tiny_sentiment.tsv` | held-out accuracy |
| AE/VAE | 二值原型向量 | MNIST | reconstruction loss + 重建图 |
| GAN | 二维高斯混合 | MNIST | G/D loss + 生成图 |

## FashionMNIST

60,000 张训练图、10,000 张测试图，28×28 灰度服饰、10 类。适合比较 MLP 展平输入与 CNN 空间归纳偏置。数据由 TorchVision `download=True` 下载到 `data/`。

## CIFAR-10

50,000 张训练图、10,000 张测试图，32×32 RGB、10 类。教学版 TinyResNet 参数较少，短训练的准确率不是论文基准；目标是验证残差块、RGB 输入和训练流程。

## MNIST

28×28 手写数字。AE/VAE 不使用标签，比较重建；GAN 用训练图作为真实分布。必须查看保存图像，不能只根据 loss 判断生成质量。

## tiny_sentiment.tsv

仓库内 40 条平衡英文短句，标签 1/0。它足够演示 tokenization、词表、padding、训练/测试划分，但太小，不代表真实 NLP 泛化能力。数据文本为本仓库原创，可直接审阅。

## 公平验证规则

1. 训练集用于梯度更新；测试集不参与调参。
2. 固定随机种子并记录 limit、epoch、batch size。
3. 报告测试指标和训练时间，不只挑最好一次。
4. 数据太小时明确说明方差与局限。
5. 下载的数据目录、checkpoint 和输出图片默认不提交 Git。
