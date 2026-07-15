# 02｜CNN：让模型利用图像空间结构

> 数学与反向传播详解：[卷积与池化反向传播](../../math/02-convolution-backprop.md)

## 原理

卷积核在图像各位置共享同一组权重，检测局部边缘和纹理。相比将图像展平后全连接，它参数更少，并保留邻近像素关系。输出尺寸为 `floor((H+2P-K)/S)+1`。池化缩小空间尺寸、扩大后续单元感受野。

## Shape

`[B,1,28,28] → Conv(16) → [B,16,28,28] → Pool → [B,16,14,14] → Conv(32) → Pool → [B,32,7,7] → Flatten → [B,1568] → logits`。

## 代码释义

[`dl_models/cnn.py`](../../dl_models/cnn.py) 将特征提取器与分类头分开。`padding=1` 让 3×3、stride=1 卷积保持宽高；`Dropout` 只在训练时随机屏蔽。合成数据包含竖线、横线、对角线和方块，见 [`synthetic.py`](../../dl_models/synthetic.py)。

## 运行

```powershell
python examples/train_classifier.py --model cnn --epochs 8
```

## 常见错误

- PyTorch 图像顺序是 `[B,C,H,W]`，不是 `[B,H,W,C]`。
- 展平尺寸写死却修改了输入大小。
- 忘记像素归一化。
- 验证阶段未调用 `eval()`，Dropout 仍随机。

## 练习

手算每层尺寸；去掉池化比较参数量；增加一个卷积层；将合成数据替换为 FashionMNIST。
