# 03｜ResNet：用残差连接训练更深网络

## 原理

普通深网要直接学习目标映射 H(x)。残差块改为学习 F(x)=H(x)-x，输出 `ReLU(F(x)+x)`。捷径为梯度提供更直接的通路，缓解深层网络优化退化，但不保证自动避免所有过拟合或梯度问题。

## 尺寸对齐

相加两侧 shape 必须一致。当通道数或空间尺寸变化时，捷径使用 1×1 卷积和相同 stride：主支路 `[B,16,28,28]→[B,32,14,14]`，捷径也变为 `[B,32,14,14]`。

## 代码释义

[`dl_models/resnet.py`](../../dl_models/resnet.py) 的 `BasicBlock` 有两次 3×3 卷积。第二次后先与 shortcut 相加，再 ReLU。`AdaptiveAvgPool2d(1)` 把任意空间尺寸汇聚到 1×1，分类头只接收通道向量。

## 运行

```powershell
python examples/train_classifier.py --model resnet --epochs 8
```

## 常见错误

- 直接相加不同 channel/宽高的张量。
- 在第二个卷积后、相加前错误地额外 ReLU。
- 小批次时 BatchNorm 统计不稳定。
- 认为层数越深一定越好。

## 练习

打印主支路与捷径 shape；删掉 shortcut 比较；再堆两个块；统计 CNN 与 TinyResNet 参数量。
