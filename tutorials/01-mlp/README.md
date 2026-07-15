# 01｜MLP：深度学习的最小完整模型

## 原理

MLP 将线性变换与非线性激活交替堆叠：`h=ReLU(XW₁+b₁)`，`logits=hW₂+b₂`。如果没有 ReLU，多层线性变换仍等价于一层，无法学习弯曲决策边界。分类使用 `CrossEntropyLoss`，它直接接收 logits；不要先 softmax。

## Shape

`X:[B,2] → Linear → [B,32] → ReLU → [B,32] → Linear → [B,2]`。B 是批大小，2 是类别数。

## 代码释义

打开 [`dl_models/mlp.py`](../../dl_models/mlp.py)：`nn.Linear` 保存 W、b；`ReLU` 只保留正值；最后一层输出每个类别分数。打开 [`examples/train_classifier.py`](../../examples/train_classifier.py)：前向计算损失，`zero_grad()` 清旧梯度，`backward()` 反传，`step()` 更新。

## 运行

```powershell
python examples/train_classifier.py --model mlp --epochs 8
```

预期看到 loss 下降并输出测试准确率。不同机器数值可不同。

## 常见错误

- 输入 shape 写成 `[特征,B]`。
- 对 logits 手动 softmax 后再传交叉熵。
- 忘记清梯度导致意外累加。
- 只看训练准确率。

## 练习

把隐藏层改成 4/64；把 ReLU 换成 Tanh；删除隐藏层比较结果；打印每个参数的梯度 shape。
