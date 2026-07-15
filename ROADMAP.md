# 8 周学习计划

建议每周 6–8 小时。每周按“读原理 2h、画 shape 1h、运行与调试 2h、重写 2h、复盘 1h”安排。

| 周 | 内容 | 必做验收 |
|---:|---|---|
| 1 | Tensor、autograd、MLP | 不看代码写出训练循环；解释 logits |
| 2 | CNN | 手算每层尺寸；替换一个卷积核大小 |
| 3 | ResNet | 解释 shortcut 对齐；实现一个 BasicBlock |
| 4 | RNN/LSTM/GRU | 比较返回状态和参数量；处理 padding |
| 5 | Transformer | 写出 Q/K/V shape；正确构造 padding mask |
| 6 | Autoencoder/VAE | 解释重参数化和 KL 项；比较潜空间 |
| 7 | GAN | 正确交替更新 G/D；解释 detach |
| 8 | 综合项目 | 替换真实数据、记录实验、错误分析和复现命令 |

## 每个模型的通关问题

1. 输入、每个中间层、输出的 shape 是什么？
2. 为什么选择这个损失？它接收 logits 还是概率？
3. 梯度从损失经过哪些操作回到参数？
4. 哪些层在 `train()` 与 `eval()` 下行为不同？
5. 如果训练 loss 不下降，前三个检查点是什么？
6. 与更简单基线相比，它解决了什么结构性问题？
