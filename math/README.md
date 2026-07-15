# 数学推导与反向传播路线

按顺序阅读。第 0 章是所有后续模型的共同基础。

1. [链式法则与反向传播总论](00-chain-rule-and-backprop.md)
2. [MLP、Softmax 与交叉熵完整推导](01-mlp-softmax-backprop.md)
3. [卷积与池化反向传播](02-convolution-backprop.md)
4. [ResNet 残差连接梯度](03-residual-gradient.md)
5. [BPTT、LSTM 与 GRU 梯度](04-bptt-lstm-gru.md)
6. [自注意力反向传播](05-attention-backprop.md)
7. [Autoencoder 反传与 VAE ELBO](06-autoencoder-vae-elbo.md)
8. [GAN 目标函数与两次反向传播](07-gan-objective-gradients.md)

建议方法：先抄写前向公式并标 shape；遮住答案自己推局部导数；用 PyTorch `retain_grad()` 查看中间梯度；最后用有限差分检查一个小参数。
