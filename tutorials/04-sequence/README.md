# 04｜RNN、LSTM 与 GRU：按顺序处理信息

> 数学与反向传播详解：[BPTT、LSTM 与 GRU 梯度](../../math/04-bptt-lstm-gru.md)

## 原理

RNN 递推 `h_t=tanh(W_x x_t+W_h h_{t-1}+b)`，隐藏状态汇总过去。长序列中反复连乘导数容易消失/爆炸。LSTM 用输入门、遗忘门、输出门和记忆单元控制信息流；GRU 将门结构简化，通常参数更少。

## Shape

token `[B,L] → Embedding → [B,L,E] → recurrent layer → hidden [layers,B,H] → 最后一层 [B,H] → logits [B,C]`。LSTM 返回 `(h_n,c_n)`，RNN/GRU 只返回 `h_n`。

## 代码释义

[`dl_models/sequence.py`](../../dl_models/sequence.py) 用同一分类头切换三种 cell。`batch_first=True` 让输入顺序保持 `[B,L,E]`。当前示例没有可变长度；真实文本要同时处理 padding、长度或 mask。

## 运行

```powershell
python examples/train_classifier.py --model rnn --epochs 8
python examples/train_classifier.py --model lstm --epochs 8
python examples/train_classifier.py --model gru --epochs 8
```

## 常见错误

- 混淆 sequence output 与 final hidden。
- LSTM 忘记拆 `(h,c)`。
- padding token 参与最终表示。
- 直接用很长序列却不处理梯度。

## 练习

比较三种模型参数量；增加序列长度；改成双向 LSTM；使用所有时间步平均而非最后 hidden。
