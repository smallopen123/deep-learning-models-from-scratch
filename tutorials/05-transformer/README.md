# 05｜Transformer Encoder：用注意力直接连接所有位置

> 数学与反向传播详解：[注意力反向传播](../../math/05-attention-backprop.md)

## 原理

缩放点积注意力：`Attention(Q,K,V)=softmax(QKᵀ/√d_k)V`。Q 决定“我要找什么”，K 决定“我包含什么”，V 是被汇总的信息。多头注意力在不同子空间并行建模。Encoder 层还包含前馈网络、残差和 LayerNorm。

自注意力本身不知道顺序，因此必须加入位置编码。padding mask 防止模型关注补齐位置；语言生成还需要 causal mask 禁止查看未来，本分类示例不需要 causal mask。

## Shape

`tokens:[B,L] → embedding:[B,L,D] → positional encoding → encoder:[B,L,D] → masked mean:[B,D] → logits:[B,C]`。`batch_first=True` 与当前 PyTorch API 保持批次在前。

## 代码释义

[`dl_models/transformer.py`](../../dl_models/transformer.py) 注册正弦位置编码为 buffer，因此会随模型移动 device，但不会被优化器更新。`tokens.eq(0)` 生成 padding mask，最后只平均有效 token。

## 运行

```powershell
python examples/train_classifier.py --model transformer --epochs 8
```

## 常见错误

- 忘记位置编码。
- mask 的 True/False 语义写反。
- `model_dim` 不能被 `num_heads` 整除。
- 序列很长时注意力矩阵 O(L²) 占用大量显存。

## 练习

打印 attention 输入 shape；改变头数；删除位置编码比较；构造含 padding 的批次并检查结果。
