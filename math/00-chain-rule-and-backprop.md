# 00｜链式法则与反向传播总论

## 1. 反向传播到底是什么

反向传播不是新的求导规则，而是把链式法则按计算图反向、重复使用，并缓存中间结果。若 `L=f(g(x))`，则：

```text
dL/dx = dL/dg · dg/dx
```

前向传播计算并保存每个节点的值；反向传播从标量损失 `L` 的梯度 1 开始，将“上游梯度 × 当前操作的局部导数”传给输入。

## 2. 一个完整标量例子

设 `a=wx+b`、`ŷ=a²`、`L=(ŷ-y)²`。局部导数：

```text
∂L/∂ŷ = 2(ŷ-y)
∂ŷ/∂a = 2a
∂a/∂w = x
∂a/∂b = 1
```

所以：

```text
∂L/∂w = 2(ŷ-y) · 2a · x
∂L/∂b = 2(ŷ-y) · 2a
```

每一项都对应计算图的一条边。多条路径汇合时梯度相加，因为总微分是各路径影响之和。

## 3. 向量与 Jacobian

若 `y=f(x)`，`x∈R^n,y∈R^m`，局部导数是 Jacobian `J_ij=∂y_i/∂x_j`。实际框架不显式构造巨大 Jacobian，而计算 vector-Jacobian product：上游梯度乘局部 Jacobian。

## 4. 常用局部导数

```text
加法 y=a+b:       ∂L/∂a=g, ∂L/∂b=g
乘法 y=ab:        ∂L/∂a=gb, ∂L/∂b=ga
ReLU y=max(0,x):  ∂L/∂x=g·1[x>0]
sigmoid σ(x):     ∂L/∂x=g·σ(x)(1-σ(x))
tanh:             ∂L/∂x=g·(1-tanh²x)
```

`g=∂L/∂y` 是上游梯度。

## 5. 参数更新不是反向传播

`loss.backward()` 只计算并累积梯度；`optimizer.step()` 才使用梯度更新参数。以 SGD 为例：`θ ← θ-η∇θL`。因为 PyTorch 默认累积梯度，每个训练 step 必须清零。

## 6. 数值梯度检查

```text
∂L/∂θ ≈ [L(θ+ε)-L(θ-ε)]/(2ε)
relative_error = |g_numeric-g_analytic| / max(1,|g_numeric|,|g_analytic|)
```

有限差分只适合小模型调试。ε 太大近似粗糙，太小会受浮点舍入影响。

## 7. PyTorch 对照

```python
w = torch.tensor(2.0, requires_grad=True)
x, y = torch.tensor(3.0), torch.tensor(40.0)
a = w*x + 1
loss = (a**2-y)**2
loss.backward()
print(w.grad)
```

练习：手算上例梯度，与 `w.grad` 比较；连续调用两次 backward，观察累积。
