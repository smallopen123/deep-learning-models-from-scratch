# 真实数据验证实验

所有命令从仓库根目录运行。首次视觉实验会下载数据到 `data/`。

## MLP 与 CNN：FashionMNIST

```powershell
python experiments/train_vision.py --model mlp --dataset fashion_mnist --epochs 5 --limit 12000
python experiments/train_vision.py --model cnn --dataset fashion_mnist --epochs 5 --limit 12000
```

`--limit 0` 使用完整训练集。比较两者测试准确率，解释 CNN 为什么利用局部结构。

## ResNet：CIFAR-10

```powershell
python experiments/train_vision.py --model resnet --dataset cifar10 --epochs 10 --limit 20000
```

CPU 可将 limit 降到 5000；GPU 再使用完整数据。

## RNN/LSTM/GRU/Transformer：固定文本数据

```powershell
python experiments/train_text.py --model rnn --epochs 30
python experiments/train_text.py --model lstm --epochs 30
python experiments/train_text.py --model gru --epochs 30
python experiments/train_text.py --model transformer --epochs 30
```

脚本固定保留每类最后 4 条做测试，并输出预测、真实标签和准确率。

## AE/VAE：MNIST 重建

```powershell
python experiments/train_mnist_autoencoder.py --model ae --epochs 5
python experiments/train_mnist_autoencoder.py --model vae --epochs 5
```

查看 `outputs/ae_reconstruction.png` 或 `vae_reconstruction.png`：第一行是原图，第二行是重建。

## GAN：MNIST 生成

```powershell
python experiments/train_mnist_gan.py --epochs 10 --limit 20000
```

逐轮查看 `outputs/gan_epoch_XX.png`。若只出现同一种数字或相似笔画，可能发生 mode collapse。

## 结果记录模板

记录：commit、设备、PyTorch 版本、数据集/limit、随机种子、模型、参数量、epoch、学习率、训练 loss、测试指标、失败样本和结论局限。
