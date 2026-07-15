# 安装与运行

## 1. 创建独立环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS 激活命令：`source .venv/bin/activate`。

## 2. 检查

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

全部示例默认自动选择 CUDA 或 CPU；CPU 足够完成教学示例。GPU 用户应通过 PyTorch 官方安装选择器获得与驱动匹配的命令，不要复制旧教程的 CUDA 版本。

## 3. 模块找不到

必须在仓库根目录执行命令。确认终端当前目录包含 `README.md`、`dl_models` 和 `examples`。如果 `import dl_models` 失败，先执行 `pwd` 或 `Get-Location` 检查路径。

## 4. 训练输出怎么看

`loss` 应总体下降，但不要求每批单调；`test_accuracy` 是未用于参数更新的数据结果。先尝试过拟合很小的数据，再增加数据和模型复杂度。

## 5. 固定随机性

示例使用 `torch.manual_seed(42)`。完全可复现还会受到设备、算子和版本影响，因此实验报告必须记录 Python、PyTorch、硬件和数据版本。
