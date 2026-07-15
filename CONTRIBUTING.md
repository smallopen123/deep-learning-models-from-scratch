# 贡献指南

新增模型必须同时提供：原理、输入输出 shape、注释实现、最小运行命令、常见错误、练习和至少一个 forward/backward 测试。示例默认应能在 CPU 和小型合成数据上运行；大型数据集作为可选扩展。

提交前运行：

```powershell
python -m compileall -q dl_models examples tests
python -m unittest discover -s tests -v
```
