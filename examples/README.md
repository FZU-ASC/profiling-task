# 两份实际生成的火焰图

在WSL/Linux中使用默认程序参数、flamegraph 0.6.14和CPU-clock采样生成，具体环境见[验证记录](../VALIDATION.md)。

- [单文件矩阵乘法](flamegraph-single.svg)：重点找`main → multiply`。
- [多文件信号流水线](flamegraph-multi.svg)：重点找`run_pipeline → run_round → extract_features → spectral_features`。

GitHub可能将SVG作为静态图片展示；下载后用浏览器打开，可以点击放大调用路径。它们用于示范如何阅读，不是所有机器都要复现的固定样本比例。
