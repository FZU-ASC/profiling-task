# 实际验证记录

首次验证：2026-10-16（Asia/Shanghai）。以下是这次实际运行的结果，不是所有机器上的性能承诺。

## 环境

- WSL2 Ubuntu 24.04.4，Linux `6.18.54.1-microsoft-standard-WSL2`，x86_64。
- CPU：AMD Ryzen 9 7945HX；WSL 提供4个逻辑CPU。
- Python 3.12.3，PyTorch 2.14.1+cpu，NumPy 2.5.2。
- GCC 13.3.0，Rust 1.95.0，flamegraph 0.6.14。
- 最终采样与解析使用 Ubuntu 打包的 perf 6.8.12，采样事件为软件 `cpu-clock`，997Hz，DWARF 调用栈。它在本次 WSL 内核上经过 record/script 与实际 SVG 生成验证。

## 四部分实际运行

| 部分 | 条件 | 结果 |
| --- | --- | --- |
| PyTorch 单文件 | 默认CPU参数，batch64、12个采集步骤 | 生成统计表、summary与trace，trace含1290个事件；输出形状64×64，checksum约24.3090 |
| PyTorch 多文件 | 默认CPU参数，batch16、32×32图像、8个训练步骤及1次评估 | 生成三份结果，trace含22525个事件；训练loss约0.46757，合成数据评估accuracy=0.75 |
| flamegraph 单文件 | size256、80轮 | 实际采集2058条样本记录，生成有效SVG；图中找到main与multiply，后者约占99.81%的采样权重 |
| flamegraph 多文件 | 1600条信号、长度256、24频点、64查询、5轮 | 实际采集610条样本记录，生成有效SVG；找到run_pipeline、run_round、extract_features、spectral_features |

`scripts/smoke_test.py` 已通过：四个程序都实际运行；矩阵结果与独立小规模参考值一致；常量信号、已知余弦DFT与距离检查通过；非法参数会失败；真实PyTorch trace中的多级嵌套范围经过检查。

SVG 中显示的 `samples` 数字可能是事件周期的累计权重，并非 perf 日志中那种样本记录条数；本次 cpu-clock 周期近似一致，比例可用于观察CPU活动分布。

两份可打开的真实火焰图保存在 [examples/](examples/)。原始trace与perf.data保留在本地results目录，没有把包含本机路径或体积较大的采样文件放进Git。

## WSL 上实际遇到的问题

本机原有perf 6.18的精简构建能够record，但script/report报`incompatible file format`；构建信息中ELF和DWARF支持均关闭。

本次从Ubuntu包提取完整perf及缺少的libtraceevent到项目本地`.tools/`，通过`PERF`和`LD_LIBRARY_PATH`指定，没有替换系统perf，也没有修改采样权限。完整perf能够读取先前采样文件，并重新采集、生成两份火焰图。

其他机器不应照搬本机包版本；应先按系统发行版安装合适的工具，再验证采集与读回两条链路。

## 验证边界

- 本地已验证WSL/Linux的CPU路径；未在本机执行CUDA或macOS/xctrace采集。
- 仓库配置了Linux与macOS的GitHub Actions自检，实际状态以仓库Actions为准。自检不宣称已执行需要系统权限的perf/xctrace采样。
- 采集会带来开销，上述事件数、占比、loss和accuracy是一次运行记录，版本、硬件或参数改变后可能不同，不能当作统一验收数值。
