# 04 · 分析一条跨模块的信号处理流水线

从矩阵乘法继续往前走。这次是另一项更复杂的工作：生成带噪声的信号，归一化、平滑，提取时域与频域特征，建索引、做近邻查询，最后汇总结果。

| 源文件 | 实际工作 |
| --- | --- |
| `src/main.cpp` | 解析参数、计时、输出 JSON |
| `src/dataset.cpp` | 固定种子生成多条带噪声信号 |
| `src/preprocess.cpp` | 标准化与五点滑动平均 |
| `src/features.cpp` | 时域能量/均值，以及直接 DFT 的部分频点功率 |
| `src/index.cpp` | 按能量排序建索引，取局部候选，计算特征距离 |
| `src/pipeline.cpp` | 组织各阶段、重复分析、汇总结果 |
| `include/pipeline.hpp` | 参数、类型与跨文件函数声明 |

索引查询是教学用的**近似邻域检索**：按能量找位置，再在左右最多32个候选中比较特征距离，不保证全局最近邻。频谱采用直接 DFT，而非高效 FFT，目的是提供可分析的数值热点；不要把它当成生产算法的最佳实现。

在仓库根目录运行：

```bash
make build/signal_pipeline
./build/signal_pipeline
bash scripts/profile_cpp.sh multi
```

默认1600条信号、每条256个采样点、24个频点、64次查询、5轮分析。原始数据只构造一次，每轮重新执行预处理、特征提取、建索引和查询。

源代码的实际调用关系：

```text
main
└── run_pipeline
    ├── generate_dataset
    └── run_round
        ├── preprocess_records
        │   ├── normalize_signal
        │   └── smooth_signal
        ├── extract_features
        │   ├── time_features
        │   └── spectral_features → sin/cos 等库实现
        ├── build_index → sort 相关实现
        ├── query_batch
        │   └── query_neighborhood
        │       └── squared_distance
        └── aggregate_results
```

这张树用于定位代码，不保证每个短函数都能出现在统计采样图中。请在真实 SVG 中重点寻找 `run_pipeline → run_round → extract_features → spectral_features`，并检查是否能够看到更深的数值库调用。

```bash
OUTPUT_DIR="$PWD/results/flamegraph-multi-bins12" bash scripts/profile_cpp.sh multi --bins 12
```

只改 bins，其他默认条件保持一致。为什么它主要影响频域计算？如果时域阶段的占比变大，能否直接推断它变慢？再考虑只改 records，哪些阶段也会一起受到影响？

正确性检查见 `tests/numeric_test.cpp`：包含常量信号的归一化/平滑、已知余弦信号的 DFT 功率和已知距离。修改算法后先运行自检；不能通过删掉计算阶段“优化”掉任务。
