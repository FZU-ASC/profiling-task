# 03 · 用一个文件，找到矩阵乘法的热点

这部分只有 `main.cpp`：初始化两份稠密矩阵，执行朴素的三重循环矩阵乘法，再汇总结果，防止计算被优化器当成无用工作移除。

在仓库根目录的 Linux/macOS 终端运行：

```bash
make build/matrix_demo
./build/matrix_demo --size 256 --rounds 80
bash scripts/profile_cpp.sh single --size 256 --rounds 80
```

用浏览器打开 `results/flamegraph-single/flamegraph.svg`，找到 `main → multiply`。核心循环在 `multiply` 内，不会为每个 for 循环单独生成一层调用框。`matrix_checksum` 很短，采样未必恰好采到它。

程序输出 JSON，包含 size、rounds、checksum、wall_seconds。固定 size 时，rounds 翻倍，校验和也应近似翻倍；浮点结果比较应留出容差。改变 size 会改变输入与计算量，不能要求校验和保持相同。

对照实验可以只改变 size，另存结果：

```bash
OUTPUT_DIR="$PWD/results/flamegraph-single-size384" bash scripts/profile_cpp.sh single --size 384 --rounds 80
```

请解释：为何 `multiply` 很可能较宽？`--size` 翻倍后，理论计算量怎样变化？图中样本占比能否直接告诉你最终运行速度？先回答，再决定是否尝试改变循环顺序。
