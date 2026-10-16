# 第二个任务：给程序做一次 Profile

Hello Everyone！上一个任务里，我们把环境搭起来，也交出了第一个 PR。现在，我们来看看一份程序运行的时候，到底把时间花在了哪里。

程序能跑当然是好事。不过，直接让 agent “帮我优化一下”，总得先知道要优化什么。这一次，我们先把性能分析做明白：**能采集、能读图、能结合源码解释，再谈优化。**

> 所有实验都必须在 **Linux / macOS** 中完成。Windows 同学请进入 WSL2 的 Linux 终端，不要在 PowerShell、CMD 或 Git Bash 中运行实验。GPU 不是必需条件，CPU 就能完成全部基础任务。

## 四部分，两个方向

| 部分 | 实际任务 | 你应该看到的层次 |
| --- | --- | --- |
| [01 · PyTorch 单文件](01-pytorch-single/) | 小型 MLP 前向推理，固定输入 | 推理步骤 → MLP → Linear / ReLU 算子 |
| [02 · PyTorch 多文件](02-pytorch-multi/) | 合成图像预处理、残差 CNN 训练、验证与指标计算 | 训练流水线 → 数据/训练/评估 → 前向/反向/更新 → 主干/残差块 → 算子 |
| [03 · flamegraph 单文件](03-flamegraph-single/) | 朴素稠密矩阵乘法 | main → multiply → 核心循环 |
| [04 · flamegraph 多文件](04-flamegraph-multi/) | 信号数据构造、预处理、时频特征、索引、近邻查询与汇总 | 主流程 → 每轮分析 → 各业务阶段 → 具体数值函数/库调用 |

AI 方向完成 **01 + 02**；优化方向完成 **03 + 04**。都感兴趣的同学可以四部分都做。

**单文件与多文件是不同的计算任务。** 多文件版增加了真正的工作阶段、算法与调用深度，不是把同一份代码拆开。请比较分析视角和调用层次，不要拿两种不同任务的运行时间宣称“多文件更慢/更快”。

## 先把工作台准备好

在 Linux / macOS 终端中克隆仓库并进入目录。WSL 用户建议放在 Linux 自己的目录，例如 `~/projects/`。

```bash
git clone https://github.com/FZU-ASC/profiling-task.git
cd profiling-task
uname -s
```

输出应为 `Linux` 或 `Darwin`。下面的命令均在仓库根目录运行。

### AI 方向

建议使用 Python 3.10+ 和独立虚拟环境。先按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择兼容当前系统的版本，避免把 CUDA 版、CPU 版和不同解释器混在一起。以下是 CPU 环境的参考方式：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Linux：使用 CPU wheels，避免下载不需要的 CUDA 运行依赖。
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# macOS：使用默认 PyPI（与上面一条按系统二选一）。
# python -m pip install -r requirements.txt

python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

示例默认 CPU；macOS 也使用 CPU 完成本次实验。Linux 有可用 CUDA 环境的同学可以用 `--device cuda` 扩展，但需另行按官网安装对应 PyTorch，并确认 profiler 能采到 CUDA kernel。`torch.cuda.is_available()` 为真不等于设备采集一定可用。

### 优化方向

需要支持 C++17 的 GCC / Clang、make、Rust stable 和 flamegraph。

```bash
c++ --version
make --version
cargo --version
cargo install flamegraph --locked
flamegraph --version
make all
```

Rust 安装见 [rustup](https://rustup.rs/)。Linux 的 flamegraph 依赖 **perf**；macOS 依赖 **xctrace / Xcode**，按 [flamegraph 官方 README](https://github.com/flamegraph-rs/flamegraph) 配置。macOS 可以先检查 `xcrun xctrace version`；只装 Apple Command Line Tools 不保证已具备 xctrace。

WSL 的 perf 必须适配实际 Linux 内核。遇到 `linux-tools-$(uname -r)` 没有安装候选时，不要不断重复同一条 apt 命令；带着 `uname -r` 和报错请 agent 排查。公共环境中的采样权限请联系管理员，不要自行放宽系统限制。

如果 perf 能 record 却在 script/report 阶段报 `incompatible file format`，检查 `perf version --build-options`。过度精简、缺少 ELF/栈展开支持的构建可能无法处理结果，应换用具备所需支持的 perf，再验证能否采集和读回；不要只检查版本号。临时使用另一份 perf 时可用环境变量 `PERF=/实际路径/perf` 指定，flamegraph 与采集脚本会沿用它。

采集脚本优先查找项目本地的 `.tools/bin`（若存在），再使用系统 PATH；也可用 `FLAMEGRAPH=/实际路径/flamegraph` 指定工具。这些本地工具不提交到仓库，克隆后仍需按上述说明安装自己的依赖。

## OK，Let's begin！

### 先做一个小规模自检

配置好两条路线的依赖后，运行：

```bash
source .venv/bin/activate
python scripts/smoke_test.py
```

这会实际编译并运行两个 C++ 程序、检查已知数值结果、运行两份 PyTorch 代码，并检查生成 trace 中的嵌套区间。自检规模刻意很小，不适合拿来制作最终火焰图；它也不代替 perf/xctrace 的采集检查。

### AI 路线：从简单推理走向完整训练流程

```bash
python 01-pytorch-single/profile_single.py
python 02-pytorch-multi/main.py
```

分别打开 `results/pytorch-single/` 与 `results/pytorch-multi/`：

- `trace.json`：用 [Perfetto UI](https://ui.perfetto.dev/) 的 Open trace file 打开。
- `operators.txt`：按 Self CPU 时间排序的算子统计。
- `summary.json`：参数、环境、数值结果与采集耗时。

单文件版先找到 `inference_step → mlp_forward`；多文件版再找到 `training_pipeline → train_step → forward → backbone → residual_block_1 → conv_path`，并分别查看 `backward`、`optimizer_step` 和 `evaluation`。每个子目录的 README 会带你回到实际源码。

多文件数据是内存中的合成图像，`num_workers=0`，目的是先看清当前进程的层次。它没有测磁盘下载，也没有测真实多进程 DataLoader；别从中推出“所有数据加载都很快”。合成数据的评估准确率只用于检查流程，不代表真实任务的模型效果。

### 优化路线：从一个热点走向完整调用树

```bash
bash scripts/profile_cpp.sh single
bash scripts/profile_cpp.sh multi
```

分别用浏览器打开 `results/flamegraph-single/flamegraph.svg` 和 `results/flamegraph-multi/flamegraph.svg`。每个结果目录也保存程序输出和环境记录；Linux 下保留 `perf.data` 供进一步分析。

Linux 脚本采用 `cpu-clock` 软件事件和 DWARF 调用栈，便于没有硬件 PMU 的 WSL 使用。这个事件分析 CPU 上的活动，不是 cache miss 或 GPU 性能。macOS 走工具默认的 xctrace 后端。

多文件版优先找 `run_pipeline → run_round → extract_features → spectral_features` 这条路径，再观察其他阶段。采样是统计性的；很短的函数可能采不到，编译器也可能改变可见栈，不能要求图里机械地出现每一行源码。若多数都是 `[unknown]` 或根本没有有效采样，应先修好采集。

> 火焰图不是时间线：横向宽度表示样本占比，纵向表示调用深度。默认颜色不表示“越红越慢”。父框包含子调用，不能把整棵树的宽度或时间简单相加。

## 给自己的猜测找一点证据

每个所选方向都需要完成单文件和多文件两个实验，并在其中至少一个实验上做一次**只改变一个条件**的对照：

- AI 单文件：改变 `--batch-size`；多文件：改变 `--image-size` 或 `--batch-size`，每次只改一个。
- flamegraph 单文件：改变 `--size`；多文件：改变 `--bins` 或 `--records`，每次只改一个。

换一个输出目录保存第二次结果，避免覆盖第一次。例如：

```bash
python 02-pytorch-multi/main.py --batch-size 32 --output results/pytorch-multi-batch32
OUTPUT_DIR="$PWD/results/flamegraph-multi-bins12" bash scripts/profile_cpp.sh multi --bins 12
```

输入较小时火焰图可能样本不足，可以先增大 `--rounds`，再固定它做实验。调整后把所有实际参数记下来，其他参数和构建方式保持一致。

你可以提出这样的猜测：“频域特征的计算量随 bins 增大而增加”，然后去查代码和采集结果。图中某阶段占比变小，不代表绝对时间也变小；不同工作负载的图也不能直接比速度。

**本阶段不要求优化成功。** 如果尝试优化，必须保留正确性检查，并在关闭 profiler 后独立、重复测量性能。`profiled_wall_seconds` 和采集过程中的 `wall_seconds` 都带有采集开销，不能直接当成最终 benchmark。

## 最后，把你的发现讲给别人听

将代码改动、运行命令和分析记录放进自己的学习仓库。可复制 [报告提纲](REPORT_TEMPLATE.md)，每个实验一小节：

1. 你跑了什么：环境、代码版本、参数、构建方式和采集命令。
2. 你看到了什么：主要热点、具体指标、原始 trace / SVG 和对应源码位置。
3. 多文件版比单文件版多出了哪些实际阶段？选一条嵌套路径解释各层职责。
4. 对照实验改了什么，观察到什么，哪些解释仍然只是猜测。

原始文件较大时可以提供可访问的下载链接，不必把 `perf.data` 等大文件直接塞进 Git。提交到团队的具体位置与时间以后续通知为准，不要往第一个任务的 `PR-test` 仓库里塞这次的全部代码和采集文件。

- [ ] 我在 Linux / macOS 中完成了实验，并记录了具体环境。
- [ ] 我完成了所选路线的单文件与多文件两部分，原始结果可以打开。
- [ ] 我能结合源码解释多文件任务中的一条实际嵌套调用路径。
- [ ] 我做了一次只改变一个条件的对照实验。
- [ ] 另一位同学可以按我的记录复现。

到这里，我们开始从“让程序跑起来”，走向“理解程序怎样跑”。先把你的第一份性能分析做出来。See you in the profile！~

## 工具参考与验证范围

- [PyTorch Profiler 教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)、[API](https://docs.pytorch.org/docs/stable/profiler.html)。CPU total 包含子调用；Self CPU 排名不能替代 GPU 排名。模型和设备预热在正式采集前完成。
- [flamegraph 官方仓库](https://github.com/flamegraph-rs/flamegraph)。默认 `-O2 -g`、保留帧指针并禁用尾调用消除；关键教学函数保留调用边界，便于读图。这些构建设置也属于实验条件。
- 四部分的 WSL 实际验证记录见 [VALIDATION.md](VALIDATION.md)。GitHub Actions 配置了 Linux/macOS 数值与 trace 自检；perf/xctrace 采样在具备权限的实际系统上单独验证。
