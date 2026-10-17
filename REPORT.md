# 我的第二个任务：性能分析记录

## 环境与复现

- 系统、内核、CPU / GPU：Ubuntu 22.04.1 LTS on WSL2；内核 6.18.54.1-microsoft-standard-WSL2；Intel(R) Core(TM) Ultra 9 185H（22 逻辑核）；机器配备 RTX 4060 Laptop 8G，但本次全程使用 CPU-only 环境（PyTorch CPU 版，`torch.cuda.is_available()` 为 False）
- 代码 commit：a856d82
- Python / PyTorch / 编译器 / flamegraph / perf 或 xctrace 版本：Python 3.10.6；PyTorch 2.14.1+cpu；numpy 2.2.6；flamegraph / perf / xctrace 不适用（AI 方向，本次未使用采样工具）
- 编译选项、运行参数、实际采样事件：无编译（纯 Python）；单文件默认 `--batch-size 64 --steps 12 --warmup 4 --threads 2 --device cpu`，多文件默认 `--batch-size 16 --image-size 32 --steps 8 --warmup 2 --threads 2 --device cpu`；采集事件均为 PyTorch Profiler CPU activity（`record_shapes=True`，`with_stack=True`）
- 完整命令：

  ```bash
  cd ~/profiling-task
  source .venv/bin/activate
  python 01-pytorch-single/profile_single.py
  python 02-pytorch-multi/main.py
  ```

## 单文件任务

- 运行目标与采集范围：固定的两层 MLP 前向推理。输入是 `torch.randn(64, 256)`，仅在程序启动时生成一次，12 次正式采集复用同一个张量。采集范围只覆盖 `with profile(...)` 上下文内的 12 次前向；**不包括**模型构建、输入生成和 4 次预热（`profile_single.py` 第 46-47 行），预热用于让线程池与内存分配器就绪。本次是纯前向推理：`inference_mode()` 关闭了 autograd，无反向传播、无参数更新；无数据加载；仅 CPU activity，无 CUDA kernel。
- 原始 trace / SVG、统计表、程序输出的链接：均在本地 `results/pytorch-single/` 目录下（该目录被 `.gitignore` 忽略，未随仓库提交）——`trace.json`（Chrome trace）、`operators.txt`（按 Self CPU 排序的算子统计）、`summary.json`（运行参数、环境与校验和）。
- 主要热点与实际指标：按 Self CPU 排序，`aten::addmm` 占 **70.40%**，12 步共调用 24 次，Self CPU 合计 1.822 ms，CPU total 2.004 ms，是唯一显著热点。其余算子占比均不足 6%：`aten::clamp_min`（即 ReLU）5.44%、`aten::copy_` 5.17%、`aten::t` 2.97%、`aten::transpose` 1.90%、`aten::linear` 1.78%。热点高度集中在矩阵乘上。
- 对应源码与解释：`aten::addmm` 对应 `01-pytorch-single/profile_single.py` 第 36 行的两个 `nn.Linear`；`nn.ReLU()` 在底层实现为 `aten::clamp_min`（即 `clamp_min(x, 0)`），因此在 trace 中找不到名为 `relu` 的条目。24 次 `addmm` 按 trace 中的 `Input Dims` 可干净地分成两组：
  - `[64, 256] @ [256, 512]`——对应第一个 `Linear(256, 512)`，每步 1 次，单次均值约 0.127 ms，12 步合计 1.523 ms；
  - `[64, 512] @ [512, 64]`——对应第二个 `Linear(512, 64)`，单次均值约 0.040 ms，合计 0.481 ms。

  **第一层约占 addmm 总时间的 76%**，因为其乘加规模（256×512 = 131072）是第二层（512×64 = 32768）的 4 倍。因此值得重点关注的是第一层 Linear 的矩阵乘。

## 多文件任务

- 与单文件任务有什么实际不同：这是**不同的计算任务**，不是把单文件代码拆开。单文件是固定输入、无状态的纯前向推理，只有 `inference_step → mlp_forward → 算子` 三层；多文件是完整的图像训练流水线，包含合成数据取 batch、预处理（增强 + 归一化）、残差 CNN 前向、交叉熵、反向传播、Adam 参数更新，以及末尾一次评估。调用层级从 3 层加深到 6 层以上，且引入了反向传播与参数更新这两类单文件完全没有的计算。实测 `training_pipeline` 总时长 90.614 ms。前向、反向、更新三者在 `train_step` 中的占比分别为 **37.3%、51.0%、8.3%**，即**反向传播是最大开销**——这与单文件「热点全在矩阵乘」的结论完全不同。
- 一条真实的嵌套调用路径，以及每层的文件/函数：`training_pipeline → train_step → forward → backbone → residual_block_1 → conv_path`（trace 中验证为逐层包含，嵌套深度依次为 5 / 10 / 14 / 21 / 31 / 35）。对应源码：`training_pipeline` 与 `train_step` 在 `02-pytorch-multi/main.py`（组织采集）与 `engine.py`（训练步骤）；`forward`、`backbone`、`residual_block_1`、`conv_path` 在 `model.py`。同理 `backward`、`optimizer_step`、`loss` 位于 `engine.py`，数据侧的 `preprocess → augmentation / normalize`、`device_transfer` 位于 `data.py`。
- 原始结果链接、主要热点与实际指标：本地 `results/pytorch-multi/`（同样被 `.gitignore` 忽略）——`trace.json`、`operators.txt`、`summary.json`。Self CPU 前三位为 `aten::convolution_backward` **33.66%**（40 次，25.602 ms）、`aten::mkldnn_convolution` 19.42%（45 次，14.772 ms）、`aten::native_batch_norm_backward` 4.87%。按阶段统计：`backward` 合计 42.837 ms（占 pipeline 47.3%）、`forward` 31.295 ms、`optimizer_step` 6.984 ms、`evaluation` 仅 2.633 ms。模型参数量 9860，末次训练 loss 0.4676，评估准确率 0.75（仅用于确认流程跑通）。
- 已观察到的事实与尚未验证的推测：
  - **事实**：反向相关算子（`convolution_backward` + `native_batch_norm_backward` + `threshold_backward` + `avg_pool2d_backward`）合计占 Self CPU 的 45% 以上，反向整体时长约为前向的 1.37 倍；`residual_block_1` 耗时（18.382 ms）是 `residual_block_2`（7.285 ms）的约 2.5 倍，开销明显集中在第一个残差块；`preprocess` 共出现 9 次，其中 8 次在训练步骤内、1 次在评估内，说明训练与评估共用同一套预处理。
  - **推测（未验证）**：`residual_block_1` 更慢可能与其处于更高分辨率特征图有关，但本次未采集各层张量形状，也无逐层输出尺寸对照，**不能确认**。`optimizer_step` 仅占 8.3% 是 CPU 上的结果，在 GPU 上可能因 kernel 启动与同步方式不同而差异很大，**未验证**。
- 同样的模型在训练和评估中出现两次（`forward` 8 次、`eval_forward` 1 次），是因为训练时参数在更新（需 `zero_grad` 与反向），而评估时只做前向且不累积梯度；两者共用模型结构但执行模式不同。

## 一次对照实验

- 我只改变了哪个条件，为什么：按 README 要求，AI 单文件方向通过改变 `--batch-size` 做对照。**只改了 batch size**（64 → 128），`--steps`、`--warmup`、`--threads`、`--device`、随机种子（固定为 42）以及代码版本均保持不变。选择该变量是因为它直接改变每次前向的输入规模，而不会改动模型结构或执行路径。
- 两次运行的完整参数与输出位置：
  - 基线：`--batch-size 64 --steps 12 --warmup 4 --threads 2 --device cpu` → `results/pytorch-single`（输出形状 `[64, 64]`，checksum 24.308988571166992）
  - 对照：`--batch-size 128 --steps 12 --warmup 4 --threads 2 --device cpu` → `results/pytorch-single-batch128`（输出形状 `[128, 64]`，checksum 6.746490478515625）
- 实际结果与解释：
  - **调用次数不变**：两次均为 24 次 `addmm`（每步 2 次）和 12 次 `clamp_min`（每步 1 次）。batch size 只改变单个张量的行数，不改变模型层数和每步的算子数量。
  - **输入规模变化**：两个 `addmm` 的第一个输入从 `[64, 256]` / `[64, 512]` 变为 `[128, 256]` / `[128, 512]`，即 batch 维翻倍；权重形状 `[256, 512]` 与 `[512, 64]` 不变。因此计算量理论上是原来的 2 倍。
  - **单次耗时显著增加，但并非线性**：第一层 `addmm` 单次均值 0.127 ms → 0.633 ms（约 5.0 倍），第二层 0.040 ms → 0.140 ms（约 3.5 倍），Self CPU 总和 2.004 ms → 9.267 ms。计算量翻倍而耗时增长超过 2 倍，说明在本环境下小规模矩阵乘的耗时并不随 batch 线性增长，可能受缓存层级、线程调度与固定开销等因素影响；**具体原因本次采集无法确定，尚未验证**。
  - **ReLU 的相对占比明显上升**：`aten::clamp_min` 单次均值 0.012 ms → 0.157 ms（约 13 倍），在 Self CPU 中的占比由 5.44% 升至 16.37%。这是本次对照中最反直觉的观察，其原因同样未能从当前数据中得到解释。
  - **热点排序未变**：两次运行中 `aten::addmm` 始终是第一热点（70.40% → 68.02%），第一层 Linear 仍然占据主要的 `addmm` 时间（76% → 约 82%）。
- 还有什么不能从当前采集中得出：
  - 两个实验都是固定流程与固定规模的合成数据，**不能据此推断真实数据集、真实训练收敛过程或线上服务的性能**。
  - 耗时数据带有 profiler 开销（`record_shapes=True`、`with_stack=True` 会显著增加开销），`summary.json` 中的 `profiled_wall_seconds`（单文件 0.644 s → 1.230 s）**不能当作性能基准（benchmark）**。
  - 多文件实验使用内存中的合成图像且 `num_workers=0`，**没有测量磁盘或网络 I/O，也没有测量多进程 DataLoader**，不能据此得出「所有数据加载都很快」的结论；`eval_accuracy`（0.75）只用于检查流程跑通，不代表真实任务的模型效果。
  - 使用 CPU-only 环境，**无法反映该机器 RTX 4060 GPU 上的性能特征**；CPU total 包含子调用、父子区间会重复计数，Self CPU 排名不等同于 GPU 上的排名。
  - 未验证耗时随 batch 增长非线性的原因，也未确认该现象在不同线程数、不同 PyTorch 版本下是否重现。

## 下一步

- 我还想验证什么：
  - 在关闭 profiler 的条件下，对 batch 64 与 128 做重复计时，验证「耗时非线性增长」是否只是 profiler 开销造成的假象。
  - 通过改变 `--image-size`（如 32 → 48）观察卷积类算子的计算量变化，并采集各层张量形状以验证「`residual_block_1` 更慢是因为分辨率更高」这一推测。
  - 在 CPU-only 之外，尝试在 RTX 4060 上使用 `--device cuda`，对比 GPU 与 CPU 的算子分布与热点排序差异（需先安装 CUDA 版 PyTorch 并确认 profiler 能采到 CUDA kernel）。
- 如尝试优化：如何确认正确，以及如何在关闭 profiler 后独立测量性能：本阶段不要求优化成功。若后续尝试优化，将保留现有 `checksum` / `output_shape`（单文件）与 `last_train_loss` / `eval_accuracy`（多文件）作为正确性校验，并在**关闭 profiler** 的前提下，对每个版本独立重复多次计时（至少 5 次取中位数），同时固定线程数、batch size、图像尺寸与随机种子，避免把采集开销混入性能比较。
