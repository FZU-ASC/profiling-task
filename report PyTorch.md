# 实验环境
-OS：macOS
-CPU:Apple 系列
-软件环境：PyTorch

# 单文件实验
- 从Perfetto图中可知，aten: : addmm阶段duration久，占绝大多数时间
- 以下是 将batch-size-64 修改为 batch-size-32 的operators.txt对比图
<img width="1335" height="433" alt="截屏2026-10-20 23 06 44" src="https://github.com/user-attachments/assets/3305c8df-e6e7-4b88-9413-6fe3eca8853c" />
- 由图中数据📊
- 当batch-size减小后，aten: : addmm耗时有所减少，即矩阵乘法变快了（total time从437.699us减少到326.213us）；
- but'forward'耗时并未减少，甚至反而增多。（对此解释，求deepsseek帮助：“这证明了在 AI 性能调优中，盲目减小 batch-size 并不一定能带来等比例的端到端加速，需要权衡“计算量减少”与“固定开销占比上升”之间的矛盾。“）
# conclusion
当其他条件不变时，将batch-size减小，即批次减小，单次计算量减小，aten: : addmm即矩阵乘法耗时减少。

# 多文件实验
- 从Perfetto图中可知，沿着training_pipeline - train_step - forward- backbone - residual_block_1 -conv_path等观察可知，aten:: convolution_backward耗时久，且backward的1耗时竟是forward多超出一倍
- 1. 以下是 将batch-size-16 修改为 batch-size-32 的operators.txt对比图
<img width="1591" height="699" alt="截屏2026-10-20 23 25 36" src="https://github.com/user-attachments/assets/b459e724-6909-47ed-aed2-116922a6acd0" />
- 由图中数据📊
- aten:: convolution_backward的time avg&&total time增大了，即反向传播的计算量增大，耗时变大；
aten:: select从2080次变成4160次，即同batch-size一样翻倍了，一次处理数据变多了。
# conclusion1
当其他条件不变时，将batch-size增大，即批次增大，单次处理数据增大，aten:: convolution_backward（卷积反向传播）的耗时增大；select次数按比例增大。
- 2.以下是 将image-size-32 修改为 image-size-64 的operators.txt对比图
<img width="1658" height="621" alt="截屏2026-10-20 23 45 25" src="https://github.com/user-attachments/assets/dbe7c12c-958f-4496-8ca1-e4032fae4b10" />
- 由图中数据📊
- aten:: convolution_backward的平均单次耗时明显增大（从1.877ms变成6.25ms)；
'backward'反向传播总耗时增多（从80.18ms变成264.19ms)。
# conclusion2
当其他条件不变时，当图像尺寸增大，像素面积扩大，反向卷积计算量增大，耗时增大。



  
