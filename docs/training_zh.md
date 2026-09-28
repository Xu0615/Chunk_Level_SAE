# Chunk-Level SAE 中文训练指南

[返回项目主页](../README.md) · [English](training.md) · [方法说明](methods.md)

本项目实现 Mean-Chunk、Cross-Chunk 和 Joint-Chunk 三种 chunk-level SAE，并提供 Token BatchTopK、Temporal 基线。完整训练流程为：**流式读取文本 → 采样相邻 chunk → 独立提取激活 → 缓存 → 训练 SAE → 评测**。

## 1. 先选择训练规模

| 用途 | 推荐入口 | 运行条件 |
| :--- | :--- | :--- |
| 理解三个方法、检查安装 | `examples/train_synthetic.py` | CPU，无须下载模型或数据 |
| 跑通真实语言模型上的训练流程 | `configs/quickstart_2b.env` | Linux、CUDA GPU、Qwen3.5-2B 权重、数据集访问 |
| 使用仓库提供的 9B 论文规模配置 | `configs/paper_9b.env` | Linux、8 张 CUDA GPU、大容量激活缓存存储 |

先运行小规模配置，再根据实际显存和磁盘占用扩大实验。模型启动脚本的默认值面向大显存集群；直接运行未覆盖参数的默认脚本会启动大规模任务。

## 2. 安装环境

```bash
git clone https://github.com/Xu0615/Chunk_Level_SAE.git
cd Chunk_Level_SAE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先根据服务器驱动，使用 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 安装对应的 CUDA 版本，再安装项目：

```bash
python -m pip install -e '.[dev]'
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果仅运行 CPU 示例，可先执行 `python -m pip install 'torch>=2.6'`。项目固定使用 `transformers==5.14.1`，因为激活提取器依赖 Qwen3.5 的内部实现。完整 shell 流程使用 Linux 的 `flock`、GNU `date` 和 `readlink`；macOS 可运行 CPU 示例和测试。

Flash Linear Attention 为可选加速依赖，默认不启用。需要时在兼容的 Linux CUDA 环境中安装 `python -m pip install -e '.[cuda]'`，并设置 `CHUNK_SAES_ENABLE_FLA=1`。

## 3. 无下载的 CPU 演示

```bash
python examples/train_synthetic.py --steps 100 --output-dir outputs/synthetic
```

示例使用带有共享潜在因素的合成 chunk 对，分别训练 Mean、Cross 和嵌套式 Joint，并输出验证误差、平均激活特征数，以及可以重新加载的三个 checkpoint。

这一步验证模型接口和保存/加载流程。论文结果仍需使用真实语言模型激活与完整实验配置。

## 4. 下载 backbone

```bash
hf download Qwen/Qwen3.5-2B-Base --local-dir models/Qwen3.5-2B-Base
```

使用完整的本地模型目录，保留 config 和 safetensors 索引。提取器使用这些文件只加载所需的模型层；缺少文件可能触发更大的完整模型加载或直接报错。

## 5. 首次真实训练：Mean + Cross

以下命令均在仓库根目录执行。切换配置文件时使用新的已激活终端，避免上一组环境变量影响下一组实验。

```bash
source configs/quickstart_2b.env

# 如果权重已经放在其他位置，在 source 之后覆盖：
# export MODEL=/absolute/path/to/Qwen3.5-2B-Base
# export PIPELINE_ROOT=/absolute/path/to/your/run
# export CUDA_VISIBLE_DEVICES=1

bash run_scripts/train_qwen35_2b_chunk_saes.sh corpus
bash run_scripts/train_qwen35_2b_chunk_saes.sh cache
bash run_scripts/train_qwen35_2b_chunk_saes.sh train
```

三个阶段分别检查数据流、生成训练/验证激活缓存、训练 SAE。默认语料为固定 revision 的 Pile-Deduplicated，通过 Hugging Face streaming 读取。`corpus` 阶段只检查数据访问，不下载完整语料。

小规模配置如下：

| 参数 | 数值 |
| :--- | :--- |
| GPU 数量 | 1 |
| 模型层，零起始索引 | 16 |
| 训练 / 验证 token occurrence 数量 | 32,768 / 8,192 |
| chunk 长度 | 32、64、128 |
| 字典宽度 / BatchTopK 的 K | 4,096 / 32 |
| 全局 batch size / 更新步数 | 256 / 128 |
| 训练方法 | Mean-Chunk、Cross-Chunk |
| 输出目录 | `outputs/quickstart_2b/` |

小规模配置用于检查完整流程，具体显存需求仍取决于 backbone 和软件环境。

## 6. 使用同一份缓存训练 Joint-Chunk

在上一步配置仍然生效的终端中执行：

```bash
SAE_MODES=joint_chunk JOINT_CROSS_PREFIX=2048 JOINT_CHUNK_ALPHA=0.25 \
RUN_NAME=quickstart_joint_chunk \
bash run_scripts/train_qwen35_2b_chunk_saes.sh train
```

Mean-Chunk 重建当前 chunk 的均值，Cross-Chunk 预测相邻 chunk 的均值。Joint-Chunk 使用同一个编码器和共享字典：完整字典重建当前 chunk，前缀子字典预测相邻 chunk。

`JOINT_CROSS_PREFIX` 必须大于 0 且小于字典宽度。这里使用 `2048/4096`，仓库提供的 9B 嵌套式配置使用 `32768/65536`。**如果 prefix 为 0，代码会选择旧版的两个独立解码器结构。** 训练器的 `--joint-modes` 表示多个方法共享数据加载流程，本身不等同于 Joint-Chunk。

## 7. 论文规模的 9B 配置

在新的已激活终端中执行，并先确认资源充足：

```bash
hf download Qwen/Qwen3.5-9B-Base --local-dir models/Qwen3.5-9B-Base
source configs/paper_9b.env

bash run_scripts/train_qwen35_9b_chunk_saes.sh corpus
bash run_scripts/train_qwen35_9b_chunk_saes.sh cache
bash run_scripts/train_qwen35_9b_chunk_saes.sh train

RUN_NAME=paper_9b_joint_chunk \
bash run_scripts/train_qwen35_9b_joint_chunk_nested.sh
```

主要参数为：Qwen3.5-9B-Base 的第 21 层（零起始索引）、8 张 GPU、10 亿训练 token occurrences、字典宽度 65,536、K=128、global batch size 32,000、31,250 步、学习率 `1e-4`。第一组训练包含 token、temporal、mean、cross，最后单独训练嵌套式 Joint。

上述 Joint 专用脚本固定了 8 卡和对应 batch/步数。小规模 Joint 实验使用上一节的通用 2B 启动方式。

## 8. 修改配置时必须保持的约束

```text
TRAIN_STEPS × GLOBAL_BATCH_SIZE = TARGET_TRAIN_TOKENS
GLOBAL_BATCH_SIZE 可以被 GPU_COUNT 整除
生成缓存时的 world size = 训练时的 world size
```

这里的 token 数量指采样的 **token occurrences**，不表示去重后的 token 总数，也不表示 chunk 个数。chunk 均值在训练中保留对应的 occurrence 权重。只修改步数、GPU 数量或缓存路径，可能破坏协议一致性。

改变模型、层、采样协议或 rank 分区时，应生成匹配的新缓存。每个新实验使用独立 `RUN_NAME`；原实验的重启则保持配置不变。

## 9. 结果、监控与断点恢复

训练结果位于：

```text
${PIPELINE_ROOT}/checkpoints/${RUN_NAME}/${mode}/
```

`mode` 为 `mean`、`cross`、`joint_chunk`、`token` 或 `temporal`。目录包含模型权重、配置、训练指标和完成标记：

- 根目录的 `sae.safetensors`：最终权重。
- `checkpoints/best/`：验证集选出的推理权重。
- `checkpoints/latest/`：包含优化器和训练进度的可恢复 checkpoint。
- `metrics.jsonl`：训练/验证指标；`complete.json`：该模式完成训练的标记。

```bash
tensorboard --logdir "${PIPELINE_ROOT}/runs"

# 使用原配置和 RUN_NAME 恢复中断的 2B 训练：
RESUME=1 bash run_scripts/train_qwen35_2b_chunk_saes.sh train
```

9B 使用对应的 9B 脚本。`best` 权重不能恢复优化器和数据位置，断点恢复需要完整的 `latest` 状态。

## 10. 显存和存储规划

缓存会同时保存 token 激活与 chunk 均值。即使只训练 Mean/Cross/Joint，缓存仍包含 token 激活。仅 token 张量的近似存储量为：

```text
occurrence 数量 × hidden dimension × 每个数值的字节数
```

例如 `10^9 × 4096 × 2 ≈ 8.2 TB`，尚未包含均值、元数据、验证缓存、模型权重和其他副本。请先测量小规模缓存，再规划完整实验的存储空间。

| 问题 | 优先调整 |
| :--- | :--- |
| 提取激活时显存不足 | 减小 `FORWARD_BATCH_SIZE`、`FORWARD_TOKEN_BUDGET` |
| SAE 训练显存不足 | 减小 `LOADER_GPU_SHARDS`；减少同时训练的方法；减小 batch 并相应增加步数 |
| 主机内存占用高 | 减小预取 shard 数量、预取 batch 数量和 tokenizer batch |
| exact coverage 报错 | 核对 occurrence 预算、batch×步数、缓存 world size |
| 输出目录已存在 | 原实验使用 `RESUME=1`，新实验更换 `RUN_NAME` |
| 评测缺少 reference/data | 先按 [评测指南](evaluation.md) 准备相应输入 |

## 11. 评测与发布范围

仓库提供保真度、字典健康、特征一致性、文档检索、分类迁移、推理和可视化工具。完整评测还需要对应 benchmark、证据缓存、部分外部 judge 输出等输入。训练配置默认关闭自动评测，便于先完成训练。

此次发布包含源码、配置、示例、文档和测试，不附带完整激活缓存、预训练 SAE 权重或全部实验产物。`steering` 目录目前尚无完整独立 runner。CPU 检查覆盖模型/缓存协议测试、合成数据训练与 checkpoint 加载；真实 Qwen3.5 的 CUDA 提取和论文规模训练需要在目标硬件上运行验证。
