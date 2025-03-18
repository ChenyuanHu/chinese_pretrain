# 中文预训练模型项目

## 项目简介

这是一个中文语言模型预训练项目，基于Transformer架构，支持分布式训练和混合精度计算。该项目实现了一个高效的中文预训练和微调框架，具有以下特点：

- 支持多节点分布式训练
- 支持FlashAttention加速
- 支持梯度累积
- 支持检查点保存和恢复
- 实现了RoPE位置编码，并支持位置编码缩放
- 支持预训练和微调数据处理
- 高效的数据加载和处理

## 环境要求

```
urllib3<2.0.0
transformers>=4.36.0
tiktoken>=0.5.0
torch>=2.0.0
matplotlib==3.8.3
datasets>=3.3.2
flash-attn
```

## 项目结构

```
chinese_pretrain/
├── src/                       # 源代码目录
│   ├── configs/               # 配置文件
│   │   ├── h20x8_config.py    # H20x8 GPU配置
│   │   └── rtx4080s_config.py # RTX4080S GPU配置
│   ├── experiments/           # 实验相关目录
│   ├── dataset_cache/         # 数据集缓存
│   ├── checkpoint.py          # 检查点管理
│   ├── config.py              # 配置导入
│   ├── dataloader.py          # 数据加载器
│   ├── env.py                 # 训练环境
│   ├── eval.py                # 评估
│   ├── generate.py            # 文本生成
│   ├── log.py                 # 日志工具
│   ├── module.py              # 模型定义
│   ├── rope.py                # RoPE位置编码
│   ├── tokenizer.py           # 分词器
│   ├── train.py               # 训练入口
│   ├── train.sh               # 单节点训练脚本
│   └── train_nnodes.sh        # 多节点训练脚本
├── tools/                     # 工具脚本
├── .gitignore                 # Git忽略文件
└── requirements.txt           # 依赖包列表
```

## 核心组件

1. **模型架构** (module.py)：
   - 实现了基于Transformer的语言模型
   - 支持GQA (Grouped Query Attention)
   - 集成FlashAttention加速
   - 支持梯度检查点降低显存占用

2. **分布式训练** (env.py)：
   - 支持PyTorch FSDP (Fully Sharded Data Parallel)
   - 支持多节点训练
   - 支持混合精度训练

3. **位置编码** (rope.py)：
   - 实现RoPE (Rotary Position Embedding)
   - 支持位置编码缩放，提高长文本处理能力

4. **数据处理** (dataloader.py)：
   - 支持HuggingFace datasets格式数据集
   - 数据预处理和缓存
   - 分布式数据加载

5. **检查点管理** (checkpoint.py)：
   - 支持普通检查点保存和恢复
   - 支持分布式检查点 (DCP) 保存和恢复

6. **分词器** (tokenizer.py)：
   - 基于预训练中文分词器
   - 添加了特殊标记支持对话格式

## 使用说明

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据预处理

数据预处理需要在训练前完成，使用下列命令处理：

```bash
cd src
python dataloader.py
```

### 修改配置

在 `src/configs/` 目录下可以找到不同硬件的配置文件，选择适合自己设备性能的配置，可以根据需要修改以下参数：

- `TrainConfig`：训练参数，如批次大小、梯度累积步数等
- `ModuleConfig`：模型参数，如层数、注意力头数等
- `PretrainConfig` 和 `SftConfig`：数据集配置


### 训练

1. 单节点单GPU训练：

```bash
cd src
python train.py
```

2. 单节点多GPU训练：
```bash
cd src
./train_8gpus.sh
```

3. 多节点训练：

```bash
cd src
./train_nnodes.sh
```

## 训练结果
训练的checkpoint以及日志都放在experiments目录下

## 特性说明

1. **预训练与微调支持**：
   - 支持大规模无监督预训练
   - 支持SFT对话数据微调

2. **高效数据处理**：
   - 使用内存映射高效读取大数据集
   - 多进程数据预处理

3. **资源优化**：
   - 梯度检查点降低显存占用
   - 混合精度训练提高训练速度
   - FlashAttention减少内存占用并提高速度

4. **训练稳定性**：
   - 定时保存检查点
   - 多种检查点格式支持
   - 训练恢复功能 