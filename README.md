# 中文预训练语言模型

这是一个基于 PyTorch 实现的中文预训练语言模型项目。该项目使用 Fineweb-Edu-Chinese-V2.1 数据集进行训练，支持单机训练和分布式训练。

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (用于 GPU 训练)
- tiktoken
- datasets
- numpy
- pandas

## 安装依赖

```bash
pip install torch datasets tiktoken numpy pandas matplotlib
```

## 模型配置

### 单机版配置 (train.py)
- 批次大小：1
- 序列长度：512
- 词表大小：50257
- 层数：36
- 注意力头数：20
- 嵌入维度：1080

### 分布式版配置 (train_ddp.py)
- 批次大小：4
- 序列长度：1024
- 词表大小：50257
- 层数：46
- 注意力头数：25
- 嵌入维度：1600

## 训练方法

### 1. 单机训练
适用于单个 GPU（4080 Super可以跑起来）：

```bash

# 启动训练
python train.py
```

### 2. 分布式训练
适用于多 GPU 环境：

```bash
# 设置线程数
export OMP_NUM_THREADS=1

# 使用 torchrun 启动分布式训练，这里使用 8 个 GPU
torchrun --nproc_per_node=8 train_ddp.py
```

## 训练监控

- 训练日志保存在 `logs` 目录下
- 模型检查点保存在 `checkpoints` 目录下
- 使用 `plot_training_logs.py` 可以绘制训练曲线：
  ```bash
  python plot_training_logs.py
  ```

## 特性

- 支持自动断点续训
- 定期保存检查点（默认每 30 分钟）
- 训练过程中定期生成示例文本
- 支持验证集评估
- 支持梯度累积
- 训练曲线可视化

## 数据集

使用 Fineweb-Edu-Chinese-V2.1 数据集，仅选择评分在 4-5 分范围内的高质量内容进行训练。数据集通过 Hugging Face datasets 库以流式方式加载，节省内存使用。