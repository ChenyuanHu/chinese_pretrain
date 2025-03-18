#! /bin/bash

export OMP_NUM_THREADS=1
# 预处理数据集
python dataloader.py

export OMP_NUM_THREADS=10

# 使用torchrun启动分布式训练
torchrun --nproc_per_node=8 train.py 