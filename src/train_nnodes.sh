#! /bin/bash

# 训练前记得手动调用下面脚本，预处理数据集
# python dataloader.py

export OMP_NUM_THREADS=10

# 调试信息，确认是否使用IB网卡
# export NCCL_DEBUG=INFO
#
# 设置一些重要的环境变量。启动IB网卡
# export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
#
# export NCCL_IB_DISABLE=0            # 启用 RDMA
# export NCCL_IB_HCA=mlx5_2           # 指定 RDMA 设备
# export NCCL_IB_GID_INDEX=3          # RoCE v2 需指定 GID 索引
#

torchrun \
--nnodes=2 \
--nproc_per_node=8 \
--node_rank=0 \
--master_addr=127.0.0.1 \
--master_port=29500 \
train.py