#! /bin/bash

export OMP_NUM_THREADS=10

torchrun \
--nnodes=2 \
--nproc_per_node=8 \
--node_rank=0 \
--master_addr=127.0.0.1 \
--master_port=29500 \
train.py