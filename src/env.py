import os
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp import CPUOffload
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from log import tprint
from module import Block

class TorchrunEnv:
    def __init__(self):
        tprint("check torchrun env...")
        self.enabled = int(os.environ.get('RANK', -1)) != -1 # is this a torchrun run?
        if not self.enabled:
            tprint("not torchrun env")
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.local_world_size = 1
            self.num_nodes = 1
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            tprint(f"使用设备: {self.device}")
            self.device_type = self.device
            self.master_process = True
        else:
            assert torch.cuda.is_available(), "for now i think we need CUDA for torch distributed training"
            dist.init_process_group(backend="nccl")
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
            self.num_nodes = self.world_size // self.local_world_size
            self.device = f'cuda:{self.local_rank}'
            torch.cuda.set_device(self.device)
            self.device_type = "cuda"
            self.master_process = self.rank == 0 # this process will do logging, checkpointing etc.
            tprint(f"torchrun rank: {self.rank}, local rank: {self.local_rank}, world size: {self.world_size}")

    # 如果显存不是特别紧张，用这个，速度快
    def model_init_hybrid(self, model):
        if self.enabled:
            device_mesh = init_device_mesh(
                "cuda", 
                mesh_shape=(self.num_nodes, self.local_world_size),  # 单维度网格
                mesh_dim_names=("node", "gpu")          # 数据并行维度
            )
            self.model = FSDP(model,
                              sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                              mixed_precision=MixedPrecision(
                                  param_dtype=torch.bfloat16,
                                  reduce_dtype=torch.bfloat16,
                                  buffer_dtype=torch.bfloat16,
                              ),
                              device_mesh=device_mesh,
                              device_id=self.local_rank,
                              use_orig_params=True,  # 关键改进：支持非连续参数
                              limit_all_gathers=True)
        else:
            self.model = model

        return self.model

    # 如果显存特别紧张，用这个，速度慢
    def model_init_full(self, model):
        if not self.enabled:
            self.model = model
            return self.model

        # 定义自动包装策略，这个对显存节省非常有帮助****，省了近一半显存
        # 但是它跟generate.py中的推理不兼容，跟模型编译也有一些兼容问题。pytorch版本2.5.1
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block}
        )

        self.model = FSDP(model,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    mixed_precision=MixedPrecision(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.bfloat16,
                        buffer_dtype=torch.bfloat16,
                    ),
                    use_orig_params=True,  # 关键改进：支持非连续参数
                    limit_all_gathers=True,
                    cpu_offload=CPUOffload(offload_params=True),
                    auto_wrap_policy=auto_wrap_policy)

        return self.model

    def model_init(self, model, full_shard=False):
        if full_shard:
            return self.model_init_full(model)
        else:
            return self.model_init_hybrid(model)

    def barrier(self):
        if self.enabled:
            tprint("wait barrier")
            dist.barrier(device_ids=[self.local_rank])  # 指定当前进程的GPU设备
            tprint("barrier done")