import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from log import tprint

class TorchrunEnv:
    def __init__(self):
        tprint("check torchrun env...")
        self.enabled = int(os.environ.get('RANK', -1)) != -1 # is this a torchrun run?
        if not self.enabled:
            tprint("not torchrun env")
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
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
            self.device = f'cuda:{self.local_rank}'
            torch.cuda.set_device(self.device)
            self.device_type = "cuda"
            self.master_process = self.rank == 0 # this process will do logging, checkpointing etc.
            tprint(f"torchrun rank: {self.rank}, local rank: {self.local_rank}, world size: {self.world_size}")

    def model_init(self, model):
        model.to(self.device)
        if self.enabled:
            self.model = FSDP(model)
        else:
            self.model = model

    def get_model(self):
        return self.model

    def barrier(self):
        if self.enabled:
            tprint("wait barrier")
            dist.barrier(device_ids=[self.local_rank])  # 指定当前进程的GPU设备
            tprint("barrier done")