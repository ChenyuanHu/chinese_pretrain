import torch
import torch.optim as optim
import torch.distributed as dist
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from env import TorchrunEnv
from module import MyModule
from dataloader import MixTrainDataLoader
from generate import TextGenerator
from checkpoint import CheckpointManager
from config import TrainConfig, ModuleConfig, TrainDataConfig
from log import tprint

class ChatBot:
    def __init__(self, train_config, module_config, train_data_config):
        self.env = TorchrunEnv()
        tprint(f"torchrun环境初始化完成")
        model = MyModule(module_config)
        tprint(f"模型初始化完成")
        model.to(self.env.device)
        tprint(f"模型移动到设备完成")
        model = self.env.model_init(model, full_shard=True if hasattr(train_config, "full_shard") and train_config.full_shard else False)
        tprint(f"模型分布式训练环境初始化完成")
        if train_config.compile:
            model = torch.compile(model)
            tprint(f"模型编译完成")
        self.model = model
        # 使用32位的AdamW优化器，设置betas和权重衰减
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=1e-4,  # 最大学习率
            betas=(0.9, 0.95),  # beta1和beta2
            weight_decay=0.1,  # 权重衰减
            eps=1e-8,
            foreach=True
        )
        # 创建学习率调度器，从预热到最大学习率1e-4，最终降至3e-5
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-4,
            total_steps=train_config.scheduler_epochs * train_config.steps_per_epoch,
            pct_start=0.05,  # 预热阶段占总步数的5%
            final_div_factor=3,  # 确保最终学习率为3e-5 (1e-4/3)
            div_factor=3,  # 起始学习率为max_lr/3
            anneal_strategy='cos'  # 余弦退火
        )
        tprint(f"优化器初始化完成")

        self.data_loader = MixTrainDataLoader(self.env.world_size, self.env.rank, self.env.local_rank, train_config.batch_size, module_config.block_size)
        tprint(f"数据加载器初始化完成")
        # self.evaluate_runner = EvaluateRunner(self.data_loader, train_config.batch_size)
        # tprint(f"评估器初始化完成")

        self.checkpoint_manager = CheckpointManager(self.env, train_config)
        tprint(f"检查点管理器初始化完成")
        self.train_config = train_config
        self.module_config = module_config

        assert self.module_config.dtype in {"float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"}, f"dtype must be float32, float16, bfloat16 or float8_e4m3fn or float8_e5m2"
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float8_e4m3fn': torch.float8_e4m3fn, 'float8_e5m2': torch.float8_e5m2}[self.module_config.dtype]
        
        self.amp = torch.amp.autocast(device_type=self.env.device_type, dtype=ptdtype)
        self.amp_scaler = torch.amp.GradScaler(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=(self.env.device_type != 'cpu' and self.env.device_type != 'mps')
        )

        self.text_generator = TextGenerator(self.model, module_config.block_size, train_data_config, device=self.env.device, amp=self.amp)
        tprint(f"文本生成器初始化完成")

        # 计算并打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        tprint(f"模型总参数量(当前GPU): {total_params:,}")
        tprint(f"可训练参数量(当前GPU): {trainable_params:,}")
        tprint(f"模型总大小(当前GPU): {total_params * 4 / (1024**2):.2f} MB")  # 假设每个参数是4字节（float32）
        
    def chat(self):
        start_epoch, progress_percentage = self.checkpoint_manager.try_load_checkpoint(self.model, self.optimizer)
        self.data_loader.set_data_progress_percentage(progress_percentage)
        self.env.barrier()

        for epoch in range(10):
            self.text_generator.generate_examples()

    def cleanup(self):
        if self.env.enabled:
            dist.destroy_process_group()


if __name__ == "__main__":

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train_config = TrainConfig()
    module_config = ModuleConfig()
    train_data_config = TrainDataConfig()
    chatbot = ChatBot(train_config, module_config, train_data_config)
    chatbot.chat()
    chatbot.cleanup()