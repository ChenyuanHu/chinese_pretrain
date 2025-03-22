import torch
import torch.optim as optim
import torch.distributed as dist
import time
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
        if self.env.enabled:
            tprint(f"torchrun环境不支持chat")
            exit()

        model = MyModule(module_config)
        tprint(f"模型初始化完成")
        # model.to(self.env.device)
        model.to("cpu")
        tprint(f"模型移动到设备完成")
        model = self.env.model_init(model)
        tprint(f"模型分布式训练环境初始化完成")
        self.model = model

        self.text_generator = TextGenerator(self.model, module_config.block_size, train_data_config, device=self.env.device)
        tprint(f"文本生成器初始化完成")

        self.checkpoint_manager = CheckpointManager(self.env, train_config)
        tprint(f"检查点管理器初始化完成")
        self.train_config = train_config
        self.module_config = module_config

        # 计算并打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        tprint(f"模型总参数量: {total_params:,}")
        tprint(f"可训练参数量: {trainable_params:,}")
        tprint(f"模型总大小: {total_params * 4 / (1024**2):.2f} MB")  # 假设每个参数是4字节（float32）
        
    def chat(self):
        start_epoch, progress_percentage = self.checkpoint_manager.try_load_checkpoint(self.model, self.optimizer)
        self.data_loader.set_data_progress_percentage(progress_percentage)
        self.env.barrier()

        for _ in range(10):
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