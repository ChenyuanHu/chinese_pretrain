import torch
import torch.distributed as dist
import argparse  # 添加argparse导入
from env import TorchrunEnv
from module import MyModule
from generate import TextGenerator
from checkpoint import CheckpointManager
from config import TrainConfig, ModuleConfig, TrainDataConfig
from log import tprint

class ChatBot:
    def __init__(self, train_config, module_config, train_data_config, checkpoint_path=None):
        if checkpoint_path and checkpoint_path.endswith(".pt"):
            self.env = TorchrunEnv(force_cpu=True)
        else:
            self.env = TorchrunEnv()
        tprint(f"env ready")

        model = MyModule(module_config)
        tprint(f"模型初始化完成")
        model.to(self.env.device)
        tprint(f"模型移动到设备完成")

        model = self.env.model_init(model)
        tprint(f"模型环境初始化完成")
        self.model = model

        self.text_generator = TextGenerator(self.model, module_config.block_size, train_data_config, device=self.env.device)
        tprint(f"文本生成器初始化完成")

        self.train_config = train_config
        self.module_config = module_config
        self.checkpoint_path = checkpoint_path

        # 计算并打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        tprint(f"模型总参数量: {total_params:,}")
        tprint(f"可训练参数量: {trainable_params:,}")
        tprint(f"模型总大小: {total_params * 4 / (1024**2):.2f} MB")  # 假设每个参数是4字节（float32）
        
    def chat(self):
        tprint(f"正在加载检查点: {self.checkpoint_path}")

        if self.checkpoint_path and self.checkpoint_path.endswith(".pt"):
            checkpoint = torch.load(self.checkpoint_path, weights_only=True)
            model_state_dict = checkpoint["app"]["model_state_dict"]
            model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()}

            self.model.load_state_dict(model_state_dict)
        else:
            self.checkpoint_manager = CheckpointManager(self.env, self.train_config)
            tprint(f"检查点管理器初始化完成")
            self.checkpoint_manager.try_load_checkpoint(self.model, None)

        for _ in range(10):
            self.text_generator.generate_examples()

    def cleanup(self):
        if self.env.enabled:
            dist.destroy_process_group()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Chat")
    parser.add_argument("--checkpoint", type=str, help="模型检查点路径, dcp类型可以使用 python3 -m torch.distributed.checkpoint.format_utils dcp_to_torch checkpoints_epoch_1 checkpoints_epoch_1.pt 转换")
    args = parser.parse_args()
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train_config = TrainConfig()
    module_config = ModuleConfig()
    train_data_config = TrainDataConfig()
    
    chatbot = ChatBot(train_config, module_config, train_data_config, checkpoint_path=args.checkpoint)
    chatbot.chat()
    chatbot.cleanup()