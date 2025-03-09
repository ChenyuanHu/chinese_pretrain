import torch
import torch.optim as optim
import torch.distributed as dist
import time
from env import TorchrunEnv
from module import MyModule
from tokenizer import Tokenizer
from dataloader import TrainDataLoader
from generate import TextGenerator
from checkpoint import CheckpointManager
from eval import EvaluateRunner
from config import TrainConfig, ModuleConfig, DemoConfig
from log import tprint

class Trainer:
    def __init__(self, train_config, module_config, demo_config):
        self.env = TorchrunEnv()
        tprint(f"DDP环境初始化完成")
        self.env.model_init(MyModule(module_config))
        self.model = self.env.get_model()
        tprint(f"模型初始化完成")
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.data_loader = TrainDataLoader(self.env, train_config.batch_size, module_config.block_size,
                                                       tokenizer=None, use_data_percent=train_config.use_data_percent,
                                                       is_sft=train_config.is_sft)
        tprint(f"数据加载器初始化完成")
        self.tokenizer = Tokenizer()
        tprint(f"分词器初始化完成")
        self.data_loader.set_tokenizer(self.tokenizer) # huggingface tokenizer要求在DataLoader后初始化
        self.evaluate_runner = EvaluateRunner(self.data_loader, train_config.batch_size)
        tprint(f"评估器初始化完成")

        self.text_generator = TextGenerator(self.model, module_config.block_size, self.tokenizer, demo_config, device=self.env.device)
        tprint(f"文本生成器初始化完成")
        self.checkpoint_manager = CheckpointManager(self.env, train_config.save_interval_sec)
        tprint(f"检查点管理器初始化完成")
        self.train_config = train_config
        self.module_config = module_config

        assert self.module_config.dtype in {"float32", "float16", "bfloat16"}, f"dtype must be float32, float16 or bfloat16"
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.module_config.dtype]
        self.amp = torch.amp.autocast(device_type=self.env.device_type, dtype=ptdtype)

        # 计算并打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        tprint(f"模型总参数量: {total_params:,}")
        tprint(f"可训练参数量: {trainable_params:,}")
        tprint(f"模型大小: {total_params * 4 / (1024**2):.2f} MB")  # 假设每个参数是4字节（float32）

    def train(self):
        start_epoch = self.checkpoint_manager.try_load_checkpoint(self.model, self.optimizer)
        self.env.barrier()

        for epoch in range(start_epoch, self.train_config.num_epochs):
            self.model.train()
            t0 = time.time()
            total_train_loss = 0
            total_train_tokens = 0
            
            self.optimizer.zero_grad()  # 在epoch开始时重置梯度
            
            last_print_time = time.time()
            for step in range(self.train_config.steps_per_epoch):
                try:
                    # 获取下一批数据
                    x, y = self.data_loader.next(self.env.device)
                    
                    # 前向传播
                    with self.amp:
                        _, loss = self.model(x, y)
                    
                        # 确保损失是标量
                        loss = loss.mean()  # 添加这行来确保损失是标量
                        
                        # 缩放损失以适应梯度累积
                        scaled_loss = loss / self.train_config.gradient_accumulation_steps

                    scaled_loss.backward()
                    
                    # 累计损失和token数
                    total_train_loss += loss.item() * y.numel()
                    total_train_tokens += y.numel()
                    
                    # 梯度累积：每 gradient_accumulation_steps 步进行一次更新
                    if (step + 1) % self.train_config.gradient_accumulation_steps == 0 or (step + 1 == self.train_config.steps_per_epoch):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    current_time = time.time()
                    if self.env.master_process and current_time - last_print_time >= 30:  # 每30秒打印一次
                        tokens_per_sec = total_train_tokens / (current_time - t0)
                        tprint(f"Epoch {epoch+1}, Step {step+1}/{self.train_config.steps_per_epoch}, Loss: {loss.item():.4f}, Tokens/s: {tokens_per_sec:.2f}")
                        last_print_time = current_time
                        
                except Exception as e:
                    tprint(f"进程 {self.env.rank} 在训练步骤中遇到错误: {str(e)}")
                    raise e
            
            # 在epoch结束时同步所有进程
            self.env.barrier()
            

            total_train_loss_tensor = torch.tensor(total_train_loss, device=self.env.device)
            total_train_tokens_tensor = torch.tensor(total_train_tokens, device=self.env.device)
            if self.env.enabled:
                dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_train_tokens_tensor, op=dist.ReduceOp.SUM)
            global_train_loss = total_train_loss_tensor.item()
            global_train_tokens = total_train_tokens_tensor.item()
            global_avg_train_loss = global_train_loss / global_train_tokens
            global_train_ppl = torch.exp(torch.tensor(global_avg_train_loss)).item()

            # 计算整个训练集群的tokens/s
            global_tokens_per_sec = global_train_tokens / (time.time() - t0)

            # 在验证集上评估
            global_eval_avg_loss, global_eval_ppl = self.evaluate_runner.evaluate(self.model, self.env.device, self.env)

            t1 = time.time()
            tprint(f"Epoch [{epoch+1}/{self.train_config.num_epochs}], 用时: {(t1-t0):.2f}秒")
            tprint(f"全局训练损失: {global_avg_train_loss:.4f}, 困惑度: {global_train_ppl:.4f}")
            tprint(f"全局验证损失: {global_eval_avg_loss:.4f}, 验证困惑度: {global_eval_ppl:.4f}")
            tprint(f"训练集群处理速度: {global_tokens_per_sec:.2f} tokens/s")
            
            # 检查是否需要保存检查点
            if self.env.master_process:
                self.checkpoint_manager.check_save_checkpoint(self.model, self.optimizer, epoch, global_avg_train_loss, global_eval_avg_loss)

            self.env.barrier()

            # 每个epoch结束后生成示例文本
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
    demo_config = DemoConfig()
    demo_config.prompts = demo_config.sft_prompts if train_config.is_sft else demo_config.pretrain_prompts
    trainer = Trainer(train_config, module_config, demo_config)
    trainer.train()
    trainer.cleanup()