import torch
import torch.optim as optim
import torch.distributed as dist
import time
import random
from env import TorchrunEnv
from module import MyModule
from dataloader import MixTrainDataLoader
from generate import TextGenerator
from checkpoint import CheckpointManager
from eval import EvaluateRunner
from config import TrainConfig, ModuleConfig, TrainDataConfig
from log import tprint
import gc

torch._dynamo.config.cache_size_limit = 64  # 默认是8

class Trainer:
    def __init__(self, train_config, module_config, train_data_config):
        self.env = TorchrunEnv()
        tprint(f"torchrun环境初始化完成")
        model = MyModule(module_config)
        tprint(f"模型初始化完成")
        model.to(self.env.device)
        tprint(f"模型移动到设备完成")
        model = self.env.model_init(model, full_shard=True if hasattr(train_config, "full_shard") and train_config.full_shard else False)
        tprint(f"模型分布式训练环境初始化完成")
        if train_config.compile == "FULL":
            model = torch.compile(model)
            tprint(f"模型编译完成, FULL")
        elif train_config.compile == "PARTIAL":
            model = torch.compile(model,
                    # 关闭导致类型冲突的优化
                    dynamic=True,  
                    disable=["max-autotune", "cudagraphs"])
            tprint(f"模型编译完成, PARTIAL")
        self.model = model
        # 使用32位的AdamW优化器，设置betas和权重衰减
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=3e-4,  # 最大学习率
            betas=(0.9, 0.95),  # beta1和beta2
            weight_decay=0.1,  # 权重衰减
            eps=1e-8,
            foreach=True
        )
        # 创建学习率调度器，从预热到最大学习率3e-4，最终降至3e-5
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=3e-4,
            total_steps=train_config.scheduler_epochs * train_config.steps_per_epoch,
            pct_start=0.1,  # 预热阶段占总步数的10%
            final_div_factor=10,  # 确保最终学习率为3e-5 (3e-4/10)
            div_factor=10,  # 起始学习率为max_lr/10
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

        if hasattr(train_config, "disable_text_generator") and train_config.disable_text_generator:
            self.text_generator = None
        else:
            self.text_generator = TextGenerator(self.model, module_config.block_size, train_data_config, device=self.env.device, amp=self.amp)
            tprint(f"文本生成器初始化完成")

        # 计算并打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        tprint(f"模型总参数量(当前GPU): {total_params:,}")
        tprint(f"可训练参数量(当前GPU): {trainable_params:,}")
        tprint(f"模型总大小(当前GPU): {total_params * 4 / (1024**2):.2f} MB")  # 假设每个参数是4字节（float32）
        
    def train(self):
        start_epoch, progress_percentage = self.checkpoint_manager.try_load_checkpoint(self.model, self.optimizer)
        self.data_loader.set_data_progress_percentage(progress_percentage)
        self.env.barrier()

        steps_done = start_epoch * self.train_config.steps_per_epoch
        for _ in range(steps_done):
            self.scheduler.step()
        tprint(f"lr scheduler 初始化完成")

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
                    x, y = self.data_loader.next()
                    x = torch.tensor(x, dtype=torch.long, device=self.env.device)
                    y = torch.tensor(y, dtype=torch.long, device=self.env.device)
                    
                    # 前向传播
                    with self.amp:
                        _, loss = self.model(x, y)
                    
                        # 确保损失是标量
                        loss = loss.mean()  # 添加这行来确保损失是标量
                        
                        # 缩放损失以适应梯度累积
                        scaled_loss = self.amp_scaler.scale(loss / self.train_config.gradient_accumulation_steps)

                    scaled_loss.backward()
                    
                    # 累计损失和token数
                    total_train_loss += loss.item() * y.numel()
                    total_train_tokens += y.numel()
                    
                    # 梯度累积：每 gradient_accumulation_steps 步进行一次更新，或者最后一个step
                    if (step + 1) % self.train_config.gradient_accumulation_steps == 0 or (step + 1 == self.train_config.steps_per_epoch):
                        self.amp_scaler.step(self.optimizer)
                        self.amp_scaler.update()
                        self.scheduler.step()  # 更新学习率
                        self.optimizer.zero_grad()
                    
                    current_time = time.time()
                    if self.env.local_rank == 0 and current_time - last_print_time >= 30:  # 每30秒打印一次
                        tokens_per_sec = total_train_tokens / (current_time - t0)
                        current_lr = int(self.scheduler.get_last_lr()[0] * 1e5)
                        tprint(f"Epoch {epoch+1}, Step {step+1}/{self.train_config.steps_per_epoch}, Loss: {loss.item():.4f}, LR: {current_lr}e-5, tokens/s/gpu: {tokens_per_sec:.2f}")
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
            # global_eval_avg_loss, global_eval_ppl = self.evaluate_runner.evaluate(self.model, self.env.device, self.env)

            t1 = time.time()
            data_progress_percentage = self.data_loader.get_data_progress_percentage()
            current_lr = int(self.scheduler.get_last_lr()[0] * 1e5)
            tprint(f"Epoch [{epoch+1}/{self.train_config.num_epochs}], {(t1-t0):.2f}sec, "
                f"world {global_tokens_per_sec:.2f} tokens/s, "
                f"训练损失: {global_avg_train_loss:.4f}, 困惑度: {global_train_ppl:.4f}, "
                f"LR: {current_lr}e-5")


            #tprint(f"全局验证损失: {global_eval_avg_loss:.4f}, 困惑度: {global_eval_ppl:.4f}")
            tprint(f"数据集使用度: {data_progress_percentage}")
            
            # 检查是否需要保存检查点
            self.checkpoint_manager.check_save_checkpoint(self.model, self.optimizer, epoch, data_progress_percentage)

            self.env.barrier()
            # 每个epoch结束后生成示例文本
            if self.text_generator is not None:
                self.text_generator.generate_examples()
                if self.train_config.compile:
                    torch.compiler.reset()
                gc.collect()
                torch.cuda.empty_cache()
            self.env.barrier()

    def cleanup(self):
        if self.env.enabled:
            dist.destroy_process_group()


if __name__ == "__main__":

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    random.seed(42)

    train_config = TrainConfig()
    module_config = ModuleConfig()
    train_data_config = TrainDataConfig()
    trainer = Trainer(train_config, module_config, train_data_config)
    trainer.train()
    trainer.cleanup()