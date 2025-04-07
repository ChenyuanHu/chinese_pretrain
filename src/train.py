import torch
import torch.optim as optim
import torch.distributed as dist
import time
import random
from env import TorchrunEnv
from generate import TextGenerator
from checkpoint import CheckpointManager
from eval import EvaluateRunner
from config import TrainConfig, model_config as input_model_config, TrainDataConfig, Model
from log import tprint
import gc

if TrainDataConfig().dataloader_mode == "padding":
    from dataloader_padding import MixTrainDataLoader
else:
    from dataloader import MixTrainDataLoader

torch._dynamo.config.cache_size_limit = 64  # 默认是8

class Trainer:
    def __init__(self, train_config, model_config, train_data_config):
        self.env = TorchrunEnv()
        tprint(f"torchrun环境初始化完成")
        model = Model(model_config)
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
            lr=train_config.max_lr,  # 最大学习率
            betas=train_config.betas,  # beta1和beta2
            weight_decay=train_config.weight_decay,  # 权重衰减
            eps=train_config.eps,
            foreach=True
        )
        # 创建学习率调度器，从预热到最大学习率3e-4，最终降至3e-5
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=train_config.max_lr,
            total_steps=train_config.scheduler_epochs * train_config.steps_per_epoch,
            pct_start=train_config.pct_start,
            final_div_factor=train_config.final_div_factor,
            div_factor=train_config.div_factor,
            anneal_strategy=train_config.anneal_strategy
        )
        tprint(f"优化器初始化完成")

        self.data_loader = MixTrainDataLoader(self.env.world_size, self.env.rank, self.env.local_rank, train_config.batch_size, train_config.block_size)
        tprint(f"数据加载器初始化完成")
        # self.evaluate_runner = EvaluateRunner(self.data_loader, train_config.batch_size)
        # tprint(f"评估器初始化完成")

        self.checkpoint_manager = CheckpointManager(self.env, train_config)
        tprint(f"检查点管理器初始化完成")
        self.train_config = train_config
        self.model_config = model_config

        assert self.train_config.dtype in {"float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"}, f"dtype must be float32, float16, bfloat16 or float8_e4m3fn or float8_e5m2"
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float8_e4m3fn': torch.float8_e4m3fn, 'float8_e5m2': torch.float8_e5m2}[self.train_config.dtype]
        
        self.amp = torch.amp.autocast(device_type=self.env.device_type, dtype=ptdtype)
        self.amp_scaler = torch.amp.GradScaler(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=(self.env.device_type != 'cpu' and self.env.device_type != 'mps')
        )

        self.run_mode = "both" if not hasattr(train_config, "run_mode") else train_config.run_mode

        assert self.run_mode in {"train", "generate", "both"}, f"run_mode must be 'train' or 'generate' or 'both'"

        if self.run_mode == "train":
            self.text_generator = None
        else:
            self.text_generator = TextGenerator(self.model, self.train_config.block_size, train_data_config, device=self.env.device, amp=self.amp)
            tprint(f"文本生成器初始化完成")

        # 计算并打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        tprint(f"模型总参数量(当前GPU): {total_params:,}")
        tprint(f"可训练参数量(当前GPU): {trainable_params:,}")
        tprint(f"模型总大小(当前GPU): {total_params * 4 / (1024**2):.2f} MB")  # 假设每个参数是4字节（float32）

    def train_one_epoch(self, epoch):
        self.model.train()
        t0 = time.time()
        total_train_loss = 0
        total_train_tokens = 0
        
        self.optimizer.zero_grad()  # 在epoch开始时重置梯度
        
        last_print_time = time.time()
        # 初始化取样本时间统计列表
        sample_times = []
        
        for step in range(self.train_config.steps_per_epoch):
            try:
                # 获取下一批数据，并统计时间
                sample_start_time = time.time()
                x, _ = self.data_loader.next()
                sample_end_time = time.time()
                sample_times.append(sample_end_time - sample_start_time)
                
                x = torch.tensor(x, dtype=torch.long, device=self.env.device)
                
                # 前向传播
                with self.amp:
                    outputs = self.model(input_ids=x, labels=x)
                    loss = outputs.loss
                    
                    # 缩放损失以适应梯度累积
                    scaled_loss = self.amp_scaler.scale(loss / self.train_config.gradient_accumulation_steps)

                scaled_loss.backward()
                
                # 累计损失和token数
                total_train_loss += loss.item() * x.numel()
                total_train_tokens += x.numel()
                
                # 梯度累积：每 gradient_accumulation_steps 步进行一次更新，或者最后一个step
                if (step + 1) % self.train_config.gradient_accumulation_steps == 0 or (step + 1 == self.train_config.steps_per_epoch):
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                    self.scheduler.step()  # 更新学习率
                    self.optimizer.zero_grad()
                
                current_time = time.time()
                if self.env.local_rank == 0 and current_time - last_print_time >= 30:  # 每30秒打印一次
                    tokens_per_sec = total_train_tokens / (current_time - t0)
                    current_lr = self.scheduler.get_last_lr()[0] * 1e5
                    tprint(f"Epoch {epoch+1}, Step {step+1}/{self.train_config.steps_per_epoch}, "
                           f"Loss: {loss.item():.4f}, "
                           f"LR: {current_lr:.2f}e-5, "
                           f"tokens/s/gpu: {tokens_per_sec:.2f}")
                    last_print_time = current_time
                    
            except Exception as e:
                tprint(f"进程 {self.env.rank} 在训练步骤中遇到错误: {str(e)}")
                raise e
        
        # 在epoch结束时同步所有进程
        self.env.barrier()
        
        # 计算样本时间的统计信息
        if len(sample_times) > 0:
            avg_sample_time = sum(sample_times) / len(sample_times)
            max_sample_time = max(sample_times)
            min_sample_time = min(sample_times)
            # 计算方差
            variance = sum((t - avg_sample_time) ** 2 for t in sample_times) / len(sample_times)

            sample_times = []
            
            # 收集所有进程的样本时间统计
            if self.env.local_rank == 0:
                tprint(f"样本获取时间统计 (进程 {self.env.rank}) - 平均: {avg_sample_time*1000:.2f}ms, "
                       f"方差: {variance*1000*1000:.2f}ms², "
                       f"最大: {max_sample_time*1000:.2f}ms, "
                       f"最小: {min_sample_time*1000:.2f}ms")

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
        current_lr = self.scheduler.get_last_lr()[0] * 1e5
        tprint(f"Epoch [{epoch+1}/{self.train_config.num_epochs}], {(t1-t0):.2f}sec, "
            f"world {global_tokens_per_sec:.2f} tokens/s, "
            f"训练损失: {global_avg_train_loss:.4f}, 困惑度: {global_train_ppl:.4f}, "
            f"LR: {current_lr:.2f}e-5")


        #tprint(f"全局验证损失: {global_eval_avg_loss:.4f}, 困惑度: {global_eval_ppl:.4f}")
        tprint(f"数据集使用度: {data_progress_percentage}")
        if hasattr(self.data_loader, "get_processed_tokens_count"):
            tprint(f"数据集处理token数: {self.data_loader.get_processed_tokens_count()}")
        
        # 检查是否需要保存检查点
        use_nfs = hasattr(self.train_config, "use_nfs") and self.train_config.use_nfs
        tprint(f"use_nfs: {use_nfs}")
        self.checkpoint_manager.check_save_checkpoint(self.model, self.optimizer, epoch, data_progress_percentage, use_nfs)
        
    def train(self):
        start_epoch, progress_percentage = self.checkpoint_manager.try_load_checkpoint(self.model, self.optimizer)
        self.data_loader.set_data_progress_percentage(progress_percentage)
        self.env.barrier()

        steps_done = start_epoch * self.train_config.steps_per_epoch
        for _ in range(steps_done):
            # 创建一个空的优化器步骤
            self.optimizer.zero_grad()
            self.optimizer.step()
            self.scheduler.step()
        tprint(f"lr scheduler 初始化完成")

        for epoch in range(start_epoch, self.train_config.num_epochs):
            if self.run_mode == "train" or self.run_mode == "both":
                self.train_one_epoch(epoch)

            self.env.barrier()
            if self.run_mode == "generate" or self.run_mode == "both":
                # 每个epoch结束后生成示例文本
                self.text_generator.generate_examples()

                # 仅仅生成则不需要清除内存
                if self.run_mode != "generate":
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
    model_config = input_model_config
    train_data_config = TrainDataConfig()
    def config_to_dict(config):
        result = {}
        # 获取对象属性
        for k in dir(config):
            if k.startswith('_') or callable(getattr(config, k, None)):
                continue
            result[k] = getattr(config, k, None)
        return result
            
    tprint(f"model_config: {config_to_dict(model_config)}")
    tprint(f"train_config: {config_to_dict(train_config)}")
    trainer = Trainer(train_config, model_config, train_data_config)
    trainer.train()
    trainer.cleanup()