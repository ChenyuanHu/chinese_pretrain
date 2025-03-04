import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tiktoken
from datasets import load_dataset
import glob
import math
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# 导入日志模块
from log import tprint

class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        FLASH = False
        if FLASH:
            # flashattention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MyModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_logits=True):
        device = idx.device
        _, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # 修改损失计算，对每个样本分别计算损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='none')
            # 重塑损失以匹配批次大小
            loss = loss.view(targets.shape)
            # 对每个序列取平均
            loss = loss.mean(dim=1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss



class DDPEnv:
    def __init__(self):
        tprint("check ddp env...")
        self.enabled = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
        if not self.enabled:
            tprint("not ddp env")
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            tprint(f"使用设备: {device}")
            self.device = device
            self.master_process = True
        else:
            assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
            dist.init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0 # this process will do logging, checkpointing etc.
            tprint(f"ddp rank: {self.ddp_rank}, local rank: {self.ddp_local_rank}, world size: {self.ddp_world_size}")

    def ddp_model_init(self, model):
        model.to(self.device)
        if self.enabled:
            self.model = DDP(model, device_ids=[self.ddp_local_rank])
        else:
            self.model = model

    def get_model(self):
        return self.model

    def barrier(self):
        if self.enabled:
            tprint("wait barrier")
            dist.barrier()
            tprint("barrier done")


class Tokenizer:
    def __init__(self):
        # 使用tiktoken编码器
        tprint("init tiktoken...")
        self.enc = tiktoken.get_encoding("gpt2")
        # 配置特殊token的处理
        self.allowed_special = {"<|endoftext|>"}  # 允许的特殊token
        self.disallowed_special = ()  # 禁用所有特殊token的检查
        self.eot_token = self.enc.eot_token
    
    def encode(self, text):
        return self.enc.encode(text, allowed_special=self.allowed_special, disallowed_special=self.disallowed_special)

    def decode(self, tokens):
        return self.enc.decode(tokens)


class FineWebEduChineseDataLoader:
    def __init__(self, ddp_env, batch_size, block_size, tokenizer, use_data_percent=100):
        percent_per_process = int(use_data_percent / ddp_env.ddp_world_size)
        offset_start = ddp_env.ddp_rank * percent_per_process
        offset_end = offset_start + percent_per_process
        assert offset_end <= 100, f"offset_end({offset_end}) must be less than 100"

        tprint(f"加载方式处理数据集，只下载和处理4-5评分范围的高质量内容. 第{ddp_env.ddp_rank}个进程，从{offset_start}%到{offset_end}%")
        self.dataset = load_dataset("opencsg/Fineweb-Edu-Chinese-V2.1", data_dir = "4_5", split=f"train[{offset_start}%:{offset_end}%]")
        dataset = dataset.shuffle(seed=42)
        self.dataset_batch = iter(dataset.batch(batch_size=batch_size))
        self.block_size = block_size
        self.tokenizer = tokenizer

    def next(self, device):
        while True:
            try:
                item = next(self.dataset_batch)
                texts = item["text"]

                xs = []
                ys = []
                for text in texts:
                    tokens = self.tokenizer.encode(text)
                    if len(tokens) < self.block_size + 1:
                        tokens = tokens + [self.tokenizer.eot_token] * (self.block_size + 1 - len(tokens))
                    else:
                        tokens = tokens[:self.block_size + 1]
                    
                    x = tokens[:-1]
                    y = tokens[1:]
                    xs.append(x)
                    ys.append(y)

                xs = torch.tensor(xs, dtype=torch.long, device=device)
                ys = torch.tensor(ys, dtype=torch.long, device=device)

                return xs, ys
                
            except StopIteration:
                # 如果数据集遍历完了，重新开始
                dataset = self.dataset.shuffle(seed=42)
                self.dataset_batch = iter(dataset.batch(batch_size=self.batch_size))
                continue

            except Exception as e:
                tprint(f"处理批次时出错: {str(e)}，跳过此批次")
                time.sleep(120)
                continue


class EvaluateRunner:
    def __init__(self, data_loader, batch_size):
        # 创建一个验证集，方便模型评估
        val_dataset = []
        for i in range(50):
            x, y = data_loader.next(None)
            val_dataset.append((x[0], y[0]))

        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def evaluate(self, model, device):
        model.eval()  # 设置模型为评估模式
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():  # 关闭梯度计算
            for x, y in self.val_loader:
                x, y = x.to(device), y.to(device)
                
                # 前向传播
                _, loss = model(x, y)
                
                # 确保损失是标量
                loss = loss.mean()
                
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()

        # 计算平均损失和困惑度
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics = {
            "loss": avg_loss,
            "perplexity": perplexity
        }

        return metrics


class TextGenerator:
    def __init__(self):
        pass
        
    # 定义文本生成函数
    def generate_text(self, model, block_size, tokenizer, prompt="", max_tokens=100, temperature=1.0, top_k=50, device="cpu"):
        model.eval()
        
        # 编码输入提示
        if len(prompt) > 0:
            tokens = tokenizer.encode(prompt)
            if len(tokens) > block_size - max_tokens:
                # 如果提示太长，只保留后面部分
                tokens = tokens[-(block_size - max_tokens):]
        else:
            # 对于空提示，使用一个起始token作为种子
            tokens = [tokenizer.eot_token]  # GPT-2的<|endoftext|> token，可作为起始点
        
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_len]
        
        # 确保tokens不是空张量
        if tokens.size(1) == 0:
            tokens = torch.tensor([[tokenizer.eot_token]], dtype=torch.long, device=device)  # 使用<|endoftext|>作为备选起始点
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # 获取预测
                if tokens.size(1) > block_size:  # 使用size(1)直接获取序列长度
                    # 如果序列太长，只保留后面的部分
                    tokens = tokens[:, -block_size:]
                
                # 前向传播
                logits, _ = model(tokens)
                
                # 获取最后一个位置的预测
                logits = logits[:, -1, :] / temperature
                
                # 应用top-k采样
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # 应用softmax获取概率分布
                probs = F.softmax(logits, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 追加到序列
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # 如果生成了结束标记，提前结束
                if next_token.item() == tokenizer.eot_token:
                    break
        
        # 解码生成的token序列
        generated_tokens = tokens[0].tolist()
        
        # 如果有提示，去掉提示部分
        if prompt:
            prompt_length = len(tokenizer.encode(prompt))
            generated_tokens = generated_tokens[prompt_length:]
        else:
            # 对于空提示，去掉我们添加的起始token
            generated_tokens = generated_tokens[1:]
        
        generated_text = tokenizer.decode(generated_tokens)
        return generated_text

    # 在训练循环中生成文本的辅助函数
    def generate_examples(self, model, block_size, tokenizer, device):
        model.eval()  # 确保模型处于评估模式
        
        # 生成不同提示的文本
        prompts = [
            "中华人民共和国是中国共产",
            ""  # 空提示，完全由模型自由生成
        ]
        
        tprint("训练完成！生成示例文本：")
        
        for prompt in prompts:
            tprint(f"\n提示: {prompt if prompt else '(无提示)'}")
            try:
                generated = self.generate_text(
                    model=model,
                    block_size=block_size,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=min(300, block_size),  # 减少生成的token数，加快生成速度
                    temperature=0.1,
                    top_k=40,
                    device=device
                )
                tprint(f"生成: {generated}")
            except Exception as e:
                tprint(f"生成文本时发生错误: {str(e)}")
            tprint("-"*50)


class CheckpointManager:
    def __init__(self, ddp_env, save_interval_sec):
        self.ddp_env = ddp_env
        # 记录上次保存模型的时间
        self.last_save_time = time.time()
        self.checkpoint_dir = "checkpoints"
        self.save_interval_sec = save_interval_sec
        # 创建检查点目录（如果不存在）
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        tprint(f"创建检查点目录: {self.checkpoint_dir}")

    # 检查是否存在checkpoint文件
    def get_latest_checkpoint(self):
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt"))
        if not checkpoint_files:
            return None
        # 按文件修改时间排序，获取最新的checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint

    def try_load_checkpoint(self, model, optimizer):
        # 添加ModuleConfig到安全globals列表中
        torch.serialization.add_safe_globals([ModuleConfig])

        # 尝试加载最新的checkpoint
        latest_checkpoint = self.get_latest_checkpoint()
        start_epoch = 0
        if latest_checkpoint:
            tprint(f"发现最新的checkpoint: {latest_checkpoint}")
            if self.ddp_env.enabled:
                map_location = { "cuda:%d" % 0 : "cuda:%d" % self.ddp_env.ddp_local_rank }
            else:
                map_location = None
            try:
                # 首先尝试使用weights_only=True加载
                checkpoint = torch.load(latest_checkpoint, weights_only=True, map_location=map_location)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                tprint(f"成功加载checkpoint，将从epoch {start_epoch} 继续训练")
            except Exception as e:
                tprint(f"使用weights_only=True模型加载失败...")
                exit()
        else:
            tprint("未找到checkpoint，将从头开始训练")

        return start_epoch

    def check_save_checkpoint(self, model, optimizer, epoch, avg_train_loss, metrics):
        current_time = time.time()
        time_since_last_save = current_time - self.last_save_time
        
        if time_since_last_save > self.save_interval_sec:  # 如果超过n秒
            tprint(f"start save checkpoint")
            try:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': metrics['loss'],
                }
                torch.save(save_dict, checkpoint_path)
                tprint(f"检查点已保存到 {checkpoint_path}，距上次保存: {time_since_last_save:.2f}秒")
                self.last_save_time = current_time
                tprint(f"删除旧的checkpoint")
                if os.path.exists(os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")):
                    os.remove(os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))
            except Exception as e:
                tprint(f"保存checkpoint时出错: {str(e)}")


class Trainer:
    def __init__(self, train_config, model_config):
        self.ddp_env = DDPEnv()
        self.ddp_env.ddp_model_init(MyModule(model_config))
        self.model = self.ddp_env.get_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        tokenizer = Tokenizer()
        self.data_loader = FineWebEduChineseDataLoader(self.ddp_env, train_config.batch_size, model_config.block_size,
                                                       tokenizer, use_data_percent=train_config.use_data_percent)
        self.evaluate_runner = EvaluateRunner(self.data_loader, train_config.batch_size)
        self.text_generator = TextGenerator()
        self.checkpoint_manager = CheckpointManager(self.ddp_env, train_config.save_interval_sec)

        # 计算并打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        tprint(f"模型总参数量: {total_params:,}")
        tprint(f"可训练参数量: {trainable_params:,}")
        tprint(f"模型大小: {total_params * 4 / (1024**2):.2f} MB")  # 假设每个参数是4字节（float32）

    def train(self):
        start_epoch = self.checkpoint_manager.try_load_checkpoint(self.model, self.optimizer)
        self.ddp_env.barrier()

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
                    x, y = self.data_loader.next(self.ddp_env.device)
                    
                    # 前向传播
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
                    if self.ddp_env.master_process and current_time - last_print_time >= 30:  # 每30秒打印一次
                        tprint(f"Epoch {epoch+1}, Step {step+1}/{self.train_config.steps_per_epoch}, Loss: {loss.item():.4f}")
                        last_print_time = current_time
                        
                except Exception as e:
                    tprint(f"进程 {self.ddp_env.ddp_rank} 在训练步骤中遇到错误: {str(e)}")
                    raise e
            
            # 在epoch结束时同步所有进程
            self.ddp_env.barrier()
            
            # 计算平均训练损失和困惑度
            avg_train_loss = total_train_loss / total_train_tokens
            train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()
                
            t1 = time.time()
            
            # 在验证集上评估
            metrics = self.evaluate_runner.evaluate(self.model, self.ddp_env.device)
            
            tprint(f"Epoch [{epoch+1}/{self.train_config.num_epochs}], 用时: {(t1-t0):.2f}秒")
            tprint(f"训练损失: {avg_train_loss:.4f}, 训练困惑度: {train_ppl:.4f}")
            tprint(f"验证损失: {metrics['loss']:.4f}, 验证困惑度: {metrics['perplexity']:.4f}")
            
            # 检查是否需要保存检查点
            if self.ddp_env.master_process:
                self.checkpoint_manager.check_save_checkpoint(self.model, self.optimizer, epoch, avg_train_loss, metrics)

            self.ddp_env.barrier()

            # 每个epoch结束后生成示例文本
            self.text_generator.generate_examples(self.model, self.model_config.block_size, self.tokenizer, self.ddp_env.device)


# 训练循环
class TrainConfig:
    use_data_percent = 80
    batch_size = 4
    num_epochs = 10000
    steps_per_epoch = 1000  # 每个epoch训练多少批次
    gradient_accumulation_steps = 1  # 梯度累积步数
    save_interval_sec = 1800  # 每n秒保存一次模型

# 模型参数
class ModuleConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 46
    n_head: int = 25
    n_embd: int = 1600

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train_config = TrainConfig()
    model_config = ModuleConfig()
    trainer = Trainer(train_config, model_config)
    trainer.train()

