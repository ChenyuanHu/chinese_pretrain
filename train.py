import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, IterableDataset
import tiktoken
from datasets import load_dataset
import numpy as np
from itertools import islice
import glob
import os
# 导入日志模块
from log import tprint

tprint("开始训练...")

# 设置随机种子以确保可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 使用tiktoken编码器
enc = tiktoken.get_encoding("gpt2")
# 配置特殊token的处理
allowed_special = {"<|endoftext|>"}  # 允许的特殊token
disallowed_special = ()  # 禁用所有特殊token的检查

def encode_with_special(text):
    return enc.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)

def decode_with_special(tokens):
    return enc.decode(tokens)

# 数据集参数
batch_size = 1
block_size = 512

# 训练循环
num_epochs = 10000
steps_per_epoch = 1000  # 每个epoch训练多少批次
gradient_accumulation_steps = 4  # 梯度累积步数
save_interval_sec = 1800  # 每n秒保存一次模型

# 模型参数
class ModuleConfig:
    block_size: int = block_size
    vocab_size: int = 50257
    n_layer: int = 36
    n_head: int = 20
    n_embd: int = 1080

# using a global to toggle flash-attention
FLASH = False

config = ModuleConfig()


tprint("使用流式加载方式处理数据集，只下载和处理4-5评分范围的高质量内容...")
dataset = load_dataset("opencsg/Fineweb-Edu-Chinese-V2.1", data_dir = "4_5", split="train", streaming=True)
dataset_batch = iter(dataset.batch(batch_size=batch_size))

def next_x_y(device):
    item = next(dataset_batch)
    texts = item["text"]

    xs = []
    ys = []
    for text in texts:
        tokens = encode_with_special(text)
        if len(tokens) < block_size + 1:
            tokens = tokens + [enc.eot_token] * (block_size + 1 - len(tokens))
        else:
            tokens = tokens[:block_size + 1]
        
        x = tokens[:-1]
        y = tokens[1:]
        xs.append(x)
        ys.append(y)

    xs = torch.tensor(xs, dtype=torch.long, device=device)
    ys = torch.tensor(ys, dtype=torch.long, device=device)

    return xs, ys

# 创建一个验证集，方便模型评估
val_dataset = []
for i in range(50):
    x, y = next_x_y(None)
    val_dataset.append((x[0], y[0]))

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss


def evaluate_model(model, val_loader, device):
    """
    评估自回归语言模型在验证集上的性能，计算困惑度(Perplexity)和平均损失(Loss)
    
    Args:
        model (nn.Module): 要评估的模型
        val_loader (DataLoader): 验证集数据加载器
        device (str): 使用的设备('cpu' 或 'cuda')
        
    Returns:
        metrics (dict): 包含验证集的困惑度和损失
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():  # 关闭梯度计算
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            logits, loss = model(x, y)
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


# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
tprint(f"使用设备: {device}")

# 初始化模型
model = MyModule(config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model.to(device)


# 定义文本生成函数
def generate_text(model, enc, prompt="", max_tokens=100, temperature=1.0, top_k=50, device="cpu"):
    """
    使用训练好的模型生成文本
    
    Args:
        model: 训练好的模型
        enc: tokenizer编码器
        prompt: 起始提示文本，可以为空
        max_tokens: 最大生成token数量
        temperature: 温度参数，控制生成文本的随机性，越高越随机
        top_k: 只考虑概率最高的top_k个token
        device: 计算设备
        
    Returns:
        生成的文本
    """
    model.eval()
    
    # 编码输入提示
    if prompt:
        tokens = encode_with_special(prompt)
        if len(tokens) > model.config.block_size - max_tokens:
            # 如果提示太长，只保留后面部分
            tokens = tokens[-(model.config.block_size - max_tokens):]
    else:
        # 对于空提示，使用一个起始token作为种子
        tokens = [enc.eot_token]  # GPT-2的<|endoftext|> token，可作为起始点
    
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_len]
    
    # 确保tokens不是空张量
    if tokens.size(1) == 0:
        tokens = torch.tensor([[enc.eot_token]], dtype=torch.long, device=device)  # 使用<|endoftext|>作为备选起始点
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # 获取预测
            if len(tokens[0]) > model.config.block_size:
                # 如果序列太长，只保留后面的部分
                tokens = tokens[:, -model.config.block_size:]
            
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
            if next_token.item() == enc.eot_token:
                break
    
    # 解码生成的token序列
    generated_tokens = tokens[0].tolist()
    
    # 如果有提示，去掉提示部分
    if prompt:
        prompt_length = len(encode_with_special(prompt))
        generated_tokens = generated_tokens[prompt_length:]
    else:
        # 对于空提示，去掉我们添加的起始token
        generated_tokens = generated_tokens[1:]
    
    generated_text = decode_with_special(generated_tokens)
    return generated_text

# 在训练循环中生成文本的辅助函数
def generate_examples(model, enc, device, block_size, epoch=None):
    """在当前epoch结束后生成示例文本"""
    model.eval()  # 确保模型处于评估模式
    
    # 生成不同提示的文本
    prompts = [
        "今天天气真好，",
        # "人工智能在教育领域的应用，",
        # "中国传统文化是指",
        # "学习编程的最佳方法是",
        ""  # 空提示，完全由模型自由生成
    ]
    
    if epoch is not None:
        tprint("\n" + "="*50)
        tprint(f"Epoch {epoch+1} 生成文本示例：")
        tprint("="*50)
    else:
        tprint("\n" + "="*50)
        tprint("训练完成！生成示例文本：")
        tprint("="*50)
    
    for prompt in prompts:
        tprint(f"\n提示: {prompt if prompt else '(无提示)'}")
        try:
            generated = generate_text(
                model=model,
                enc=enc,
                prompt=prompt,
                max_tokens=min(100, block_size),  # 减少生成的token数，加快生成速度
                temperature=0.1,
                top_k=40,
                device=device
            )
            tprint(f"生成: {generated}")
        except Exception as e:
            tprint(f"生成文本时发生错误: {str(e)}")
        tprint("-"*50)

# 记录上次保存模型的时间
last_save_time = time.time()
checkpoint_dir = "checkpoints"
# 创建检查点目录（如果不存在）
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    tprint(f"创建检查点目录: {checkpoint_dir}")

# 检查是否存在checkpoint文件
def get_latest_checkpoint():
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        return None
    # 按文件修改时间排序，获取最新的checkpoint
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint

# 尝试加载最新的checkpoint
latest_checkpoint = get_latest_checkpoint()
start_epoch = 0
if latest_checkpoint:
    tprint(f"发现最新的checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    tprint(f"成功加载checkpoint，将从epoch {start_epoch} 继续训练")
else:
    tprint("未找到checkpoint，将从头开始训练")

for epoch in range(start_epoch, num_epochs):
    model.train()
    t0 = time.time()
    total_train_loss = 0
    total_train_tokens = 0
    
    optimizer.zero_grad()  # 在epoch开始时重置梯度
    
    for step in range(steps_per_epoch):
        # 获取下一批数据
        x, y = next_x_y(device)
        
        # 前向传播
        logits, loss = model(x, y)
        
        # 缩放损失以适应梯度累积
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()
        
        # 累计损失和token数
        total_train_loss += loss.item() * y.numel()
        total_train_tokens += y.numel()
        
        # 梯度累积：每 gradient_accumulation_steps 步进行一次更新
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1 == steps_per_epoch):
            flag = True
            optimizer.step()
            optimizer.zero_grad()
            
        # tprint(f"Epoch {epoch+1}, Step {step+1}/{steps_per_epoch}, Loss: {loss.item():.4f}, 有效批次大小: {batch_size*gradient_accumulation_steps}")
    
    # 计算平均训练损失和困惑度
    avg_train_loss = total_train_loss / total_train_tokens
    train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()
        
    t1 = time.time()
    
    # 在验证集上评估
    metrics = evaluate_model(model, val_loader, device)
    
    tprint(f"Epoch [{epoch+1}/{num_epochs}], 用时: {(t1-t0):.2f}秒")
    tprint(f"训练损失: {avg_train_loss:.4f}, 训练困惑度: {train_ppl:.4f}")
    tprint(f"验证损失: {metrics['loss']:.4f}, 验证困惑度: {metrics['perplexity']:.4f}")
    
    # 检查是否需要保存检查点
    current_time = time.time()
    time_since_last_save = current_time - last_save_time
    
    if time_since_last_save > save_interval_sec:  # 如果超过n秒
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': metrics['loss'],
            'config': config,
        }, checkpoint_path)
        tprint(f"检查点已保存到 {checkpoint_path}，距上次保存: {time_since_last_save:.2f}秒")
        last_save_time = current_time
    
    # 每个epoch结束后生成示例文本
    generate_examples(model, enc, device, block_size, epoch)

# 保存模型
model_save_path = "chinese_lm_model.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
}, model_save_path)
tprint(f"模型已保存到 {model_save_path}")

# 训练结束后生成示例文本
generate_examples(model, enc, device, block_size)
