"""
与GPT2有以下几个不同
1) RoPE: Relative Positional Encoding
2) GQA: Grouped Query Attention
3) SwiGLU: Swish-Gated Linear Unit
4) RMSNorm: Root Mean Square Layer Normalization
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import glob
import math
import typing
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import random

# 导入日志模块
from log import tprint


class RoPE:
    def __init__(self, dim: int, theta: float = 10000.0, use_scaled: bool = False):
        # 不再预计算block_size相关参数
        self.dim = dim
        self.theta = theta
        self.use_scaled = use_scaled
        # 预计算频率基底（与序列长度无关）
        self.base_freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2)[: (dim//2)].float() / dim))
        if self.use_scaled:
            self.base_freqs = self.apply_scaling(self.base_freqs)

    def apply_rotary_emb_warp(self, xq: torch.Tensor, xk: torch.Tensor):
        """动态生成当前序列长度的位置编码"""
        B, T, H, D = xq.shape  # 假设输入形状 [B, T, H, D]
        device = xq.device
        
        # 这里可以考虑使用lru_cache缓存频率基底
        # from functools import lru_cache
        # class RoPE:
        #     @lru_cache(maxsize=32)
        #     def get_freqs_cis(self, T: int, device):
        #         t = torch.arange(T, dtype=torch.float32, device=device)
        #         return torch.polar(torch.ones_like(t), t[:, None] * self.base_freqs.to(device))

        # 根据当前序列长度生成t
        t = torch.arange(T, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.base_freqs.to(device))
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        return self.apply_rotary_emb(xq, xk, freqs_cis)

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        """删除原断言，适配动态形状"""
        ndim = x.ndim
        shape = [1] * ndim
        shape[1] = freqs_cis.size(0)  # 序列维度
        shape[-1] = freqs_cis.size(1) # 特征维度
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_scaling(freqs: torch.Tensor):
        # Values obtained from grid search
        scale_factor = 8
        low_freq_factor = 1
        high_freq_factor = 4
        old_context_len = 8192  # original llama3 length

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = RoPE.reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, rope):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_kv_head = config.n_kv_head
        self.hd = self.n_embd // self.n_head
        self.n_rep = self.n_head // self.n_kv_head

        self.rope = rope

        # key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, (config.n_head + 2 * config.n_kv_head) * self.hd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)

        q, k, v = qkv.split([self.n_head * self.hd, self.n_kv_head * self.hd, self.n_kv_head * self.hd], dim=-1)
        q, k, v = map(lambda t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH, HD)

        q, k = self.rope.apply_rotary_emb_warp(q, k)

        k = self.repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        v = self.repeat_kv(v, self.n_rep)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

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
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        # SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))  <-- 3. difference compared to GPT-2
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, rope):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, rope)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MyModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rope = RoPE(
            dim = config.n_embd // config.n_head,
            theta = config.rope_theta,
            use_scaled = config.use_scaled_rope,
        )

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, self.rope) for _ in range(config.n_layer)]),
            ln_f = nn.RMSNorm(config.n_embd),
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
        _, t = idx.size()

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
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
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            tprint(f"使用设备: {self.device}")
            self.device_type = self.device
            self.master_process = True
        else:
            assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
            dist.init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.device_type = "cuda"
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
        # 使用 flagalpha/llama3-chinese-8b-instruct 的分词器
        tprint("正在加载分词器...")
        self.raw_tokenizer = AutoTokenizer.from_pretrained("flagalpha/llama3-chinese-8b-instruct", trust_remote_code=True)
        tprint(f"分词器加载成功！词汇表大小：{self.raw_tokenizer.vocab_size}")
        self.bos_token = self.raw_tokenizer.bos_token
        self.eos_token_id = self.raw_tokenizer.eos_token_id
        tprint(f"BOS token: {self.bos_token}")
        tprint(f"EOS token ID: {self.eos_token_id}")
    
    def encode(self, text):
        return self.raw_tokenizer.encode(text)

    def decode(self, tokens):
        return self.raw_tokenizer.decode(tokens)


class TrainDataLoader:
    def __init__(self, ddp_env, batch_size, block_size, tokenizer=None, use_data_percent=100, is_sft=False):
        self.fineweb_edu_chinese_v2_1_iter = None
        self.chinese_deepseek_r1_distill_data_110k_sft_iter = None
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.ddp_env = ddp_env
        self.use_data_percent = use_data_percent
        self.batch_size = batch_size
        self.is_sft = is_sft
        self.reload()

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def reload(self):
        if self.is_sft:
            self.chinese_deepseek_r1_distill_data_110k_sft_iter = self.load_chinese_deepseek_r1_distill_data_110k_sft(self.batch_size)
        else:
            self.fineweb_edu_chinese_v2_1_iter = self.load_fineweb_edu_chinese_v2_1(self.ddp_env, self.batch_size, self.use_data_percent)

    @staticmethod
    def load_fineweb_edu_chinese_v2_1(ddp_env, batch_size, use_data_percent):
        percent_per_process = int(use_data_percent / ddp_env.ddp_world_size)
        offset_start = ddp_env.ddp_rank * percent_per_process
        offset_end = offset_start + percent_per_process
        assert offset_end <= 100, f"offset_end({offset_end}) must be less than 100"

        tprint(f"加载FineWebEduChinese数据集，只下载和处理4-5评分范围的高质量内容. 第{ddp_env.ddp_rank}个进程，从{offset_start}%到{offset_end}%")
        raw_dataset = load_dataset("opencsg/Fineweb-Edu-Chinese-V2.1", data_dir = "4_5", split=f"train[{offset_start}%:{offset_end}%]")
        dataset_batch = DataLoader(raw_dataset, batch_size=batch_size, shuffle=True)
        return iter(dataset_batch)
    
    def next_fineweb_edu_chinese_v2_1(self):
        items = next(self.fineweb_edu_chinese_v2_1_iter)
        texts = items["text"]
        return texts

    @staticmethod
    def load_chinese_deepseek_r1_distill_data_110k_sft(batch_size):
        tprint(f"加载ChineseDeepSeekR1DistillData数据集")
        raw_dataset = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", split="train")
        dataset_batch = DataLoader(raw_dataset, batch_size=batch_size, shuffle=True)
        return iter(dataset_batch)
    
    def next_chinese_deepseek_r1_distill_data_110k_sft(self):
        items = next(self.chinese_deepseek_r1_distill_data_110k_sft_iter)
        texts = []
        for i in range(len(items["instruction"])):
            text = "系统提示：你是一个叫小伽的人工智能小助手，你的思考过程放在<think></think>标签中" + "\n" + "用户：" + items["instruction"][i] + "\n" + "助手：" + items["output"][i]
            texts.append(text)
        return texts

    def next_router(self):
        if self.is_sft:
            return self.next_chinese_deepseek_r1_distill_data_110k_sft()
        else:
            return self.next_fineweb_edu_chinese_v2_1()

    def next(self, device):
        while True:
            try:
                texts = self.next_router()

                xs = []
                ys = []
                for text in texts:
                    tokens = self.tokenizer.encode(self.tokenizer.bos_token + text)
                    if len(tokens) < self.block_size + 1:
                        tokens = tokens + [self.tokenizer.eos_token_id] * (self.block_size + 1 - len(tokens))
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
                tprint("数据集遍历完了，重新开始")
                time.sleep(120)
                self.reload()
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

    def evaluate(self, model, device, ddp_env):
        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        total_tokens = torch.tensor(0, device=device)

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                total_loss += loss.sum().detach()
                total_tokens += y.numel()

        # 同步所有进程的总损失和总token数
        if ddp_env.enabled:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

        total_tokens = max(total_tokens, 1)
        avg_loss = (total_loss / total_tokens).item()
        perplexity = torch.exp(total_loss / total_tokens).item()

        return avg_loss, perplexity

class TextGenerator:
    def __init__(self, model, block_size, tokenizer, demo_config, device="cpu"):
        self.model = model
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.prompts = demo_config.prompts
        self.max_tokens = demo_config.max_tokens
        self.temperature = demo_config.temperature
        self.top_k = demo_config.top_k
        self.device = device
        
    # 定义文本生成函数
    def generate_text(self, prompt):
        self.model.eval()
        
        prompt = self.tokenizer.bos_token + prompt
        # 编码输入提示
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > self.block_size - self.max_tokens:
            # 如果提示太长，只保留后面部分
            tokens = tokens[-(self.block_size - self.max_tokens):]
        
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, seq_len]
        
        with torch.no_grad():
            while tokens.size(1) < self.max_tokens:
                # 获取预测
                if tokens.size(1) > self.block_size:  # 使用size(1)直接获取序列长度
                    # 如果序列太长，只保留后面的部分
                    tokens = tokens[:, -(self.block_size):]
                
                # 前向传播
                logits, _ = self.model(tokens)
                
                # 获取最后一个位置的预测
                logits = logits[:, -1, :] / self.temperature
                
                # 应用top-k采样
                if self.top_k > 0:
                    v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # 应用softmax获取概率分布
                probs = F.softmax(logits, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 追加到序列
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # 如果生成了结束标记，提前结束
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # 解码生成的token序列
        generated_tokens = tokens[0].tolist()
        
        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text

    # 在训练循环中生成文本的辅助函数
    def generate_examples(self):
        tprint("生成示例文本：")
        
        self.model.eval()  # 确保模型处于评估模式
        # 随机选择一个提示
        prompt = random.choice(self.prompts)
        tprint(f"\n提示: {prompt if prompt else '(无提示)'}")
        try:
            generated = self.generate_text(prompt)
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

    # 兼容DP/DDP的模型dict，他们的key会多一个module.前缀
    def _load_model_state_dict(self, model, model_state_dict):
        # 检查是否存在 "module." 前缀的键名
        has_module_prefix = any(k.startswith("module.") for k in model_state_dict.keys())
        
        # 如果当前模型是 DataParallel/DDP 但权重没有前缀，添加前缀
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            if not has_module_prefix:
                return {"module." + k: v for k, v in model_state_dict.items()}
        # 如果当前模型是单机但权重有前缀，去除前缀
        else:
            if has_module_prefix:
                return {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        
        return model_state_dict

    def _get_model_state_dict_to_save(self, model):
        # 如果模型被 DataParallel 或 DDP 包装，提取内部模型
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            return model.module.state_dict()
        else:
            return model.state_dict()

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
                state_dict = torch.load(latest_checkpoint, weights_only=True, map_location=map_location)
                # 首先尝试使用weights_only=True加载
                model.load_state_dict(self._load_model_state_dict(model, state_dict['model_state_dict']))
                optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                start_epoch = state_dict.get('epoch', 0)
                tprint(f"成功加载checkpoint，将从epoch {start_epoch} 继续训练")
            except Exception as e:
                tprint(f"使用weights_only=True模型加载失败: {str(e)}")
                exit()
        else:
            tprint("未找到checkpoint，将从头开始训练")

        return start_epoch

    def check_save_checkpoint(self, model, optimizer, epoch, avg_train_loss, avg_eval_loss):
        current_time = time.time()
        time_since_last_save = current_time - self.last_save_time
        
        if time_since_last_save > self.save_interval_sec:  # 如果超过n秒
            tprint(f"start save checkpoint")
            try:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': self._get_model_state_dict_to_save(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_eval_loss,
                }
                torch.save(save_dict, checkpoint_path)
                tprint(f"检查点已保存到 {checkpoint_path}，距上次保存: {time_since_last_save:.2f}秒")
                self.last_save_time = current_time
                tprint(f"删除旧的checkpoint")
                if os.path.exists(os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")):
                    os.remove(os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))
            except Exception as e:
                tprint(f"保存checkpoint时出错: {str(e)}")
                exit()


class Trainer:
    def __init__(self, train_config, module_config, demo_config):
        self.ddp_env = DDPEnv()
        tprint(f"DDP环境初始化完成")
        self.ddp_env.ddp_model_init(MyModule(module_config))
        self.model = self.ddp_env.get_model()
        tprint(f"模型初始化完成")
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.data_loader = TrainDataLoader(self.ddp_env, train_config.batch_size, module_config.block_size,
                                                       tokenizer=None, use_data_percent=train_config.use_data_percent,
                                                       is_sft=train_config.is_sft)
        tprint(f"数据加载器初始化完成")
        self.tokenizer = Tokenizer()
        tprint(f"分词器初始化完成")
        self.data_loader.set_tokenizer(self.tokenizer) # huggingface tokenizer要求在DataLoader后初始化
        self.evaluate_runner = EvaluateRunner(self.data_loader, train_config.batch_size)
        tprint(f"评估器初始化完成")

        self.text_generator = TextGenerator(self.model, module_config.block_size, self.tokenizer, demo_config, device=self.ddp_env.device)
        tprint(f"文本生成器初始化完成")
        self.checkpoint_manager = CheckpointManager(self.ddp_env, train_config.save_interval_sec)
        tprint(f"检查点管理器初始化完成")
        self.train_config = train_config
        self.module_config = module_config

        assert self.module_config.dtype in {"float32", "float16", "bfloat16"}, f"dtype must be float32, float16 or bfloat16"
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.module_config.dtype]
        self.amp = torch.amp.autocast(device_type=self.ddp_env.device_type, dtype=ptdtype)

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
                    if self.ddp_env.master_process and current_time - last_print_time >= 30:  # 每30秒打印一次
                        tokens_per_sec = total_train_tokens / (current_time - t0)
                        tprint(f"Epoch {epoch+1}, Step {step+1}/{self.train_config.steps_per_epoch}, Loss: {loss.item():.4f}, Tokens/s: {tokens_per_sec:.2f}")
                        last_print_time = current_time
                        
                except Exception as e:
                    tprint(f"进程 {self.ddp_env.ddp_rank} 在训练步骤中遇到错误: {str(e)}")
                    raise e
            
            # 在epoch结束时同步所有进程
            self.ddp_env.barrier()
            

            total_train_loss_tensor = torch.tensor(total_train_loss, device=self.ddp_env.device)
            total_train_tokens_tensor = torch.tensor(total_train_tokens, device=self.ddp_env.device)
            if self.ddp_env.enabled:
                dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_train_tokens_tensor, op=dist.ReduceOp.SUM)
            global_train_loss = total_train_loss_tensor.item()
            global_train_tokens = total_train_tokens_tensor.item()
            global_avg_train_loss = global_train_loss / global_train_tokens
            global_train_ppl = torch.exp(torch.tensor(global_avg_train_loss)).item()

            # 计算整个训练集群的tokens/s
            global_tokens_per_sec = global_train_tokens / (time.time() - t0)

            # 在验证集上评估
            global_eval_avg_loss, global_eval_ppl = self.evaluate_runner.evaluate(self.model, self.ddp_env.device, self.ddp_env)

            t1 = time.time()
            tprint(f"Epoch [{epoch+1}/{self.train_config.num_epochs}], 用时: {(t1-t0):.2f}秒")
            tprint(f"全局训练损失: {global_avg_train_loss:.4f}, 困惑度: {global_train_ppl:.4f}")
            tprint(f"全局验证损失: {global_eval_avg_loss:.4f}, 验证困惑度: {global_eval_ppl:.4f}")
            tprint(f"训练集群处理速度: {global_tokens_per_sec:.2f} tokens/s")
            
            # 检查是否需要保存检查点
            if self.ddp_env.master_process:
                self.checkpoint_manager.check_save_checkpoint(self.model, self.optimizer, epoch, global_avg_train_loss, global_eval_avg_loss)

            self.ddp_env.barrier()

            # 每个epoch结束后生成示例文本
            self.text_generator.generate_examples()

    def cleanup(self):
        if self.ddp_env.enabled:
            dist.destroy_process_group()


# 训练循环
class TrainConfig:
    is_sft = False
    use_data_percent = 80
    batch_size = 1
    num_epochs = 10000
    steps_per_epoch = 5000  # 每个epoch训练多少批次
    gradient_accumulation_steps = 4  # 梯度累积步数
    save_interval_sec = 1800  # 每n秒保存一次模型

# 模型参数
class ModuleConfig:
    block_size: int = 1024
    vocab_size: int = 128256  # 词表大小实际是128000，但是eos token id是128001，所以对齐到128256
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 2560
    n_kv_head: int = 8
    # amp
    dtype = "bfloat16"
    # MLP
    ffn_dim_multiplier: float = 1.3
    multiple_of: int = 1024
    # RoPE
    rope_theta: float = 500000.0
    use_scaled_rope: bool = True

class DemoConfig:
    # 生成不同提示的文本
    sft_prompts = [
        "系统提示：你是一个叫小伽的人工智能小助手，你的思考过程放在<think></think>标签中\n用户：请根据规律填充这两个空缺的数字。 4, 3, 4, 3, 4, 3, （），（）\n助手：",
        "系统提示：你是一个叫小伽的人工智能小助手，你的思考过程放在<think></think>标签中\n用户：中华人民共和国的现在的主席是谁？\n助手：",
        "系统提示：你是一个叫小伽的人工智能小助手，你的思考过程放在<think></think>标签中\n用户：你是谁？\n助手："
    ]
    pretrain_prompts = [
        "中华人民共和国的现在的主席是",
        ""
    ]
    max_tokens = 100
    temperature = 0.1
    top_k = 40

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
