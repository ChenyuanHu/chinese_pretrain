import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from rope import RoPEv2 as RoPE
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin

class CustomModelConfig(PretrainedConfig):
    model_type = "qwen2"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        hidden_act="silu",
        intermediate_size=4864,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        bos_token_id=151643,
        eos_token_id=151643,
        pad_token_id=151643,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=24,
        rope_scaling=None,
        use_mrope=False,
        use_sdpa=True,
        torch_dtype="float32",
        use_block_checkpoint=0,
        flash_attn="FLASH_ATTENTION|EFFICIENT_ATTENTION|CUDNN_ATTENTION|MATH",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.attention_dropout = attention_dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.rope_scaling = rope_scaling
        self.use_mrope = use_mrope
        self.use_sdpa = use_sdpa
        self.torch_dtype = torch_dtype
        self.use_block_checkpoint = use_block_checkpoint
        self.flash_attn = flash_attn


class CausalSelfAttention(nn.Module):
    def __init__(self, config, rope):
        super().__init__()
        self.config = config

        assert config.hidden_size % config.num_attention_heads == 0
        # regularization
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.hd = self.hidden_size // self.num_attention_heads
        self.n_rep = self.num_attention_heads // self.num_key_value_heads

        self.rope = rope

        self.flash_attn_backends = []
        self.flash_attn = False
        flash_attn_backends = config.flash_attn.split("|")
        if "FLASH_ATTENTION" in flash_attn_backends:
            self.flash_attn_backends.append(SDPBackend.FLASH_ATTENTION)
        if "EFFICIENT_ATTENTION" in flash_attn_backends:
            self.flash_attn_backends.append(SDPBackend.EFFICIENT_ATTENTION)
        if "CUDNN_ATTENTION" in flash_attn_backends:
            self.flash_attn_backends.append(SDPBackend.CUDNN_ATTENTION)
        if "MATH" in flash_attn_backends:
            self.flash_attn_backends.append(SDPBackend.MATH)
        if len(self.flash_attn_backends) != 0:
            self.flash_attn = True

        # key, query, value projections
        self.c_attn = nn.Linear(config.hidden_size, (config.num_attention_heads + 2 * config.num_key_value_heads) * self.hd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

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

    # 使用 torch.triu 避免初始化全 1 矩阵
    def get_causal_mask(T, device, dtype):
        return torch.triu(torch.full((T, T), float('-inf'), dtype=dtype, device=device), diagonal=1)

    def forward(self, x, attention_mask=None, past_key_value=None, **kwargs):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (hidden_size)

        # 生成因果掩码（兼容HF格式）
        if attention_mask is not None:
            # 将注意力掩码转换为 additive mask
            causal_mask = self.get_causal_mask(T, x.device, x.dtype)
            causal_mask = causal_mask[None, None, :, :]
            if self.config.use_sliding_window:
                window_mask = torch.ones_like(causal_mask)
                window_mask[:, :, :, :-self.sliding_window] = -torch.inf
                causal_mask = torch.maximum(causal_mask, window_mask)
            
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min
            attention_mask = attention_mask + causal_mask
        else:
            attention_mask = torch.ones(T, T, dtype=torch.float32, device=x.device).tril()[None, None, :, :]

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)

        q, k, v = qkv.split([self.num_attention_heads * self.hd, self.num_key_value_heads * self.hd, self.num_key_value_heads * self.hd], dim=-1)
        q, k, v = map(lambda t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH, HD)

        # 合并KV缓存（如果存在）
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)  # 时间维度拼接
            v = torch.cat([past_v, v], dim=1)

        # 保存当前KV作为新的缓存
        new_key_value = (k, v) if self.config.use_cache else None

        # 计算位置IDs
        seq_len = k.shape[1]  # 包含历史缓存后的总长度
        position_ids = torch.arange(seq_len, device=x.device).expand(x.size(0), seq_len)
        # 应用动态位置编码
        q, k = self.rope.apply_rotary_emb_warp_position_ids(q, k, position_ids=position_ids)

        k = self.repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        v = self.repeat_kv(v, self.n_rep)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        if self.flash_attn:
            # flashattention
            with sdpa_kernel(self.flash_attn_backends):
                y = F.scaled_dot_product_attention(q, k, v,
                                                   attn_mask=attention_mask,
                                                   is_causal=False # 因为已经显式处理了mask
                                                   )
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att + attention_mask
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y, new_key_value

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self, config, rope, id):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config, rope)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        self.use_checkpoint = config.use_block_checkpoint
        self.id = id

    # 原始版本，没有显存优化
    def forward_without_checkpoint(self, x, attention_mask=None, past_key_value=None):
        attn_output, new_key_value = self.self_attn(self.input_layernorm(x), attention_mask=attention_mask, past_key_value=past_key_value)
        x = x + attn_output
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_key_value

    # 使用激活检查点，性能降低18%，显存降低10%
    def forward_with_checkpoint(self, x, attention_mask=None, past_key_value=None):
        # 将前向传播分为两个检查段
        def create_custom_forward_attn(module):
            def custom_forward(*inputs):
                return module(inputs[0], attention_mask=attention_mask, past_key_value=past_key_value)
            return custom_forward
            
        def create_custom_forward_mlp(module):
            def custom_forward(*inputs):
                return module(inputs[0])
            return custom_forward
        
        # 第一个检查段：注意力机制
        attn_output, new_key_value = checkpoint(create_custom_forward_attn(self.self_attn), self.input_layernorm(x), use_reentrant=False)
        x = x + attn_output
        # 第二个检查段：MLP
        x = x + checkpoint(create_custom_forward_mlp(self.mlp), self.post_attention_layernorm(x), use_reentrant=False)
        return x, new_key_value

    def forward(self, x, attention_mask=None, past_key_value=None):
        if self.use_checkpoint > self.id:
            return self.forward_with_checkpoint(x, attention_mask, past_key_value)
        else:
            return self.forward_without_checkpoint(x, attention_mask, past_key_value)


class CustomModelForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = CustomModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.rope = RoPE(
            dim = config.hidden_size // config.num_attention_heads,
            theta = config.rope_theta,
            use_scaled = config.use_mrope,
        )

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            h = nn.ModuleList([Block(config, self.rope, i) for i in range(config.num_hidden_layers)]),
            ln_f = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = self.config.initializer_range
            if hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG'):
                std = std/math.sqrt(2 * self.config.num_hidden_layers)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

    def prepare_inputs_for_generation(
        self, 
        input_ids,
        past_key_values = None,
        attention_mask = None,
        **kwargs
    ):
        # 合并历史长度与当前长度
        past_length = past_key_values[0][0].size(1) if past_key_values else 0

        # 扩展attention_mask
        if attention_mask is not None:
            attention_mask = torch.cat([
                torch.ones(input_ids.size(0), past_length, device=input_ids.device),
                attention_mask
            ], dim=-1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def forward(self, input_ids, labels=None, past_key_values=None, attention_mask=None, **kwargs):
        # 初始化缓存（如果第一次调用）
        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers

        x = self.transformer.wte(input_ids)
        presents = []
        for (block, past) in zip(self.transformer.h, past_key_values):
            x, kv_cache = block(x, attention_mask=attention_mask, past_key_value=past)
            presents.append(kv_cache)
        x = self.transformer.ln_f(x)

        if labels is not None:
            logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

            # 更高效的标签右移与掩码处理
            shifted_labels = torch.full_like(labels, -100)  # 初始化为全 -100
            shifted_labels[:, :-1] = labels[:, 1:]       # 右移：将 labels[1:] 复制到 shifted_labels 的前 n-1 列

            # 直接计算损失（无需手动拼接）
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=-100,
                #reduction='none'
                reduction='mean'
            )

            # 如果不是全局计算loss的时候全局mean，则可以按样本平均损失，避免长短样本权重不一致问题
            # 梯度累积的时候，mean也可能会有误差
            # loss = loss.view(shifted_labels.shape)
            # valid_tokens = (shifted_labels != -1).float()
            # loss = (loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1)
            # loss = loss.mean()

        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,  # 如果实现了KV缓存，这里应该返回
        )
