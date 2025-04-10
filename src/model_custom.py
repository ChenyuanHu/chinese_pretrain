import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin

# default config, qwen2.5-0.5b
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

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 添加头数验证
        assert config.num_attention_heads % config.num_key_value_heads == 0
        assert config.hidden_size % config.num_attention_heads == 0

        self.config = config

        # regularization
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.hd = self.hidden_size // self.num_attention_heads
        self.n_rep = self.num_attention_heads // self.num_key_value_heads

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
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.hd, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.hd, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.hd, bias=True)
        # output projection
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.q_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        self.k_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        self.v_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        self.o_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(self, x, position_embeddings):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (hidden_size)

        # 添加位置编码长度验证
        assert T <= self.config.max_position_embeddings

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_attention_heads, self.hd)
        k = k.view(B, T, self.num_key_value_heads, self.hd)
        v = v.view(B, T, self.num_key_value_heads, self.hd)

        # 转置维度以进行注意力计算 [B, N, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = self.repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        v = self.repeat_kv(v, self.n_rep)

        if self.flash_attn:
            # flashattention
            with sdpa_kernel(self.flash_attn_backends):
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            att = att.masked_fill(causal_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.o_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

        self.gate_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        self.down_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        self.up_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self, config, id):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        self.use_checkpoint = config.use_block_checkpoint
        self.id = id

    # 原始版本，没有显存优化
    def forward_without_checkpoint(self, x, position_embeddings):
        attn_out = self.self_attn(self.input_layernorm(x), position_embeddings)
        x = x + attn_out
        mlp_out = self.mlp(self.post_attention_layernorm(x))
        x = x + mlp_out
        return x

    def forward_with_checkpoint(self, x, position_embeddings):
        # 重构检查点函数
        def attn_block(x):
            _x = self.input_layernorm(x)
            return self.self_attn(_x, position_embeddings)
        
        def mlp_block(x):
            _x = self.post_attention_layernorm(x)
            return self.mlp(_x)
        
        # 确保完整计算图
        x = x + checkpoint(attn_block, x, use_reentrant=False, preserve_rng_state=False)
        x = x + checkpoint(mlp_block, x, use_reentrant=False, preserve_rng_state=False)
        return x

    def forward(self, x, position_embeddings):
        if self.use_checkpoint > self.id:
            return self.forward_with_checkpoint(x, position_embeddings)
        else:
            return self.forward_without_checkpoint(x, position_embeddings)


class CustomModelForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = CustomModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.model = nn.ModuleDict(dict(
            rotary_emb = Qwen2RotaryEmbedding(config),
            embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size),
            layers = nn.ModuleList([Block(config, i) for i in range(config.num_hidden_layers)]),
            norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
            self.model.embed_tokens.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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
        return self.model.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.model.embed_tokens = new_embeddings

    def prepare_inputs_for_generation(
        self, 
        input_ids,
        past_key_values = None,
        attention_mask = None,
        **kwargs
    ):
        
        return {
            "input_ids": input_ids,
        }

    def forward(self, input_ids, labels=None, past_key_values=None, attention_mask=None, **kwargs):
        x = self.model.embed_tokens(input_ids)
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        position_embeddings = self.model.rotary_emb(x, position_ids)
        for block in self.model.layers:
            x = block(x, position_embeddings)
        x = self.model.norm(x)

        if labels is not None:
            logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
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
            past_key_values=None,  # 如果实现了KV缓存，这里应该返回
        )
