import torch
import json
import math
from pathlib import Path
from typing import Dict, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel, PretrainedConfig

# 从你的原始代码中导入必要组件
from module import MyModule, CausalSelfAttention, MLP, Block
from config import ModuleConfig
from tokenizer import Tokenizer

# ----------------------
# 自定义配置类 (兼容Hugging Face)
# ----------------------
class LlamaLikeConfig(PretrainedConfig):
    model_type = "llama"
    
    def __init__(self, 
                 block_size=2048,
                 vocab_size=152000,
                 n_layer=32,
                 n_head=32,
                 n_embd=4096,
                 n_kv_head=8,
                 flash_attn="FLASH_ATTENTION",
                 dtype="bfloat16",
                 ffn_dim_multiplier=1.3,
                 multiple_of=1024,
                 rope_theta=500000.0,
                 use_scaled_rope=True,
                 use_block_checkpoint=0,
                 **kwargs):
        super().__init__(**kwargs)
        # 原始参数
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_kv_head = n_kv_head
        self.flash_attn = flash_attn
        self.dtype = dtype
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.rope_theta = rope_theta
        self.use_scaled_rope = use_scaled_rope
        self.use_block_checkpoint = use_block_checkpoint
        
        # 转换为Hugging Face标准参数名
        self.hidden_size = n_embd
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_kv_head
        self.num_hidden_layers = n_layer
        self.rms_norm_eps = 1e-6  # 假设你的RMSNorm使用该eps值
        self.initializer_range = 0.02
        self.pretraining_tp = 1

# ----------------------
# 自定义Tokenizer包装
# ----------------------
class LlamaLikeTokenizer(PreTrainedTokenizer):
    def __init__(self, tokenizer: Tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.raw_tokenizer = tokenizer
        self.model_max_length = 2048  # 根据你的block_size设置
        
    @property
    def vocab_size(self) -> int:
        return self.raw_tokenizer.vocab_size
        
    def _tokenize(self, text: str, **kwargs) -> list:
        return self.raw_tokenizer.encode(text).tokens
        
    def _convert_token_to_id(self, token: str) -> int:
        return self.raw_tokenizer.encode(token).ids[0]
        
    def save_vocabulary(self, save_directory: str, **kwargs) -> tuple:
        output_dir = Path(save_directory)
        self.raw_tokenizer.save(str(output_dir / "vocab.txt"))
        return (str(output_dir / "vocab.txt"),)

# ----------------------
# 兼容Hugging Face的模型包装
# ----------------------
class HuggingFaceModelWrapper(PreTrainedModel):
    config_class = LlamaLikeConfig
    
    def __init__(self, original_model: MyModule, config: LlamaLikeConfig):
        super().__init__(config)
        self.model = original_model
        
        # 层名映射表 (原始层名 -> HF标准层名)
        self.layer_name_mapping = {
            'transformer.wte': 'model.embed_tokens',
            'transformer.h': 'model.layers',
            'transformer.ln_f': 'model.norm',
            'attn.c_attn': 'self_attn.qkv_proj',
            'attn.c_proj': 'self_attn.o_proj',
            'mlp.c_fc2': 'mlp.up_proj',
            'mlp.c_fc': 'mlp.gate_proj',
            'mlp.c_proj': 'mlp.down_proj',
            'ln_1': 'input_layernorm',
            'ln_2': 'post_attention_layernorm',
            'attn.bias': 'self_attn.attn_bias',
            '.attn.': '.self_attn.'
        }
        
    def forward(self, input_ids: torch.Tensor, **kwargs):
        return self.model(input_ids)
        
    # 关键方法：重写state_dict以兼容HF格式
    def state_dict(self, destination=None, prefix='', keep_vars=False) -> Dict:
        original_state_dict = self.model.state_dict()
        mapped_state_dict = {}
        
        for orig_key, tensor in original_state_dict.items():
            # 通用替换规则
            new_key = orig_key
            for k, v in self.layer_name_mapping.items():
                new_key = new_key.replace(k, v)
            
            # 特殊处理attention层
            if 'attn.c_attn' in orig_key:
                if 'weight' in orig_key:
                    # 拆分QKV投影
                    qkv_dim = self.config.hidden_size
                    q_proj = tensor[:self.config.hidden_size]
                    k_proj = tensor[self.config.hidden_size:self.config.hidden_size*2]
                    v_proj = tensor[self.config.hidden_size*2:]
                    mapped_state_dict[f'model.layers.{orig_key.split(".")[2]}.self_attn.q_proj.weight'] = q_proj
                    mapped_state_dict[f'model.layers.{orig_key.split(".")[2]}.self_attn.k_proj.weight'] = k_proj
                    mapped_state_dict[f'model.layers.{orig_key.split(".")[2]}.self_attn.v_proj.weight'] = v_proj
                    continue
                
            mapped_state_dict[new_key] = tensor
        
        return mapped_state_dict

# ----------------------
# 主转换函数
# ----------------------
def convert_to_hf_format(checkpoint_path: str, output_dir: str, tokenizer: Tokenizer):
    # 初始化原始模型
    config = ModuleConfig()
    original_model = MyModule(config)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint['app']['model_state_dict']
    
    # 去除"_orig_mod."前缀
    model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()}
    original_model.load_state_dict(model_state_dict, strict=True)
    
    # 转换为HF配置
    hf_config = LlamaLikeConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_kv_head=config.n_kv_head,
        flash_attn=config.flash_attn,
        dtype=config.dtype,
        ffn_dim_multiplier=config.ffn_dim_multiplier,
        multiple_of=config.multiple_of,
        rope_theta=config.rope_theta,
        use_scaled_rope=config.use_scaled_rope,
        use_block_checkpoint=config.use_block_checkpoint
    )
    
    # 包装模型
    hf_model = HuggingFaceModelWrapper(original_model, hf_config)
    
    # 保存模型
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 移除所有与注意力偏置相关的参数
    save_state_dict = {
        k: v for k, v in hf_model.state_dict().items()
        if "attn.bias" not in k and "self_attn.attn_bias" not in k
    }
    
    # 保存权重
    torch.save(save_state_dict, output_dir / "pytorch_model.bin")
    
    # 保存配置
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config.to_dict(), f, indent=2)
    
    # 保存tokenizer
    #hf_tokenizer = LlamaLikeTokenizer(tokenizer)
    #hf_tokenizer.save_pretrained(output_dir)
    tokenizer.raw_tokenizer.save_pretrained(output_dir)

# ----------------------
# 执行转换
# ----------------------
if __name__ == "__main__":
    # 参数设置
    checkpoint_path = "./checkpoints_epoch_64.pt"
    output_dir = "./converted_hf_model"
    tokenizer = Tokenizer()  # 初始化你的原始tokenizer
    
    # 执行转换
    convert_to_hf_format(checkpoint_path, output_dir, tokenizer)
    
    print(f"Conversion complete. Model saved to {output_dir}")

# config.json
#
#  {
#   "architectures": ["LlamaForCausalLM"],
#   "bos_token_id": 151646,
#   "eos_token_id": 151643,
#   "hidden_act": "silu",
#   "hidden_size": 4096,
#   "initializer_range": 0.02,
#   "intermediate_size": 11008,
#   "max_position_embeddings": 2048,
#   "model_type": "llama",
#   "num_attention_heads": 32,
#   "num_hidden_layers": 32,
#   "pad_token_id": 1516430,
#   "rms_norm_eps": 1e-06,
#   "tie_word_embeddings": true,
#   "torch_dtype": "float32",
#   "vocab_size": 152000
# }