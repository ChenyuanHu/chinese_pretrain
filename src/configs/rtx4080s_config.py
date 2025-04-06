# Epoch 1, Step 190/2000, Loss: 7.1797, Tokens/s: 25886.21
# 模型总参数量: 240,280,320

# 训练循环
class TrainConfig:
    batch_size = 1
    gradient_accumulation_steps = 4  # 梯度累积步数
    block_size: int = 1024

    num_epochs = 10000 # 一般不结束
    scheduler_epochs = 100 # 调度器预期训练收敛需要的epoch数
    steps_per_epoch = 2000  # 每个epoch训练多少批次

    # checkpoint config
    save_interval_sec = 180  # 每n秒保存一次模型
    save_dcp_checkpoint = False
    save_normal_checkpoint = True

    dtype = "bfloat16"

    compile = "PARTIAL"
    run_mode = "both"

# from model_qwen import QwenModel as Model
from model_custom import CustomModelForCausalLM as Model

# 模型参数
class ModuleConfig2:
    block_size: int = 1024
    vocab_size: int = 152000
    n_layer: int = 16
    n_head: int = 16
    n_embd: int = 768
    n_kv_head: int = 8
    tie_weights: bool = True
    # flash attention. 可选值为 "FLASH_ATTENTION|EFFICIENT_ATTENTION|MATH|CUDNN_ATTENTION", 用竖线多选，为空禁用
    flash_attn: str = "FLASH_ATTENTION|EFFICIENT_ATTENTION"
    # amp
    dtype = "bfloat16"
    # MLP
    ffn_dim_multiplier: float = 1.3
    multiple_of: int = 1024
    # RoPE
    rope_theta: float = 500000.0
    use_scaled_rope: bool = True
    use_block_checkpoint: int = 0 # 使用梯度检查点的block层数

from transformers import PretrainedConfig
class ModelConfig(PretrainedConfig):
    model_type = "qwen2"

    def __init__(
        self,
        vocab_size=152000,
        hidden_size=768,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=1024,
        rope_theta=1000000.0,
        hidden_act="silu",
        intermediate_size=2048,
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

from configs.data_sources import *

class PretrainConfig:
    datasets = [
        {
            "enabled": True,
            "data": fineweb_edu_chinese_2p1_1percent,
            "weight": 3
        },
        {
            "enabled": False,
            "data": codeparrot_clean_1percent,
            "weight": 1
        },
        {
            "enabled": True,
            "data": zh_en_translation,
            "weight": 1
        },
        {
            "enabled": True,
            "data": open_r1_math_220k,
            "weight": 1
        }
    ]
    

class SftConfig:
    datasets = [
        {
            "enabled": True,
            "data": sft_r1_distill,
            "weight": 1
        }
    ]

from configs.case_prompts import *

class TrainDataConfig:
    data = SftConfig()
    case_prompts = sft_case_prompts
    dataloader_mode = "packing"   # 可选值为 "padding" 或 "packing"

    # 生成样例配置
    max_tokens = 1000
    temperature = 0.8
    top_k = 60