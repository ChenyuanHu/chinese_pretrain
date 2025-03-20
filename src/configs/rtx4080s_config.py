# Epoch 1, Step 190/2000, Loss: 7.1797, Tokens/s: 25886.21
# 模型总参数量: 240,280,320

# 训练循环
class TrainConfig:
    batch_size = 4
    num_epochs = 10000
    steps_per_epoch = 2000  # 每个epoch训练多少批次
    gradient_accumulation_steps = 1  # 梯度累积步数

    # checkpoint config
    save_interval_sec = 1800  # 每n秒保存一次模型
    save_dcp_checkpoint = False
    save_normal_checkpoint = True

    compile = True

# 模型参数
class ModuleConfig:
    block_size: int = 1024
    vocab_size: int = 128512  # 基础词表128000，加上特殊标记后为128512
    n_layer: int = 16
    n_head: int = 16
    n_embd: int = 768
    n_kv_head: int = 8
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
    use_block_checkpoint: bool = False

from configs.data_sources import *

class PretrainConfig:
    datasets = [
        {
            "enabled": True,
            "data": fineweb_edu_chinese_2p1_1percent,
            "weight": 3
        },
        {
            "enabled": True,
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
    data = PretrainConfig()
    case_prompts = pretrain_case_prompts

    # 生成样例配置
    max_tokens = 100
    temperature = 0.1
    top_k = 40