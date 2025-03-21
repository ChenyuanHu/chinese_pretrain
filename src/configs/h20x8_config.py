# 训练循环
class TrainConfig:
    batch_size = 2
    gradient_accumulation_steps = 2  # 梯度累积步数

    num_epochs = 10000 # 一般不结束
    scheduler_epochs = 100 # 调度器预期训练收敛需要的epoch数
    steps_per_epoch = 2000  # 每个epoch训练多少批次

    # checkpoint config
    save_interval_sec = 1800  # 每n秒保存一次模型
    save_dcp_checkpoint = True
    save_normal_checkpoint = False

    compile = True

# 模型参数
class ModuleConfig:
    block_size: int = 4096
    vocab_size: int = 128512  # 基础词表128000，加上特殊标记后为128512
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 2560
    n_kv_head: int = 8
    # flash attention. 可选值为 "FLASH_ATTENTION|EFFICIENT_ATTENTION|MATH|CUDNN_ATTENTION", 用竖线多选，为空禁用
    flash_attn: str = "FLASH_ATTENTION"
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
            "data": fineweb_edu_chinese_2p1,
            "weight": 4
        },
        {
            "enabled": True,
            "data": codeparrot_clean,
            "weight": 1
        },
        {
            "enabled": True,
            "data": zh_en_translation,
            "weight": 0.1
        },
        {
            "enabled": True,
            "data": open_r1_math_220k,
            "weight": 0.5
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
    max_tokens = 1000
    temperature = 0.8
    top_k = 60