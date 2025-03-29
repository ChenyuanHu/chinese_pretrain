# 训练循环
class TrainConfig:
    batch_size = 6
    gradient_accumulation_steps = 1  # 梯度累积步数

    num_epochs = 10000 # 一般不结束
    scheduler_epochs = 500 # 调度器预期训练收敛需要的epoch数
    steps_per_epoch = 500  # 每个epoch训练多少批次

    # checkpoint config
    save_interval_sec = 7200  # 每n秒保存一次模型
    save_last_n_checkpoints = 5 # 保存最后n个checkpoint
    save_dcp_checkpoint = True         # **** 多几多卡的dcp一定要存储在一个所有机器都能访问到的共享存储上面
    save_normal_checkpoint = False
    use_nfs = True  # nfs的时候只能全局rank 0 才能删除checkpoint

    compile = "FULL"
    full_shard = True
    run_mode = "train" # train, generate, both


# 模型参数
class ModuleConfig:
    block_size: int = 2048
    vocab_size: int = 152000
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
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
    use_block_checkpoint: int = 0 # 使用梯度检查点的block层数

from configs.data_sources import *

class PretrainConfig:
    datasets = [
        {
            "enabled": True,
            "data": fineweb_edu_chinese_2p1,
            "weight": 3
        },
        {
            "enabled": True,
            "data": fineweb_edu_sample_100bt,
            "weight": 3.5
        },
        {
            "enabled": True,
            "data": open_web_math,
            "weight": 1
        },
        {
            "enabled": True,
            "data": codeparrot_clean,
            "weight": 1
        },
        {
            "enabled": False,
            "data": dummy_data,
            "weight": 1
        },
        {
            "enabled": True,
            "data": translation_chinese_2_english,
            "weight": 0.4
        },
        {
            "enabled": True,
            "data": zh_en_translation_v2,
            "weight": 0.1
        },
        {
            "enabled": False,
            "data": open_r1_math_220k,
            "weight": 0.03
        },
        {
            "enabled": False,
            "data": open_math_instruct_2,
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
    dataloader_mode = "packing"   # 可选值为 "padding" 或 "packing"

    # 生成样例配置
    max_tokens = 1000
    temperature = 0.8
    top_k = 60