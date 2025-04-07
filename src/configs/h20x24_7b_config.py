# 训练循环
class TrainConfig:
    batch_size = 1
    gradient_accumulation_steps = 4  # 梯度累积步数
    block_size = 2048

    num_epochs = 10000 # 一般不结束
    scheduler_epochs = 100 # 调度器预期训练收敛需要的epoch数
    steps_per_epoch = 500  # 每个epoch训练多少批次

    # adamw优化器参数
    max_lr = 3e-4
    betas = (0.9, 0.95)
    weight_decay = 0.01
    eps = 1e-8
    # 学习率调度器参数
    pct_start = 0.1 # 预热阶段占总步数的10%
    div_factor = 10 # 起始学习率为max_lr/10
    final_div_factor = 10 # 最终学习率为max_lr/10
    anneal_strategy = "cos" # 余弦退火

    # checkpoint config
    save_interval_sec = 1800  # 每n秒保存一次模型
    save_dcp_checkpoint = True         # **** 多几多卡的dcp一定要存储在一个所有机器都能访问到的共享存储上面
    save_normal_checkpoint = False
    use_nfs = True  # nfs的时候只能全局rank 0 才能删除checkpoint

    # amp
    dtype = "bfloat16"

    compile = "FULL"
    full_shard = True
    run_mode = "train" # train, generate, both

# from model_qwen import QwenModel as Model
from model_custom import CustomModelForCausalLM as Model
from model_custom import CustomModelConfig as ModelConfig

model_config = ModelConfig(
    hidden_size=2048,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=5460,
)

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
            "data": zh_en_translation_v2,
            "weight": 0.008
        },
        {
            "enabled": True,
            "data": open_r1_math_220k,
            "weight": 0.03
        },
        {
            "enabled": True,
            "data": open_math_instruct_2,
            "weight": 0.5
        },
        {
            "enabled": True,
            "data": fineweb_sample_10bt,
            "weight": 2
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