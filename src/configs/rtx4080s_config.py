# Epoch 1, Step 190/2000, Loss: 7.1797, Tokens/s: 25886.21
# 模型总参数量: 240,280,320

# 训练循环
class TrainConfig:
    batch_size = 1
    gradient_accumulation_steps = 4  # 梯度累积步数
    block_size = 1024

    num_epochs = 10000 # 一般不结束
    scheduler_epochs = 100 # 调度器预期训练收敛需要的epoch数
    steps_per_epoch = 20000  # 每个epoch训练多少批次

    # adamw优化器参数
    max_lr = 5e-5
    betas = (0.9, 0.95)
    weight_decay = 0.01
    eps = 1e-8
    # 学习率调度器参数
    pct_start = 0.1 # 预热阶段占总步数的10%
    div_factor = 10 # 起始学习率为max_lr/10
    final_div_factor = 10 # 最终学习率为max_lr/10
    anneal_strategy = "cos" # 余弦退火

    # checkpoint config
    save_interval_sec = 600  # 每n秒保存一次模型
    save_dcp_checkpoint = False
    save_normal_checkpoint = True

    # amp
    dtype = "bfloat16"

    compile = "PARTIAL"
    run_mode = "both"

from model_qwen import QwenModel as Model
# from model_custom import CustomModelForCausalLM as Model
from model_custom import CustomModelConfig as ModelConfig

model_config = ModelConfig()

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
            "enabled": False,
            "data": zh_en_translation,
            "weight": 1
        },
        {
            "enabled": False,
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