# Epoch 1, Step 28/200, Loss: 9.9251, Tokens/s: 3821.39
# 模型总参数量: 128,729,600

# 训练循环
class TrainConfig:
    batch_size = 4
    gradient_accumulation_steps = 1  # 梯度累积步数
    block_size = 1024

    num_epochs = 10000 # 一般不结束
    scheduler_epochs = 100 # 调度器预期训练收敛需要的epoch数
    steps_per_epoch = 200  # 每个epoch训练多少批次

    # checkpoint config
    save_interval_sec = 180  # 每n秒保存一次模型
    save_dcp_checkpoint = False
    save_normal_checkpoint = True

    # amp
    dtype = "bfloat16"

    compile = "PARTIAL"
    run_mode = "both"

# from model_qwen import QwenModel as Model
from model_custom import CustomModelForCausalLM as Model
from model_custom import CustomModelConfig as ModelConfig

model_config = ModelConfig(
    hidden_size=512,
    num_hidden_layers=16,
    num_attention_heads=16,
    num_key_value_heads=8,
    intermediate_size=1024,
)

from configs.data_sources import *

class PretrainConfig:
    datasets = [
        {
            "enabled": False,
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
    max_tokens = 100
    temperature = 0.8
    top_k = 60