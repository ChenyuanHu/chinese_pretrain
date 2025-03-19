# 训练循环
class TrainConfig:
    batch_size = 1
    num_epochs = 10000
    steps_per_epoch = 2000  # 每个epoch训练多少批次
    gradient_accumulation_steps = 4  # 梯度累积步数

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
    use_block_checkpoint: bool = True


class PretrainConfig:
    path = "opencsg/Fineweb-Edu-Chinese-V2.1"
    data_dir = "4_5"
    split = "train"

    # 样例Prompt
    case_prompts = [
        "中华人民共和国2020年的的中共中央总书记是",
        ""
    ]

    def text_fn(x):
        return x["text"]

class SftConfig:
    path = "Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT"
    data_dir = None
    split = "train"

    # 样例Prompt
    case_prompts = [
        "<|im_start|>用户\n请根据规律填充这两个空缺的数字。 4, 3, 4, 3, 4, 3, （），（）\n<|im_end|>\n<|im_start|>助手\n",
        "<|im_start|>用户\n中华人民共和国的2020年的总书记是谁？\n<|im_end|>\n<|im_start|>助手\n",
        "<|im_start|>用户\n你是谁？\n<|im_end|>\n<|im_start|>助手\n"
    ]

    def text_fn(x):
        return "<|im_start|>用户\n" + x["instruction"] + "\n<|im_end|>\n<|im_start|>助手\n" + x["output"] + "\n<|im_end|>"

class TrainDataConfig:
    data = PretrainConfig()

    # 生成样例配置
    max_tokens = 1000
    temperature = 0.8
    top_k = 60