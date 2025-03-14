# 训练循环
class TrainConfig:
    is_sft = False
    use_data_percent = 80
    batch_size = 6
    num_epochs = 10000
    steps_per_epoch = 2000  # 每个epoch训练多少批次
    gradient_accumulation_steps = 1  # 梯度累积步数
    save_interval_sec = 1800  # 每n秒保存一次模型

# 模型参数
class ModuleConfig:
    block_size: int = 1024
    vocab_size: int = 128256  # 词表大小实际是128000，但是eos token id是128001，所以对齐到128256
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 2560
    n_kv_head: int = 8
    # amp
    dtype = "bfloat16"
    # MLP
    ffn_dim_multiplier: float = 1.3
    multiple_of: int = 1024
    # RoPE
    rope_theta: float = 500000.0
    use_scaled_rope: bool = True

class DemoConfig:
    # 生成不同提示的文本
    sft_prompts = [
        "系统提示：你是一个叫小伽的人工智能小助手，你的思考过程放在<think></think>标签中\n用户：请根据规律填充这两个空缺的数字。 4, 3, 4, 3, 4, 3, （），（）\n助手：",
        "系统提示：你是一个叫小伽的人工智能小助手，你的思考过程放在<think></think>标签中\n用户：中华人民共和国的现在的主席是谁？\n助手：",
        "系统提示：你是一个叫小伽的人工智能小助手，你的思考过程放在<think></think>标签中\n用户：你是谁？\n助手："
    ]
    pretrain_prompts = [
        "中华人民共和国现在的中共中央总书记是",
        ""
    ]
    max_tokens = 100
    temperature = 0.1
    top_k = 40