from transformers import AutoConfig, AutoModelForCausalLM


def QwenModel(module_config):
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    config.sliding_window = None  # 完全禁用滑动窗口
    config.use_cache = True
    config.use_sdpa = False
    model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            config=config,
            device_map="auto"
    )
    return model

