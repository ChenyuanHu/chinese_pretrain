from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import torch

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
config.sliding_window = None  # 完全禁用滑动窗口
config.use_cache = True
config.use_sdpa = True

# 1. 加载模型与分词器
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    config=config,
    torch_dtype=torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

print(tokenizer.pad_token)
print(tokenizer.pad_token_id)
print(tokenizer.eos_token)
print(tokenizer.eos_token_id)
print(tokenizer.bos_token)
print(tokenizer.bos_token_id)

dataset = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", data_dir=None, split="train")

# 3. 数据预处理函数
def format_function(x):
    text = "用户：" + x["instruction"] + "\n\n模型：" + x["output"]
    return {"text": text}

format_dataset = dataset.map(format_function, remove_columns=dataset.column_names)

tokenized_dataset = format_dataset.map(
    lambda x: tokenizer(
        x["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors=None  # 不使用return_tensors，因为map不能处理批量的张量
    ),
    batched=True,
    remove_columns=format_dataset.column_names
)

# 4. 训练参数配置
training_args = TrainingArguments(
    output_dir="./qwen2.5-0.5b-finetuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=0.01,
    fp16=True,  # MPS 设备不支持fp16，需要禁用
    eval_strategy="no",
    save_strategy="epoch",
    logging_steps=1,
    remove_unused_columns=False  # 避免在数据处理时删除必要的列
)

# 5. 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=None,
    data_collator=lambda data: {
        "input_ids": torch.tensor([d["input_ids"] for d in data]),
        "attention_mask": torch.tensor([d["attention_mask"] for d in data]),
        "labels": torch.tensor([d["input_ids"] for d in data])
    }
)

# 6. 开始微调
trainer.train()

# 7. 保存模型
model.save_pretrained("./qwen2.5-0.5b-finetuned")
tokenizer.save_pretrained("./qwen2.5-0.5b-finetuned")

# 生成示例文本
input_text = "用户：你好\n\n模型："
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(
    **input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))