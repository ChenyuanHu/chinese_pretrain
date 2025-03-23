import torch
import torch.nn.functional as F
import random
from log import tprint
from tokenizer import Tokenizer

class TextGenerator:
    def __init__(self, model, block_size, train_data_config, device="cpu", amp=None):
        self.model = model
        self.block_size = block_size
        self.tokenizer = Tokenizer()
        self.prompts = train_data_config.case_prompts
        self.max_tokens = train_data_config.max_tokens
        self.temperature = train_data_config.temperature
        self.top_k = train_data_config.top_k
        self.device = device
        self.amp = amp
        tprint(f"{self.amp}")

    # 定义文本生成函数
    def generate_text(self, prompt):
        self.model.eval()
        
        prompt = self.tokenizer.bos_token + prompt
        # 编码输入提示
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > self.block_size - self.max_tokens:
            # 如果提示太长，只保留后面部分
            tokens = tokens[-(self.block_size - self.max_tokens):]
        
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, seq_len]
        
        tprint(f"{self.amp}")
        # 确保与模型使用相同的数据类型进行推理
        with self.amp:
            while tokens.size(1) < self.max_tokens:
                # 获取预测
                if tokens.size(1) > self.block_size:  # 使用size(1)直接获取序列长度
                    # 如果序列太长，只保留后面的部分
                    tokens = tokens[:, -(self.block_size):]
                
                # 前向传播
                logits, _ = self.model(tokens)
                
                # 获取最后一个位置的预测
                logits = logits[:, -1, :] / self.temperature
                
                # 应用top-k采样
                if self.top_k > 0:
                    v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # 应用softmax获取概率分布
                probs = F.softmax(logits, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 追加到序列
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # 如果生成了结束标记，提前结束
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # 解码生成的token序列
        generated_tokens = tokens[0].tolist()
        
        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text

    # 在训练循环中生成文本的辅助函数
    def generate_examples(self):
        tprint("生成示例文本：")
        
        self.model.eval()  # 确保模型处于评估模式
        # 随机选择一个提示
        prompt = random.choice(self.prompts)
        tprint(f"\n提示: {prompt if prompt else '(无提示)'}")
        try:
            with torch.no_grad():  # 确保不会计算梯度
                generated = self.generate_text(prompt)
                tprint(f"生成: {generated}")
        except Exception as e:
            tprint(f"生成文本时发生错误: {str(e)}")
            # 打印更详细的错误信息以便调试
            import traceback
            tprint(f"详细错误信息: {traceback.format_exc()}")
            tprint("继续训练而不中断...")
        tprint("-"*50)