import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='交互式对话程序')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--max_length', type=int, default=2048, help='最大序列长度')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='生成的最大token数量')
    parser.add_argument('--top_p', type=float, default=0.9, help='核采样参数')
    parser.add_argument('--top_k', type=int, default=50, help='top-k采样参数，设置为0则禁用')
    parser.add_argument('--temperature', type=float, default=0.8, help='采样温度')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    parser.add_argument('--max_history_rounds', type=int, default=0, help='保留的对话历史轮数，默认为0，不使用历史记录')
    parser.add_argument('--generate_mode', action='store_true', help='使用补全模式而不是对话模式')
    return parser.parse_args()

from module import MyModule
from configs.h20x64_7b_config import ModuleConfig
from tokenizer import Tokenizer

class ChatBot:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.tokenizer = Tokenizer()
        self.load_model()
        
    def load_model(self):
        """加载模型"""
        print(f"正在加载模型: {self.args.model_path}")
        config = ModuleConfig()
        
        self.model = MyModule(config)
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['app_state']['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()
        print("模型加载完成！")
        
    def generate(self, input_text, max_new_tokens=100):
        """生成回复"""
        input_ids = torch.tensor(self.tokenizer.encode(input_text)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            generated_ids = []
            past_tokens = input_ids
            
            for _ in range(max_new_tokens):
                # 向前传播模型
                logits, _ = self.model(past_tokens)
                
                # 获取最新的token的logits
                next_token_logits = logits[:, -1, :]
                
                # 应用温度
                next_token_logits = next_token_logits / self.args.temperature
                
                # 首先应用top-k采样（如果启用）
                if self.args.top_k > 0:
                    top_k = min(self.args.top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # 然后应用top-p核采样
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除概率质量超过top_p的token
                sorted_indices_to_remove = cumulative_probs > self.args.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for batch_idx in range(next_token_logits.shape[0]):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = -float("Inf")
                
                # 采样下一个token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 如果生成了结束标记，停止生成
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                # 添加生成的token
                generated_ids.append(next_token.item())
                
                # 更新past_tokens以包含新生成的token
                past_tokens = torch.cat([past_tokens, next_token], dim=1)
                
                # 如果序列过长，保留最近的tokens
                if past_tokens.shape[1] > self.args.max_length:
                    past_tokens = past_tokens[:, -self.args.max_length:]
            
            return self.tokenizer.decode(generated_ids)
        
    def chat(self):
        """交互式对话"""
        print("=" * 60)
        if self.args.generate_mode:
            print("欢迎使用AI补全系统，输入'退出'或'exit'结束")
        else:
            print("欢迎使用AI对话系统，输入'退出'或'exit'结束对话")
        print("=" * 60)
        
        history = []
        
        while True:
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() in ['退出', 'exit', 'quit']:
                print("谢谢使用，再见！")
                break
                
            if not user_input:
                continue
            
            # 生成提示文本
            if self.args.generate_mode:
                # 补全模式：直接使用用户输入
                prompt = user_input
            else:
                # 对话模式：使用特定格式
                prompt = ""
                # 添加历史对话
                if self.args.max_history_rounds > 0:
                    for h in history[-self.args.max_history_rounds:]:
                        prompt += f"{self.tokenizer.user_token}{h[0]}{self.tokenizer.assistant_token}{h[1]}"
                # 添加当前用户输入
                prompt += f"{self.tokenizer.user_token}{user_input}{self.tokenizer.assistant_token}"
            
            # 生成回复
            try:
                response = self.generate(prompt, max_new_tokens=self.args.max_new_tokens)
                print(f"AI: {response}")
                
                # 更新历史
                history.append((user_input, response))
                
                # 如果历史太长，移除最早的对话
                if len(history) > 20:  # 保留更长的历史用于记录，实际使用由max_history_rounds控制
                    history.pop(0)
                    
            except Exception as e:
                print(f"生成回复时出错: {e}")
                
def main():
    args = parse_arguments()
    chatbot = ChatBot(args)
    chatbot.chat()
    
if __name__ == "__main__":
    main()
