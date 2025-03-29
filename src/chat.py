import torch
import argparse
import torch.nn.functional as F

def parse_arguments():
    parser = argparse.ArgumentParser(description='交互式对话程序')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--max_length', type=int, default=2048, help='最大序列长度')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='生成的最大token数量')
    parser.add_argument('--top_p', type=float, default=1.0, help='核采样参数')
    parser.add_argument('--top_k', type=int, default=60, help='top-k采样参数，设置为0则禁用')
    parser.add_argument('--temperature', type=float, default=0.8, help='采样温度')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    parser.add_argument('--max_history_rounds', type=int, default=0, help='保留的对话历史轮数，默认为0，不使用历史记录')
    parser.add_argument('--generate_mode', action='store_true', help='使用补全模式而不是对话模式')
    parser.add_argument('--compile', action='store_true', help='使用compile')
    return parser.parse_args()

from module import MyModule
from config import ModuleConfig
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
        checkpoint = torch.load(self.args.model_path, map_location=self.device, weights_only=True)
        if 'app' in checkpoint:
            model_state_dict = checkpoint['app']['model_state_dict']
        else:
            model_state_dict = checkpoint['model_state_dict']

        if self.args.compile:
            self.model = torch.compile(self.model)
        else:
            # 去除 "_orig_mod." 前缀，因为compile时会自动添加
            model_state_dict = {
                key.replace("_orig_mod.", ""): value 
                for key, value in model_state_dict.items()
            }
        self.model.load_state_dict(model_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("模型加载完成！")
        
    def generate(self, input_text, max_new_tokens=100, stream_callback=None):
        """生成回复，支持流式输出
        
        参数:
            input_text: 输入文本
            max_new_tokens: 最大生成token数
            stream_callback: 流式回调函数，接收当前生成的token作为参数
        返回:
            完整的生成文本
        """
        input_ids = torch.tensor(self.tokenizer.encode(input_text)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            generated_ids = []
            tokens = input_ids
            
            for _ in range(max_new_tokens):
                # 如果序列太长，只保留后面的部分
                if tokens.size(1) > self.args.max_length:
                    tokens = tokens[:, -(self.args.max_length):]
                
                # 前向传播
                logits, _ = self.model(tokens)
                
                # 获取最后一个位置的预测
                next_token_logits = logits[:, -1, :] / self.args.temperature
                
                # 应用top-k采样
                if self.args.top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(self.args.top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
                
                # 应用top-p核采样（如果启用）
                if self.args.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除概率质量超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > self.args.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for batch_idx in range(next_token_logits.shape[0]):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = -float("Inf")
                
                # 应用softmax获取概率分布
                probs = F.softmax(next_token_logits, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 追加到序列
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # 如果生成了结束标记，停止生成
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                # 添加生成的token到结果列表
                token_id = next_token.item()
                generated_ids.append(token_id)
                
                # 流式输出
                if stream_callback is not None:
                    # 解码当前生成的token
                    token_text = self.tokenizer.decode([token_id])
                    stream_callback(token_text)
            
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
            
            # 流式输出回调函数
            print("AI: ", end="", flush=True)
            full_response = ""
            
            def stream_output(text):
                nonlocal full_response
                full_response += text
                print(text, end="", flush=True)
            
            # 生成回复
            try:
                response = self.generate(prompt, max_new_tokens=self.args.max_new_tokens, stream_callback=stream_output)
                print()  # 添加换行
                
                # 更新历史
                history.append((user_input, full_response))
                
                # 如果历史太长，移除最早的对话
                if len(history) > 20:  # 保留更长的历史用于记录，实际使用由max_history_rounds控制
                    history.pop(0)
                    
            except Exception as e:
                print(f"\n生成回复时出错: {e}")
                
def main():
    args = parse_arguments()
    chatbot = ChatBot(args)
    chatbot.chat()
    
if __name__ == "__main__":
    main()
