from log import tprint
from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self):
        # 使用 flagalpha/llama3-chinese-8b-instruct 的分词器
        tprint("正在加载分词器...")
        self.raw_tokenizer = AutoTokenizer.from_pretrained("flagalpha/llama3-chinese-8b-instruct", trust_remote_code=True)
        
        # 打印初始特殊标记信息
        tprint(f"初始词表大小: {self.raw_tokenizer.vocab_size}")
        tprint(f"所有特殊标记: {self.raw_tokenizer.all_special_tokens}")
        tprint(f"所有特殊标记ID: {self.raw_tokenizer.all_special_ids}")
        tprint(f"特殊标记映射: {self.raw_tokenizer.special_tokens_map}")
        
        # 添加特殊标记 <|im_start|> 和 <|im_end|>
        special_tokens_dict = {'additional_special_tokens': ['<|im_start|>', '<|im_end|>']}
        self.raw_tokenizer.add_special_tokens(special_tokens_dict)
        tprint(f"添加了特殊标记: <|im_start|> 和 <|im_end|>")
        
        tprint(f"添加特殊标记后词表大小：{self.raw_tokenizer.vocab_size}")
        tprint(f"添加后所有特殊标记: {self.raw_tokenizer.all_special_tokens}")
        tprint(f"添加后所有特殊标记ID: {self.raw_tokenizer.all_special_ids}")
        
        self.bos_token = self.raw_tokenizer.bos_token
        self.bos_token_id = self.raw_tokenizer.bos_token_id
        self.eos_token = self.raw_tokenizer.eos_token
        self.eos_token_id = self.raw_tokenizer.eos_token_id
        
        # 获取新添加的特殊标记的ID
        self.im_start_id = self.raw_tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.im_end_id = self.raw_tokenizer.convert_tokens_to_ids('<|im_end|>')
        
        tprint(f"BOS token: {self.bos_token}")
        tprint(f"BOS token ID: {self.bos_token_id}")
        tprint(f"EOS token: {self.eos_token}")
        tprint(f"EOS token ID: {self.eos_token_id}")
        tprint(f"IM_START token ID: {self.im_start_id}")
        tprint(f"IM_END token ID: {self.im_end_id}")
    
    def encode(self, text):
        return self.raw_tokenizer.encode(text)

    def decode(self, tokens):
        return self.raw_tokenizer.decode(tokens)


if __name__ == "__main__":
    tokenizer = Tokenizer()
    print(tokenizer.encode("你好"))
    print(tokenizer.decode(tokenizer.encode("你好")))
    
    # 测试新添加的特殊标记
    print(tokenizer.encode("<|im_start|>你好<|im_end|>"))
    print(tokenizer.decode(tokenizer.encode("<|im_start|>你好<|im_end|>")))
