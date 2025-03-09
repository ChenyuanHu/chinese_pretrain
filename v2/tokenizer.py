from log import tprint
from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self):
        # 使用 flagalpha/llama3-chinese-8b-instruct 的分词器
        tprint("正在加载分词器...")
        self.raw_tokenizer = AutoTokenizer.from_pretrained("flagalpha/llama3-chinese-8b-instruct", trust_remote_code=True)
        tprint(f"分词器加载成功！词汇表大小：{self.raw_tokenizer.vocab_size}")
        self.bos_token = self.raw_tokenizer.bos_token
        self.eos_token_id = self.raw_tokenizer.eos_token_id
        tprint(f"BOS token: {self.bos_token}")
        tprint(f"EOS token ID: {self.eos_token_id}")
    
    def encode(self, text):
        return self.raw_tokenizer.encode(text)

    def decode(self, tokens):
        return self.raw_tokenizer.decode(tokens)