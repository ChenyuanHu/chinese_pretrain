from log import tprint
from transformers import AutoTokenizer

# flagalpha/llama3-chinese-8b-instruct
# deepseek-ai/deepseek-r1-distill-qwen-32b
class Tokenizer:
    def __init__(self, model_name="deepseek-ai/deepseek-r1-distill-qwen-32b"):
        # 使用 flagalpha/llama3-chinese-8b-instruct 的分词器
        tprint("正在加载分词器...")
        self.raw_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
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
        tprint(f"bos_token: {self.bos_token} bos_token_id: {self.bos_token_id} eos_token: {self.eos_token} eos_token_id: {self.eos_token_id}")
        tprint(f"tokenizer.model_max_length: {self.raw_tokenizer.model_max_length}")
        
        # 获取新添加的特殊标记的ID
        self.im_start_id = self.raw_tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.im_end_id = self.raw_tokenizer.convert_tokens_to_ids('<|im_end|>')
    
    def decode(self, tokens):
        return self.raw_tokenizer.decode(tokens)

# 有些分词器处理不了太长的序列，会报错
# Token indices sequence length is longer than the specified maximum sequence length for this model (17692 > 16384). Running this sequence through the model will result in indexing errors
# 不过这种分快处理没有考虑到边界情况下，拿到的分词不是最优的情况
    def encode_split(self, text, max_length=16384):
        if text is None:
            raise ValueError("text is None")
        
        if len(text) <= max_length:
            return self.raw_tokenizer.encode(text)

        n_blocks = len(text) // max_length
        last_n_chars = len(text) % max_length
        
        chunks = []
        for i in range(n_blocks):
            chunk = text[i * max_length:(i + 1) * max_length]
            chunk_tokens = self.raw_tokenizer.encode(chunk)
            chunks.extend(chunk_tokens)
        
        if last_n_chars > 0:
            chunk = text[-last_n_chars:]
            chunk_tokens = self.raw_tokenizer.encode(chunk)
            chunks.extend(chunk_tokens)
        
        return chunks

# 大部分情况下按最大长度分块是OK的，小部分情况得退避
    def encode(self, text):
        count = 0
        max_length = int(self.raw_tokenizer.model_max_length * 0.8)
        return self.encode_split(text, max_length)


if __name__ == "__main__":
    tokenizer = Tokenizer()
    print(tokenizer.encode("你好"))
    print(tokenizer.decode(tokenizer.encode("你好")))
    
    # 测试新添加的特殊标记
    print(tokenizer.encode("<|im_start|>你好<|im_end|>"))
    print(tokenizer.decode(tokenizer.encode("<|im_start|>你好<|im_end|>")))
