from log import tprint
from transformers import AutoTokenizer

# flagalpha/llama3-chinese-8b-instruct
# deepseek-ai/deepseek-r1-distill-qwen-32b
class Tokenizer:
    def __init__(self, model_name="deepseek-ai/deepseek-r1-distill-qwen-32b"):
        # 使用 flagalpha/llama3-chinese-8b-instruct 的分词器
        tprint("正在加载分词器...")
        self.raw_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        self.bos_token = self.raw_tokenizer.bos_token
        self.bos_token_id = self.raw_tokenizer.bos_token_id
        self.eos_token = self.raw_tokenizer.eos_token
        self.eos_token_id = self.raw_tokenizer.eos_token_id
        
        self.user_token = '<｜User｜>'
        self.assistant_token = '<｜Assistant｜>'
        self.user_token_id = self.raw_tokenizer.convert_tokens_to_ids(self.user_token)
        self.assistant_token_id = self.raw_tokenizer.convert_tokens_to_ids(self.assistant_token)
        self.pad_token = self.raw_tokenizer.pad_token
        self.pad_token_id = self.raw_tokenizer.convert_tokens_to_ids(self.pad_token)

        tprint(f"bos_token: {self.bos_token} bos_token_id: {self.bos_token_id} eos_token: {self.eos_token} eos_token_id: {self.eos_token_id}")
        tprint(f"pad_token: {self.pad_token} pad_token_id: {self.pad_token_id}")
        tprint(f"user_token: {self.user_token} user_token_id: {self.user_token_id}")
        tprint(f"assistant_token: {self.assistant_token} assistant_token_id: {self.assistant_token_id}")
        tprint(f"tokenizer.vocab_size: {self.raw_tokenizer.vocab_size}")

    def decode(self, tokens):
        return self.raw_tokenizer.decode(tokens)

    # Token indices sequence length is longer than the specified maximum sequence length for this model (17692 > 16384). Running this sequence through the model will result in indexing errors
    # 这个告警不会影响分词
    def encode(self, text):
        return self.raw_tokenizer.encode(text)[1:]  # qwen-32b 分词器会多一个<｜begin▁of▁sentence｜>

    def dump(self):
        self.raw_tokenizer.save_pretrained("save_tokenizer")


if __name__ == "__main__":
    tokenizer = Tokenizer()
    print(tokenizer.encode("你好"))
    print(len(tokenizer.encode("你好" * 17000)))
    print(tokenizer.encode("你好" * 17000)[:10])
    print(tokenizer.encode("你好" * 17000)[-10:])
    print(tokenizer.decode(tokenizer.encode("你好")))
    
    # 测试新添加的特殊标记
    print(tokenizer.encode(tokenizer.user_token + "你好" + tokenizer.assistant_token))
    print(tokenizer.decode(tokenizer.encode(tokenizer.user_token + "你好" + tokenizer.assistant_token)))

    print(tokenizer.encode(tokenizer.bos_token + "你好" + tokenizer.eos_token))
    print(tokenizer.decode(tokenizer.encode(tokenizer.bos_token + "你好" + tokenizer.eos_token)))
