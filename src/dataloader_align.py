import os
from log import tprint
import multiprocessing as mp

from config import PretrainConfig, SftConfig, TrainDataConfig
from tokenizer import Tokenizer

# 创建包装器类用于处理函数参数
class TextFnWrapper:
    def __init__(self, fn):
        self.fn = fn
        
    def __call__(self, x):
        return self.fn(x)

class DataPreparer:
    def __init__(self, source, tokenizer, cache_dir="./dataset_cache", num_workers=None):
        self.source = source
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.num_workers = num_workers if num_workers is not None else max(1, mp.cpu_count() - 2)

        self.text_fn = TextFnWrapper(source["text_fn"])
        self.file_path = os.path.join(self.cache_dir, f"{source['name']}.bin")

    def _process_chunk(self, chunk_data, worker_id):
        encoded_samples = []
        for item in chunk_data:
            text = self.text_fn(item)
            if text is None or text == "":
                continue
            encoded = self.tokenizer.encode(text)
            tokens = [self.tokenizer.bos_token_id] + encoded + [self.tokenizer.eos_token_id]
            encoded_samples.append(tokens)

        return encoded_samples

    def _merge_samples(self, encoded_samples, file_path):
        sorted_encoded_samples = sorted(encoded_samples, key=lambda x: len(x))
        bucket = {}
        for tokens in sorted_encoded_samples:
            bucket[len(tokens)] = bucket.get(len(tokens), 0) + 1
        for length, count in sorted(bucket.items()):
            tprint(f"长度为 {length} 的样本数量: {count}")
    
    def preprocess_to_file(self):
        # 确保目录存在
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if os.path.exists(self.file_path):
            tprint(f"数据集预处理文件已存在: {self.file_path}")
            return self.file_path

        tprint(f"正在加载数据集: {self.source['name']}...")
        raw_dataset = self.source["ds_fn"]()
        tprint(f"数据集长度: {len(raw_dataset)}")
        
        # 计算每个进程处理的数据量
        num_workers = min(self.num_workers, len(raw_dataset))
        if num_workers <= 1:
            # 如果只有一个工作进程或数据集很小，使用单进程处理
            tprint(f"使用单进程处理数据集")
            encoded_samples = self._process_chunk(raw_dataset, 0)
        else:
            # 将数据集分成多个块
            chunk_size = len(raw_dataset) // num_workers
            chunks = []
            for i in range(num_workers):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < num_workers - 1 else len(raw_dataset)
                chunks.append(raw_dataset.select(range(start_idx, end_idx)))
            
            tprint(f"将数据集分成 {num_workers} 个块进行并行处理")
            
            # 创建进程池并并行处理数据
            with mp.Pool(processes=num_workers) as pool:
                encoded_samples = []
                
                # 使用迭代器获取结果，这样可以在处理完成后立即获取结果
                for encoded_samples in pool.starmap(self._process_chunk, [(chunk, i) for i, chunk in enumerate(chunks)]):
                    encoded_samples.extend(encoded_samples)
            
        # 合并所有临时文件
        tprint(f"分析处理样本")
        self._merge_samples(encoded_samples, self.file_path)
        
        tprint(f"数据集预处理完毕，已保存到: {self.file_path}")
        return self.file_path
        


if __name__ == "__main__":
    tokenizer = Tokenizer()
    for source in SftConfig.datasets:
        if not source["enabled"]:
            continue
        data_preparer = DataPreparer(source["data"], tokenizer)
        path = data_preparer.preprocess_to_file()