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
    def __init__(self, source, tokenizer, block_size, cache_dir="./dataset_cache", num_workers=None):
        self.source = source
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache_dir = cache_dir
        self.num_workers = num_workers if num_workers is not None else max(1, mp.cpu_count() - 2)

        self.text_fn = TextFnWrapper(source["text_fn"])
        self.file_path = os.path.join(self.cache_dir, f"{source['name']}_padding.bin")

        self.bucket = None

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
        bucket = {}
        for tokens in encoded_samples:
            length = len(tokens)
            if length not in bucket:
                bucket[length] = []
            bucket[length].append(tokens)
        self.bucket = bucket
    
    def preprocess_to_file(self):
        # 确保目录存在
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

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


class MixTrainDataLoader:
    def __init__(self, world_size, rank, local_rank, batch_size, block_size, cache_dir="./dataset_cache"):
        self.train_data_loaders = {}
        self.loader_names = []
        self.loader_weights = []
        self.cache_dir = cache_dir
        self.block_size = (block_size + 1) * batch_size

        self.tokenizer = Tokenizer()

        data_preparer = DataPreparer(PretrainConfig.datasets[0]["data"], self.tokenizer, self.block_size)
        data_preparer.preprocess_to_file()
        self.data_preparer = data_preparer

        self.bucket = self.data_preparer.bucket
        self.iterator = self.iter()

    def inter_iter(self):
        # 获取所有长度并排序
        sorted_lengths = sorted(self.bucket.keys())
        samples = []
        for length in sorted_lengths:
            tokenss = self.bucket[length]
            for tokens in tokenss:
                if len(tokens) * (len(samples) + 1) > self.block_size:
                    yield samples
                    samples = []
                samples.append(tokens)
        if len(samples) > 0:
            yield samples
    
    def iter(self):
        inter_iter = self.inter_iter()

        for xs in inter_iter:
            assert len(xs) > 0, "len(xs) == 0"
            
            # 检查是否有单个样本长度超过block_size
            if len(xs[0]) > self.block_size:
                # 确保xs只有一个超长样本
                assert len(xs) == 1, "超长样本批次应该只包含一个样本"
                
                # 计算stride（步长），重叠区域为10%
                stride = int(self.block_size * 0.9)
                tokens = xs[0]
                
                last_end = 0
                # 滑动窗口切分长样本
                for start in range(0, len(tokens) - self.block_size + 1, stride):
                    # 提取当前窗口
                    last_end = start + self.block_size
                    window = tokens[start:last_end]
                    yield [window]  # 将window包装在列表中，保持一致的返回格式
                
                # 处理最后一个窗口（如果需要）
                if last_end < len(tokens):
                    last_window = tokens[-self.block_size:]
                    yield [last_window]  # 将last_window包装在列表中
                
            else:
                yield xs

    def next(self):
        xs = next(self.iterator)
        xs_out = []
        ys_out = []
        length = len(xs[-1])
        for i in range(len(xs)):
            if len(xs[i]) != length:
                xs[i].extend([self.tokenizer.eos_token_id] * (length - len(xs[i])))
            xs_out.append(xs[i][:-1])
            ys_out.append(xs[i][1:])

        # tprint(f"batch_size: {len(xs_out)}, sample_len: {len(xs_out[0])}, all_len: {len(xs_out) * len(xs_out[0])}")
        return xs_out, ys_out

    def __iter__(self):
        for xs in self.iter():
            length = len(xs[-1])
            for i in range(len(xs)):
                if len(xs[i]) != length:
                    xs[i].extend([self.tokenizer.eos_token_id] * (length - len(xs[i])))
            yield xs


    def set_data_progress_percentage(self, progress_percentage):
        pass

    def get_data_progress_percentage(self):
        return "{}"

if __name__ == "__main__":
    block_size = 1024
    mix_train_data_loader = MixTrainDataLoader(world_size=1, rank=0, local_rank=0, batch_size=1, block_size=block_size, cache_dir="./dataset_cache")
    
    print("\n按长度从小到大遍历数据：")
    print("-" * 50)
    
    token_count = 0
    pad_count = 0
    for xs in mix_train_data_loader:
        print(len(xs), len(xs[0]), len(xs) * len(xs[0]))
        for x in xs:
            for token in x:
                if token == mix_train_data_loader.tokenizer.pad_token_id:
                    pad_count += 1
                token_count += 1
        print("-" * 50)

    print(f"token_count: {token_count}, pad_count: {pad_count}")
    print(f"pad_ratio: {pad_count / token_count}")