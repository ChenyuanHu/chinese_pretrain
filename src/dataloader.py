from datasets import load_dataset
import os
import mmap
from log import tprint
import multiprocessing as mp
import uuid
import glob
import time
import re

def sanitize_path(path_str):
    """清理路径字符串，使其对Windows和Unix系统都友好"""
    # 替换Windows不允许的字符
    sanitized = re.sub(r'[\\/*?:"<>|[\]%]', '_', str(path_str))
    return sanitized

# 创建包装器类用于处理函数参数
class TextFnWrapper:
    def __init__(self, fn):
        self.fn = fn
        
    def __call__(self, x):
        return self.fn(x)

class DataMapper:
    def __init__(self, path, data_dir, split, tokenizer, text_fn, cache_dir="./dataset_cache", num_workers=None):
        self.path = path
        self.data_dir = data_dir
        self.split = split

        self.tokenizer = tokenizer
        self.text_fn = TextFnWrapper(text_fn)

        self.cache_dir = cache_dir
        # 确保文件路径是系统兼容的
        safe_path = sanitize_path(path)
        safe_data_dir = sanitize_path(data_dir)
        safe_split = sanitize_path(split)
        self.file_path = os.path.join(self.cache_dir, safe_path, f"{safe_data_dir}_{safe_split}.bin")
        self.num_workers = num_workers if num_workers is not None else max(1, mp.cpu_count() - 2)

    def _process_chunk(self, chunk_data, worker_id, temp_dir):
        """处理数据集的一个分片"""
        buffer_size = 10000000
        token_buffer = []
        temp_file_path = os.path.join(temp_dir, f"chunk_{worker_id}_{uuid.uuid4().hex}.bin")
        
        # 记录开始时间和上次打印时间
        start_time = time.time()
        last_log_time = start_time
        total_items = len(chunk_data)
        
        with open(temp_file_path, "wb") as f:
            for i, item in enumerate(chunk_data):
                text = self.text_fn(item)
                encoded = self.tokenizer.encode(text)
                tokens = [self.tokenizer.bos_token_id] + encoded + [self.tokenizer.eos_token_id]
                
                token_buffer.extend(tokens)
                
                # 每30秒打印一次进度
                current_time = time.time()
                if current_time - last_log_time >= 30:
                    progress = (i + 1) / total_items * 100
                    elapsed = current_time - start_time
                    tprint(f"Worker {worker_id}: 已处理 {i+1}/{total_items} 项 ({progress:.2f}%), 用时 {elapsed:.2f}秒")
                    last_log_time = current_time
                
                if len(token_buffer) >= buffer_size:
                    # 使用3字节存储每个token ID
                    bytes_data = bytearray()
                    for token in token_buffer:
                        # 将token ID转换为3字节
                        bytes_data.extend(token.to_bytes(3, byteorder='little'))
                    f.write(bytes_data)
                    token_buffer = []
            
            if token_buffer:
                # 使用3字节存储每个token ID
                bytes_data = bytearray()
                for token in token_buffer:
                    # 将token ID转换为3字节
                    bytes_data.extend(token.to_bytes(3, byteorder='little'))
                f.write(bytes_data)
                
        # 完成时打印最终进度
        total_time = time.time() - start_time
        tprint(f"Worker {worker_id}: 已完成全部 {total_items} 项处理, 总用时 {total_time:.2f}秒")
        
        return temp_file_path
    
    def _merge_temp_files(self, temp_files, output_file):
        """合并所有临时文件到最终输出文件"""
        with open(output_file, "wb") as outf:
            for temp_file in temp_files:
                with open(temp_file, "rb") as inf:
                    outf.write(inf.read())
                # 合并后删除临时文件
                os.remove(temp_file)
    
    def preprocess_to_file(self):
        # 确保目录存在
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if os.path.exists(self.file_path):
            tprint(f"数据集预处理文件已存在: {self.file_path}")
            return

        tprint(f"正在加载数据集: {self.path}, {self.data_dir}, {self.split}...")
        raw_dataset = load_dataset(self.path, data_dir=self.data_dir, split=self.split)
        tprint(f"数据集长度: {len(raw_dataset)}")
        
        # 创建临时目录
        temp_dir = os.path.join(self.cache_dir, "temp", f"{uuid.uuid4()}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 计算每个进程处理的数据量
        num_workers = min(self.num_workers, len(raw_dataset))
        if num_workers <= 1:
            # 如果只有一个工作进程或数据集很小，使用单进程处理
            tprint(f"使用单进程处理数据集")
            temp_file = self._process_chunk(raw_dataset, 0, temp_dir)
            os.rename(temp_file, self.file_path)
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
                temp_files = []
                
                # 显示进度的计数器
                completed = 0
                
                # 使用迭代器获取结果，这样可以在处理完成后立即获取结果
                for _, temp_file in enumerate(pool.starmap(self._process_chunk, [(chunk, i, temp_dir) for i, chunk in enumerate(chunks)])):
                    temp_files.append(temp_file)
                    completed += 1
                    tprint(f"已完成 {completed}/{num_workers} 个数据块, 占比 {completed/num_workers:.2%}")
            
            # 合并所有临时文件
            tprint(f"正在合并临时文件...")
            self._merge_temp_files(temp_files, self.file_path)
            
            # 清理临时目录
            try:
                remaining_files = glob.glob(os.path.join(temp_dir, "*"))
                for f in remaining_files:
                    os.remove(f)
                os.rmdir(temp_dir)
            except Exception as e:
                tprint(f"清理临时文件时出错: {e}")
        
        tprint(f"数据集预处理完毕，已保存到: {self.file_path}")

    def map_to_array(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"找不到预处理文件：{self.file_path}. 请使用python dataloader.py预处理数据集")
            
        # 使用mmap将文件映射到内存
        class MemoryMappedTokens:
            def __init__(self, filename):
                self.filename = filename
                self.file_size = os.path.getsize(filename)
                self.bytes_per_token = 3  # 每个token现在是3字节
                self.num_tokens = self.file_size // self.bytes_per_token
                
                # 打开文件并创建内存映射
                self.file = open(filename, 'rb')
                self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
                
            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    # 处理切片操作
                    start, stop, step = idx.indices(self.num_tokens)
                    return [self[i] for i in range(start, stop, step)]
                
                if idx < 0:  # 处理负索引
                    idx += self.num_tokens
                
                if not 0 <= idx < self.num_tokens:
                    raise IndexError("索引超出范围")
                
                # 计算在文件中的位置并读取token
                pos = idx * self.bytes_per_token
                self.mm.seek(pos)
                chunk = self.mm.read(self.bytes_per_token)
                # 将3字节转换为整数
                return int.from_bytes(chunk, byteorder='little')
                
            def __len__(self):
                return self.num_tokens
                
            def __del__(self):
                # 确保在对象被销毁时关闭资源
                if hasattr(self, 'mm') and self.mm:
                    self.mm.close()
                if hasattr(self, 'file') and self.file:
                    self.file.close()
                    
        return MemoryMappedTokens(self.file_path)


from config import PretrainConfig, SftConfig, TrainDataConfig
from tokenizer import Tokenizer

class TrainDataLoader:
    def __init__(self, world_size, rank, local_rank, batch_size, block_size, tokenizer):
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.batch_size = batch_size
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.data = TrainDataConfig().data
        self.data_mapper = DataMapper(self.data.path, self.data.data_dir, self.data.split, self.tokenizer, self.data.text_fn, num_workers=None)

        self.tokens = self.data_mapper.map_to_array()
        self.all_tokens_len = len(self.tokens)

        node_data_len = self.all_tokens_len // self.world_size
        tprint(f"总体样本token数量: {self.all_tokens_len}, 每个节点样本token数量: {node_data_len}")

        self.offset_start = self.rank * node_data_len
        self.offset_end = self.offset_start + node_data_len
        tprint(f"当前节点token offset: {self.offset_start} - {self.offset_end}")
        
        # 初始化当前位置指针
        self.current_position = self.offset_start
        # 计算当前节点数据的总长度
        self.node_data_len = self.offset_end - self.offset_start

    def next(self):
        # 准备batch数据
        xs = []
        ys = []
        
        for _ in range(self.batch_size):
            # 如果剩余数据不足block_size + 1, 回到起点，因为要预留一个的y位置
            if self.current_position + self.block_size + 1 > self.offset_end:
                self.current_position = self.offset_start
            
            # 获取一个数据块
            x = self.tokens[self.current_position:self.current_position + self.block_size + 1]
            xs.append(x[:-1])
            ys.append(x[1:])
            
            # 移动指针
            self.current_position += self.block_size
        
        # 计算遍历进度百分比
        progress_percentage = ((self.current_position - self.offset_start) % self.node_data_len) / self.node_data_len * 100
        
        return xs, ys, progress_percentage


if __name__ == "__main__":
    tokenizer = Tokenizer()
    data_mapper = DataMapper(PretrainConfig.path, PretrainConfig.data_dir, PretrainConfig.split, tokenizer, PretrainConfig.text_fn, num_workers=None)
    data_mapper.preprocess_to_file()
    tokens = data_mapper.map_to_array()
    tprint(f"pretrain tokens length: {len(tokens)}")

    data_mapper = DataMapper(SftConfig.path, SftConfig.data_dir, SftConfig.split, tokenizer, SftConfig.text_fn, num_workers=None)
    data_mapper.preprocess_to_file()
    tokens = data_mapper.map_to_array()
    tprint(f"sft tokens length: {len(tokens)}")