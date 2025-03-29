import os
import mmap
from log import tprint
import multiprocessing as mp
import uuid
import glob
import time
import random
import json

from config import PretrainConfig, SftConfig, TrainDataConfig
from tokenizer import Tokenizer

# 创建包装器类用于处理函数参数
class TextFnWrapper:
    def __init__(self, fn):
        self.fn = fn
        
    def __call__(self, x):
        return self.fn(x)

class DataPreparer:
    def __init__(self, source, tokenizer, num_workers=None):
        self.source = source
        self.tokenizer = tokenizer
        self.cache_dir = os.environ.get('dataset_cache', "./dataset_cache")
        self.num_workers = num_workers if num_workers is not None else max(1, mp.cpu_count() - 2)

        self.text_fn = TextFnWrapper(source["text_fn"])
        self.file_path = os.path.join(self.cache_dir, f"{source['name']}.bin")

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
                if text is None or text == "":
                    continue
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
            return self.file_path

        tprint(f"正在加载数据集: {self.source['name']}...")
        raw_dataset = self.source["ds_fn"]()
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
                
                # 使用迭代器获取结果，这样可以在处理完成后立即获取结果
                for _, temp_file in enumerate(pool.starmap(self._process_chunk, [(chunk, i, temp_dir) for i, chunk in enumerate(chunks)])):
                    temp_files.append(temp_file)
            
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
        return self.file_path
        

class DataMapper:
    def __init__(self, path):
        self.file_path = path

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
                self.fd = os.open(filename, os.O_RDONLY)
                # 尝试使用posix_fadvise，如果系统支持的话
                try:
                    if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_SEQUENTIAL'):
                        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                        tprint(f"使用posix_fadvise(os.POSIX_FADV_SEQUENTIAL)")
                except (AttributeError, OSError):
                    # macOS等系统可能不支持posix_fadvise
                    tprint(f"使用posix_fadvise(os.POSIX_FADV_SEQUENTIAL)失败")
                    pass

                self.mm = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
                
                # 尝试使用madvise，如果系统支持的话
                try:
                    if hasattr(mmap, 'MADV_SEQUENTIAL'):
                        self.mm.madvise(mmap.MADV_SEQUENTIAL)
                        tprint(f"使用madvise(mmap.MADV_SEQUENTIAL)")
                except (AttributeError, OSError):
                    # 某些系统可能不支持madvise或MADV_SEQUENTIAL
                    tprint(f"使用madvise(mmap.MADV_SEQUENTIAL)失败")
                    pass
                
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
                if hasattr(self, 'mm') and self.mm is not None:
                    self.mm.close()
                if hasattr(self, 'fd') and self.fd is not None:
                    # 检查os模块是否仍然可用
                    if 'os' in globals() and os is not None and hasattr(os, 'close'):
                        os.close(self.fd)
                    
        return MemoryMappedTokens(self.file_path)


class TrainDataLoader:
    def __init__(self, path, world_size, rank, local_rank, batch_size, block_size):
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.batch_size = batch_size
        self.block_size = block_size
        self.path = path
        self.data_mapper = DataMapper(self.path)

        self.tokens = self.data_mapper.map_to_array()
        self.all_tokens_len = len(self.tokens)

        node_data_len = self.all_tokens_len // self.world_size
        tprint(f"{self.path} 总体样本token数量: {self.all_tokens_len}, 每个节点样本token数量: {node_data_len}")

        self.offset_start = self.rank * node_data_len
        self.offset_end = self.offset_start + node_data_len
        tprint(f"{self.path} 当前节点token offset: {self.offset_start} - {self.offset_end}")
        
        # 初始化当前位置指针
        self.current_position = self.offset_start
        self.data_epoch = 0
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
                self.data_epoch += 1

            # 获取一个数据块
            x = self.tokens[self.current_position:self.current_position + self.block_size + 1]
            xs.append(x[:-1])
            ys.append(x[1:])
            
            # 移动指针
            self.current_position += self.block_size
        
        return xs, ys

    def set_data_progress_percentage(self, progress_percentage):
        self.data_epoch = progress_percentage // 100
        self.current_position = int(self.offset_start + (progress_percentage % 100) * self.node_data_len / 100)
        tprint(f"{self.path} 设置数据进度百分比: {progress_percentage}, 当前位置: {self.current_position}")

    def get_data_progress_percentage(self):
        return ((self.current_position - self.offset_start) % self.node_data_len) / self.node_data_len * 100 + self.data_epoch * 100


class MixTrainDataLoader:
    def __init__(self, world_size, rank, local_rank, batch_size, block_size):
        self.train_data_loaders = {}
        self.loader_names = []
        self.loader_weights = []
        self.cache_dir = os.environ.get('dataset_cache', "./dataset_cache")

        for source in TrainDataConfig().data.datasets:
            if not source["enabled"]:
                continue
            name = source["data"]["name"]
            file_path = os.path.join(self.cache_dir, f"{name}.bin")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到预处理文件：{file_path}. 请使用python dataloader.py预处理数据集")
            self.train_data_loaders[name] = TrainDataLoader(file_path, world_size, rank, local_rank, batch_size, block_size)
            self.loader_names.append(name)
            # 如果没有指定权重，默认为1.0
            self.loader_weights.append(source.get("weight", 1.0))
            
        # 确保权重总和为1.0
        weight_sum = sum(self.loader_weights)
        if weight_sum != 1.0 and weight_sum > 0:
            self.loader_weights = [w / weight_sum for w in self.loader_weights]
            tprint(f"数据加载器权重已归一化: {list(zip(self.loader_names, self.loader_weights))}")

    def next(self):
        # 根据权重随机选择一个loader
        chosen_name = random.choices(self.loader_names, weights=self.loader_weights, k=1)[0]
        return self.train_data_loaders[chosen_name].next()

    def set_data_progress_percentage(self, progress_percentage):
        try:
            # 尝试解析JSON字符串
            progress_data = json.loads(progress_percentage)
            
            # 为每个数据加载器设置进度
            for name, loader in self.train_data_loaders.items():
                # 如果在JSON中找到对应的数据集进度，则使用它
                # 否则默认为0
                progress = progress_data.get(name, 0)
                loader.set_data_progress_percentage(progress)
                tprint(f"为数据集 {name} 设置进度: {progress}")
        except json.JSONDecodeError:
            # 如果输入不是有效的JSON，将所有进度设为0
            tprint(f"无效的进度JSON字符串，将所有数据集进度重置为0")
            for name, loader in self.train_data_loaders.items():
                loader.set_data_progress_percentage(0)

    def get_data_progress_percentage(self):
        # 收集所有数据加载器的进度
        progress_data = {}
        for name, loader in self.train_data_loaders.items():
            progress_data[name] = loader.get_data_progress_percentage()
            
        # 将进度数据转换为JSON字符串并返回
        return json.dumps(progress_data)
        
    def get_processed_tokens_count(self):
        # 收集所有数据加载器处理的绝对token数量
        tokens_count = {}
        for name, loader in self.train_data_loaders.items():
            # 获取进度百分比
            progress_percentage = loader.get_data_progress_percentage()
            # 计算绝对token数量 = 总token数量 * 进度百分比 / 100
            processed_tokens = loader.all_tokens_len * progress_percentage / 100
            tokens_count[name] = int(processed_tokens)
            
        # 将token数量数据转换为JSON字符串并返回
        return json.dumps(tokens_count)


if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenss = []
    for source in PretrainConfig.datasets:
        if not source["enabled"]:
            continue
        data_preparer = DataPreparer(source["data"], tokenizer)
        path = data_preparer.preprocess_to_file()
        data_mapper = DataMapper(path)
        tokens = data_mapper.map_to_array()
        tprint(f"{source['data']['name']} tokens length: {len(tokens)}")
        tokenss.append(tokens)

    for source in SftConfig.datasets:
        if not source["enabled"]:
            continue
        data_preparer = DataPreparer(source["data"], tokenizer)
        path = data_preparer.preprocess_to_file()
        data_mapper = DataMapper(path)
        tokens = data_mapper.map_to_array()
        tprint(f"{source['data']['name']} tokens length: {len(tokens)}")
        tokenss.append(tokens)

    train_data_loader = MixTrainDataLoader(1, 0, 0, 1, 1024)
    tprint("打印100个case检查一下")
    for _ in range(100):
        xs, ys = train_data_loader.next()
        tprint("="*80)
        tprint(f"xs: {tokenizer.decode(xs[0] + ys[0][-1:])}")

    # 统计下样本里面的token频率, sample_num 为0则不统计
    sample_num = 0
    if sample_num > 0:
        count_dict = {}
        for _ in range(sample_num):
            xs, _ = train_data_loader.next()
            for x in xs:
                for token in x:
                    count_dict[token] = count_dict.get(token, 0) + 1
        with open("count_dict.txt", "w") as f:
            for k, v in sorted(count_dict.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{v} {int(k)} {tokenizer.decode(k)}\n")
        