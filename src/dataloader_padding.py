import os
from log import tprint
import multiprocessing as mp
import pickle
import random

from config import PretrainConfig, SftConfig, TrainDataConfig
from tokenizer import Tokenizer

# 创建包装器类用于处理函数参数
class TextFnWrapper:
    def __init__(self, fn):
        self.fn = fn
        
    def __call__(self, x):
        return self.fn(x)

class DataPreparer:
    def __init__(self, source, tokenizer, cache_dir="./dataset_cache/padding", num_workers=None):
        self.source = source
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.num_workers = num_workers if num_workers is not None else max(1, mp.cpu_count() - 2)

        self.text_fn = TextFnWrapper(source["text_fn"])
        self.file_path_prefix = os.path.join(self.cache_dir, f"{source['name']}_padding_")

    def _process_chunk(self, chunk_data):
        encoded_samples = []
        for item in chunk_data:
            text = self.text_fn(item)
            if text is None or text == "":
                continue
            encoded = self.tokenizer.encode(text)
            tokens = [self.tokenizer.bos_token_id] + encoded + [self.tokenizer.eos_token_id]
            encoded_samples.append(tokens)

        return encoded_samples

    def preprocess_to_file(self):
        # 检查是否已经生成过数据文件
        first_file_path = self.file_path_prefix + "0.bin"
        if os.path.exists(first_file_path):
            tprint(f"数据集 {self.source['name']} 已经处理过，直接使用现有文件")
            file_paths = [self.file_path_prefix + f"{i}.bin" for i in range(64)]
            return file_paths
            
        # 确保目录存在
        os.makedirs(os.path.dirname(self.file_path_prefix), exist_ok=True)

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
                for tmp_encoded_samples in pool.starmap(self._process_chunk, [(chunk,) for chunk in chunks]):
                    encoded_samples.extend(tmp_encoded_samples)

        tprint(f"开始存储数据")
        # 随机打乱编码样本并分成64份
        random.shuffle(encoded_samples)
        
        num_files = 64
        samples_per_file = len(encoded_samples) // num_files
        
        file_paths = []
        
        # 分成64个文件存储
        for i in range(num_files):
            start_idx = i * samples_per_file
            end_idx = start_idx + samples_per_file if i < num_files - 1 else len(encoded_samples)
            file_samples = encoded_samples[start_idx:end_idx]
            
            # 创建bucket字典，按长度分组
            bucket = {}
            for tokens in file_samples:
                length = len(tokens)
                if length not in bucket:
                    bucket[length] = []
                bucket[length].append(tokens)
            
            # 保存处理后的数据到二进制文件
            file_path = self.file_path_prefix + f"{i}.bin"
            with open(file_path, 'wb') as f:
                pickle.dump(bucket, f)
            
            file_paths.append(file_path)
            
        tprint(f"数据集预处理完毕，已保存到64个文件: {self.file_path_prefix}[0-63].bin")
        return file_paths

class TrainDataLoader:
    def __init__(self, source, world_size, rank, batch_size, block_size, tokenizer, cache_dir="./dataset_cache/padding"):
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size
        self.block_size = block_size
        self.batch_token_size = (self.block_size + 1) * batch_size
        self.source = source
        self.tokenizer = tokenizer
        self.file_path_prefix = os.path.join(cache_dir, f"{source['name']}_padding_")
        self.samples = self.init_samples()
        self.iterator = self.__iter__()
        
        # 添加进度跟踪变量
        self.data_epoch = 0
        self.current_position = 0
        self.total_samples = len(self.samples)

    def load_data(self):
        tprint(f"进程 {self.rank}/{self.world_size} 开始加载数据")
        
        # 计算当前进程需要处理的文件索引
        total_files = 64
        files_per_rank = total_files // self.world_size
        start_file_idx = self.rank * files_per_rank
        end_file_idx = start_file_idx + files_per_rank if self.rank < self.world_size - 1 else total_files
        
        # 记录此进程处理的文件编号
        my_file_indices = list(range(start_file_idx, end_file_idx))
        tprint(f"进程 {self.rank} 负责处理文件: {my_file_indices}")
        
        # 加载数据
        merged_bucket = {}
        for file_idx in my_file_indices:
            file_path = self.file_path_prefix + f"{file_idx}.bin"
            try:
                with open(file_path, 'rb') as f:
                    bucket = pickle.load(f)
                    
                # 合并bucket
                for length, tokens_list in bucket.items():
                    if length not in merged_bucket:
                        merged_bucket[length] = []
                    merged_bucket[length].extend(tokens_list)
                
                tprint(f"进程 {self.rank} 成功加载文件: {file_path}")
            except Exception as e:
                tprint(f"进程 {self.rank} 加载文件 {file_path} 失败: {e}")
        
        tprint(f"进程 {self.rank} 加载完成，共加载 {sum(len(tokens_list) for tokens_list in merged_bucket.values())} 个样本")

        self.bucket = merged_bucket
        return merged_bucket

    # 确保所有大于block_size + 1的样本都被切分
    def init_samples(self):
        bucket = self.load_data()
        samples = []
        # 因为有x=[:-1], y=[1:], 所以需要+1
        raw_tokens_size = self.block_size + 1
        for length in sorted(bucket.keys()):
            if length <= raw_tokens_size:
                for tokens in bucket[length]:
                    samples.append(tokens)
            else:
                tokenss = bucket[length]
                for tokens in tokenss:
                    last_end = 0
                    # 滑动窗口切分长样本, 重叠区域为10%
                    for start in range(0, len(tokens) - raw_tokens_size, int(raw_tokens_size * 0.9)):
                        # 提取当前窗口
                        last_end = start + raw_tokens_size
                        window = tokens[start:last_end]
                        samples.append(window)
                    
                    # 处理最后一个窗口（如果需要）
                    if last_end < len(tokens):
                        last_window = tokens[-raw_tokens_size:]
                        samples.append(last_window)  # 将last_window包装在列表中

        return samples

    def inter_iter(self):
        # 获取所有长度并排序
        samples = []
        for tokens in self.samples:
            if (len(tokens) - 1) * (len(samples) + 1) >= self.batch_token_size:
                yield samples
                samples = []
            samples.append(tokens)
        if len(samples) > 0:
            yield samples
    
    def next(self):
        return next(self.iterator)

    def __iter__(self):
        # 根据current_position开始迭代
        samples_iter = self.inter_iter()
        # 跳过之前已处理的batch
        position_tracker = 0
        for xs in samples_iter:
            position_tracker += len(xs)
            if position_tracker <= self.current_position:
                continue
                
            xs_out = []
            ys_out = []
            length = len(xs[-1])
            for i in range(len(xs)):
                if len(xs[i]) != length:
                    xs[i].extend([self.tokenizer.pad_token_id] * (length - len(xs[i])))
                xs_out.append(xs[i][:-1])
                ys_out.append(xs[i][1:])
            
            # 更新当前位置
            self.current_position = position_tracker
            
            yield xs_out, ys_out
            
        # 一个epoch结束，更新状态
        self.data_epoch += 1
        self.current_position = 0

    def set_data_progress_percentage(self, progress_percentage):
        self.data_epoch = progress_percentage // 100
        self.current_position = int((progress_percentage % 100) * self.total_samples / 100)
        tprint(f"设置数据进度百分比: {progress_percentage}, epoch: {self.data_epoch}, 当前位置: {self.current_position}/{self.total_samples}")
        # 重新初始化迭代器
        self.iterator = self.__iter__()

    def get_data_progress_percentage(self):
        # 计算当前位置百分比加上已完成的epoch数*100
        position_percentage = (self.current_position / self.total_samples) * 100 if self.total_samples > 0 else 0
        return position_percentage + self.data_epoch * 100

class MixTrainDataLoader:
    def __init__(self, world_size, rank, local_rank, batch_size, block_size, cache_dir="./dataset_cache/padding"):
        _ = local_rank
        self.train_data_loaders = {}
        self.loader_names = []
        self.loader_weights = []
        self.cache_dir = cache_dir
        self.tokenizer = Tokenizer()
        
        # 初始化各个数据集的加载器
        for source in TrainDataConfig().data.datasets:
            if not source["enabled"]:
                continue
            name = source["data"]["name"]
            self.train_data_loaders[name] = TrainDataLoader(source["data"], world_size, rank, 
                                                           batch_size, block_size, self.tokenizer, cache_dir)
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
            import json
            progress_data = json.loads(progress_percentage)
            
            # 为每个数据加载器设置进度
            for name, loader in self.train_data_loaders.items():
                # 如果在JSON中找到对应的数据集进度，则使用它
                # 否则默认为0
                progress = progress_data.get(name, 0)
                loader.set_data_progress_percentage(progress)
                tprint(f"为数据集 {name} 设置进度: {progress}")
        except (json.JSONDecodeError, ValueError):
            # 如果输入不是有效的JSON，将所有进度设为0
            tprint(f"无效的进度JSON字符串，将所有数据集进度重置为0")
            for name, loader in self.train_data_loaders.items():
                loader.set_data_progress_percentage(0)

    def get_data_progress_percentage(self):
        # 收集所有数据加载器的进度
        import json
        progress_data = {}
        for name, loader in self.train_data_loaders.items():
            progress_data[name] = loader.get_data_progress_percentage()
            
        # 将进度数据转换为JSON字符串并返回
        return json.dumps(progress_data)


if __name__ == "__main__":
    tokenizer = Tokenizer()
    
    # 预处理预训练数据集
    for source in PretrainConfig.datasets:
        if not source["enabled"]:
            continue
        data_preparer = DataPreparer(source["data"], tokenizer)
        file_paths = data_preparer.preprocess_to_file()
        tprint(f"预处理完成: {source['data']['name']}, 文件保存在: {file_paths}")
    
    # 预处理SFT数据集
    for source in SftConfig.datasets:
        if not source["enabled"]:
            continue
        data_preparer = DataPreparer(source["data"], tokenizer)
        file_paths = data_preparer.preprocess_to_file()
        tprint(f"预处理完成: {source['data']['name']}, 文件保存在: {file_paths}")

    # 测试数据加载
    train_data_loader = MixTrainDataLoader(world_size=1, rank=0, batch_size=1, block_size=1024)
    tprint("打印10个case检查一下")
    for _ in range(10):
        xs, ys = train_data_loader.next()
        tprint("="*80)
        tprint(f"xs: {tokenizer.decode(xs[0])}")
        tprint(f"ys: {tokenizer.decode(ys[0])}")

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
                f.write(f"{v} {int(k)} {tokenizer.decode([k])}\n")
