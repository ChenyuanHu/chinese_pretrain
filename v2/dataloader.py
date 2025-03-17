from torch.utils.data import DataLoader
import torch
import time
from log import tprint
from datasets import load_dataset
from tokenizer import Tokenizer


class DataLoaderProcess:
    def __init__(self, path, data_dir, env, batch_size, block_size, use_data_percent=100, shuffle=True, tokenizer=None, text_fn=None):
        self.path = path
        self.data_dir = data_dir
        self.env = env
        self.batch_size = batch_size
        self.block_size = block_size
        self.use_data_percent = use_data_percent
        self.shuffle = shuffle

        self.tokenizer = tokenizer
        self.generator = torch.Generator()
        self.generator.manual_seed(42 + int(time.time()))  # 每次重启时使用不同的种子，避免断点续训时数据重复
        self.text_fn = text_fn
        self.iter = None
        self.reload()

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        xs = []
        ys = []
        for item in batch:
            text = self.text_fn(item)
            tokens = self.tokenizer.encode(text)
            if len(tokens) < self.block_size + 1:
                tokens = tokens + [self.tokenizer.eos_token_id] * (self.block_size + 1 - len(tokens))
            else:
                # 超长的文本，随机选择一个起始点
                max_start_idx = len(tokens) - (self.block_size + 1)
                start_idx = torch.randint(0, max_start_idx + 1, (1,), generator=self.generator).item()
                tokens = tokens[start_idx:start_idx + self.block_size + 1]

            xs.append(tokens[:-1])
            ys.append(tokens[1:])
        xs = torch.tensor(xs, dtype=torch.long)
        ys = torch.tensor(ys, dtype=torch.long)
        # 不使用pin_memory，避免设备不匹配
        return xs, ys

    def reload(self):
        percent_per_process = int(self.use_data_percent / self.env.world_size)
        offset_start = self.env.rank * percent_per_process
        offset_end = offset_start + percent_per_process
        assert offset_end <= 100, f"offset_end({offset_end}) must be less than 100"

        tprint(f"加载数据集{self.path}. 第{self.env.rank}个进程，从{offset_start}%到{offset_end}%")
        raw_dataset = load_dataset(self.path, data_dir=self.data_dir, split=f"train[{offset_start}%:{offset_end}%]")

        if self.env.device == "mps":
            num_workers = 0
            prefetch_factor = None
            persistent_workers = False
        else:
            num_workers = 1
            prefetch_factor = 4
            persistent_workers = True

        dataset_batch = DataLoader(raw_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            generator=self.generator,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=self.collate_fn
        )
        self.iter = iter(dataset_batch)

    def next(self, device):
        while True:
            try:
                xs, ys = next(self.iter)

                xs = xs.to(device)
                ys = ys.to(device)

                return xs, ys
                
            except StopIteration:
                # 如果数据集遍历完了，重新开始
                tprint("数据集遍历完了，重新开始")
                time.sleep(120)
                self.reload()
                continue

            except Exception as e:
                tprint(f"处理批次时出错: {str(e)}，跳过此批次")
                time.sleep(120)
                continue


def text_fn_pretrain(x, tokenizer):
    return tokenizer.bos_token + x["text"]

def text_fn_sft(x, tokenizer):
    return tokenizer.bos_token + "<|im_start|>用户\n" + x["instruction"] + "\n<|im_end|>\n<|im_start|>助手\n" + x["output"] + "\n<|im_end|>"

# 创建包装器类用于处理函数参数
class TextFnWrapper:
    def __init__(self, fn, tokenizer):
        self.fn = fn
        self.tokenizer = tokenizer
        
    def __call__(self, x):
        return self.fn(x, self.tokenizer)


class TrainDataLoader:
    def __init__(self, env, batch_size, block_size, tokenizer=None, use_data_percent=100, is_sft=False):
        self.tokenizer = tokenizer
        
        if not is_sft:
            self.train_data_loader_process = DataLoaderProcess(
            path="opencsg/Fineweb-Edu-Chinese-V2.1",
            data_dir="4_5",
            env=env,
            batch_size=batch_size,
            block_size=block_size,
            use_data_percent=use_data_percent,
            shuffle=True,
            tokenizer=tokenizer,
            text_fn=TextFnWrapper(text_fn_pretrain, tokenizer))
        else:
            self.train_data_loader_process = DataLoaderProcess(
            path="Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT",
            data_dir=None,
            env=env,
            batch_size=batch_size,
            block_size=block_size,
            use_data_percent=use_data_percent,
            shuffle=True,
            tokenizer=tokenizer,
            text_fn=TextFnWrapper(text_fn_sft, tokenizer))

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.train_data_loader_process.tokenizer = tokenizer
        # 更新 text_fn 中的 tokenizer
        if hasattr(self.train_data_loader_process.text_fn, 'tokenizer'):
            self.train_data_loader_process.text_fn.tokenizer = tokenizer

    def next(self, device):
        return self.train_data_loader_process.next(device)


# 将MockEnv移到文件顶层
class MockEnv:
    def __init__(self):
        self.rank = 0
        self.world_size = 1

if __name__ == "__main__":
    # 用来单独预缓存数据集
    env = MockEnv()
    tokenizer = Tokenizer()
    dataloader = TrainDataLoader(env, 1, 1024, tokenizer=tokenizer, use_data_percent=100, is_sft=True)

    for i in range(10):
        xs, ys = dataloader.next(device="cpu")
        print(xs)
        print(ys)

