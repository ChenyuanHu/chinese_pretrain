from torch.utils.data import DataLoader
import torch
import time
from log import tprint
from datasets import load_dataset


class TrainDataLoader:
    def __init__(self, env, batch_size, block_size, tokenizer=None, use_data_percent=100, is_sft=False):
        self.fineweb_edu_chinese_v2_1_iter = None
        self.chinese_deepseek_r1_distill_data_110k_sft_iter = None
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.env = env
        self.use_data_percent = use_data_percent
        self.batch_size = batch_size
        self.is_sft = is_sft
        self.reload()

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def reload(self):
        if self.is_sft:
            self.chinese_deepseek_r1_distill_data_110k_sft_iter = self.load_chinese_deepseek_r1_distill_data_110k_sft(self.batch_size)
        else:
            self.fineweb_edu_chinese_v2_1_iter = self.load_fineweb_edu_chinese_v2_1(self.env, self.batch_size, self.use_data_percent)

    @staticmethod
    def load_fineweb_edu_chinese_v2_1(env, batch_size, use_data_percent):
        percent_per_process = int(use_data_percent / env.world_size)
        offset_start = env.rank * percent_per_process
        offset_end = offset_start + percent_per_process
        assert offset_end <= 100, f"offset_end({offset_end}) must be less than 100"

        tprint(f"加载FineWebEduChinese数据集，只下载和处理4-5评分范围的高质量内容. 第{env.rank}个进程，从{offset_start}%到{offset_end}%")
        raw_dataset = load_dataset("opencsg/Fineweb-Edu-Chinese-V2.1", data_dir = "4_5", split=f"train[{offset_start}%:{offset_end}%]")

        generator = torch.Generator()
        generator.manual_seed(42 + int(time.time()))  # 每次重启时使用不同的种子，避免断点续训时数据重复
        dataset_batch = DataLoader(raw_dataset, batch_size=batch_size, shuffle=True, generator=generator)
        return iter(dataset_batch)
    
    def next_fineweb_edu_chinese_v2_1(self):
        items = next(self.fineweb_edu_chinese_v2_1_iter)
        texts = items["text"]
        return texts

    @staticmethod
    def load_chinese_deepseek_r1_distill_data_110k_sft(batch_size):
        tprint(f"加载ChineseDeepSeekR1DistillData数据集")
        raw_dataset = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", split="train")

        generator = torch.Generator()
        generator.manual_seed(42 + int(time.time()))  # 每次重启时使用不同的种子，避免断点续训时数据重复
        dataset_batch = DataLoader(raw_dataset, batch_size=batch_size, shuffle=True, generator=generator)
        return iter(dataset_batch)
    
    def next_chinese_deepseek_r1_distill_data_110k_sft(self):
        items = next(self.chinese_deepseek_r1_distill_data_110k_sft_iter)
        texts = []
        for i in range(len(items["instruction"])):
            text = "系统提示：你是一个叫小伽的人工智能小助手，你的思考过程放在<think></think>标签中" + "\n" + "用户：" + items["instruction"][i] + "\n" + "助手：" + items["output"][i]
            texts.append(text)
        return texts

    def next_router(self):
        if self.is_sft:
            return self.next_chinese_deepseek_r1_distill_data_110k_sft()
        else:
            return self.next_fineweb_edu_chinese_v2_1()

    def next(self, device):
        while True:
            try:
                texts = self.next_router()

                xs = []
                ys = []
                for text in texts:
                    tokens = self.tokenizer.encode(self.tokenizer.bos_token + text)
                    if len(tokens) < self.block_size + 1:
                        tokens = tokens + [self.tokenizer.eos_token_id] * (self.block_size + 1 - len(tokens))
                    else:
                        tokens = tokens[:self.block_size + 1]
                    
                    x = tokens[:-1]
                    y = tokens[1:]
                    xs.append(x)
                    ys.append(y)

                xs = torch.tensor(xs, dtype=torch.long, device=device)
                ys = torch.tensor(ys, dtype=torch.long, device=device)

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