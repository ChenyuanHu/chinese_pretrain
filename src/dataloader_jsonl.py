import argparse
import time
import os
import json
from tokenizer import Tokenizer
from dataloader import DataMapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='输入文件路径', default="input.jsonl")
    parser.add_argument('--output', type=str, help='输出文件名', default="output.bin")
    args = parser.parse_args()

    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")

    tokenizer = Tokenizer()
    start_time = time.time()
    last_log_time = start_time
    total_items = 0
    failed_items = 0

    with open(args.output, 'wb') as f_out:
        try:
            with open(args.input, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    try:
                        item = json.loads(line.strip())
                        
                        # 提取jsonl中的字段
                        original_text = item.get('original_text', '')
                        equation = item.get('equation', '')
                        ans = item.get('ans', '')
                        
                        # 构建处理后的文本
                        chunk = f"数学应用题: {original_text}\n\n数学表达式抽象: {equation}\n\n答案: {ans}\n\n"
                        
                        encoded = tokenizer.encode(chunk)
                        tokens = [tokenizer.bos_token_id] + encoded + [tokenizer.eos_token_id]

                        # 使用3字节存储每个token ID
                        bytes_data = bytearray()
                        for token in tokens:
                            # 将token ID转换为3字节
                            bytes_data.extend(token.to_bytes(3, byteorder='little'))
                        f_out.write(bytes_data)

                        total_items += 1

                        # 每30秒打印一次进度
                        current_time = time.time()
                        if current_time - last_log_time >= 30:
                            print(f"已处理 {total_items} 项, 失败 {failed_items} 项, 用时 {current_time - start_time:.2f}秒")
                            last_log_time = current_time
                        
                    except Exception as e:
                        failed_items += 1
                        print(f"处理JSONL行失败: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"读取输入文件 {args.input} 失败: {str(e)}")
    
    current_time = time.time()
    print(f"已处理 {total_items} 项, 失败 {failed_items} 项, 用时 {current_time - start_time:.2f}秒")

    data_mapper = DataMapper(args.output)
    tokens = data_mapper.map_to_array()
    print(f"tokens length: {len(tokens)}")
    print(f"sample: {tokenizer.decode(tokens[:1024])}")
