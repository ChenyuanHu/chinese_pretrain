import argparse
import time
import os
from tokenizer import Tokenizer
from dataloader import DataMapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='输入目录路径', default=".")
    parser.add_argument('--base_dir', type=str, help='基础目录路径，样本的路径描述都会基于这个目录', default=".")
    parser.add_argument('--output', type=str, help='输出文件名', default="output.txt")
    args = parser.parse_args()

    # 获取规范化的工作目录和输入目录绝对路径
    target_dir = os.path.abspath(args.dir)
    base_dir = os.path.abspath(args.base_dir)

    print(f"base目录: {base_dir}")
    print(f"输入目录: {target_dir}")
    print(f"输出文件: {args.output}")

    # 常见文本文件扩展名
    text_extensions = {'.go', '.txt', '.py', '.java', '.c', '.cpp', '.h', '.hpp', '.js', '.css', '.html',
                       '.xml', '.json', '.yaml', '.yml', '.md', '.rst', '.ini', '.conf', '.sh',
                       '.php', '.rb', '.pl', '.scala', '.swift', '.ts', '.jsx', '.tsx',
                       '.csv', '.log', '.sql', '.ipynb', '.properties', '.toml', '.env',
                       '.gitignore', '.svg'}

    tokenizer = Tokenizer()
    start_time = time.time()
    last_log_time = start_time
    total_items = 0
    failed_items = 0

    with open(args.output, 'wb') as f_out:
        # 递归遍历目录
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # 获取相对路径
                relative_path = os.path.relpath(file_path, start=base_dir)
                
                # 检查文件扩展名
                _, ext = os.path.splitext(file.lower())
                if ext not in text_extensions:
                    continue
                
                print(f"处理文件: {relative_path}")
                
                # 读取并写入文件内容
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_in:
                        content = f_in.read()

                except Exception as e:
                    failed_items += 1
                    print(f"读取文件 {relative_path} 失败: {str(e)}")
                    continue


                chunk = f"=== 文件路径: {relative_path} ===\n{content}\n\n"
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
                
    print(f"已处理 {total_items} 项, 失败 {failed_items} 项, 用时 {current_time - start_time:.2f}秒")

    data_mapper = DataMapper(args.output)
    tokens = data_mapper.map_to_array()
    print(f"tokens length: {len(tokens)}")
    for i in range(10):
        print(f"sample {i}: {tokenizer.decode(tokens[i*1024:(i+1)*1024])}")
