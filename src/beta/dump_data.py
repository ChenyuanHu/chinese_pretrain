import argparse
import sys
from tokenizer import Tokenizer
from dataloader import DataMapper

def main():
    parser = argparse.ArgumentParser(description="转储数据集所有样本")
    parser.add_argument('--datafile', type=str, help='数据文件路径')
    args = parser.parse_args()

    print(f"数据文件: {args.datafile}")

    # 初始化分词器和数据映射器
    tokenizer = Tokenizer()
    data_mapper = DataMapper(args.datafile)
    
    try:
        tokens = data_mapper.map_to_array()

        i = 0
        while True:
            if i*16384 >= len(tokens):
                break
            data = tokenizer.decode(tokens[i*16384:(i+1)*16384])
            print(data, end="")
            i += 1
        
        # for token in tokens:
        #     print(tokenizer.decode([token]), end="")
        
    except Exception as e:
        print(f"处理数据文件时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
