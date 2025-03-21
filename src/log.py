import os
from datetime import datetime

rank = int(os.environ.get('RANK', -1))
local_rank = int(os.environ.get('LOCAL_RANK', -1))

def get_log_file():
    """获取日志文件路径"""
    log_dir = 'experiments/logs'
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f'train_{rank}.log')

# 创建日志文件
LOG_FILE = get_log_file()

def tprint(*args, **kwargs):
    """带时间戳的打印函数，同时输出到文件和标准输出"""
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if rank != -1:
        log_str = f'[RANK:{rank}][{time_str}] ' + ' '.join(map(str, args))
    else:
        log_str = f'[{time_str}] ' + ' '.join(map(str, args))
    
    # 打印到标准输出
    if local_rank == 0 or local_rank == -1:
        print(log_str, **kwargs)
    
    # 写入到日志文件
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_str + '\n') 