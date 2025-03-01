import os
from datetime import datetime

def get_log_file():
    """获取日志文件路径"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(log_dir, f'train_{timestamp}.log')

# 创建日志文件
LOG_FILE = get_log_file()

def tprint(*args, **kwargs):
    """带时间戳的打印函数，同时输出到文件和标准输出"""
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_str = f'[{time_str}] ' + ' '.join(map(str, args))
    
    # 打印到标准输出
    print(log_str, **kwargs)
    
    # 写入到日志文件
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_str + '\n') 