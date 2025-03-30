import matplotlib.pyplot as plt
import datetime
import re
import json
import numpy as np

# Read and parse the log file
timestamps = []
losses = []
perplexities = []
learning_rates = []
dataset_usage = {}
dataset_tokens = {}  # 新增：存储每个数据集处理的token数
throughputs = []  # For tokens/s
epochs = []  # For epoch numbers

with open('train.log', 'r', encoding='utf-8') as f:
    for line in f:
        # Extract timestamp, loss, perplexity, learning rate, throughput and epoch using regex
        match = re.search(r'(?:\[RANK:0\])?\[(.*?)\].*?Epoch \[(\d+)/\d+\].*?([\d.]+)sec.*?world ([\d.]+) tokens/s.*?(?:训练损失|loss): ([\d.]+).*?(?:困惑度|perplexity): ([\d.]+).*?LR: ([\d.e-]+)', line, re.IGNORECASE)
        if match:
            timestamp_str = match.group(1)
            epoch = int(match.group(2))
            epoch_time = float(match.group(3))
            throughput = float(match.group(4))
            loss = float(match.group(5))
            perplexity = float(match.group(6))
            lr = float(match.group(7))
            
            # Parse timestamp
            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            timestamps.append(timestamp)
            losses.append(loss)
            perplexities.append(perplexity)
            learning_rates.append(lr)
            throughputs.append(throughput)
            epochs.append(epoch)
        
        # Extract dataset usage statistics
        dataset_match = re.search(r'(?:\[RANK:0\])?\[(.*?)\].*?数据集使用度: (\{.*\})', line)
        if dataset_match:
            timestamp_str = dataset_match.group(1)
            dataset_json = dataset_match.group(2)
            
            # Parse timestamp
            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            # Parse dataset usage JSON
            try:
                usage_data = json.loads(dataset_json)
                # Initialize dataset keys if they don't exist
                for key in usage_data:
                    if key not in dataset_usage:
                        dataset_usage[key] = []
                
                # For each dataset, add the usage value and timestamp
                for key, value in usage_data.items():
                    # 找到最接近的时间戳，而不是要求精确匹配
                    if timestamps:
                        # 计算时间差，找到最接近的索引
                        time_diffs = [(abs((t - timestamp).total_seconds()), i) for i, t in enumerate(timestamps)]
                        closest_idx = min(time_diffs, key=lambda x: x[0])[1]
                        # 仅当时间差在合理范围内（如60秒）才添加数据点
                        if time_diffs[closest_idx][0] <= 10:  # 允许10秒误差
                            # Extend the list to match the current timestamp index if needed
                            while len(dataset_usage[key]) < closest_idx:
                                dataset_usage[key].append(None)
                            dataset_usage[key].append(value)
            except json.JSONDecodeError:
                print(f"Error parsing JSON: {dataset_json}")
                
        # 新增：提取数据集处理token数统计
        tokens_match = re.search(r'(?:\[RANK:0\])?\[(.*?)\].*?数据集处理token数: (\{.*\})', line)
        if tokens_match:
            timestamp_str = tokens_match.group(1)
            tokens_json = tokens_match.group(2)
            
            # Parse timestamp
            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            # Parse dataset tokens JSON
            try:
                tokens_data = json.loads(tokens_json)
                # Initialize dataset keys if they don't exist
                for key in tokens_data:
                    if key not in dataset_tokens:
                        dataset_tokens[key] = []
                
                # For each dataset, add the tokens value and timestamp
                for key, value in tokens_data.items():
                    # 找到最接近的时间戳，而不是要求精确匹配
                    if timestamps:
                        # 计算时间差，找到最接近的索引
                        time_diffs = [(abs((t - timestamp).total_seconds()), i) for i, t in enumerate(timestamps)]
                        closest_idx = min(time_diffs, key=lambda x: x[0])[1]
                        # 仅当时间差在合理范围内（如60秒）才添加数据点
                        if time_diffs[closest_idx][0] <= 10:  # 允许10秒误差
                            # Extend the list to match the current timestamp index if needed
                            while len(dataset_tokens[key]) < closest_idx:
                                dataset_tokens[key].append(None)
                            dataset_tokens[key].append(value)
            except json.JSONDecodeError:
                print(f"Error parsing JSON: {tokens_json}")

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 24))  # 增加高度以容纳新图表
fig.suptitle('Training Metrics', fontsize=16)

# Create GridSpec for more control over subplot layout
from matplotlib.gridspec import GridSpec
gs = GridSpec(5, 2, figure=fig)  # 5行而不是4行

# 计算所有图表共享的 X 轴范围
if timestamps:
    x_min = min(timestamps)
    x_max = max(timestamps)
    # 添加一点边距
    time_range = x_max - x_min
    padding = time_range * 0.05  # 5% 的边距
    x_min = x_min - padding
    x_max = x_max + padding
else:
    x_min = x_max = None

# Global loss plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(timestamps, losses, 'b-', label='Loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.legend()
ax1.set_title('Global Training Loss')
if x_min and x_max:
    ax1.set_xlim(x_min, x_max)

# Global perplexity plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(timestamps, perplexities, 'r-', label='Perplexity')
ax2.set_ylabel('Perplexity')
ax2.set_xlabel('Time')
ax2.grid(True)
ax2.legend()
ax2.set_title('Global Perplexity')
if x_min and x_max:
    ax2.set_xlim(x_min, x_max)

# Learning rate plot
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(timestamps, learning_rates, 'g-', marker='o', label='Learning Rate')
ax3.set_ylabel('Learning Rate')
ax3.set_xlabel('Time')
ax3.grid(True)
ax3.legend()
ax3.set_title('Learning Rate Over Time')
if x_min and x_max:
    ax3.set_xlim(x_min, x_max)

# Dataset usage plot
ax4 = fig.add_subplot(gs[1, 1])
colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_usage)))
for (dataset_name, usage_values), color in zip(dataset_usage.items(), colors):
    # Filter out None values and get corresponding timestamps
    valid_indices = [i for i, v in enumerate(usage_values) if v is not None]
    valid_timestamps = [timestamps[i] for i in valid_indices]
    valid_values = [usage_values[i] for i in valid_indices]
    
    ax4.plot(valid_timestamps, valid_values, marker='o', linestyle='-', 
             label=dataset_name, color=color)

ax4.set_ylabel('Usage')
ax4.set_xlabel('Time')
ax4.grid(True)
ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax4.set_title('Dataset Usage Over Time')
if x_min and x_max:
    ax4.set_xlim(x_min, x_max)

# Throughput (tokens/s) plot
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(timestamps, throughputs, 'c-', marker='o', label='Throughput')
ax5.set_ylabel('Tokens per Second')
ax5.set_xlabel('Time')
ax5.grid(True)
ax5.legend()
ax5.set_title('Training Throughput Over Time')
if x_min and x_max:
    ax5.set_xlim(x_min, x_max)

# Epoch progress plot
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(timestamps, epochs, 'm-', marker='o', label='Epoch')
ax6.set_ylabel('Epoch')
ax6.set_xlabel('Time')
ax6.grid(True)
ax6.legend()
ax6.set_title('Epoch Progress Over Time')
if x_min and x_max:
    ax6.set_xlim(x_min, x_max)

# 新增：数据集处理token数图表
ax9 = fig.add_subplot(gs[3, :])  # 使用整行
colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_tokens)))
for (dataset_name, token_values), color in zip(dataset_tokens.items(), colors):
    # Filter out None values and get corresponding timestamps
    valid_indices = [i for i, v in enumerate(token_values) if v is not None]
    valid_timestamps = [timestamps[i] for i in valid_indices]
    valid_values = [token_values[i] for i in valid_indices]
    
    ax9.plot(valid_timestamps, valid_values, marker='o', linestyle='-', 
             label=dataset_name, color=color)

ax9.set_ylabel('Processed Tokens')
ax9.set_xlabel('Time')
ax9.grid(True)
ax9.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax9.set_title('Dataset Processed Tokens Over Time')
# 对y轴使用科学计数法
ax9.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
if x_min and x_max:
    ax9.set_xlim(x_min, x_max)

# Get the most recent 30 data points (or all if less than 30)
recent_count = min(30, len(timestamps))
recent_timestamps = timestamps[-recent_count:]
recent_losses = losses[-recent_count:]
recent_perplexities = perplexities[-recent_count:]

# 计算最近数据点的 X 轴范围
if recent_timestamps:
    recent_x_min = min(recent_timestamps)
    recent_x_max = max(recent_timestamps)
    # 添加一点边距
    recent_time_range = recent_x_max - recent_x_min
    recent_padding = recent_time_range * 0.05  # 5% 的边距
    recent_x_min = recent_x_min - recent_padding
    recent_x_max = recent_x_max + recent_padding
else:
    recent_x_min = recent_x_max = None

# Zoomed-in loss plot
ax7 = fig.add_subplot(gs[4, 0])
ax7.plot(recent_timestamps, recent_losses, 'b-', marker='o', label='Loss')
ax7.set_ylabel('Loss')
ax7.set_xlabel('Time')
ax7.grid(True)
ax7.legend()
ax7.set_title('Last {} Data Points - Loss'.format(recent_count))
if recent_x_min and recent_x_max:
    ax7.set_xlim(recent_x_min, recent_x_max)

# Zoomed-in perplexity plot
ax8 = fig.add_subplot(gs[4, 1])
ax8.plot(recent_timestamps, recent_perplexities, 'r-', marker='o', label='Perplexity')
ax8.set_ylabel('Perplexity')
ax8.set_xlabel('Time')
ax8.grid(True)
ax8.legend()
ax8.set_title('Last {} Data Points - Perplexity'.format(recent_count))
if recent_x_min and recent_x_max:
    ax8.set_xlim(recent_x_min, recent_x_max)

# Rotate x-axis labels for better readability
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    plt.setp(ax.get_xticklabels(), rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle

# Save the plot
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
print("Plot has been saved as 'training_metrics.png'")