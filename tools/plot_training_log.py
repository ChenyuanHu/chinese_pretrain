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

with open('../src/experiments/logs/train_-1.log', 'r', encoding='utf-8') as f:
    for line in f:
        # Extract timestamp, loss, perplexity and learning rate using regex
        match = re.search(r'(?:\[RANK:0\])?\[(.*?)\].*?(?:训练损失|loss): ([\d.]+).*?(?:困惑度|perplexity): ([\d.]+).*?LR: ([\d.e-]+)', line, re.IGNORECASE)
        if match:
            timestamp_str = match.group(1)
            loss = float(match.group(2))
            perplexity = float(match.group(3))
            lr = float(match.group(4))
            
            # Parse timestamp
            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            timestamps.append(timestamp)
            losses.append(loss)
            perplexities.append(perplexity)
            learning_rates.append(lr)
        
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
                    # Find the corresponding timestamp index
                    if timestamp in timestamps:
                        idx = timestamps.index(timestamp)
                        # Extend the list to match the current timestamp index if needed
                        while len(dataset_usage[key]) < idx:
                            dataset_usage[key].append(None)
                        dataset_usage[key].append(value)
            except json.JSONDecodeError:
                print(f"Error parsing JSON: {dataset_json}")

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 16))
fig.suptitle('Training Metrics', fontsize=16)

# Create GridSpec for more control over subplot layout
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 2, figure=fig)

# Global loss plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(timestamps, losses, 'b-', label='Loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.legend()
ax1.set_title('Global Training Loss')

# Global perplexity plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(timestamps, perplexities, 'r-', label='Perplexity')
ax2.set_ylabel('Perplexity')
ax2.set_xlabel('Time')
ax2.grid(True)
ax2.legend()
ax2.set_title('Global Perplexity')

# Learning rate plot
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(timestamps, learning_rates, 'g-', marker='o', label='Learning Rate')
ax3.set_ylabel('Learning Rate')
ax3.set_xlabel('Time')
ax3.grid(True)
ax3.legend()
ax3.set_title('Learning Rate Over Time')

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

# Get the most recent 30 data points (or all if less than 30)
recent_count = min(30, len(timestamps))
recent_timestamps = timestamps[-recent_count:]
recent_losses = losses[-recent_count:]
recent_perplexities = perplexities[-recent_count:]

# Zoomed-in loss plot
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(recent_timestamps, recent_losses, 'b-', marker='o', label='Loss')
ax5.set_ylabel('Loss')
ax5.set_xlabel('Time')
ax5.grid(True)
ax5.legend()
ax5.set_title('Last {} Data Points - Loss'.format(recent_count))

# Zoomed-in perplexity plot
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(recent_timestamps, recent_perplexities, 'r-', marker='o', label='Perplexity')
ax6.set_ylabel('Perplexity')
ax6.set_xlabel('Time')
ax6.grid(True)
ax6.legend()
ax6.set_title('Last {} Data Points - Perplexity'.format(recent_count))

# Rotate x-axis labels for better readability
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    plt.setp(ax.get_xticklabels(), rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle

# Save the plot
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
print("Plot has been saved as 'training_metrics.png'")