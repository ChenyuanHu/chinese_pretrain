import re
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def parse_log_file(filename):
    epochs = []
    train_losses = []
    train_perplexities = []
    val_losses = []
    val_perplexities = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Use regex to match epoch data
    pattern = r'Epoch \[(\d+)/\d+\].*?训练损失: ([\d.]+), 训练困惑度: ([\d.]+).*?验证损失: ([\d.]+), 验证困惑度: ([\d.]+)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        train_perp = float(match.group(3))
        val_loss = float(match.group(4))
        val_perp = float(match.group(5))
        
        epochs.append(epoch)
        train_losses.append(train_loss)
        train_perplexities.append(train_perp)
        val_losses.append(val_loss)
        val_perplexities.append(val_perp)
    
    return epochs, train_losses, train_perplexities, val_losses, val_perplexities

# Get all training log files from logs directory
log_files = sorted(glob.glob('logs/train_*.log'))
if not log_files:
    print("No training log files found in logs directory!")
    exit(1)

print(f"Found {len(log_files)} log files:")
for log_file in log_files:
    print(f"- {os.path.basename(log_file)}")

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']  # 增加更多颜色以支持更多日志文件

# Process each log file
for i, log_file in enumerate(log_files):
    try:
        epochs, train_losses, train_perp, val_losses, val_perp = parse_log_file(log_file)
        
        # Get training session number from filename
        file_date = os.path.basename(log_file).split('_')[1]
        
        # Plot loss curves
        label = f'Training {file_date}'
        ax1.plot(epochs, train_losses, color=colors[i % len(colors)], linestyle='-', label=f'{label} - Train Loss')
        ax1.plot(epochs, val_losses, color=colors[i % len(colors)], linestyle='--', label=f'{label} - Val Loss')
        
        # Plot perplexity curves
        ax2.plot(epochs, train_perp, color=colors[i % len(colors)], linestyle='-', label=f'{label} - Train Perplexity')
        ax2.plot(epochs, val_perp, color=colors[i % len(colors)], linestyle='--', label=f'{label} - Val Perplexity')
        
    except Exception as e:
        print(f"Error processing file {log_file}: {str(e)}")

# Set plot properties
ax1.set_title('Loss During Training')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax1.legend()

ax2.set_title('Perplexity During Training')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Perplexity')
ax2.grid(True)
ax2.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Save the plot
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print("\nPlot has been saved as training_curves.png") 