import matplotlib.pyplot as plt
import datetime
import re

# Read and parse the log file
timestamps = []
losses = []
perplexities = []

with open('../src/experiments/logs/train_-1.log', 'r', encoding='utf-8') as f:
    for line in f:
        # Extract timestamp, loss and perplexity using regex
        # Match both formats: with [RANK:0] prefix and without
        match = re.search(r'(?:\[RANK:0\])?\[(.*?)\].*?(?:训练损失|loss): ([\d.]+).*?(?:困惑度|perplexity): ([\d.]+)', line, re.IGNORECASE)
        if match:
            timestamp_str = match.group(1)
            loss = float(match.group(2))
            perplexity = float(match.group(3))
            
            # Parse timestamp
            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            timestamps.append(timestamp)
            losses.append(loss)
            perplexities.append(perplexity)

# Create figure with four subplots (2x2 grid)
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Training Metrics', fontsize=16)

# Global loss plot
ax1 = plt.subplot(2, 2, 1)
ax1.plot(timestamps, losses, 'b-', label='Loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.legend()
ax1.set_title('Global Training Loss')

# Global perplexity plot
ax2 = plt.subplot(2, 2, 2)
ax2.plot(timestamps, perplexities, 'r-', label='Perplexity')
ax2.set_ylabel('Perplexity')
ax2.set_xlabel('Time')
ax2.grid(True)
ax2.legend()
ax2.set_title('Global Perplexity')

# Get the most recent 30 data points (or all if less than 30)
recent_count = min(30, len(timestamps))
recent_timestamps = timestamps[-recent_count:]
recent_losses = losses[-recent_count:]
recent_perplexities = perplexities[-recent_count:]

# Zoomed-in loss plot
ax3 = plt.subplot(2, 2, 3)
ax3.plot(recent_timestamps, recent_losses, 'b-', marker='o', label='Loss')
ax3.set_ylabel('Loss')
ax3.set_xlabel('Time')
ax3.grid(True)
ax3.legend()
ax3.set_title('Last {} Data Points - Loss'.format(recent_count))

# Zoomed-in perplexity plot
ax4 = plt.subplot(2, 2, 4)
ax4.plot(recent_timestamps, recent_perplexities, 'r-', marker='o', label='Perplexity')
ax4.set_ylabel('Perplexity')
ax4.set_xlabel('Time')
ax4.grid(True)
ax4.legend()
ax4.set_title('Last {} Data Points - Perplexity'.format(recent_count))

# Rotate x-axis labels for better readability
for ax in [ax1, ax2, ax3, ax4]:
    plt.setp(ax.get_xticklabels(), rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle

# Save the plot
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
print("Plot has been saved as 'training_metrics.png'")