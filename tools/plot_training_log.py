import matplotlib.pyplot as plt
import datetime
import re

# Read and parse the log file
timestamps = []
losses = []
perplexities = []

with open('../logs/train.log', 'r') as f:
    for line in f:
        # Extract timestamp, loss and perplexity using regex
        match = re.search(r'\[RANK:0\]\[(.*?)\] .*?: ([\d.]+), .*?: ([\d.]+)', line)
        if match:
            timestamp_str = match.group(1)
            loss = float(match.group(2))
            perplexity = float(match.group(3))
            
            # Parse timestamp
            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            timestamps.append(timestamp)
            losses.append(loss)
            perplexities.append(perplexity)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Training Metrics Over Time', fontsize=14)

# Plot loss
ax1.plot(timestamps, losses, 'b-', label='Loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.legend()

# Plot perplexity
ax2.plot(timestamps, perplexities, 'r-', label='Perplexity')
ax2.set_ylabel('Perplexity')
ax2.set_xlabel('Time')
ax2.grid(True)
ax2.legend()

# Rotate x-axis labels for better readability
plt.setp(ax1.get_xticklabels(), rotation=45)
plt.setp(ax2.get_xticklabels(), rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
print("Plot has been saved as 'training_metrics.png'")