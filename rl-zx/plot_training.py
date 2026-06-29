import pandas as pd
import matplotlib.pyplot as plt

# Load the structured CSV file
df = pd.read_csv("training_logs.csv")

# Create a figure with two subplots side-by-side or stacked
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Reward
ax1.plot(df['Episode'], df['Total Reward'], color='blue', label='Total Reward')
ax1.set_ylabel('Reward')
ax1.set_title('Training Performance Over Time')
ax1.grid(True)

# Plot Loss
ax2.plot(df['Episode'], df['Final Loss'], color='red', label='Final Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Episode')
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_performance.png', dpi=300)
plt.show()

