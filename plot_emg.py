# offline plotting of EMG data from CSV file
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV (replace with your filename)
df = pd.read_csv("emg_20250922_160729.csv")

# Show the first rows to confirm structure
print(df.head())

# Plot all channels
plt.figure(figsize=(12,6))
for col in df.columns:
    if col != "timestamp":
        plt.plot(df["timestamp"], df[col], label=col)

plt.xlabel("Timestamp")
plt.ylabel("Signal Value")
plt.title("Offline EMG Signal from CSV")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
