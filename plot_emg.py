import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === CONFIG ===
CSV_FILE = "emg_raw_20251102_184520.csv"  # <-- change to your actual file
BYTES_PER_SAMPLE = 3  # 3 bytes = 24-bit sample
CHANNEL_COUNT = 2     # adjust if needed

# === LOAD CSV ===
df = pd.read_csv(CSV_FILE)

# === Decode function ===
def decode_24bit_signed(hex_string):
    """Convert a hex string into signed 24-bit integers."""
    try:
        raw_bytes = bytes.fromhex(hex_string)
        samples = []
        for i in range(0, len(raw_bytes), BYTES_PER_SAMPLE):
            chunk = raw_bytes[i:i+BYTES_PER_SAMPLE]
            if len(chunk) < BYTES_PER_SAMPLE:
                continue
            val = int.from_bytes(chunk, byteorder="big", signed=False)
            if val & 0x800000:  # check sign bit
                val -= 1 << 24
            samples.append(val)
        return samples
    except Exception:
        return []

# === Apply decoding ===
df["samples"] = df["hex_data"].apply(decode_24bit_signed)

# === Flatten and normalize ===
flat = [v for row in df["samples"] for v in row]
flat = np.array(flat, dtype=float)

# Optional normalization for visualization
flat = flat / np.max(np.abs(flat))

# === Plot ===
plt.figure(figsize=(12, 5))
plt.plot(flat, lw=0.8)
plt.title("Decoded 24-bit EMG Stream")
plt.xlabel("Sample index")
plt.ylabel("Normalized amplitude")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"[INFO] Decoded {len(flat)} samples from {len(df)} BLE packets.")
