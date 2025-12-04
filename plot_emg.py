import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Optional Band-Pass Filter ----------------
ENABLE_FILTER = False
FILTER_BAND = (20, 450)  # Hz

def apply_bandpass(sig, fs, low=20, high=450, order=4):
    """Apply Butterworth bandpass filter."""
    from scipy.signal import butter, filtfilt
    b, a = butter(order, [low, high], btype="bandpass", fs=fs)
    return filtfilt(b, a, sig)

# ---------------- Plot Function ----------------
def plot_emg(csv_path: str):
    # --- Load data ---
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} samples from {csv_path}")

    # --- Detect channel columns ---
    channel_cols = [c for c in df.columns if c.lower().startswith("ch")]
    if not channel_cols:
        print("[ERROR] No channel columns found! Columns:", df.columns.tolist())
        return

    print(f"[INFO] Found {len(channel_cols)} channels: {channel_cols}")

    # ---------------- Timestamp processing ----------------s
    # fallback if no timestamp
    fs = 2000
    print("Len is =:", len(df))
    time = np.arange(len(df)) / fs
    print("[WARN] No timestamp column found — assuming 1000 Hz")

    # ---------------- Create subplots ----------------
    fig, axes = plt.subplots(len(channel_cols), 1, sharex=True,
                             figsize=(12, 2.3 * len(channel_cols)))
    if len(channel_cols) == 1:
        axes = [axes]

    # ---------------- Plot each channel ----------------
    for i, ch in enumerate(channel_cols):
        sig = df[ch].values.astype(float)

        # Remove DC offset (recommended for EMG)
        sig = sig - np.mean(sig)

        # Optional band-pass filter
        if ENABLE_FILTER:
            try:
                sig = apply_bandpass(sig, fs, *FILTER_BAND)
            except Exception as e:
                print(f"[WARN] Filtering failed for {ch}: {e}")

        axes[i].plot(time, sig, lw=0.8)
        axes[i].set_ylabel(ch)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"EMG Data – {csv_path}")
    plt.tight_layout()

    # Plot power spectrum of every channel in a separate figure
    fig_psd, axes_psd = plt.subplots(len(channel_cols), 1, sharex=True,
                                     figsize=(12, 2.3 * len(channel_cols)))
    if len(channel_cols) == 1:
        axes_psd = [axes_psd]
    for i, ch in enumerate(channel_cols):
        sig = df[ch].values.astype(float)
        sig = sig - np.mean(sig)

        # Compute power spectrum and plot in logarithmic scale in db/Hz
        freqs = np.fft.rfftfreq(len(sig), d=1/fs)
        psd = np.abs(np.fft.rfft(sig))**2 / len(sig)
        axes_psd[i].semilogx(freqs, 10 * np.log10(psd), lw=0.8)
        axes_psd[i].set_ylabel(f"{ch} PSD (dB/Hz)")
        axes_psd[i].grid(True, alpha=0.3)
    axes_psd[-1].set_xlabel("Frequency (Hz)")
    fig_psd.suptitle(f"EMG Power Spectrum – {csv_path}")
    plt.tight_layout()
    plt.show()


# ---------------- Main Entry ----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_emg.py <path_to_csv>")
        sys.exit(1)

    plot_emg(sys.argv[1])