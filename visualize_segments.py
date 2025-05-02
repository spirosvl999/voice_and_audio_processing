import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

# --- SETTINGS ---
AUDIO_PATH = "data/test/file1.wav"
PREDICTIONS_CSV = "outputs/predictions.csv"
TRANSCRIPTIONS_JSON = "data/test/transcriptions/test_file.json"

# --- Load Audio ---
y, sr = librosa.load(AUDIO_PATH, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
time = np.linspace(0, duration, num=len(y))

# --- Load Predictions ---
preds = pd.read_csv(PREDICTIONS_CSV)

# --- Load Ground Truth ---
with open(TRANSCRIPTIONS_JSON, "r") as f:
    gt = json.load(f)["annotations"]

# --- Plot ---
plt.figure(figsize=(15, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.title("Speech Segmentation: Ground Truth vs Predictions")

# Plot ground truth
for seg in gt:
    color = "green" if seg["class"] == "foreground" else "grey"
    plt.axvspan(seg["start"], seg["end"], color=color, alpha=0.3, label="GT: " + seg["class"])

# Plot predictions
for _, row in preds.iterrows():
    color = "red" if row["class"] == "foreground" else "blue"
    plt.axvspan(row["start"], row["end"], color=color, alpha=0.3, label="PR: " + row["class"])

# Avoid duplicate labels in legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()