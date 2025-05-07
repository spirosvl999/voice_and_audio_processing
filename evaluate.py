import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# --- SETTINGS ---
PREDICTIONS_CSV = "outputs/predictions.csv"
TRANSCRIPTIONS_JSON = "data/test/transcriptions/test_file.json"
FRAME_SIZE = 0.01  # 10ms
DURATION = None  # αν ξέρεις τη διάρκεια του test αρχείου, βάλε το εδώ (σε sec)

# --- Load Predictions ---
pred_df = pd.read_csv(PREDICTIONS_CSV)

# Βρίσκουμε τη διάρκεια
if DURATION is None:
    DURATION = pred_df["end"].max()

# Μετατροπή predictions σε frame-level
n_frames = int(DURATION / FRAME_SIZE)
pred_labels = ["background"] * n_frames

for _, row in pred_df.iterrows():
    start_frame = int(row["start"] / FRAME_SIZE)
    end_frame = int(row["end"] / FRAME_SIZE)
    for i in range(start_frame, min(end_frame, n_frames)):
        pred_labels[i] = row["class"]

# --- Load Ground Truth ---
with open(TRANSCRIPTIONS_JSON, "r") as f:
    gt_data = json.load(f)

annotations = gt_data["annotations"]
gt_labels = ["background"] * n_frames

for segment in annotations:
    start = segment["start"]
    end = segment["end"]
    label = segment["class"]
    start_frame = int(start / FRAME_SIZE)
    end_frame = int(end / FRAME_SIZE)
    for i in range(start_frame, min(end_frame, n_frames)):
        gt_labels[i] = label

# Evaluation
print("\nClassification Report (frame-level):")
print(classification_report(gt_labels, pred_labels, digits=4))