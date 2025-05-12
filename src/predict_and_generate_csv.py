import os
import librosa
import numpy as np
import pandas as pd
import torch
from models.mlp_classifier import MLP
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from scipy.signal import medfilt
import pickle


def predict_and_export_csv(filepath, model_type="mlp", sr=16000):
    if not os.path.isfile(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return

    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)

    try:
        y, _ = librosa.load(filepath, sr=sr)
    except Exception as e:
        print(f"[ERROR] Could not load audio file: {e}")
        return

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_len, n_fft=frame_len).T

    if model_type == "mlp":
        model = MLP(mfccs.shape[1])
        model.load_state_dict(torch.load("voice_and_audio_processing/models/mlp_weights.pth"))
        model.eval()

        with open("voice_and_audio_processing/models/mlp_label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)

        with torch.no_grad():
            inputs = torch.tensor(mfccs, dtype=torch.float32)
            preds = model(inputs).numpy().flatten()
    else:
        model = Ridge(alpha=1.0)
        try:
            data = np.load("voice_and_audio_processing/data/features_dataset.npz")
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(data['y'])
            model.fit(data['X'], y_encoded)
            preds = model.predict(mfccs)
        except Exception as e:
            print(f"[ERROR] Failed to load or use ridge model: {e}")
            return

    preds_bin = (preds > 0.5).astype(int)
    preds_bin = medfilt(preds_bin, kernel_size=5)

    timestamps = []
    class_map = {0: "background", 1: "foreground"}
    last_class = preds_bin[0]
    start_time = 0.0

    for i in range(1, len(preds_bin)):
        if preds_bin[i] != last_class:
            end_time = i * 0.010  # hop length in seconds
            timestamps.append([os.path.basename(filepath), round(start_time, 2), round(end_time, 2), class_map[last_class]])
            start_time = end_time
            last_class = preds_bin[i]

    timestamps.append([os.path.basename(filepath), round(start_time, 2), round(len(preds_bin) * 0.010, 2), class_map[last_class]])

    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(timestamps, columns=["Audiofile", "start", "end", "class"])
    df.to_csv("voice_and_audio_processing/outputs/predictions.csv", index=False)
    print("Saved to outputs/predictions.csv")
