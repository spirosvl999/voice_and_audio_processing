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


def predict_and_export_csv(filepath, model_type="mlp", sr=16000):                                                               # We got the .wav file, we use MLP and we use 16kHz sample rate.
    if not os.path.isfile(filepath):
        print(f"[ERROR] File not found: {filepath}")                                                                            # Error Message.
        return

    frame_len = int(sr * 0.025)                                                                                                 # 25 ms frames,
    hop_len = int(sr * 0.010)                                                                                                   # 10 ms hop

    try:
        y, _ = librosa.load(filepath, sr=sr)                                                                                    # We select the file we want to predict and generate the csv for.
    except Exception as e:
        print(f"[ERROR] Could not load audio file: {e}")                                                                        # Error Message.
        return

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_len, n_fft=frame_len).T                                  # MFCC for every frame, with .% we get it into the format: [frames, features].

    if model_type == "mlp":                                                                                                     # For using MLP.
        model = MLP(mfccs.shape[1])
        model.load_state_dict(torch.load("voice_and_audio_processing/models/mlp_weights.pth"))
        model.eval()                                                                                                            # Do eval(), for not updating into weights.

        with open("voice_and_audio_processing/models/mlp_label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)

        with torch.no_grad():
            inputs = torch.tensor(mfccs, dtype=torch.float32)
            preds = model(inputs).numpy().flatten()                                                                             # Predictions.
    else:
        model = Ridge(alpha=1.0)                                                                                                # For using Ridge.
        try:
            data = np.load("voice_and_audio_processing/data/features_dataset.npz")
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(data['y'])
            model.fit(data['X'], y_encoded)
            preds = model.predict(mfccs)
        except Exception as e:
            print(f"[ERROR] Failed to load or use ridge model: {e}")
            return

    preds_bin = (preds > 0.5).astype(int)                                                                                       # Every frame gets a 0 or 1.
    preds_bin = medfilt(preds_bin, kernel_size=5)                                                                               # Clearing small differences

    timestamps = []
    class_map = {0: "background", 1: "foreground"}
    last_class = preds_bin[0]
    start_time = 0.0

    for i in range(1, len(preds_bin)):
        if preds_bin[i] != last_class:
            end_time = i * 0.010                                                                                                # Hop length in seconds.
            timestamps.append([os.path.basename(filepath), round(start_time, 2), round(end_time, 2), class_map[last_class]])
            start_time = end_time
            last_class = preds_bin[i]

    timestamps.append([os.path.basename(filepath), round(start_time, 2), round(len(preds_bin) * 0.010, 2), class_map[last_class]])

    os.makedirs("outputs", exist_ok=True)                                                                                       # Extract into .csv file.
    df = pd.DataFrame(timestamps, columns=["Audiofile", "start", "end", "class"])
    df.to_csv("voice_and_audio_processing/outputs/predictions.csv", index=False)
    print("Saved to outputs/predictions.csv")