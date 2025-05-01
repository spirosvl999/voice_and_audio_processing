import librosa
import numpy as np
import pandas as pd
import torch
from models.mlp_classifier import MLP
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

# Simple median filter
from scipy.signal import medfilt

def predict_and_export_csv(filepath, model_type="mlp", sr=16000):
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)

    y, _ = librosa.load(filepath, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_len, n_fft=frame_len).T

    if model_type == "mlp":
        model = MLP(mfccs.shape[1])
        # Προσομοίωση εκπαιδευμένου μοντέλου - εσύ φόρτωσε weights αν έχεις αποθηκεύσει
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(mfccs, dtype=torch.float32)
            preds = model(inputs).numpy().flatten()
    else:
        model = Ridge(alpha=1.0)
        # Load & fit το ίδιο dataset για now (σε κανονική χρήση: load pretrained)
        data = np.load("data/features_dataset.npz")
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(data['y'])
        model.fit(data['X'], y_encoded)
        preds = model.predict(mfccs)

    preds_bin = (preds > 0.5).astype(int)
    preds_bin = medfilt(preds_bin, kernel_size=5)

    timestamps = []
    class_map = {0: "background", 1: "foreground"}
    last_class = preds_bin[0]
    start_time = 0

    for i in range(1, len(preds_bin)):
        if preds_bin[i] != last_class:
            end_time = i * 0.010
            timestamps.append([filepath, round(start_time, 2), round(end_time, 2), class_map[last_class]])
            start_time = end_time
            last_class = preds_bin[i]

    # Τελευταίο segment
    timestamps.append([filepath, round(start_time, 2), round(len(preds_bin)*0.010, 2), class_map[last_class]])

    df = pd.DataFrame(timestamps, columns=["Audiofile", "start", "end", "class"])
    df.to_csv("outputs/predictions.csv", index=False)
    print("Saved to outputs/predictions.csv")
