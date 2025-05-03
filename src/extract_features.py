import os
import librosa
import numpy as np
from tqdm import tqdm

def extract_features_from_directory(directory, label, sr=16000, frame_length=0.025, hop_length=0.010, n_mfcc=13):
    data = []
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]

    for file in tqdm(files, desc=f"Processing {label}"):
        filepath = os.path.join(directory, file)
        y, _ = librosa.load(filepath, sr=sr)
        frame_len = int(sr * frame_length)
        hop_len = int(sr * hop_length)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_len, n_fft=frame_len).T

        for mfcc in mfccs:
            data.append((mfcc, label, file))

    return data

def create_dataset(speech_dir, noise_dir):
    speech_data = extract_features_from_directory(speech_dir, 'foreground')
    noise_data = extract_features_from_directory(noise_dir, 'background')

    all_data = speech_data + noise_data
    np.random.shuffle(all_data)

    features = np.array([x[0] for x in all_data])
    labels = np.array([x[1] for x in all_data])
    filenames = np.array([x[2] for x in all_data])

    # Create 'data' directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save to consistent relative path
    np.savez("data/features_dataset.npz", X=features, y=labels, filenames=filenames)