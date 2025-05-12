import os
import librosa
import numpy as np
from tqdm import tqdm

def extract_features_from_directory(directory, label, sr=16000, frame_length=0.025, hop_length=0.010, n_mfcc=13):
    import librosa
    import numpy as np
    from tqdm import tqdm
    import os

    data = []
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.wav'):
                files.append(os.path.join(root, filename))

    for filepath in tqdm(files, desc=f"Processing {label}"):
        y, _ = librosa.load(filepath, sr=sr)
        frame_len = int(sr * frame_length)
        hop_len = int(sr * hop_length)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_len, n_fft=frame_len).T

        for mfcc in mfccs:
            data.append((mfcc, label, os.path.basename(filepath)))

    return data

def create_dataset(speech_dir, noise_dir):
    print("[Dataset] Loading speech files...")
    speech_data = extract_features_from_directory(speech_dir, 'foreground')                                             # We use the extract_features_from_directory function for foreground.
    print(f"[Dataset] Loaded {len(speech_data)} speech samples.")

    print("[Dataset] Loading noise files...")
    noise_data = extract_features_from_directory(noise_dir, 'background')                                               # We use the extract_features_from_directory function for background.
    print(f"[Dataset] Loaded {len(noise_data)} noise samples.")

    all_data = speech_data + noise_data
    print(f"[Dataset] Total samples: {len(all_data)}")

    if len(all_data) == 0:
        raise ValueError("No data extracted. Check your file paths or .wav files.")                                     # An Error if we dont have files for data.

    np.random.shuffle(all_data)

    features = np.array([x[0] for x in all_data])
    labels = np.array([x[1] for x in all_data])
    filenames = np.array([x[2] for x in all_data])

    os.makedirs("data", exist_ok=True)                                                                                  # Εnsure directory exists.
    np.savez("data/features_dataset.npz", X=features, y=labels, filenames=filenames)                                    # Use the extracted files to a new dataset.

    print("[Dataset] Dataset saved to data/features_dataset.npz")                                                       # Print message that everything went right.