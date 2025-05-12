import os
from src.extract_features import create_dataset
from models.least_squares_classifier import train_least_squares
from models.mlp_classifier import train_mlp
from src.predict_and_generate_csv import predict_and_export_csv

if __name__ == "__main__":
    os.makedirs("voice_and_audio_processing/data", exist_ok=True)                           # Ensure output data folder exists.

    print("[1] Extracting features...")
    create_dataset(                                                                         # Creating dataset of background noise and speech.
        "voice_and_audio_processing/data/train/speech",
        "voice_and_audio_processing/data/train/noise"
    )

    print("[2] Training Least Squares model...")
    train_least_squares()                                                                   # Binary classification (background vs foreground) using Least Squares Regression.

    print("[3] Training MLP model...")
    train_mlp()                                                                             # Training MLP model for 10 Epoch.

    print("[4] Predicting and generating CSV...")
    predict_and_export_csv(                                                                 # Checking the .wav file and creating the final .csv file of the output.
        "voice_and_audio_processing/data/test/file1.wav",
        model_type="mlp"
    )