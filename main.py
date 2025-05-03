import os
from src.extract_features import create_dataset
from models.least_squares_classifier import train_least_squares
from models.mlp_classifier import train_mlp
from src.predict_and_generate_csv import predict_and_export_csv

if __name__ == "__main__":
    # Ensure output data folder exists
    os.makedirs("voice_and_audio_processing/data", exist_ok=True)

    print("[1] Extracting features...")
    create_dataset(
        "voice_and_audio_processing/data/train/speech",
        "voice_and_audio_processing/data/train/noise"
    )

    print("[2] Training Least Squares model...")
    train_least_squares()

    print("[3] Training MLP model...")
    train_mlp()

    print("[4] Predicting and generating CSV...")
    predict_and_export_csv(
        "voice_and_audio_processing/data/test/file1.wav",
        model_type="mlp"
    )