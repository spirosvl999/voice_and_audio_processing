# Voice And Audio Processing Final Project
This repository contains the final project for the Speech Segmentation course. The goal is to detect and segment speech (foreground) versus background noise from audio files using classification and post-processing techniques.


![GitHub repo size](https://img.shields.io/github/repo-size/spirosvl999/voice_and_audio_processing)
![GitHub last commit](https://img.shields.io/github/last-commit/spirosvl999/voice_and_audio_processing)
![GitHub language count](https://img.shields.io/github/languages/count/spirosvl999/voice_and_audio_processing)


## Programming Languages & Libraries:
- Python 3.8+
- Scikit-learn
- Librosa
- Numpy
- Matplotlib
- Pandas
- PyTorch
- SciPy

## Features
- MFCC feature extraction
- MLP-based classifier
- Ridge regression (Least Squares) classifier
- Frame-level evaluation and classification
- CSV export of segment classifications
- JSON ground truth comparison
- Waveform & prediction visualization

## File Structure:
```
speech_segmentation_project/
├── data/
|   ├── features_dataset.npz
│   ├── train/
│   │   ├── speech/
│   │   └── noise/
│   └── test/
│       └── file1.wav
├── src/
│   ├── extract_features.py
│   └── predict_and_generate_csv.py
├── outputs/
│   └── predictions.csv
├── main.py
├── LICENCE.txt
├── requirements.txt
└── README.md
```

## How to Run:
Install the required libraries:
```bash
pip install -r requirements.txt
```

Then run:
```bash
python main.py
```

## Export：
```
outputs/predictions.csv
```

## Data Sources:
- [MUSAN Corpus](https://www.openslr.org/17)
- [CHiME Dataset](https://www.openslr.org/26)

## Last Updated
This README was last updated on [5/16/2025].
