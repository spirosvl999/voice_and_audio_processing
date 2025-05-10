# Voice And Audio Processing Final Project
This repository contains the final project for the Speech Segmentation course. The goal is to detect and segment speech (foreground) versus background noise from audio files using classification and post-processing techniques.

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
│   ├── train/
│   │   ├── speech/
│   │   └── noise/
│   └── test/
│       ├── file1.wav
│       └── transcriptions/
│           └── test_file.json
├── outputs/
│   └── predictions.csv
├── main.py
├── evaluate.py
├── visualize_segments.py
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
python main.py             # Train & predict
python evaluate.py         # Evaluate predictions against ground truth
python visualize_segments.py   # Visualize waveform and segments
```

## Data Sources:
- [MUSAN Corpus](https://www.openslr.org/17)
- [CHiME Dataset](https://www.openslr.org/26)

## Last Updated
This README was last updated on [5/11/2025].
