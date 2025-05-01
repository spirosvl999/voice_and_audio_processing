# Voice And Audio Processing Final Project
This repository contains the final project for the Speech Segmentation course. The goal is to detect and segment speech (foreground) versus background noise from audio files using classification and post-processing techniques.

## Programming Languages & Libraries:
- Python 3.8+
- Scikit-learn
- Librosa
- Numpy
- Matplotlib
- Pandas

## Tools Used
- MLP Classifier (3-layer)
- Least Squares Classifier
- Frame-level Evaluation
- CSV Export of Segments
- JSON Ground Truth Comparison
- Waveform Visualization

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
```bash
pip install -r requirements.txt
python main.py       # Train & predict
```

## Data Sources:
- [MUSAN Corpus](https://www.openslr.org/17)
- [CHiME Dataset](https://www.openslr.org/26)

## Last Updated
This README was last updated on [5/1/2025].
