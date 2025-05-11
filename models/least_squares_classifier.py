import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def train_least_squares():
    print("[Least Squares] Loading dataset...")
    data = np.load("data/features_dataset.npz")

    X = data['X']
    y = data['y']

    if X.size == 0 or y.size == 0:
        raise ValueError("Dataset Empty!")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("[Least Squares] Training model...")
    model = RidgeClassifier(class_weight='balanced') ####
    model.fit(X, y_encoded)

    preds = model.predict(X)

    print("\n[Least Squares] Report:\n")
    print(classification_report(y_encoded, preds, target_names=encoder.classes_))