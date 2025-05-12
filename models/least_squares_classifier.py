import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def train_least_squares():
    print("[Least Squares] Loading dataset...")
    data = np.load("data/features_dataset.npz")                                     # Use the dataset we created on extract_features.py.

    X = data['X']                                                                   
    y = data['y']                                                                  

    if X.size == 0 or y.size == 0:
        raise ValueError("Dataset Empty!")                                          # Error message if the dataset is Empty.

    encoder = LabelEncoder()                                                        # 0 or 1 depends on background or foreground.
    y_encoded = encoder.fit_transform(y)

    print("[Least Squares] Training model...")
    model = RidgeClassifier(class_weight='balanced')                                # Linear model trained with Least Squares.
    model.fit(X, y_encoded)                                                         # Model training for X data depends the y_encoded tags 

    preds = model.predict(X)                                                        # Predictions on the same data.

    print("\n[Least Squares] Report:\n")
    print(classification_report(y_encoded, preds, target_names=encoder.classes_))   # Printing full report on the least squares.