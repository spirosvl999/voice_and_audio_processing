import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def train_least_squares():
    data = np.load("data/features_dataset.npz")
    X = data['X']
    y = data['y']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = Ridge(alpha=1.0)
    model.fit(X, y_encoded)

    preds = model.predict(X)
    preds_binary = (preds > 0.5).astype(int)
    print("\n[Least Squares] Report:\n")
    print(classification_report(y_encoded, preds_binary, target_names=encoder.classes_))