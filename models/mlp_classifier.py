import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import os

class MLP(nn.Module):
    def __init__(self, input_size):                                             # We set a MLP with 3 secret levels
        super().__init__()                                                      # giving us an output with Sigmoid
        self.model = nn.Sequential(                                             # predicting foreground or background pobability.
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):                                                       # Provides the datas.
        return self.model(x)

def train_mlp():
    data = np.load("voice_and_audio_processing/data/features_dataset.npz")                                 # Get the datas and their tags.
    X = data['X']
    y = data['y']

    encoder = LabelEncoder()                                                    # Converts tags (background & foreground) into 0 and 1.
    y_encoded = encoder.fit_transform(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)                             # Converts into PyTorch tensors
    y_tensor = torch.tensor(y_encoded, dtype=torch.float32).view(-1, 1)         # for use into the model.

    dataset = TensorDataset(X_tensor, y_tensor)                                 # Batch Loader
    loader = DataLoader(dataset, batch_size=64, shuffle=True)                   # 64 examples every time.

    model = MLP(X.shape[1])                                                     # Creates the MLP.
    criterion = nn.BCELoss()                                                    # Use Binary Classification.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)                  # Optimizer for learning rate.

    for epoch in range(10):                                                     # 10 Epoch for all the datas.
        for xb, yb in loader:
            preds = model(xb)
            loss = criterion(preds, yb)                                         # The loss of each one.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")                       # Printing the Results.

    with torch.no_grad():
        preds = model(X_tensor).numpy()                                         # Predicting the dataset.
        preds_binary = (preds > 0.5).astype(int)
        print("\n[MLP] Report:\n")
        print(classification_report(y_encoded, preds_binary, target_names=encoder.classes_))

    os.makedirs("models", exist_ok=True)                                        # Save the MLP into .pth file.
    torch.save(model.state_dict(), "voice_and_audio_processing/models/mlp_weights.pth")

    with open("voice_and_audio_processing/models/mlp_label_encoder.pkl", "wb") as f:                       # Save the encoder.
        pickle.dump(encoder, f)