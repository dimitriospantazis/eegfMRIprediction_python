import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------- Device Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Dataset ----------
class EEGfMRIDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ---------- Model ----------
class EEGtofMRI(nn.Module):
    def __init__(self, input_dim, output_dim=300):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.model(x)

# ---------- Training Function ----------
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f}")

# ---------- Evaluation Function ----------
def evaluate_model(model, dataloader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            preds.append(output.cpu().numpy())
            targets.append(y_batch.numpy())
    return np.vstack(preds), np.vstack(targets)

# ---------- Main LOSO Loop ----------
def run_loso(X, Y, subject_ids, epochs=20, batch_size=32):
    logo = LeaveOneGroupOut()
    all_r2 = []

    X_flat = X.reshape(X.shape[0], -1)
    input_dim = X_flat.shape[1]

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_flat, Y, groups=subject_ids)):
        print(f"\n--- Fold {fold+1}/{len(np.unique(subject_ids))} ---")

        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        train_ds = EEGfMRIDataset(X_train, Y_train)
        test_ds = EEGfMRIDataset(X_test, Y_test)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size)

        model = EEGtofMRI(input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train_model(model, train_dl, criterion, optimizer, epochs=epochs)
        y_pred, y_true = evaluate_model(model, test_dl)

        r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
        all_r2.append(r2)
        print(f"LOSO Fold R²: {r2:.4f}")

    print(f"\nAverage R² across subjects: {np.mean(all_r2):.4f}")
    return all_r2

# ---------- Load and Preprocess Data ----------
load_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025', 'processed_data', 'eeg_fmri_data.npz')
data = np.load(load_path, allow_pickle=True)
X = data['X']  # shape: (n_samples, n_channels, n_freqs, n_bins)
Y = data['Y']  # shape: (n_samples, 300)
subject_ids = data['subject_ids']

# Log-transform
epsilon = 1e-10
X_log = np.log10(X + epsilon)

'''
# Plot histogram of raw and log-transformed data
plt.figure(figsize=(10, 4))
plt.hist(X.flatten(), bins=100, alpha=0.5, label="Raw", color='gray')
plt.hist(X_log.flatten(), bins=100, alpha=0.5, label="Log10", color='steelblue')
plt.title("Histogram of EEG Power (Raw vs. Log10)")
plt.legend()
plt.tight_layout()
plt.show()
'''

# Normalize
n_samples, n_channels, n_freqs, n_bins = X_log.shape
X_reshaped = X_log.reshape(n_samples, -1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X_normalized = X_scaled.reshape(n_samples, n_channels, n_freqs, n_bins)

'''
# Histogram after normalization
plt.figure(figsize=(10, 4))
plt.hist(X_normalized.flatten(), bins=100, color='seagreen', alpha=0.8)
plt.title("Histogram of Z-score Normalized EEG Power")
plt.xlabel("Z-score")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()
'''

# ---------- Run the LOSO Cross-validation ----------
r2_scores = run_loso(X_normalized, Y, subject_ids, epochs=20)
