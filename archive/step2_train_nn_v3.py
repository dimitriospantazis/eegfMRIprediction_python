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
from sklearn.model_selection import train_test_split


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

def train_model(model, train_dl, val_dl, criterion, optimizer, epochs=10, patience=5):
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_dl)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_dl:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                output = model(x_val)
                loss = criterion(output, y_val)
                val_loss += loss.item()
        val_loss /= len(val_dl)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

    return train_losses, val_losses

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

def run_loso(X, Y, subject_ids, epochs=20, batch_size=32, plot_losses=False):
    logo = LeaveOneGroupOut()
    all_r2 = []

    X_flat = X.reshape(X.shape[0], -1)
    input_dim = X_flat.shape[1]

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_flat, Y, groups=subject_ids)):
        print(f"\n--- Fold {fold+1}/{len(np.unique(subject_ids))} ---")

        X_trainval, X_test = X_flat[train_idx], X_flat[test_idx]
        Y_trainval, Y_test = Y[train_idx], Y[test_idx]

        # Split train into train/val
        X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=42)

        train_ds = EEGfMRIDataset(X_train, Y_train)
        val_ds = EEGfMRIDataset(X_val, Y_val)
        test_ds = EEGfMRIDataset(X_test, Y_test)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        test_dl = DataLoader(test_ds, batch_size=batch_size)

        model = EEGtofMRI(input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train_losses, val_losses = train_model(model, train_dl, val_dl, criterion, optimizer, epochs=epochs)

        y_pred, y_true = evaluate_model(model, test_dl)
        r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
        all_r2.append(r2)
        print(f"LOSO Fold R²: {r2:.4f}")

        # Optional: plot learning curves
        if plot_losses:
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.title(f'Fold {fold+1} Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.show()

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
r2_scores = run_loso(X_normalized, Y, subject_ids, epochs=50, plot_losses=True)
