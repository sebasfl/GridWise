# src/forecasting/train_lstm_gpu.py
import argparse, pathlib, time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------- Dataset --------------------
class EnergyDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len=168):
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(X), torch.tensor(y)

# -------------------- Modelo --------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # último paso
        return self.fc(out).squeeze()

# -------------------- Entrenamiento --------------------
def train_model(model, train_loader, valid_loader, device, epochs=20, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)

        # validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss = criterion(preds, y)
                val_loss += loss.item() * X.size(0)
        val_loss /= len(valid_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} - Train RMSE: {np.sqrt(train_loss):.4f} - Val RMSE: {np.sqrt(val_loss):.4f}")

# -------------------- Script principal --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/app/data/processed/bdg2_electricity_long.parquet")
    ap.add_argument("--building_id", default=None)
    ap.add_argument("--seq_len", type=int, default=168)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model_out", default="/app/models/lstm_model.pth")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # cargar datos
    df = pd.read_parquet(args.parquet)
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp_local"
    df = df[df["meter"] == "electricity"].copy()

    if args.building_id:
        df = df[df["building_id"].astype(str) == str(args.building_id)].copy()

    # ordenar por tiempo
    df = df.sort_values(ts_col)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    # features numéricas (aquí puedes incluir tus lags, rollings, calendario)
    from train_catboost_gpu import add_calendar, make_lags_and_rolls, parse_int_list
    df = add_calendar(df, ts_col=ts_col)
    df = make_lags_and_rolls(df, ts_col=ts_col, y_col="value", by="building_id")

    df = df.dropna()
    drop_cols = { "value", ts_col, "meter", "site_id", "timezone", "building_id" }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    target_col = "value"

    # dividir train/val por tiempo
    cut = df[ts_col].quantile(0.8)
    train_df = df[df[ts_col] <= cut]
    val_df = df[df[ts_col] > cut]

    train_dataset = EnergyDataset(train_df, feature_cols, target_col, seq_len=args.seq_len)
    val_dataset = EnergyDataset(val_df, feature_cols, target_col, seq_len=args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # modelo
    model = LSTMRegressor(input_dim=len(feature_cols)).to(device)
    train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)

    # guardar
    torch.save(model.state_dict(), args.model_out)
    print(f"[OK] Saved model to {args.model_out}")

if __name__ == "__main__":
    main()
