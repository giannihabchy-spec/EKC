import calendar
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from ml.modeling import _split, _add_holiday_features
from ml.config import nhits_tuning


# ── Covariates (calendar + holidays only, no lags/rolling stats) ─────────

def _make_covariates(index: pd.DatetimeIndex, freq: str) -> pd.DataFrame:
    df = pd.DataFrame(index=index)

    if freq == "D":
        df["day_of_week"] = index.dayofweek
        df["day_of_month"] = index.day
        df["is_weekend"] = (index.dayofweek >= 5).astype(int)

    df["month"] = index.month
    df["weekofyear"] = index.isocalendar().week.astype(int).values

    df = _add_holiday_features(df, freq)
    return df


# ── Dataset ──────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, sales: np.ndarray, covariates: np.ndarray, lookback: int):
        self.sales = sales
        self.covariates = covariates
        self.lookback = lookback
        self.n = len(sales) - lookback

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        s_window = self.sales[idx: idx + self.lookback]
        c_window = self.covariates[idx: idx + self.lookback]
        x = np.column_stack([s_window, c_window])
        y = self.sales[idx + self.lookback]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32),
        )


# ── Model ────────────────────────────────────────────────────────────────

class NHiTSBlock(nn.Module):
    def __init__(self, input_dim, lookback, hidden_size, pool_kernel):
        super().__init__()
        pooled_len = lookback // pool_kernel
        flat_dim = pooled_len * input_dim

        self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.backcast_fc = nn.Linear(hidden_size, lookback)
        self.forecast_fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        pooled = self.pool(x.permute(0, 2, 1))
        flat = pooled.reshape(pooled.size(0), -1)
        h = self.mlp(flat)
        return self.backcast_fc(h), self.forecast_fc(h)


class NHiTS(nn.Module):
    def __init__(self, input_dim, lookback, n_stacks=3, hidden_size=128):
        super().__init__()
        ratios = self._get_ratios(n_stacks, lookback)
        self.blocks = nn.ModuleList([
            NHiTSBlock(input_dim, lookback, hidden_size, r) for r in ratios
        ])

    @staticmethod
    def _get_ratios(n_stacks, lookback):
        ratios = []
        for i in range(n_stacks):
            r = max(1, lookback // (2 ** (n_stacks - 1 - i)))
            while lookback % r != 0:
                r -= 1
                if r < 1:
                    r = 1
                    break
            ratios.append(r)
        return ratios

    def forward(self, x):
        residual = x[:, :, 0]
        forecast = torch.zeros(x.size(0), 1, device=x.device)
        for block in self.blocks:
            bc, fc = block(x)
            residual = residual - bc
            forecast = forecast + fc
            x = x.clone()
            x[:, :, 0] = residual
        return forecast


# ── Training ─────────────────────────────────────────────────────────────

def _train(model, train_loader, val_loader, lr, epochs=100, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = None
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            loss = nn.functional.l1_loss(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = np.mean([
                nn.functional.l1_loss(model(xb), yb).item()
                for xb, yb in val_loader
            ])

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ── Fit function ─────────────────────────────────────────────────────────

def fit_nhits(s: pd.Series, freq: str = "D") -> dict:
    train, val, test = _split(s)

    sales_mean = train.mean()
    sales_std = train.std() + 1e-8

    full_sales_norm = (s.values - sales_mean) / sales_std

    covariates = _make_covariates(s.index, freq)
    cov_values = covariates.values.astype(np.float32)
    input_dim = 1 + cov_values.shape[1]

    n_train = len(train)
    n_val = len(val)
    n_train_val = n_train + n_val

    best_wape = float("inf")
    best_params = None
    best_state = None

    for lookback in nhits_tuning["lookback"]:
        if lookback >= n_train:
            continue

        for hidden_size in nhits_tuning["hidden_size"]:
            for n_stacks in nhits_tuning["n_stacks"]:
                for lr in nhits_tuning["lr"]:

                    train_ds = WindowDataset(
                        full_sales_norm[:n_train],
                        cov_values[:n_train],
                        lookback,
                    )
                    if len(train_ds) == 0:
                        continue

                    val_ds = WindowDataset(
                        full_sales_norm[n_train - lookback: n_train_val],
                        cov_values[n_train - lookback: n_train_val],
                        lookback,
                    )
                    if len(val_ds) == 0:
                        continue

                    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=64)

                    model = NHiTS(input_dim, lookback, n_stacks, hidden_size)
                    model = _train(model, train_loader, val_loader, lr)

                    # compute val WAPE in original scale
                    model.eval()
                    val_preds = []
                    with torch.no_grad():
                        for xb, _ in val_loader:
                            val_preds.append(model(xb).squeeze(-1))
                    val_preds_norm = torch.cat(val_preds).numpy()
                    val_preds_orig = val_preds_norm * sales_std + sales_mean

                    y_val = val.values
                    denom = np.sum(np.abs(y_val))
                    if denom == 0:
                        continue

                    wape = np.sum(np.abs(y_val[:len(val_preds_orig)] - val_preds_orig)) / denom

                    if wape < best_wape:
                        best_wape = wape
                        best_params = {
                            "lookback": lookback,
                            "hidden_size": hidden_size,
                            "n_stacks": n_stacks,
                            "lr": lr,
                        }
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_params is None:
        raise ValueError("No valid configuration found")

    val_wape = round(best_wape * 100, 2)
    lookback = best_params["lookback"]

    # Retrain on train+val, evaluate on test
    train_val_ds = WindowDataset(
        full_sales_norm[:n_train_val],
        cov_values[:n_train_val],
        lookback,
    )
    test_ds = WindowDataset(
        full_sales_norm[n_train_val - lookback:],
        cov_values[n_train_val - lookback:],
        lookback,
    )

    model = NHiTS(input_dim, lookback, best_params["n_stacks"], best_params["hidden_size"])
    model.load_state_dict(best_state)
    train_val_loader = DataLoader(train_val_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)
    model = _train(model, train_val_loader, test_loader, best_params["lr"])

    model.eval()
    test_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            test_preds.append(model(xb).squeeze(-1))
    test_preds_norm = torch.cat(test_preds).numpy()
    test_preds_orig = test_preds_norm * sales_std + sales_mean

    y_test = test.values[:len(test_preds_orig)]
    final_mae = round(float(mean_absolute_error(y_test, test_preds_orig)), 2)
    final_rmse = round(float(np.sqrt(np.mean((y_test - test_preds_orig) ** 2))), 2)
    final_wape = round(float(np.sum(np.abs(y_test - test_preds_orig)) / np.sum(np.abs(y_test)) * 100), 2)

    test_pred = {
        "dates": [d.strftime("%Y-%m-%d") for d in test.index[:len(test_preds_orig)]],
        "values": [round(float(v), 2) for v in test_preds_orig],
    }

    # Retrain on full data (warm start from current weights)
    full_ds = WindowDataset(full_sales_norm, cov_values, lookback)
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=True)
    model = _train(model, full_loader, full_loader, best_params["lr"], epochs=50, patience=5)

    # Recursive forecast
    forecast = _recursive_forecast_nhits(
        model, s, covariates, lookback, sales_mean, sales_std, freq,
    )

    metrics = {
        "val_wape": val_wape,
        "final_mae": final_mae,
        "final_rmse": final_rmse,
        "final_wape": final_wape,
    }

    return {
        "model": model,
        "best_params": best_params,
        "metrics": metrics,
        "final_features": None,
        "norm_params": {"mean": float(sales_mean), "std": float(sales_std)},
        "test_pred": test_pred,
        "forecast": forecast,
        "from": s.index.min().strftime("%Y-%m-%d"),
        "to": s.index.max().strftime("%Y-%m-%d"),
    }


def _recursive_forecast_nhits(model, s, covariates, lookback,
                               sales_mean, sales_std, freq, months=8):
    step = pd.Timedelta(days=1) if freq == "D" else pd.Timedelta(weeks=1)
    max_date = s.index.max()
    end_month = max_date.month + months
    end_year = max_date.year + (end_month - 1) // 12
    end_month = (end_month - 1) % 12 + 1
    last_day = calendar.monthrange(end_year, end_month)[1]
    horizon_end = pd.Timestamp(year=end_year, month=end_month, day=last_day)
    future_dates = pd.date_range(start=max_date + step, end=horizon_end, freq=freq)

    future_cov = _make_covariates(future_dates, freq)
    all_cov = pd.concat([covariates, future_cov]).values.astype(np.float32)

    all_sales_norm = np.concatenate([
        (s.values - sales_mean) / sales_std,
        np.zeros(len(future_dates)),
    ])

    model.eval()
    values = []
    offset = len(s)

    with torch.no_grad():
        for i in range(len(future_dates)):
            idx = offset + i
            s_window = all_sales_norm[idx - lookback: idx]
            c_window = all_cov[idx - lookback: idx]
            x = np.column_stack([s_window, c_window])
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            pred_norm = model(xt)[0, 0].item()

            pred = pred_norm * sales_std + sales_mean
            pred = max(0.0, round(pred))
            all_sales_norm[idx] = (pred - sales_mean) / sales_std
            values.append(pred)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "values": values,
    }
