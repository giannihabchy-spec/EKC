import calendar
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from ml.modeling import _split, _add_holiday_features
from ml.config import tft_tuning
from ml.forecasting.nhits import _make_covariates, WindowDataset, MultiWindowDataset, _train


# ── TFT Components ─────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        h = torch.nn.functional.elu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h) * torch.sigmoid(self.gate(h))
        return self.layer_norm(out + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, n_vars, hidden_dim, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.per_var_dim = input_dim // n_vars
        self.grns = nn.ModuleList([
            GatedResidualNetwork(self.per_var_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(n_vars)
        ])
        self.softmax_grn = GatedResidualNetwork(input_dim, hidden_dim, n_vars, dropout)
        self.output_dim = hidden_dim

    def forward(self, x):
        # x: (batch, seq, input_dim)
        weights = torch.softmax(self.softmax_grn(x), dim=-1)  # (batch, seq, n_vars)
        var_outputs = []
        for i, grn in enumerate(self.grns):
            start = i * self.per_var_dim
            end = start + self.per_var_dim
            var_outputs.append(grn(x[..., start:end]))
        var_outputs = torch.stack(var_outputs, dim=-1)  # (batch, seq, hidden_dim, n_vars)
        selected = (var_outputs * weights.unsqueeze(-2)).sum(dim=-1)  # (batch, seq, hidden_dim)
        return selected


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, self.head_dim)
        self.out_proj = nn.Linear(self.head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        batch = q.size(0)
        seq_len = q.size(1)

        Q = self.q_proj(q).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v)  # (batch, seq, head_dim) — shared across heads

        scale = self.head_dim ** 0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # interpretable: average attention across heads, apply to shared V
        attn_avg = attn.mean(dim=1)  # (batch, seq, seq)
        out = torch.matmul(attn_avg, V)  # (batch, seq, head_dim)
        return self.out_proj(out)


# ── TFT Model ──────────────────────────────────────────────────────────

class TFT(nn.Module):
    def __init__(self, input_dim, lookback, hidden_size=128, n_heads=4,
                 dropout=0.1, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.lookback = lookback

        self.input_proj = nn.Linear(input_dim, hidden_size)

        n_vars = input_dim
        self.vsn = VariableSelectionNetwork(
            hidden_size, n_vars=min(n_vars, hidden_size), hidden_dim=hidden_size, dropout=dropout,
        )

        self.lstm_encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.post_lstm_gate = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.post_lstm_norm = nn.LayerNorm(hidden_size)

        self.attention = InterpretableMultiHeadAttention(hidden_size, n_heads, dropout)
        self.post_attn_gate = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.post_attn_norm = nn.LayerNorm(hidden_size)

        self.output_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.output_fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x: (batch, lookback, input_dim)
        h = self.input_proj(x)  # (batch, lookback, hidden_size)

        vsn_out = self.vsn(h)

        lstm_out, _ = self.lstm_encoder(vsn_out)

        gated = self.post_lstm_gate(lstm_out)
        temporal = self.post_lstm_norm(gated + vsn_out)

        attn_out = self.attention(temporal, temporal, temporal)
        attn_gated = self.post_attn_gate(attn_out)
        enriched = self.post_attn_norm(attn_gated + temporal)

        last = enriched[:, -1, :]  # (batch, hidden_size)
        out = self.output_grn(last)
        return self.output_fc(out)  # (batch, output_dim)


# ── Single-category fit ────────────────────────────────────────────────

def fit_tft(s: pd.Series, freq: str = "D") -> dict:
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

    for lookback in tft_tuning["lookback"]:
        if lookback >= n_train:
            continue

        for hidden_size in tft_tuning["hidden_size"]:
            for n_heads in tft_tuning["n_heads"]:
                for lr in tft_tuning["lr"]:

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

                    model = TFT(input_dim, lookback, hidden_size, n_heads)
                    model = _train(model, train_loader, val_loader, lr)

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
                            "n_heads": n_heads,
                            "lr": lr,
                        }
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_params is None:
        raise ValueError("No valid configuration found")

    val_wape = round(best_wape * 100, 2)
    lookback = best_params["lookback"]

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

    model = TFT(input_dim, lookback, best_params["hidden_size"], best_params["n_heads"])
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

    full_ds = WindowDataset(full_sales_norm, cov_values, lookback)
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=True)
    model = _train(model, full_loader, full_loader, best_params["lr"], epochs=50, patience=5)

    forecast = _recursive_forecast_tft(
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


def _recursive_forecast_tft(model, s, covariates, lookback,
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


# ── Multi-category fit ─────────────────────────────────────────────────

def fit_single_tft(series: dict, freq: str = "D") -> dict:
    categories = sorted(series.keys())
    n_cat = len(categories)

    ref = series[categories[0]]
    index = ref.index

    sales_df = pd.DataFrame({cat: series[cat] for cat in categories}, index=index).fillna(0)

    n = len(index)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_train_val = n_train + n_val

    train_sales = sales_df.iloc[:n_train]
    sales_mean = train_sales.mean().values
    sales_std = train_sales.std().values + 1e-8

    sales_norm = (sales_df.values - sales_mean) / sales_std

    covariates = _make_covariates(index, freq)
    cov_values = covariates.values.astype(np.float32)
    input_dim = n_cat + cov_values.shape[1]

    best_wape = float("inf")
    best_params = None
    best_state = None

    for lookback in tft_tuning["lookback"]:
        if lookback >= n_train:
            continue

        for hidden_size in tft_tuning["hidden_size"]:
            for n_heads in tft_tuning["n_heads"]:
                for lr in tft_tuning["lr"]:

                    train_ds = MultiWindowDataset(
                        sales_norm[:n_train],
                        cov_values[:n_train],
                        lookback,
                    )
                    if len(train_ds) == 0:
                        continue

                    val_ds = MultiWindowDataset(
                        sales_norm[n_train - lookback: n_train_val],
                        cov_values[n_train - lookback: n_train_val],
                        lookback,
                    )
                    if len(val_ds) == 0:
                        continue

                    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=64)

                    model = TFT(input_dim, lookback, hidden_size, n_heads, output_dim=n_cat)
                    model = _train(model, train_loader, val_loader, lr)

                    model.eval()
                    val_preds = []
                    with torch.no_grad():
                        for xb, _ in val_loader:
                            val_preds.append(model(xb))
                    val_preds_norm = torch.cat(val_preds).numpy()
                    val_preds_orig = val_preds_norm * sales_std + sales_mean

                    y_val = sales_df.values[n_train: n_train_val]
                    denom = np.sum(np.abs(y_val))
                    if denom == 0:
                        continue

                    wape = np.sum(np.abs(y_val[:len(val_preds_orig)] - val_preds_orig)) / denom

                    if wape < best_wape:
                        best_wape = wape
                        best_params = {
                            "lookback": lookback,
                            "hidden_size": hidden_size,
                            "n_heads": n_heads,
                            "lr": lr,
                        }
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_params is None:
        raise ValueError("No valid configuration found")

    val_wape = round(best_wape * 100, 2)
    lookback = best_params["lookback"]

    train_val_ds = MultiWindowDataset(
        sales_norm[:n_train_val], cov_values[:n_train_val], lookback,
    )
    test_ds = MultiWindowDataset(
        sales_norm[n_train_val - lookback:], cov_values[n_train_val - lookback:], lookback,
    )

    model = TFT(input_dim, lookback, best_params["hidden_size"], best_params["n_heads"], output_dim=n_cat)
    model.load_state_dict(best_state)
    train_val_loader = DataLoader(train_val_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)
    model = _train(model, train_val_loader, test_loader, best_params["lr"])

    model.eval()
    test_preds_list = []
    with torch.no_grad():
        for xb, _ in test_loader:
            test_preds_list.append(model(xb))
    test_preds_norm = torch.cat(test_preds_list).numpy()
    test_preds_orig = test_preds_norm * sales_std + sales_mean

    test_dates = index[n_train_val: n_train_val + len(test_preds_orig)]

    full_ds = MultiWindowDataset(sales_norm, cov_values, lookback)
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=True)
    model = _train(model, full_loader, full_loader, best_params["lr"], epochs=50, patience=5)

    # Recursive forecast
    step = pd.Timedelta(days=1) if freq == "D" else pd.Timedelta(weeks=1)
    max_date = index.max()
    months = 8
    end_month = max_date.month + months
    end_year = max_date.year + (end_month - 1) // 12
    end_month = (end_month - 1) % 12 + 1
    last_day = calendar.monthrange(end_year, end_month)[1]
    horizon_end = pd.Timestamp(year=end_year, month=end_month, day=last_day)
    future_dates = pd.date_range(start=max_date + step, end=horizon_end, freq=freq)

    future_cov = _make_covariates(future_dates, freq)
    all_cov = np.vstack([cov_values, future_cov.values.astype(np.float32)])
    all_sales_norm = np.vstack([sales_norm, np.zeros((len(future_dates), n_cat))])

    model.eval()
    forecast_values = {cat: [] for cat in categories}
    offset = len(index)

    with torch.no_grad():
        for i in range(len(future_dates)):
            idx = offset + i
            s_window = all_sales_norm[idx - lookback: idx]
            c_window = all_cov[idx - lookback: idx]
            x = np.hstack([s_window, c_window])
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            pred_norm = model(xt)[0].numpy()
            pred_orig = pred_norm * sales_std + sales_mean
            pred_orig = np.maximum(pred_orig, 0).round()
            all_sales_norm[idx] = (pred_orig - sales_mean) / sales_std
            for j, cat in enumerate(categories):
                forecast_values[cat].append(float(pred_orig[j]))

    # Build per-category results
    output = {}
    y_test_all = sales_df.values[n_train_val: n_train_val + len(test_preds_orig)]

    for j, cat in enumerate(categories):
        cat_test_preds = test_preds_orig[:, j]
        cat_y_test = y_test_all[:, j]

        cat_mae = round(float(mean_absolute_error(cat_y_test, cat_test_preds)), 2)
        cat_rmse = round(float(np.sqrt(np.mean((cat_y_test - cat_test_preds) ** 2))), 2)
        denom = np.sum(np.abs(cat_y_test))
        cat_wape = round(float(np.sum(np.abs(cat_y_test - cat_test_preds)) / denom * 100), 2) if denom > 0 else 0.0

        output[cat] = {
            "best_params": best_params,
            "metrics": {
                "val_wape": val_wape,
                "final_mae": cat_mae,
                "final_rmse": cat_rmse,
                "final_wape": cat_wape,
            },
            "norm_params": {"mean": float(sales_mean[j]), "std": float(sales_std[j])},
            "test_pred": {
                "dates": [d.strftime("%Y-%m-%d") for d in test_dates],
                "values": [round(float(v), 2) for v in cat_test_preds],
            },
            "forecast": {
                "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
                "values": forecast_values[cat],
            },
            "from": index.min().strftime("%Y-%m-%d"),
            "to": index.max().strftime("%Y-%m-%d"),
        }

    return output
