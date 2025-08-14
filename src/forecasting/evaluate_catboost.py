# src/forecasting/evaluate_catboost.py
import argparse, pathlib, time
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

def add_calendar(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    t = pd.to_datetime(df[ts_col])
    df["hour"] = t.dt.hour.astype("int16")
    df["dow"] = t.dt.dayofweek.astype("int8")
    df["month"] = t.dt.month.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    return df

def make_lags_and_rolls(
    df: pd.DataFrame,
    ts_col="timestamp_local",
    y_col="value",
    by="building_id",
    lags=(1,24,168),
    roll_windows=(24,168),
):
    df = df.sort_values([by, ts_col]).copy()
    for L in lags:
        df[f"lag_{L}"] = df.groupby(by, sort=False)[y_col].shift(L)
    for W in roll_windows:
        past = df.groupby(by, sort=False)[y_col].shift(1)
        df[f"roll_mean_{W}"] = past.groupby(df[by]).rolling(W).mean().reset_index(level=0, drop=True)
        df[f"roll_std_{W}"] = past.groupby(df[by]).rolling(W).std().reset_index(level=0, drop=True)
    return df

def time_based_split(df: pd.DataFrame, ts_col: str, valid_frac: float = 0.2):
    df = df.sort_values(ts_col)
    cut = df[ts_col].quantile(1 - valid_frac)
    tr = df[df[ts_col] <= cut]
    va = df[df[ts_col] > cut]
    return tr, va

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lags", default="1,24,168")
    ap.add_argument("--roll_windows", default="24,168")
    ap.add_argument("--valid_frac", type=float, default=0.2)
    ap.add_argument("--out_dir", default="/app/data/eval")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp_local"
    df = df[df["meter"]=="electricity"].copy()
    df["building_id"] = df["building_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = add_calendar(df, ts_col)
    lags = tuple(int(x) for x in args.lags.split(",") if x.strip())
    rolls = tuple(int(x) for x in args.roll_windows.split(",") if x.strip())
    df = make_lags_and_rolls(df, ts_col, "value", "building_id", lags, rolls)

    needed = ["value"] + [f"lag_{L}" for L in lags] + [f"roll_mean_{W}" for W in rolls] + [f"roll_std_{W}" for W in rolls]
    df = df.dropna(subset=needed)
    tr, va = time_based_split(df, ts_col, args.valid_frac)

    drop_cols = { "value", ts_col, "meter", "site_id", "timezone" }
    X_cols = [c for c in df.columns if c not in drop_cols]
    y_col = "value"
    cat_features = [c for c in ["building_id"] if c in X_cols]

    model = CatBoostRegressor()
    model.load_model(args.model)

    va_pred = model.predict(va[X_cols])
    va_out = va[[ts_col, "building_id", y_col]].copy()
    va_out["pred"] = va_pred
    va_out["error"] = va_out["pred"] - va_out[y_col]

    # overall metrics
    overall = {
        "rmse": rmse(va_out[y_col], va_out["pred"]),
        "mae": mean_absolute_error(va_out[y_col], va_out["pred"]),
        "mape": float((np.abs((va_out[y_col] - va_out["pred"]) / np.clip(va_out[y_col].replace(0,np.nan), 1e-9, None))).median())
    }

    # per-building metrics
    per_bldg = (va_out
                .groupby("building_id")
                .apply(lambda g: pd.Series({
                    "rmse": rmse(g[y_col], g["pred"]),
                    "mae": mean_absolute_error(g[y_col], g["pred"]),
                    "mape": float((np.abs((g[y_col] - g["pred"]) / np.clip(g[y_col].replace(0,np.nan), 1e-9, None))).median())
                }))
                .reset_index())

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    va_out.to_parquet(out_dir / "validation_predictions.parquet", index=False)
    per_bldg.to_csv(out_dir / "per_building_metrics.csv", index=False)

    print("[EVAL] Overall:", overall)
    print(f"[EVAL] Saved validation predictions → {out_dir/'validation_predictions.parquet'}")
    print(f"[EVAL] Saved per-building metrics → {out_dir/'per_building_metrics.csv'}")

if __name__ == "__main__":
    main()
