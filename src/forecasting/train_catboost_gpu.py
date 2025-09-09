# src/forecasting/train_catboost_gpu.py
import argparse, pathlib
import pandas as pd
from catboost import CatBoostRegressor, Pool
import time

def parse_int_list(s: str, default):
    if s is None:
        return tuple(default)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def add_calendar(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    t = pd.to_datetime(df[ts_col])
    df["hour"] = t.dt.hour.astype("int16")
    df["dow"] = t.dt.dayofweek.astype("int8")
    df["month"] = t.dt.month.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    return df


def make_lags_and_rolls(
    df: pd.DataFrame,
    ts_col: str = "timestamp_local",
    y_col: str = "value",
    by: str = "building_id",
    lags=(1, 24, 168),
    roll_windows=(24, 168),
) -> pd.DataFrame:
    """
    Create per-building lag features and rolling stats (no leakage).
    Assumes hourly granularity (as in BDG2 electricity).
    """
    df = df.sort_values([by, ts_col]).copy()
    # Lags
    for L in lags:
        df[f"lag_{L}"] = df.groupby(by, sort=False)[y_col].shift(L)
    # Rolling means/stds using only past data (shift before rolling)
    for W in roll_windows:
        past = df.groupby(by, sort=False)[y_col].shift(1)
        df[f"roll_mean_{W}"] = past.groupby(df[by]).rolling(W).mean().reset_index(level=0, drop=True)
        df[f"roll_std_{W}"] = past.groupby(df[by]).rolling(W).std().reset_index(level=0, drop=True)
    return df


def time_based_split(df: pd.DataFrame, ts_col: str, valid_frac: float = 0.2):
    """Split by timestamp quantile to avoid leakage."""
    df = df.sort_values(ts_col)
    cut = df[ts_col].quantile(1 - valid_frac)
    tr = df[df[ts_col] <= cut]
    va = df[df[ts_col] > cut]
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/app/data/processed/bdg2_electricity_cleaned.parquet")
    ap.add_argument("--building_id", default=None, help="If set, train on a single building. Omit for global model.")
    ap.add_argument("--date_from", default=None, help="Optional ISO date filter start (e.g., 2016-01-01)")
    ap.add_argument("--date_to", default=None, help="Optional ISO date filter end (e.g., 2018-12-31)")
    ap.add_argument("--lags", default="1,24,168", help="Comma-separated lags, default 1,24,168")
    ap.add_argument("--roll_windows", default="24,168", help="Comma-separated rolling windows, default 24,168")
    ap.add_argument("--valid_frac", type=float, default=0.2, help="Validation fraction by time (default 0.2)")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--used_ram_limit", default="12gb")
    ap.add_argument("--model_out", default="/app/models/catboost_gpu.cbm")
    args = ap.parse_args()

    # Load
    df = pd.read_parquet(args.parquet)

    # Basic schema handling
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp_local"
    if ts_col not in df.columns:
        raise SystemExit(f"Timestamp column not found. Expected 'timestamp_local' or 'timestamp_utc'. Got: {df.columns.tolist()}")

    # Filter meter and optional building/date
    df = df[df["meter"] == "electricity"].copy()
    if args.building_id:
        df = df[df["building_id"].astype(str) == str(args.building_id)].copy()

    # Date filters (optional)
    if args.date_from:
        df = df[pd.to_datetime(df[ts_col]) >= pd.to_datetime(args.date_from)]
    if args.date_to:
        df = df[pd.to_datetime(df[ts_col]) <= pd.to_datetime(args.date_to)]

    # Ensure types
    df["building_id"] = df["building_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Feature engineering
    df = add_calendar(df, ts_col=ts_col)
    lags = parse_int_list(args.lags, default=(1, 24, 168))
    rolls = parse_int_list(args.roll_windows, default=(24, 168))
    df = make_lags_and_rolls(df, ts_col=ts_col, y_col="value", by="building_id", lags=lags, roll_windows=rolls)

    # Drop rows with missing target or features
    needed = ["value"] + [f"lag_{L}" for L in lags] + [f"roll_mean_{W}" for W in rolls] + [f"roll_std_{W}" for W in rolls]
    df = df.dropna(subset=needed)

    if df.empty:
        raise SystemExit("No data after filtering/feature engineering. Check building_id/date filters and data availability.")

    # Time-based split
    train_df, valid_df = time_based_split(df, ts_col=ts_col, valid_frac=args.valid_frac)

    # Build feature matrix
    drop_cols = { "value", ts_col, "meter", "site_id", "timezone" }  # drop if present
    X_cols = [c for c in df.columns if c not in drop_cols]
    y_col = "value"

    # Pools (mark building_id as categorical)
    cat_features = [c for c in ["building_id"] if c in X_cols]
    train_pool = Pool(train_df[X_cols], train_df[y_col], cat_features=cat_features)
    valid_pool = Pool(valid_df[X_cols], valid_df[y_col], cat_features=cat_features)
    train_dir = f"/app/models/cbtrain_{int(time.time())}"

    model = CatBoostRegressor(
    task_type="GPU",
    devices="0",
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=args.iterations,
    depth=min(args.depth, 6),       # ↓ menos profundidad = menos VRAM
    learning_rate=args.learning_rate,
    random_seed=42,
    od_type="Iter",
    od_wait=200,

    bootstrap_type="Bernoulli",     # permite subsample en GPU
    subsample=0.8,                  # muestreo por filas
    max_ctr_complexity=1,           # ↓ sin combinaciones de CTR (menos VRAM)
    # ctr_leaf_count_limit=8,       # ❌ QUITAR: causa el error de "change of option"
    leaf_estimation_iterations=1,   # ↓ coste por árbol
    gpu_ram_part=0.85,              # respeta VRAM
    gpu_cat_features_storage="CpuPinnedMemory",  # ✅ mueve parte del manejo de categóricas a RAM de CPU
    max_bin=128,                        # ✅ menos bins → menos memoria

    used_ram_limit=args.used_ram_limit,
    train_dir=train_dir,            # ya lo tienes único por run
    save_snapshot=False,            # no reanudar
    verbose=200,
    )



    model.fit(train_pool, eval_set=valid_pool)

    # Save
    out_path = pathlib.Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path))
    print(f"[OK] Saved model to {out_path}")

    # Optional: print best scores if available
    try:
        print("Best scores:", model.get_best_score())
    except Exception:
        pass


if __name__ == "__main__":
    main()
