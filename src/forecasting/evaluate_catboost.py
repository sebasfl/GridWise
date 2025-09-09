# src/forecasting/evaluate_catboost.py
import argparse, pathlib, time
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

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
    """Root Mean Square Error with input validation."""
    return mean_squared_error(y_true, y_pred, squared=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lags", default="1,24,168")
    ap.add_argument("--roll_windows", default="24,168")
    ap.add_argument("--valid_frac", type=float, default=0.2)
    ap.add_argument("--out_dir", default="/app/data/eval")
    ap.add_argument("--use_gpu", action="store_true", help="Usar GPU para inferencia")
    ap.add_argument("--use_existing_features", action="store_true", help="Usar features existentes en dataset limpio")
    args = ap.parse_args()
    
    print(f"🔍 Starting model evaluation {'with GPU' if args.use_gpu else 'on CPU'}...")
    start_time = time.time()

    print("📥 Loading data...")
    df = pd.read_parquet(args.parquet)
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp_local"
    
    # Filter electricity data if meter column exists
    if "meter" in df.columns:
        df = df[df["meter"]=="electricity"].copy()
    
    df["building_id"] = df["building_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    
    # Check if features already exist (from cleaned dataset)
    lags = tuple(int(x) for x in args.lags.split(",") if x.strip())
    rolls = tuple(int(x) for x in args.roll_windows.split(",") if x.strip())
    
    lag_cols = [f"lag_{L}" for L in lags]
    roll_cols = [f"roll_mean_{W}" for W in rolls] + [f"roll_std_{W}" for W in rolls]
    
    if args.use_existing_features or all(col in df.columns for col in lag_cols + roll_cols):
        print("✅ Using existing features from cleaned dataset")
        # Just add calendar features if missing
        if "hour" not in df.columns:
            df = add_calendar(df, ts_col)
    else:
        print("🔄 Creating lag and rolling features...")
        df = add_calendar(df, ts_col)
        df = make_lags_and_rolls(df, ts_col, "value", "building_id", lags, rolls)
    
    needed = ["value"] + lag_cols + roll_cols
    print(f"🧹 Filtering complete records (need {len(needed)} features)...")
    df = df.dropna(subset=needed)
    print(f"📊 Dataset size after filtering: {len(df):,} records, {df['building_id'].nunique():,} buildings")
    tr, va = time_based_split(df, ts_col, args.valid_frac)

    drop_cols = { "value", ts_col, "meter", "site_id", "timezone" }
    X_cols = [c for c in df.columns if c not in drop_cols]
    y_col = "value"
    cat_features = [c for c in ["building_id"] if c in X_cols]

    print("🤖 Loading model...")
    model = CatBoostRegressor(task_type="GPU", devices="0") if args.use_gpu else CatBoostRegressor()
    model.load_model(args.model)

    print(f"🔮 Generating predictions on {len(va):,} validation samples...")
    if args.use_gpu:
        # For GPU, use Pool for better memory management
        va_pool = Pool(va[X_cols], cat_features=cat_features)
        va_pred = model.predict(va_pool)
    else:
        va_pred = model.predict(va[X_cols])
    
    va_out = va[[ts_col, "building_id", y_col]].copy()
    va_out["pred"] = va_pred
    va_out["error"] = va_out["pred"] - va_out[y_col]
    va_out["abs_error"] = np.abs(va_out["error"])
    va_out["pct_error"] = (va_out["error"] / np.clip(va_out[y_col], 1e-9, None)) * 100  # Avoid division by zero

    print("📈 Computing evaluation metrics...")
    # Overall metrics - using more robust calculations
    y_true = va_out[y_col].values
    y_pred = va_out["pred"].values
    
    # Remove any infinite or NaN values for metric calculation
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    overall = {
        "rmse": rmse(y_true_clean, y_pred_clean),
        "mae": mean_absolute_error(y_true_clean, y_pred_clean),
        "r2": r2_score(y_true_clean, y_pred_clean),
        "mape": float(np.median(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100),  # Median APE as %
        "mean_actual": float(np.mean(y_true_clean)),
        "mean_predicted": float(np.mean(y_pred_clean)),
        "total_samples": len(y_true_clean),
        "buildings_evaluated": va_out["building_id"].nunique()
    }

    print("🏢 Computing per-building metrics...")
    # Per-building metrics - optimized calculation
    def compute_building_metrics(group):
        y_true = group[y_col].values
        y_pred = group["pred"].values
        
        # Filter valid values
        mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
        if mask.sum() < 5:  # Need at least 5 valid samples
            return pd.Series({
                "rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape": np.nan,
                "mean_actual": np.nan, "mean_predicted": np.nan, "sample_count": mask.sum()
            })
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        return pd.Series({
            "rmse": rmse(y_true_clean, y_pred_clean),
            "mae": mean_absolute_error(y_true_clean, y_pred_clean),
            "r2": r2_score(y_true_clean, y_pred_clean),
            "mape": float(np.median(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100),
            "mean_actual": float(np.mean(y_true_clean)),
            "mean_predicted": float(np.mean(y_pred_clean)),
            "sample_count": len(y_true_clean)
        })
    
    # Use parallel processing for per-building metrics if many buildings
    if va_out["building_id"].nunique() > 100:
        print("🚀 Using parallel processing for per-building metrics...")
        building_groups = list(va_out.groupby("building_id"))
        
        def process_group(name_group):
            name, group = name_group
            metrics = compute_building_metrics(group)
            metrics.name = name
            return metrics
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            per_bldg_list = list(executor.map(process_group, building_groups))
        
        per_bldg = pd.DataFrame(per_bldg_list).reset_index().rename(columns={"index": "building_id"})
    else:
        per_bldg = va_out.groupby("building_id").apply(compute_building_metrics).reset_index()

    print("💾 Saving evaluation results...")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed predictions
    va_out.to_parquet(out_dir / "validation_predictions.parquet", index=False)
    
    # Save per-building metrics
    per_bldg.to_csv(out_dir / "per_building_metrics.csv", index=False)
    
    # Save overall metrics as JSON for easy parsing
    import json
    with open(out_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    
    # Performance summary
    evaluation_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("🏆 EVALUATION RESULTS")
    print("="*60)
    print(f"📊 Dataset: {len(va_out):,} samples, {overall['buildings_evaluated']} buildings")
    print(f"⏱️  Evaluation time: {evaluation_time:.2f} seconds")
    print(f"🎯 RMSE: {overall['rmse']:.4f}")
    print(f"📏 MAE: {overall['mae']:.4f}")
    print(f"🎖️  R²: {overall['r2']:.4f}")
    print(f"📈 MAPE: {overall['mape']:.2f}%")
    print(f"📊 Mean Actual: {overall['mean_actual']:.2f}")
    print(f"📊 Mean Predicted: {overall['mean_predicted']:.2f}")
    print("\n📂 Files saved:")
    print(f"   • Predictions: {out_dir/'validation_predictions.parquet'}")
    print(f"   • Building metrics: {out_dir/'per_building_metrics.csv'}")
    print(f"   • Overall metrics: {out_dir/'overall_metrics.json'}")
    print("\n🔍 Building Performance Summary:")
    
    # Building performance summary
    valid_buildings = per_bldg.dropna(subset=["rmse"])
    if len(valid_buildings) > 0:
        print(f"   • Best RMSE: {valid_buildings['rmse'].min():.4f}")
        print(f"   • Worst RMSE: {valid_buildings['rmse'].max():.4f}")
        print(f"   • Median RMSE: {valid_buildings['rmse'].median():.4f}")
        print(f"   • Buildings with R² > 0.5: {(valid_buildings['r2'] > 0.5).sum()}/{len(valid_buildings)}")
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()
