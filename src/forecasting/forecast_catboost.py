# src/forecasting/forecast_catboost.py
import argparse, pathlib
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def add_calendar(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    t = pd.to_datetime(df[ts_col])
    df["hour"] = t.dt.hour.astype("int16")
    df["day_of_week"] = t.dt.dayofweek.astype("int8")
    df["month"] = t.dt.month.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_working_hours"] = (
        (df["hour"] >= 8) & (df["hour"] <= 18) & (df["is_weekend"] == 0)
    ).astype("int8")
    df["quarter"] = t.dt.quarter.astype("int8")
    df["day_of_year"] = t.dt.dayofyear.astype("int16")
    return df

def step_features_batch(series_history, lags, rolls, horizon):
    """Generate features for entire forecast horizon at once."""
    features_list = []
    current_series = series_history.copy()
    
    for step in range(horizon):
        feats = {}
        # Lag features
        for L in lags:
            feats[f"lag_{L}"] = current_series[-L] if len(current_series) >= L else np.nan
        
        # Rolling features (excluding current value)
        arr = np.array(current_series[:-1], dtype=float) if len(current_series) > 1 else np.array([], dtype=float)
        for W in rolls:
            if len(arr) >= W:
                feats[f"roll_mean_{W}"] = float(arr[-W:].mean())
                feats[f"roll_std_{W}"] = float(arr[-W:].std(ddof=1)) if W > 1 else 0.0
            else:
                feats[f"roll_mean_{W}"] = np.nan
                feats[f"roll_std_{W}"] = np.nan
        
        features_list.append(feats)
        # Add placeholder for next prediction (will be updated with actual prediction)
        current_series.append(0.0)  # Placeholder
    
    return features_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--horizon", type=int, default=168)
    ap.add_argument("--freq", default="h")  # usar 'h' para evitar el FutureWarning
    ap.add_argument("--lags", default="1,24,168")
    ap.add_argument("--roll_windows", default="24,168")
    ap.add_argument("--building_id", default=None)
    ap.add_argument("--out", default="/app/data/forecasts/cb_forecast.parquet")
    ap.add_argument("--use_gpu", action="store_true", help="Usar GPU para inferencia (opcional)")
    ap.add_argument("--batch_size", type=int, default=24, help="Tamaño de batch para predicciones (default: 24)")
    ap.add_argument("--max_workers", type=int, default=4, help="Número máximo de workers paralelos (default: 4)")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp_local"
    df = df[df["meter"] == "electricity"].copy()
    df["building_id"] = df["building_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if args.building_id:
        df = df[df["building_id"] == str(args.building_id)].copy()

    lags  = tuple(int(x) for x in args.lags.split(",") if x.strip())
    rolls = tuple(int(x) for x in args.roll_windows.split(",") if x.strip())

    BASE_COLS = ["building_id", "hour", "day_of_week", "month", "is_weekend", "is_working_hours", "quarter", "day_of_year"]
    LAG_COLS  = [f"lag_{L}" for L in lags]
    ROLL_COLS = sum(([f"roll_mean_{W}", f"roll_std_{W}"] for W in rolls), [])
    FEATURES  = BASE_COLS + LAG_COLS + ROLL_COLS
    CAT_COLS  = ["building_id"]

    df = df.sort_values(["building_id", ts_col])
    last_ts = df[ts_col].max()
    freq = args.freq.lower()
    future_index = pd.date_range(
        last_ts + pd.tseries.frequencies.to_offset(freq),
        periods=args.horizon, freq=freq
    )

    print(f"Loading model {'with GPU acceleration' if args.use_gpu else 'on CPU'}...")
    start_time = time.time()
    model = CatBoostRegressor(task_type="GPU", devices="0") if args.use_gpu else CatBoostRegressor()
    model.load_model(args.model)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    def forecast_building(bld_hist_tuple):
        """Forecast single building with batch optimization."""
        bld, hist = bld_hist_tuple
        hist_vals = hist["value"].tolist()
        if len(hist_vals) < keep + 1:
            return None
        
        # Pre-generate calendar features for entire horizon
        fut_df = pd.DataFrame({ts_col: future_index})
        fut_df = add_calendar(fut_df, ts_col)
        
        # Initialize prediction list and working series
        preds = []
        current_series = hist_vals.copy()
        
        # Sequential prediction with optimized feature computation
        for i in range(len(future_index)):
            # Calendar features
            feat = {"building_id": bld, **fut_df.iloc[i].to_dict()}
            
            # Time series features
            ts_feats = {}
            for L in lags:
                ts_feats[f"lag_{L}"] = current_series[-L] if len(current_series) >= L else np.nan
            
            # Rolling features (use only historical data, not current point)
            arr = np.array(current_series, dtype=float)
            for W in rolls:
                if len(arr) >= W:
                    ts_feats[f"roll_mean_{W}"] = float(arr[-W:].mean())
                    ts_feats[f"roll_std_{W}"] = float(arr[-W:].std(ddof=1)) if W > 1 else 0.0
                else:
                    ts_feats[f"roll_mean_{W}"] = np.nan
                    ts_feats[f"roll_std_{W}"] = np.nan
            
            feat.update(ts_feats)
            
            # Single prediction - much faster than creating DataFrame each time
            X = pd.DataFrame([feat], columns=FEATURES)
            yhat = float(model.predict(X)[0])  # Direct prediction without Pool for single row
            preds.append(yhat)
            current_series.append(yhat)
        
        return pd.DataFrame({
            "building_id": bld,
            ts_col: future_index,
            "pred": preds
        })
    
    forecasts = []
    keep = max([1] + list(lags) + list(rolls))  # mínimo de historia que necesito
    
    # Process buildings in parallel
    building_groups = list(df.groupby("building_id", sort=False))
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_building = {executor.submit(forecast_building, bg): bg[0] for bg in building_groups}
        
        for future in as_completed(future_to_building):
            bld_id = future_to_building[future]
            try:
                result = future.result()
                if result is not None:
                    forecasts.append(result)
            except Exception as exc:
                print(f"Building {bld_id} generated an exception: {exc}")

    if not forecasts:
        raise SystemExit("No forecasts were produced (insufficient history?).")

    print(f"\nConcatenating {len(forecasts)} building forecasts...")
    fcst = pd.concat(forecasts, ignore_index=True)
    
    print(f"Saving forecasts to {args.out}...")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fcst.to_parquet(args.out, index=False)
    
    total_time = time.time() - start_time
    print(f"\n[FORECAST] ✅ Complete!")
    print(f"📊 Generated forecasts for {fcst['building_id'].nunique()} buildings")
    print(f"🔮 Total predictions: {len(fcst):,}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"🚀 Avg time per building: {total_time/fcst['building_id'].nunique():.3f} seconds")
    print(f"💾 Saved → {args.out}")

if __name__ == "__main__":
    main()
