# src/forecasting/forecast_catboost.py
import argparse, pathlib
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool

def add_calendar(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    t = pd.to_datetime(df[ts_col])
    df["hour"] = t.dt.hour.astype("int16")
    df["dow"] = t.dt.dayofweek.astype("int8")
    df["month"] = t.dt.month.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    return df

def step_features(state, lags, rolls):
    feats = {}
    series = state["series"]
    for L in lags:
        feats[f"lag_{L}"] = series[-L] if len(series) >= L else np.nan
    arr = np.array(series[:-1], dtype=float)
    for W in rolls:
        if len(arr) >= W:
            feats[f"roll_mean_{W}"] = float(arr[-W:].mean())
            feats[f"roll_std_{W}"]  = float(arr[-W:].std(ddof=1)) if W > 1 else 0.0
        else:
            feats[f"roll_mean_{W}"] = np.nan
            feats[f"roll_std_{W}"]  = np.nan
    return feats

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

    BASE_COLS = ["building_id", "hour", "dow", "month", "is_weekend"]
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

    model = CatBoostRegressor(task_type="GPU", devices="0") if args.use_gpu else CatBoostRegressor()
    model.load_model(args.model)

    forecasts = []
    keep = max([1] + list(lags) + list(rolls))  # mínimo de historia que necesito

    for bld, hist in df.groupby("building_id", sort=False):
        hist_vals = hist["value"].tolist()
        if len(hist_vals) < keep + 1:
            continue

        state = {"series": hist_vals.copy()}
        fut_df = pd.DataFrame({ts_col: future_index})
        fut_df = add_calendar(fut_df, ts_col)

        preds = []
        for i in range(len(future_index)):
            feat = {"building_id": bld, **fut_df.iloc[i].to_dict()}
            feat.update(step_features(state, lags, rolls))
            X = pd.DataFrame([feat]).reindex(columns=FEATURES)
            pool = Pool(X, cat_features=CAT_COLS)   # consistente con el entrenamiento
            yhat = float(model.predict(pool)[0])
            preds.append(yhat)
            state["series"].append(yhat)            # recursivo

        forecasts.append(pd.DataFrame({
            "building_id": bld,
            ts_col: future_index,
            "pred": preds
        }))

    if not forecasts:
        raise SystemExit("No forecasts were produced (insufficient history?).")

    fcst = pd.concat(forecasts, ignore_index=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fcst.to_parquet(args.out, index=False)
    print(f"[FORECAST] Saved → {args.out}")

if __name__ == "__main__":
    main()
