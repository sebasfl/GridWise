# dash/pages/02_forecast_viewer.py
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq

RAW_DEFAULT  = "/app/data/processed/bdg2_electricity_long.parquet"
FCST_DEFAULT = "/app/data/forecasts/cb_forecast.parquet"

st.set_page_config(page_title="Pronóstico vs histórico", layout="wide")
st.title("Pronóstico vs histórico")

# -------------------- helpers --------------------
def parquet_columns(path: str) -> set:
    """Return column names from a Parquet file without loading data."""
    return set(pq.ParquetFile(path).schema.names)

def pick_ts_from_names(names: set) -> str | None:
    """Pick a sensible timestamp column name from a set of names."""
    for c in ("timestamp_local", "timestamp_utc", "ts", "datetime"):
        if c in names:
            return c
    return None

def resample(df: pd.DataFrame, ts: str, col: str, freq: str, how: str) -> pd.Series:
    s = df.set_index(ts)[col]
    if how == "mean":
        return s.resample(freq).mean()
    if how == "sum":
        return s.resample(freq).sum()
    if how == "median":
        return s.resample(freq).median()
    return s

# -------------------- inputs --------------------
raw_path  = st.text_input("Parquet histórico", RAW_DEFAULT)
fcst_path = st.text_input("Parquet de pronósticos", FCST_DEFAULT)
freq = st.selectbox("Frecuencia de visualización", ["1H", "1D"], index=1)
agg  = st.selectbox("Agregación (histórico)", ["mean", "sum", "median"], index=0)
hist_days = st.slider("Días de histórico a mostrar", 7, 365*2, 180)

# -------------------- forecasts --------------------
if not Path(fcst_path).exists():
    st.warning(f"No encuentro {fcst_path}. Ejecuta el script de pronóstico primero.")
    st.stop()

fc_cols_avail = parquet_columns(fcst_path)
ts_fc = pick_ts_from_names(fc_cols_avail)
need_fc_cols = [c for c in (ts_fc, "building_id", "pred") if c]
missing_fc = [c for c in ("building_id", "pred") if c not in fc_cols_avail]
if ts_fc is None or missing_fc:
    st.error("El parquet de pronóstico debe tener columnas 'building_id', 'pred' y una de "
             "'timestamp_local'/'timestamp_utc'/'ts'/'datetime'.")
    st.stop()

# Leemos sólo las columnas necesarias
fc = pd.read_parquet(fcst_path, columns=need_fc_cols)
fc["building_id"] = fc["building_id"].astype(str)
blds = sorted(fc["building_id"].unique().tolist())
bld  = st.selectbox("Edificio", blds)

# Asegurar dtype datetime (por si viene como string)
fc[ts_fc] = pd.to_datetime(fc[ts_fc], errors="coerce")

# -------------------- historical --------------------
if not Path(raw_path).exists():
    st.error(f"No encuentro histórico {raw_path}")
    st.stop()

raw_cols_avail = parquet_columns(raw_path)

# Intentamos usar el mismo ts que en pronóstico; si no está, buscamos el alterno
ts_raw = ts_fc if ts_fc in raw_cols_avail else pick_ts_from_names(raw_cols_avail)
if ts_raw is None:
    st.error("No encuentro una columna de tiempo en el histórico "
             "(esperaba 'timestamp_local' o 'timestamp_utc').")
    st.stop()

need_raw_cols = [c for c in (ts_raw, "building_id", "value", "meter") if c in raw_cols_avail]
raw = pd.read_parquet(raw_path, columns=need_raw_cols)

# Filtramos electricidad y el edificio seleccionado
if "meter" in raw.columns:
    raw = raw[raw["meter"] == "electricity"]
raw["building_id"] = raw["building_id"].astype(str)
raw = raw[raw["building_id"] == bld].copy()

# Asegurar datetime
raw[ts_raw] = pd.to_datetime(raw[ts_raw], errors="coerce")
raw = raw.dropna(subset=[ts_raw, "value"]).sort_values(ts_raw)

# Recorte del histórico
if not raw.empty:
    cut = raw[ts_raw].max() - pd.Timedelta(days=hist_days)
    raw = raw[raw[ts_raw] >= cut]

# -------------------- resample & plot --------------------
hist = resample(raw, ts_raw, "value", freq, agg) if not raw.empty else pd.Series(dtype=float)
fut  = resample(fc[fc["building_id"] == bld], ts_fc, "pred", freq, "mean")

fig, ax = plt.subplots(figsize=(14, 5))
if not hist.empty:
    hist.plot(ax=ax, label="Histórico")
fut.plot(ax=ax,  label="Pronóstico", linestyle="--")

if not hist.empty:
    ax.axvline(hist.index.max(), ls=":", alpha=0.7)

ax.set_title(f"{bld} — {freq} ({agg})")
ax.set_ylabel("kWh")
ax.legend()
st.pyplot(fig, clear_figure=True)

# -------------------- footer --------------------
hist_last = hist.index.max() if not hist.empty else "N/A"
fut_first = fut.index.min() if not fut.empty else "N/A"
st.caption(f"Último dato histórico: {hist_last} — Inicio del pronóstico: {fut_first}")

st.download_button(
    "Descargar pronóstico (CSV)",
    fc[fc["building_id"] == bld][[ts_fc, "pred"]].to_csv(index=False).encode("utf-8"),
    file_name=f"{bld}_forecast.csv",
    mime="text/csv",
)
