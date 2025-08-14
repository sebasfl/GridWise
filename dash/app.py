# dash/app.py
import streamlit as st
import pandas as pd
from pathlib import Path

DEFAULT_PARQUET = "/app/data/processed/bdg2_electricity_long.parquet"

st.set_page_config(page_title="SmartGrid Dash", layout="wide")
st.title("SmartGrid – Series por edificio")

parquet_path = st.text_input("Ruta del parquet", DEFAULT_PARQUET)
resample = st.selectbox("Frecuencia de remuestreo", ["1H", "6H", "1D"], index=2)
topn = st.slider("Top N edificios (por cantidad de datos)", 1, 50, 10)

p = Path(parquet_path)
if not p.exists():
    st.error(f"No existe el archivo: {p}")
    st.stop()

# Cargar y preparar
df = pd.read_parquet(parquet_path)
ts_col = "timestamp_local" if "timestamp_local" in df.columns else (
         "timestamp_utc" if "timestamp_utc" in df.columns else None)
if ts_col is None:
    st.error("El parquet no tiene timestamp_local ni timestamp_utc")
    st.stop()

df = df[df["meter"] == "electricity"].copy()
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["value"])

# Seleccionar Top-N edificios con más lecturas
top = (df["value"].notna().groupby(df["building_id"]).sum()
      ).sort_values(ascending=False).head(topn).index
df = df[df["building_id"].isin(top)].copy()

# Remuestrear y pivotear a formato ancho para st.line_chart
wide = (df.set_index(ts_col)
          .groupby("building_id")["value"]
          .resample(resample).mean()
          .unstack("building_id")
          .sort_index())

st.line_chart(wide)  # gráfico interactivo
st.caption(f"Parquet: {parquet_path} — Frecuencia: {resample} — Edificios: {len(top)}")
