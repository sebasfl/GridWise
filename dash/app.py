# dash/app.py
import streamlit as st
import pandas as pd
from pathlib import Path

DEFAULT_PARQUET = "/app/data/processed/bdg2_electricity_long.parquet"

st.set_page_config(page_title="SmartGrid Dash", layout="wide")
st.title("SmartGrid – Series por edificio")

parquet_path = st.text_input("Ruta del parquet", DEFAULT_PARQUET)
resample = st.selectbox("Frecuencia de remuestreo", ["1H", "6H", "1D"], index=2)
topn = st.slider("Top N (para selección por defecto)", 1, 50, 10)
name_filter = st.text_input("Filtrar lista de edificios por texto (opcional)", "")

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

if "meter" in df.columns:
    df = df[df["meter"] == "electricity"].copy()

df["value"] = pd.to_numeric(df.get("value"), errors="coerce")
df = df.dropna(subset=["value"])

# Ranking por cantidad de lecturas para defaults
counts = df.groupby("building_id")[ts_col].count().sort_values(ascending=False)
default_top = counts.head(topn).index.tolist()

# Lista completa (con filtro opcional de nombre)
all_buildings = counts.index.astype(str)
if name_filter:
    all_buildings = [b for b in all_buildings if name_filter.lower() in b.lower()]

# Selector de edificios (con búsqueda)
selected_buildings = st.multiselect(
    "Selecciona edificios a visualizar",
    options=all_buildings,
    default=[b for b in default_top if b in all_buildings],  # default sensato
)

# Si nada seleccionado, usa el default_top
if not selected_buildings:
    selected_buildings = default_top

# Filtra y grafica
plot_df = df[df["building_id"].isin(selected_buildings)].copy()

# Remuestrear y pivotear a formato ancho para st.line_chart
wide = (
    plot_df.set_index(ts_col)
           .groupby("building_id")["value"]
           .resample(resample).mean()
           .unstack("building_id")
           .sort_index()
)

st.line_chart(wide)  # gráfico interactivo
st.caption(
    f"Parquet: {parquet_path} — Frecuencia: {resample} — "
    f"Edificios seleccionados: {len(selected_buildings)}"
)
