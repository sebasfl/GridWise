# dash/app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

DEFAULT_PARQUET = "/app/data/processed/bdg2_electricity_cleaned.parquet"
EVAL_METRICS = "/app/data/evaluation/overall_metrics.json"
BUILDING_METRICS = "/app/data/evaluation/per_building_metrics.csv"

st.set_page_config(page_title="SmartGrid - Model Performance Dashboard", page_icon="🏢", layout="wide")
st.title("🏢 SmartGrid - Model Performance Dashboard")

# Load and display model performance metrics
@st.cache_data
def load_evaluation_metrics():
    try:
        with open(EVAL_METRICS, 'r') as f:
            overall = json.load(f)
        building_metrics = pd.read_csv(BUILDING_METRICS)
        return overall, building_metrics
    except FileNotFoundError:
        return None, None

overall_metrics, building_metrics = load_evaluation_metrics()

if overall_metrics:
    st.header("📈 Overall Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{overall_metrics['rmse']:.4f}")
    with col2:
        st.metric("MAE", f"{overall_metrics['mae']:.4f}")
    with col3:
        st.metric("R²", f"{overall_metrics['r2']:.4f}")
    with col4:
        st.metric("MAPE", f"{overall_metrics['mape']:.2f}%")
    
    st.markdown("---")

# Sidebar controls
st.sidebar.header("🎛️ Visualization Controls")
parquet_path = st.sidebar.text_input("Ruta del parquet", DEFAULT_PARQUET)
resample = st.sidebar.selectbox("Frecuencia de remuestreo", ["1H", "6H", "1D"], index=2)
topn = st.sidebar.slider("Top N (para selección por defecto)", 1, 50, 10)
name_filter = st.sidebar.text_input("Filtrar lista de edificios por texto (opcional)", "")

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
