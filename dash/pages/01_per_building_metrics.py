# dash/pages/01_per_building_metrics.py
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_CSV = "/app/data/evaluation/per_building_metrics.csv"
DEFAULT_PARQUET = "/app/data/processed/bdg2_electricity_cleaned.parquet"          # opcional (para NRMSE/WAPE)
DEFAULT_PREDS   = "/app/data/evaluation/validation_predictions.parquet"              # opcional (para skill)

st.set_page_config(page_title="Métricas por edificio", page_icon="🏢", layout="wide")
st.title("🏢 Métricas por edificio - Análisis de Performance del Modelo")

# ---------- Carga ----------
with st.expander("Fuentes de datos", expanded=True):
    csv_path = st.text_input("Ruta CSV de métricas (obligatorio)", DEFAULT_CSV)
    parquet_path = st.text_input("Parquet de datos (opcional, para NRMSE/WAPE)", DEFAULT_PARQUET)
    preds_path   = st.text_input("Predicciones de validación (opcional, para skill)", DEFAULT_PREDS)

p = Path(csv_path)
if not p.exists():
    st.warning(f"No encuentro el archivo: {p}. Genera el CSV con el evaluador.")
    st.stop()

df = pd.read_csv(p)
if "building_id" not in df.columns:
    st.error("El CSV no tiene la columna 'building_id'.")
    st.stop()

df["building_id"] = df["building_id"].astype(str)
if "mape" in df.columns and "mape_pct" not in df.columns:
    df["mape_pct"] = df["mape"] * 100.0

# ---------- Derivados opcionales a partir del parquet ----------
mean_load = None
if parquet_path and Path(parquet_path).exists():
    try:
        # solo las columnas necesarias para ser livianos
        raw = pd.read_parquet(parquet_path, columns=["building_id", "value"])
        raw["building_id"] = raw["building_id"].astype(str)
        mean_load = raw.groupby("building_id")["value"].mean().rename("mean_load")
        df = df.merge(mean_load.reset_index(), on="building_id", how="left")
        # evitar /0 o NaN
        safe_mean = df["mean_load"].replace(0, np.nan)
        if "rmse" in df:
            df["nrmse"] = df["rmse"] / safe_mean
        if "mae" in df:
            df["wape_pct"] = 100.0 * (df["mae"] / safe_mean)
    except Exception as e:
        st.info(f"No pude calcular NRMSE/WAPE desde {parquet_path}: {e}")

# ---------- Skill (opcional) desde predicciones ----------
# buscamos columnas de forma robusta
def _first_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

if preds_path and Path(preds_path).exists():
    try:
        pr = pd.read_parquet(preds_path)
        need_cols = list(pr.columns)
        bid = _first_col(need_cols, ["building_id", "bldg_id", "site_id"])
        ts  = _first_col(need_cols, ["timestamp_local", "timestamp_utc", "ts"])
        y   = _first_col(need_cols, ["y_true", "target", "actual", "value_true", "y"])
        yhat= _first_col(need_cols, ["y_pred", "prediction", "pred", "yhat"])

        if not all([bid, ts, y, yhat]):
            raise ValueError(f"Faltan columnas en {preds_path}. Tengo: {list(pr.columns)}")

        pr = pr[[bid, ts, y, yhat]].copy()
        pr.rename(columns={bid:"building_id", ts:"ts", y:"y_true", yhat:"y_pred"}, inplace=True)
        pr["building_id"] = pr["building_id"].astype(str)
        pr = pr.sort_values(["building_id", "ts"])
        # baseline: persistencia a 168 horas (misma hora semana anterior)
        pr["y_base"] = pr.groupby("building_id")["y_true"].shift(168)

        def _rmse(a):
            a = a.dropna()
            return float(np.sqrt(np.mean(np.square(a)))) if len(a) else np.nan

        g = pr.groupby("building_id", group_keys=False)
        rmse_model = g.apply(lambda gdf: _rmse(gdf["y_true"] - gdf["y_pred"])).rename("rmse_model")
        rmse_base  = g.apply(lambda gdf: _rmse(gdf["y_true"] - gdf["y_base"])).rename("rmse_baseline")
        skill = (1.0 - rmse_model / rmse_base).rename("skill_w168")

        df = df.merge(pd.concat([rmse_base, skill], axis=1).reset_index(), on="building_id", how="left")
        df["skill_pct"] = 100.0 * df["skill_w168"]
    except Exception as e:
        st.info(f"No pude calcular skill desde {preds_path}: {e}")

# ---------- Controles ----------
metric_choices = [c for c in ["rmse","mae","mape_pct","nrmse","wape_pct","skill_w168","skill_pct"] if c in df.columns]
if not metric_choices:
    st.error("No hay métricas disponibles para graficar.")
    st.stop()

c1, c2, c3 = st.columns([1,1,2])
with c1:
    metric = st.selectbox("Métrica", metric_choices, index=metric_choices.index("nrmse") if "nrmse" in metric_choices else 0)
with c2:
    worst_first = st.toggle("Peores primero", value=True)
with c3:
    top_k = st.slider("Cantidad a mostrar", 5, min(100, len(df)), min(20, len(df)))

substring = st.text_input("Filtrar por texto en building_id (opcional)", "")

# para métricas donde un valor MAYOR es MEJOR
higher_is_better = {"skill_w168", "skill_pct"}

# ---------- Filtro + orden ----------
d = df.copy()
if substring:
    d = d[d["building_id"].str.contains(substring, case=False, na=False)]

# decidir el orden según “peores primero”
descending = worst_first if metric not in higher_is_better else (not worst_first)
d = d.sort_values(metric, ascending=not descending).head(top_k)

# ---------- Resumen ----------
def _fmt(x, m):
    is_pct = m.endswith("_pct")
    return f"{x:.2f}%" if is_pct and pd.notna(x) else (f"{x:.3f}" if m in ("nrmse",) else f"{x:.2f}")

media = d[metric].mean()
mediana = d[metric].median()
st.caption(
    f"Edificios (total): {len(df):,} — "
    f"{metric.upper()} media: {_fmt(media, metric)} | "
    f"mediana: {_fmt(mediana, metric)}"
)

# ---------- Gráfico ----------
fig_h = max(4, 0.35 * len(d))
fig, ax = plt.subplots(figsize=(12, fig_h))
ax.barh(d["building_id"], d[metric])
ax.invert_yaxis()
xlab = metric.upper() + (" (%)" if metric.endswith("_pct") else "")
ax.set_xlabel(xlab); ax.set_ylabel("building_id")
ttl = f"{xlab} por edificio (Top {len(d)})"
if metric in higher_is_better:
    ttl += " — mayor es mejor"
else:
    ttl += " — menor es mejor"
ax.set_title(ttl)

for pch in ax.patches:
    w = pch.get_width()
    label = _fmt(w, metric)
    ax.text(w, pch.get_y() + pch.get_height()/2, label, va="center", ha="left", fontsize=8)

plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# ---------- Tabla + descarga ----------
cols_order = ["building_id","rmse","mae","mape","mape_pct","mean_load","nrmse","wape_pct","rmse_baseline","skill_w168","skill_pct"]
cols_show = [c for c in cols_order if c in d.columns]
st.subheader("Tabla (filtrada)")
st.dataframe(d[cols_show].reset_index(drop=True), use_container_width=True)

st.download_button(
    "Descargar CSV filtrado",
    d[cols_show].to_csv(index=False).encode("utf-8"),
    file_name="per_building_metrics_filtrado.csv",
    mime="text/csv"
)

with st.expander("Notas de interpretación"):
    st.markdown(
        "- **NRMSE = RMSE / media(consumo)** — comparable entre edificios.\n"
        "- **WAPE% ≈ 100·MAE / media(consumo)** — porcentaje robusto.\n"
        "- **skill_w168**: mejora relativa vs baseline de persistencia semanal `y(t-168h)`; >0 indica mejora.\n"
        "- **MAPE%** puede inflarse cuando el consumo real es muy bajo."
    )
