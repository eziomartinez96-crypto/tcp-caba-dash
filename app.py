import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# 0) CONFIG INICIAL
# =========================
st.set_page_config(page_title="Transporte Colectivo de Pasajeros en CABA", layout="wide")
st.title("Transporte Colectivo de Pasajeros en CABA")
st.write("Cantidad de Transacciones y Máximo de Vehículos en Calle por Línea y Hora")

# =========================
# 1) LECTURA SIMPLE
# =========================
RUTA_CSV = "data/cantrx_maxvh_2508_3108_25.csv"

def _leer_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

df_raw = _leer_csv(RUTA_CSV)

# =========================
# 2) VALIDACIÓN + TIPOS
# =========================
req = [
    "IDLINEA","NUM_LINEA","FECHA","HORARIO","hora_sola",
    "cantidad_internos","cantidad_transacciones",
    "max_internos_en_el_dia","max_transacciones_en_el_dia"
]
faltan = [c for c in req if c not in df_raw.columns]
if faltan:
    st.error(f"Faltan columnas requeridas: {faltan}")
    st.stop()

df = df_raw.copy()
df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date

def to_int(series):
    return pd.to_numeric(
        series.astype(str)
              .str.replace(",", ".", regex=False)
              .str.replace('"', "", regex=False)
              .str.replace("'", "", regex=False),
        errors="coerce"
    ).astype("Int64")

for c in ["IDLINEA","NUM_LINEA","hora_sola",
          "cantidad_internos","cantidad_transacciones",
          "max_internos_en_el_dia","max_transacciones_en_el_dia"]:
    df[c] = to_int(df[c])

# =========================
# 3) CONTROLES (SIDEBAR)
# =========================
with st.sidebar:
    st.header("Filtros")

    # Fecha (día/rango)
    if df["FECHA"].notna().any():
        fmin, fmax = df["FECHA"].min(), df["FECHA"].max()
    else:
        fmin = fmax = None

    modo_fecha = st.radio("Filtrar por:", ["Un día", "Rango"], horizontal=True)
    if modo_fecha == "Un día":
        dia_sel = st.date_input("Día", value=fmin, min_value=fmin, max_value=fmax)
        mask_fecha = (df["FECHA"] == dia_sel)
    else:
        rango = st.date_input("Rango", value=(fmin, fmax), min_value=fmin, max_value=fmax)
        if isinstance(rango, tuple) and len(rango) == 2:
            mask_fecha = (df["FECHA"] >= rango[0]) & (df["FECHA"] <= rango[1])
        else:
            mask_fecha = df["FECHA"].notna()

    # Rango horario (0–23)
    hmin_raw = pd.to_numeric(df["hora_sola"], errors="coerce").min()
    hmax_raw = pd.to_numeric(df["hora_sola"], errors="coerce").max()
    hmin = int(hmin_raw) if pd.notna(hmin_raw) else 0
    hmax = int(hmax_raw) if pd.notna(hmax_raw) else 23
    hmin, hmax = max(0, hmin), min(23, hmax)

    rango_horas = st.slider(
        "Rango horario",
        min_value=0, max_value=23,
        value=(hmin, hmax),
        step=1
    )

    # Líneas
    lineas_all = sorted(df["NUM_LINEA"].dropna().unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    lineas_sel = st.multiselect("Líneas", options=lineas_all, default=lineas_all)

    # Métrica (aplica a pivot y gráficos)
    metrica = st.radio(
        "Métrica:",
        [
            "Suma",
            "Promedio diario (todos los días)",
            "Promedio diario hábil (lun–vie)",
            "Promedio diario no hábil (sáb–dom)"
        ],
        index=0
    )

# Aplicar filtros base
df_fil = df.loc[mask_fecha & df["NUM_LINEA"].isin(lineas_sel)].copy()
df_fil = df_fil[(df_fil["hora_sola"] >= rango_horas[0]) & (df_fil["hora_sola"] <= rango_horas[1])]

# =========================
# 4) DÍAS ÚNICOS (TOT/H/NH)
# =========================
fechas_unicas = pd.to_datetime(pd.Series(sorted(df_fil["FECHA"].dropna().unique())))
dias_total = int(len(fechas_unicas))

# 0..4 = L a V ; 5..6 = S y D
mask_hab   = fechas_unicas.dt.weekday < 5
mask_nohab = fechas_unicas.dt.weekday >= 5

dias_hab   = int(mask_hab.sum())
dias_nohab = int(mask_nohab.sum())

fechas_hab   = set(fechas_unicas[mask_hab].dt.date)
fechas_nohab = set(fechas_unicas[mask_nohab].dt.date)

# =========================
# 5–7) PIVOT (cálculo + métrica + render) — UNA sola tabla
# =========================
# Índice de horas según el rango elegido
horas_idx = pd.Index(range(rango_horas[0], rango_horas[1] + 1), name="hora_sola")

def pivot_sum(dframe):
    return (
        pd.pivot_table(
            dframe,
            index="hora_sola",
            columns="NUM_LINEA",
            values=["cantidad_transacciones", "cantidad_internos"],
            aggfunc="sum",
            fill_value=0
        )
        .reindex(horas_idx, fill_value=0)
    )

pivot_all = pivot_sum(df_fil)
pivot_hab = pivot_sum(df_hab) if dias_hab > 0 else pivot_all*0
pivot_noh = pivot_sum(df_nohab) if dias_nohab > 0 else pivot_all*0

# Aplicar métrica seleccionada
if metrica == "Suma":
    pivot_sel = pivot_all
    etiqueta = "Suma"
elif metrica == "Promedio diario (todos los días)":
    d = max(dias_total, 1)
    pivot_sel = (pivot_all / d).round(2)
    etiqueta = f"Promedio diario (÷{d})"
elif metrica == "Promedio diario hábil (lun–vie)":
    d = max(dias_hab, 1)
    pivot_sel = (pivot_hab / d).round(2)
    etiqueta = f"Promedio diario hábil (÷{d})"
else:  # Promedio diario no hábil (sáb–dom)
    d = max(dias_nohab, 1)
    pivot_sel = (pivot_noh / d).round(2)
    etiqueta = f"Promedio diario no hábil (÷{d})"

# Ordenar columnas a (NUM_LINEA, Métrica) y renombrar subcolumnas
pivot_sel = (pivot_sel.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0))
pivot_sel.columns.set_names(["NUM_LINEA", "Métrica"], inplace=True)
pivot_sel = pivot_sel.rename(columns={"cantidad_transacciones": "Transacciones",
                                      "cantidad_internos": "Vehículos"}, level=1)

st.subheader(f"Tabla pivote (hora × línea) – {etiqueta}")
st.dataframe(pivot_sel, use_container_width=True)

st.divider()

# =========================
# 8) GRÁFICO ÚNICO CON SOLAPAS (una curva por línea; Hábiles vs No hábiles por trazo)
# =========================
st.subheader("Curvas por hora (una visual; cambiar por solapa)")

def agg_por_hora_linea(dframe, col):
    # Construye el grid solo para las horas del rango y las líneas seleccionadas
    idx = pd.MultiIndex.from_product(
        [range(rango_horas[0], rango_horas[1] + 1), lineas_sel],
        names=["hora_sola", "NUM_LINEA"]
    )
    g = (dframe.groupby(["hora_sola", "NUM_LINEA"])[col].sum()
               .reindex(idx, fill_value=0)
               .reset_index())
    return g

# a) Transacciones
tx_hab = agg_por_hora_linea(df_hab, "cantidad_transacciones")
tx_noh = agg_por_hora_linea(df_nohab, "cantidad_transacciones")

# b) Vehículos
vh_hab = agg_por_hora_linea(df_hab, "cantidad_internos")
vh_noh = agg_por_hora_linea(df_nohab, "cantidad_internos")

# c) Ajuste por métrica (promedios por grupo)
if metrica == "Promedio diario (todos los días)":
    if dias_hab > 0:
        tx_hab["cantidad_transacciones"] = (tx_hab["cantidad_transacciones"] / dias_hab).round(2)
        vh_hab["cantidad_internos"] = (vh_hab["cantidad_internos"] / dias_hab).round(2)
    if dias_nohab > 0:
        tx_noh["cantidad_transacciones"] = (tx_noh["cantidad_transacciones"] / dias_nohab).round(2)
        vh_noh["cantidad_internos"] = (vh_noh["cantidad_internos"] / dias_nohab).round(2)
elif metrica == "Promedio diario hábil (lun–vie)":
    if dias_hab > 0:
        tx_hab["cantidad_transacciones"] = (tx_hab["cantidad_transacciones"] / dias_hab).round(2)
        vh_hab["cantidad_internos"] = (vh_hab["cantidad_internos"] / dias_hab).round(2)
    # No hábiles quedan como suma (si no hubo días, serán 0)
elif metrica == "Promedio diario no hábil (sáb–dom)":
    if dias_nohab > 0:
        tx_noh["cantidad_transacciones"] = (tx_noh["cantidad_transacciones"] / dias_nohab).round(2)
        vh_noh["cantidad_internos"] = (vh_noh["cantidad_internos"] / dias_nohab).round(2)
    # Hábiles quedan como suma

def prep_long(dfA, dfB, col, labelA, labelB):
    a = dfA.rename(columns={col: "valor"}).assign(Grupo=labelA)
    b = dfB.rename(columns={col: "valor"}).assign(Grupo=labelB)
    out = pd.concat([a[["hora_sola","NUM_LINEA","valor","Grupo"]],
                     b[["hora_sola","NUM_LINEA","valor","Grupo"]]], axis=0)
    out["HORA"] = out["hora_sola"].apply(lambda h: f"{int(h):02d}:00")
    out["NUM_LINEA"] = out["NUM_LINEA"].astype(str)
    return out

plot_tx = prep_long(tx_hab, tx_noh, "cantidad_transacciones", "Hábiles", "No hábiles")
plot_vh = prep_long(vh_hab, vh_noh, "cantidad_internos", "Hábiles", "No hábiles")

# 👇 acá insertás el filtro según la métrica
if metrica == "Promedio diario hábil (lun–vie)":
    plot_tx = plot_tx[plot_tx["Grupo"] == "Hábiles"]
    plot_vh = plot_vh[plot_vh["Grupo"] == "Hábiles"]
elif metrica == "Promedio diario no hábil (sáb–dom)":
    plot_tx = plot_tx[plot_tx["Grupo"] == "No hábiles"]
    plot_vh = plot_vh[plot_vh["Grupo"] == "No hábiles"]
# si es "Suma" o "Promedio diario (todos los días)" → se mantienen ambos

yl_tx = {
    "Suma": "Transacciones (suma)",
    "Promedio diario (todos los días)": "Transacciones (promedio diario)",
    "Promedio diario hábil (lun–vie)": "Transacciones (promedio diario hábil)",
    "Promedio diario no hábil (sáb–dom)": "Transacciones (promedio diario no hábil)"
}[metrica]

yl_vh = {
    "Suma": "Vehículos (suma)",
    "Promedio diario (todos los días)": "Vehículos (promedio diario)",
    "Promedio diario hábil (lun–vie)": "Vehículos (promedio diario hábil)",
    "Promedio diario no hábil (sáb–dom)": "Vehículos (promedio diario no hábil)"
}[metrica]

tabs = st.tabs(["📈 Transacciones", "🚌 Vehículos"])

with tabs[0]:
    fig_tx = px.line(
        plot_tx, x="HORA", y="valor",
        color="NUM_LINEA", line_dash="Grupo", markers=True,
        labels={"HORA":"Hora", "valor": yl_tx, "NUM_LINEA":"Línea", "Grupo":"Tipo de día"}
    )
    fig_tx.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=440, legend_title_text="")
    st.plotly_chart(fig_tx, use_container_width=True)

with tabs[1]:
    fig_vh = px.line(
        plot_vh, x="HORA", y="valor",
        color="NUM_LINEA", line_dash="Grupo", markers=True,
        labels={"HORA":"Hora", "valor": yl_vh, "NUM_LINEA":"Línea", "Grupo":"Tipo de día"}
    )
    fig_vh.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=440, legend_title_text="")
    st.plotly_chart(fig_vh, use_container_width=True)
