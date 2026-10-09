
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

# CONFIGURACIÓN
st.set_page_config(
    page_title="GAC · Ventas Anuales",
    layout="wide"
)

# ESTÉTICA GAC (solo presentación)
ROJO = "#EF3340"
VINO = "#8F1D2C"
GRIS = "#9B9CA3"
FONDO = "#0E1117"

st.markdown("""
<style>
.stApp {background-color:#0E1117;color:#FFFFFF;}
.block-container {padding-top:2rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#262730;}
[data-testid="stMetric"] {
    background:linear-gradient(145deg,#2B0D16,#210A11);
    border:1px solid #501421;border-radius:14px;
    padding:18px 20px;min-height:130px;
}
[data-testid="stMetricLabel"] {color:#D2C7CA;}
[data-testid="stMetricValue"] {color:#FFFFFF;}
[data-testid="stAlert"] {
    background:#171A22 !important;
    background-color:#171A22 !important;
    border:1px solid #343741 !important;
    border-left:5px solid #EF3340 !important;
    border-radius:12px !important;
}
[data-testid="stAlert"] * {color:#FFFFFF !important;}
[data-baseweb="tag"] {background:#8F1D2C !important;color:white !important;}
[data-baseweb="select"] * {accent-color:#EF3340;}
.gac-ventas-titulo {font-size:48px;font-weight:750;color:#FFFFFF;line-height:1.15;}
.gac-ventas-subtitulo {font-size:15px;color:#9B9CA3;margin-top:8px;}
.gac-ventas-linea {height:4px;background:#EF3340;border-radius:6px;margin:20px 0 30px;}
</style>
""", unsafe_allow_html=True)

col_logo, col_encabezado = st.columns([1, 5], vertical_alignment="center")
with col_logo:
    logo = Path(__file__).resolve().parent / "logo_gac.png"
    if logo.exists():
        st.image(str(logo), width=125)
with col_encabezado:
    st.markdown(
        '<div class="gac-ventas-titulo">Ventas Anuales</div>'
        '<div class="gac-ventas-subtitulo">Análisis de desempeño comercial por vendedor</div>',
        unsafe_allow_html=True
    )
st.markdown('<div class="gac-ventas-linea"></div>', unsafe_allow_html=True)

# Ajuste de colores de Plotly, sin modificar datos, trazas ni dimensiones
# Se aplica únicamente a los gráficos ya construidos.
def aplicar_colores_gac(figura):
    paleta = [ROJO, VINO, "#E88D98", GRIS]
    for i, traza in enumerate(figura.data):
        color = paleta[i % len(paleta)]
        if traza.type == "scatter":
            if "lines" in (traza.mode or ""):
                traza.line.color = color
            if "markers" in (traza.mode or ""):
                traza.marker.color = color
    figura.update_layout(
        paper_bgcolor=FONDO,
        plot_bgcolor=FONDO,
        font_color="#FFFFFF",
        colorway=paleta,
        xaxis=dict(gridcolor="#30323A", zerolinecolor="#30323A"),
        yaxis=dict(gridcolor="#30323A", zerolinecolor="#30323A")
    )
    return figura

# BASE DE DATOS
def load_data():
    datos = [
        ["PEDRO GONZALEZ ESPINDOLA",88,0,22,0,45,2,21,0],
        ["VICTOR MANUEL SANCHEZ ARRO",88,0,11,0,51,0,26,0],
        ["MIGUEL ANGEL SONDEREGGER",28,6,27,1,1,25,0,17],
        ["JOSE ANTONIO LICONA MENDE",98,1,10,4,35,4,53,1],
        ["JAIME GONZALEZ CORONA",25,8,25,2,0,30,0,17],
        ["ALAN GONZÁLEZ BELANZATEGU",46,4,4,10,36,3,6,11],
        ["MAURICIO ALFREDO VAZQUEZ F",72,2,0,24,47,1,25,2],
        ["AURELIO FELIPE TORRES REV",63,3,3,12,45,2,15,3],
        ["J4OSETTE RAMON NAVA FLORES",16,11,16,3,0,30,0,17],
        ["CARLOS ALBERTO NIETO MERI",15,12,10,4,5,12,0,17],
        ["NANCY OLASCOAGA CASTILLO",20,9,3,12,17,6,0,17],
        ["DANIEL DE LEON CANDIA",12,15,9,6,3,18,0,17],
        ["DANIELA DURAN BERNARDINO",8,21,8,7,0,30,0,17],
        ["GUADALUPE PALACIOS HERNÁN",7,22,5,9,2,19,0,17],
        ["CARLOS CORTES TEXIS",7,22,1,18,6,11,0,17],
        ["RICARDO SILVA HERAS",20,9,0,24,20,5,0,17],
        ["PAUL ROSAS YRIGOYEN",6,24,6,8,0,30,0,17],
        ["OSCAR ALONSO VELÁZQUEZ HE",6,24,4,10,2,19,0,17],
        ["JOSE MANUEL LOPEZ AGUILA",3,31,3,12,0,30,0,17],
        ["JESUS SERRANO RAMIREZ",2,32,2,15,0,30,0,17],
        ["LUIS ALBERTO GOMEZ GONZAL",2,32,2,15,0,30,0,17],
        ["MONICA GISELA FLORES RODR",2,32,2,15,0,30,0,17],
        ["iGOR SEBASTIÁN BENHUMEA Z",2,32,0,24,2,19,0,17],
        ["CLAUDIA GALVÁN ARCE",2,32,0,24,2,19,0,17],
        ["MIGUEL",4,28,0,24,4,15,0,17],
        ["SERGIO",2,32,0,24,2,19,0,17],
        ["DAVID JIMENEZ CASTRO",1,41,1,18,0,30,0,17],
        ["ABIGAIL VARGAS CRUZ",1,41,1,18,0,30,0,17],
        ["NAYELI MONTES DE OCA MART",1,41,1,18,0,30,0,17],
        ["ANGELICA MAGDALENA CEDILL",1,41,1,18,0,30,0,17],
        ["JOSE CARLOS ZENTENO LOZAD",1,41,1,18,0,30,0,17],
        ["ANGEL EDUARDO VAZQUEZ REY",1,41,0,24,1,25,0,17],
        ["JOSE EDUARDO PARADA DURAN",1,41,0,24,1,25,0,17],
        ["LAURA LOPEZ MEJIA",1,41,0,24,1,25,0,17],
        ["IGNACIO",14,13,0,24,8,10,6,11],
        ["MARCOS COCA",9,19,0,24,9,9,0,17],
        ["CESAR",4,28,0,24,4,15,0,17],
        ["JULIO CESAR PESTAÑA",29,5,0,24,16,7,13,4],
        ["JULIO CESAR",12,15,0,24,5,12,7,10],
        ["JOSHLAAN",2,32,0,24,2,19,0,17],
        ["OMAR",5,26,0,24,5,12,0,17],
        ["MERCEDES",4,28,0,24,4,15,0,17],
        ["ALVARO",5,26,0,24,1,25,4,13],
        ["CARLOS MEJIA (PACHUCA)",10,18,0,24,0,30,10,8],
        ["MARIO",9,19,0,24,0,30,9,9],
        ["CARLOS",12,15,0,24,0,30,12,6],
        ["NAYELI",13,14,0,24,0,30,13,4],
        ["PEDRO ANTONIO",26,7,0,24,15,8,11,7],
        ["OBRIAN",2,32,0,24,0,30,2,14],
        ["FATIMA",2,32,0,24,0,30,2,14],
        ["RUBEN",1,41,0,24,0,30,1,16]
    ]

    columnas = [
        "Nombre vendedor", "Acum", "Ranking",
        "Acum 2024", "Ranking 2024",
        "Acum 2025", "Ranking 2025",
        "Acum 2026", "Ranking 2026"
    ]

    return pd.DataFrame(datos, columns=columnas)


df = load_data()


# VARIABLES NUMÉRICAS
variables = [
    "Acum",
    "Ranking",
    "Acum 2024",
    "Ranking 2024",
    "Acum 2025",
    "Ranking 2025",
    "Acum 2026",
    "Ranking 2026"
]


# MENÚ LATERAL
st.sidebar.title("🚗 GAC · Ventas Anuales")

tipo = st.sidebar.selectbox(
    "Tipo de regresión",
    ["Regresión Lineal Simple", "Regresión Lineal Múltiple"]
)


# REGRESIÓN LINEAL SIMPLE
if tipo == "Regresión Lineal Simple":

    st.subheader("Regresión Lineal Simple")

    y_var = st.sidebar.selectbox(
        "Variable objetivo (Y)",
        variables,
        index=0
    )

    variables_x = [
        v for v in variables
        if v != y_var and df[v].nunique() > 1
    ]

    x_var = st.sidebar.selectbox(
        "Variable independiente (X)",
        variables_x
    )

    X = df[[x_var]]
    y = df[y_var]

    modelo = LinearRegression()
    modelo.fit(X, y)

    predicciones = modelo.predict(X)
    r2 = modelo.score(X, y)

    col1, col2 = st.columns(2)
    col1.metric("R²", f"{r2:.3f}")
    col2.metric("Pendiente", f"{modelo.coef_[0]:,.3f}")

    st.write(
        f"**Ecuación:** Y = {modelo.intercept_:.2f} "
        f"+ ({modelo.coef_[0]:.3f} × X)"
    )

    grafica = px.scatter(
        df,
        x=x_var,
        y=y_var,
        hover_name="Nombre vendedor",
        title=f"{y_var} según {x_var}"
    )

    orden = np.argsort(df[x_var].values)

    grafica.add_scatter(
        x=df[x_var].values[orden],
        y=predicciones[orden],
        mode="lines",
        name="Regresión lineal"
    )

    st.plotly_chart(aplicar_colores_gac(grafica), use_container_width=True)


# REGRESIÓN LINEAL MÚLTIPLE
else:

    st.subheader("Regresión Lineal Múltiple")

    y_var = st.sidebar.selectbox(
        "Variable objetivo (Y)",
        variables,
        index=0
    )

    variables_x = [
        v for v in variables
        if v != y_var and df[v].nunique() > 1
    ]

    x_vars = st.sidebar.multiselect(
        "Variables independientes (X)",
        variables_x,
        default=[
            v for v in ["Acum 2024", "Acum 2025"]
            if v in variables_x
        ]
    )

    if not x_vars:
        st.warning("Selecciona al menos una variable independiente.")

    else:
        X = df[x_vars]
        y = df[y_var]

        modelo = LinearRegression()
        modelo.fit(X, y)

        predicciones = modelo.predict(X)
        r2 = modelo.score(X, y)

        col1, col2 = st.columns(2)
        col1.metric("R²", f"{r2:.3f}")
        col2.metric("Variables X", len(x_vars))

        st.subheader("Coeficientes")

        coeficientes = pd.DataFrame({
            "Variable": x_vars,
            "Coeficiente": modelo.coef_
        })

        st.dataframe(coeficientes, use_container_width=True)

        st.write(f"**Intercepto:** {modelo.intercept_:.3f}")

        resultados = pd.DataFrame({
            "Vendedor": df["Nombre vendedor"],
            "Valor real": y,
            "Valor predicho": predicciones
        })

        grafica = px.scatter(
            resultados,
            x="Valor real",
            y="Valor predicho",
            hover_name="Vendedor",
            title="Valores reales vs. predichos"
        )

        st.plotly_chart(aplicar_colores_gac(grafica), use_container_width=True)
