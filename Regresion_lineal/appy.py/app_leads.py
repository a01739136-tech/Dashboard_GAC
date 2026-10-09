
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ============================================================
# ESTÉTICA GAC
# ============================================================

BASE = Path(__file__).resolve().parent

ROJO = "#EF3340"
VINO = "#8F1D2C"
ROSA = "#E88D98"
GRIS = "#9B9CA3"
BLANCO = "#FFFFFF"
ROJO_MEDIO = "#B52A3B"
GRIS_CLARO = "#D4D4D8"
FONDO = "#0E1117"

COLORES_GAC = [
    ROJO,
    VINO,
    ROSA,
    GRIS,
    ROJO_MEDIO,
    GRIS_CLARO,
    "#C94354",
    "#A96A77",
    "#F36B75"
]

ESCALA_GAC = [
    [0.00, "#FFF4F1"],
    [0.20, "#F7C4C0"],
    [0.40, "#E88D98"],
    [0.65, "#EF3340"],
    [0.85, "#B52A3B"],
    [1.00, "#641322"]
]

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background-color: #262730;
    border-right: 1px solid #383A45;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #2B0D16,
        #210A11
    );
    border: 1px solid #501421;
    border-radius: 14px;
    padding: 18px 20px;
    min-height: 125px;
}

[data-testid="stMetricLabel"] {
    color: #D2C7CA;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF;
    font-weight: 700;
}

[data-testid="stMetricDelta"] {
    color: #E88D98;
}

[data-baseweb="tag"] {
    background-color: #EF3340 !important;
    color: white !important;
    border-radius: 5px !important;
}

[data-baseweb="select"] > div {
    background-color: #0E1117 !important;
    border: 1px solid #8F1D2C !important;
    border-radius: 9px !important;
}

[data-testid="stAlert"] {
    background-color: #201016;
    border: 1px solid #501421;
    border-left: 4px solid #EF3340;
    border-radius: 10px;
}

.gac-title {
    font-size: 50px;
    font-weight: 750;
    color: #FFFFFF;
    margin: 0;
    line-height: 1.2;
}

.gac-subtitle {
    color: #9B9CA3;
    font-size: 15px;
    margin-top: 10px;
}

.gac-red-line {
    height: 4px;
    background: #EF3340;
    border-radius: 6px;
    margin: 22px 0 32px;
}

</style>
""", unsafe_allow_html=True)


def encabezado_gac():
    col_logo, col_texto = st.columns(
        [1.2, 5],
        vertical_alignment="center"
    )

    with col_logo:
        logo = BASE / "logo_gac.png"

        if logo.exists():
            st.image(str(logo), width=130)

    with col_texto:
        st.markdown(
            """
            <div class="gac-title">Análisis de Leads</div>
            <div class="gac-subtitle">
            Análisis de ventas, conversión, calidad del lead
            y regresión lineal
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="gac-red-line"></div>',
        unsafe_allow_html=True
    )


def estilo_grafica(fig, altura=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color=BLANCO,
            size=12
        ),
        margin=dict(
            l=30,
            r=25,
            t=55,
            b=65
        ),
        hoverlabel=dict(
            bgcolor="#242730",
            font_color=BLANCO
        )
    )

    if altura is not None:
        fig.update_layout(height=altura)

    fig.update_xaxes(
        gridcolor="#30323A",
        zerolinecolor="#30323A",
        automargin=True
    )

    fig.update_yaxes(
        gridcolor="#30323A",
        zerolinecolor="#30323A",
        automargin=True
    )

    return fig


# ============================================================
# CARGA DE DATOS — ANÁLISIS ORIGINAL
# ============================================================

@st.cache_resource
def load_data():

    df = pd.read_csv(
        BASE / "base leads.csv"
    )

    data = df.copy()

    data.columns = data.columns.str.strip()

    data["Año"] = data["Año"].astype(int)
    data["Mes"] = data["Mes"].astype(int)

    data["Fecha"] = pd.to_datetime(
        dict(
            year=data["Año"],
            month=data["Mes"],
            day=1
        )
    )

    data = data[
        data["Total"] > 0
    ].reset_index(drop=True)

    data["Conv_Efectivos"] = (
        data["Ventas"] / data["Efectivos"] * 100
    )

    data["Conv_Total"] = (
        data["Ventas"] / data["Total"] * 100
    )

    categorias = [
        "Efectivos",
        "Ilocalizable",
        "No Contesta",
        "D. Incorrectos",
        "Duplicados"
    ]

    data["Suma_Cat"] = (
        data[categorias].sum(axis=1)
    )

    return data, categorias


data, categorias = load_data()

nombres_meses = [
    "Ene", "Feb", "Mar", "Abr",
    "May", "Jun", "Jul", "Ago",
    "Sep", "Oct", "Nov", "Dic"
]

# ============================================================
# SIDEBAR — FILTROS ORIGINALES
# ============================================================

st.sidebar.title("📈 GAC | Leads")

View = st.sidebar.selectbox(
    label="Tipo de Análisis",
    options=[
        "Ventas y Conversión",
        "Calidad del Lead",
        "Regresión Lineal"
    ]
)

Lista_Años = sorted(
    data["Año"].unique()
)

Años_sel = st.sidebar.multiselect(
    label="Años",
    options=Lista_Años,
    default=Lista_Años
)

data_f = data[
    data["Año"].isin(Años_sel)
]

if data_f.empty:
    st.warning(
        "Selecciona al menos un año en la barra lateral"
    )
    st.stop()

# ============================================================
# ENCABEZADO GAC — TODAS LAS VISTAS
# ============================================================

encabezado_gac()

# ============================================================
# VISTA 1: VENTAS Y CONVERSIÓN
# ============================================================

if View == "Ventas y Conversión":

    st.title("Ventas y Conversión")

    # KPI originales
    Kpi_1, Kpi_2, Kpi_3, Kpi_4 = st.columns(4)

    Kpi_1.metric(
        "Total leads",
        f"{data_f['Total'].sum():,.0f}"
    )

    Kpi_2.metric(
        "Efectivos",
        f"{data_f['Efectivos'].sum():,.0f}"
    )

    Kpi_3.metric(
        "Ventas",
        f"{data_f['Ventas'].sum():,.0f}"
    )

    Kpi_4.metric(
        "Conversión (% efectivos)",
        (
            f"{data_f['Ventas'].sum() / data_f['Efectivos'].sum() * 100:.1f}%"
        )
    )

    st.markdown("")

    Contenedor_A, Contenedor_B = st.columns(2)

    # ========================================================
    # GRÁFICA 1: BARRAS + LÍNEA CON DOBLE EJE
    # ========================================================

    with Contenedor_A:

        st.write("Ventas y conversión por mes")

        figure1 = make_subplots(
            specs=[[{"secondary_y": True}]]
        )

        figure1.add_trace(
            go.Bar(
                x=data_f["Fecha"],
                y=data_f["Ventas"],
                name="Ventas",
                marker_color=ROJO
            ),
            secondary_y=False
        )

        figure1.add_trace(
            go.Scatter(
                x=data_f["Fecha"],
                y=data_f["Conv_Efectivos"],
                name="Conversión (%)",
                mode="lines+markers",
                line=dict(
                    color=BLANCO,
                    width=3
                ),
                marker=dict(
                    color=BLANCO,
                    size=7
                )
            ),
            secondary_y=True
        )

        figure1.update_yaxes(
            title_text="Ventas",
            secondary_y=False
        )

        figure1.update_yaxes(
            title_text="Conversión (% de efectivos)",
            secondary_y=True
        )

        figure1.update_layout(
            height=400,
            legend=dict(
                orientation="h",
                y=-0.2
            )
        )

        st.plotly_chart(
            estilo_grafica(figure1),
            use_container_width=True
        )

    # ========================================================
    # GRÁFICA 2: EMBUDO ORIGINAL
    # ========================================================

    with Contenedor_B:

        st.write("Embudo de conversión")

        Tabla_embudo = pd.DataFrame({
            "etapa": [
                "Total leads",
                "Efectivos",
                "Ventas"
            ],
            "valor": [
                data_f["Total"].sum(),
                data_f["Efectivos"].sum(),
                data_f["Ventas"].sum()
            ]
        })

        figure2 = px.funnel(
            Tabla_embudo,
            x="valor",
            y="etapa",
            color_discrete_sequence=[ROJO]
        )

        figure2.update_traces(
            textinfo="value+percent initial"
        )

        figure2.update_layout(
            height=400
        )

        st.plotly_chart(
            estilo_grafica(figure2),
            use_container_width=True
        )

# ============================================================
# VISTA 2: CALIDAD DEL LEAD
# ============================================================

if View == "Calidad del Lead":

    st.title("Calidad del Lead")

    # ========================================================
    # GRÁFICA 3: BARRAS APILADAS ORIGINALES
    # ========================================================

    st.write("Calidad del lead (% por mes)")

    calidad = (
        data_f[categorias]
        .div(data_f["Suma_Cat"], axis=0)
        * 100
    )

    calidad["Mes"] = (
        data_f["Fecha"].dt.strftime("%b %y")
    )

    Tabla_calidad = calidad.melt(
        id_vars="Mes",
        var_name="categoria",
        value_name="porcentaje"
    )

    figure3 = px.bar(
        Tabla_calidad,
        x="Mes",
        y="porcentaje",
        color="categoria",
        color_discrete_sequence=[
            ROJO,
            VINO,
            ROSA,
            GRIS,
            ROJO_MEDIO
        ]
    )

    figure3.update_layout(
        height=450,
        barmode="stack",
        yaxis_title="% del total",
        legend=dict(
            orientation="h",
            y=-0.25
        )
    )

    st.plotly_chart(
        estilo_grafica(figure3),
        use_container_width=True
    )

    # ========================================================
    # GRÁFICA 4: HEATMAP ORIGINAL
    # ========================================================

    st.write("Conversión (%) por año y mes")

    pivot_conv = data_f.pivot(
        index="Año",
        columns="Mes",
        values="Conv_Efectivos"
    )

    pivot_conv = pivot_conv.reindex(
        columns=range(1, 13)
    )

    pivot_conv.columns = nombres_meses

    figure4 = px.imshow(
        pivot_conv,
        text_auto=".1f",
        color_continuous_scale=ESCALA_GAC,
        aspect="auto"
    )

    figure4.update_yaxes(
        type="category"
    )

    figure4.update_layout(
        height=350
    )

    st.plotly_chart(
        estilo_grafica(figure4),
        use_container_width=True
    )

# ============================================================
# VISTA 3: REGRESIÓN LINEAL
# ============================================================

if View == "Regresión Lineal":

    from sklearn.linear_model import LinearRegression

    st.title("📈 Regresión Lineal")

    # ========================================================
    # VARIABLES Y FILTROS ORIGINALES
    # ========================================================

    excluir = [
        "Año",
        "Mes",
        "Conv_Efectivos",
        "Conv_Total",
        "Suma_Cat"
    ]

    numeric_df = (
        data_f.select_dtypes("number")
        .drop(
            columns=excluir,
            errors="ignore"
        )
    )

    Lista_num = list(
        numeric_df.columns
    )

    Variable_y = st.sidebar.selectbox(
        "Variable objetivo (Y)",
        Lista_num,
        index=Lista_num.index("Ventas")
    )

    opciones_x = [
        c for c in Lista_num
        if c != Variable_y
    ]

    Variable_x = st.sidebar.selectbox(
        "Variable independiente simple (X)",
        opciones_x,
        index=opciones_x.index("Efectivos")
    )

    Variables_x = st.sidebar.multiselect(
        "Variables independientes múltiple (X)",
        opciones_x,
        default=[
            c for c in [
                "Efectivos",
                "Total"
            ]
            if c in opciones_x
        ]
    )

    if not Variables_x:
        st.warning(
            "Selecciona al menos una variable para el modelo múltiple"
        )
        st.stop()

    Contenedor_A, Contenedor_B = st.columns(2)

    # ========================================================
    # REGRESIÓN LINEAL SIMPLE — MODELO ORIGINAL
    # ========================================================

    with Contenedor_A:

        st.subheader("Regresión lineal simple")

        model = LinearRegression().fit(
            numeric_df[[Variable_x]],
            numeric_df[Variable_y]
        )

        y_pred = model.predict(
            numeric_df[[Variable_x]]
        )

        r2 = model.score(
            numeric_df[[Variable_x]],
            numeric_df[Variable_y]
        )

        r = np.corrcoef(
            numeric_df[Variable_x],
            numeric_df[Variable_y]
        )[0, 1]

        m1, m2 = st.columns(2)

        m1.metric(
            "r (correlación)",
            f"{r:.3f}"
        )

        m2.metric(
            "R²",
            f"{r2:.3f}"
        )

        # ====================================================
        # GRÁFICA 5: REGRESIÓN SIMPLE ORIGINAL
        # ====================================================

        figure5 = go.Figure()

        figure5.add_trace(
            go.Scatter(
                x=numeric_df[Variable_x],
                y=numeric_df[Variable_y],
                mode="markers",
                name="Real",
                marker=dict(
                    color=VINO,
                    size=9
                )
            )
        )

        figure5.add_trace(
            go.Scatter(
                x=numeric_df[Variable_x],
                y=y_pred,
                mode="markers",
                name="Predicho",
                marker=dict(
                    color=ROJO,
                    symbol="x",
                    size=9
                )
            )
        )

        figure5.update_layout(
            title="Modelo Lineal Simple",
            xaxis_title=Variable_x,
            yaxis_title=Variable_y,
            height=400
        )

        st.plotly_chart(
            estilo_grafica(figure5),
            use_container_width=True
        )

    # ========================================================
    # REGRESIÓN LINEAL MÚLTIPLE — MODELO ORIGINAL
    # ========================================================

    with Contenedor_B:

        st.subheader("Regresión lineal múltiple")

        model_M = LinearRegression().fit(
            numeric_df[Variables_x],
            numeric_df[Variable_y]
        )

        y_pred_M = model_M.predict(
            numeric_df[Variables_x]
        )

        r2_M = model_M.score(
            numeric_df[Variables_x],
            numeric_df[Variable_y]
        )

        m1, m2 = st.columns(2)

        m1.metric(
            "R (correlación múltiple)",
            f"{np.sqrt(max(r2_M, 0)):.3f}"
        )

        m2.metric(
            "R²",
            f"{r2_M:.3f}"
        )

        # ====================================================
        # GRÁFICA 6: REGRESIÓN MÚLTIPLE ORIGINAL
        # Un color GAC diferente por cada variable X
        # ====================================================

        colores = COLORES_GAC

        figure6 = go.Figure()

        for i, var in enumerate(Variables_x):

            color = colores[
                i % len(colores)
            ]

            figure6.add_trace(
                go.Scatter(
                    x=numeric_df[var],
                    y=numeric_df[Variable_y],
                    mode="markers",
                    name=f"{var} (real)",
                    marker=dict(
                        color=color,
                        symbol="circle",
                        size=9
                    )
                )
            )

            figure6.add_trace(
                go.Scatter(
                    x=numeric_df[var],
                    y=y_pred_M,
                    mode="markers",
                    name=f"{var} (predicho)",
                    marker=dict(
                        color=color,
                        symbol="x",
                        size=9
                    )
                )
            )

        figure6.update_layout(
            title="Modelo Lineal Múltiple",
            xaxis_title="Valor de X",
            yaxis_title=Variable_y,
            height=450,
            legend=dict(
                orientation="h",
                y=-0.25
            )
        )

        st.plotly_chart(
            estilo_grafica(figure6),
            use_container_width=True
        )
