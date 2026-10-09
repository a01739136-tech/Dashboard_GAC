
# ============================================================
# GAC | MARKETING
# Regresión Lineal Simple y Múltiple
# ============================================================

from pathlib import Path
from html import escape

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="GAC | Marketing",
    page_icon="📣",
    layout="wide",
    initial_sidebar_state="expanded"
)

CARPETA_BASE = Path(__file__).resolve().parent

BG = "#0E1117"
SIDEBAR = "#262730"
RED = "#EF3340"
RED_DARK = "#8F1D2C"
RED_MEDIUM = "#B52A3B"
RED_LIGHT = "#E88D98"
BURGUNDY = "#2B0D16"
WHITE = "#FFFFFF"
GRAY = "#9B9CA3"
GRID = "#30323A"

PREDICTORES = [
    "Leads Plaza",
    "Leads Digitales",
    "Leads Piso"
]

OBJETIVO = "Ventas Totales"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #0E1117;
    color: white;
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
    color: white;
}

.red-line {
    height: 3px;
    background: #EF3340;
    border-radius: 10px;
    margin-top: 12px;
    margin-bottom: 26px;
}

.dashboard-subtitle {
    color: #9B9CA3;
    font-size: 14px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: white;
    margin-bottom: 2px;
}

.section-subtitle {
    color: #9B9CA3;
    font-size: 13px;
    margin-bottom: 18px;
}

.kpi-card {
    background: linear-gradient(
        145deg,
        #2B0D16,
        #1B0A10
    );
    border: 1px solid #501421;
    border-radius: 13px;
    padding: 16px 18px;
    min-height: 125px;
    margin-bottom: 8px;
}

.kpi-title {
    color: #C9C9CD;
    font-size: 12px;
    margin-bottom: 7px;
}

.kpi-value {
    color: #FFFFFF;
    font-size: 27px;
    font-weight: 700;
    line-height: 1.15;
    overflow-wrap: anywhere;
}

.kpi-detail {
    color: #E88D98;
    font-size: 11px;
    margin-top: 8px;
}

.insight-card {
    background: #171A21;
    border: 1px solid #30323A;
    border-left: 4px solid #EF3340;
    border-radius: 10px;
    padding: 15px 18px;
    margin-top: 10px;
    margin-bottom: 15px;
}

.insight-title {
    color: white;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 7px;
}

.insight-text {
    color: #B8B9BE;
    font-size: 13px;
    line-height: 1.55;
}

.soft-divider {
    border-top: 1px solid #30323A;
    margin-top: 25px;
    margin-bottom: 25px;
}

div[data-baseweb="select"] > div {
    background-color: #2B0D16 !important;
    border-color: #501421 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background-color: #8F1D2C;
    color: white;
    border: 1px solid #B52A3B;
    border-radius: 9px;
    font-weight: 600;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #A62638;
    border-color: #EF3340;
    color: white;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# FUNCIONES VISUALES
# ============================================================

def tarjeta_kpi(titulo, valor, detalle):

    html = (
        '<div class="kpi-card">'
        f'<div class="kpi-title">{escape(str(titulo))}</div>'
        f'<div class="kpi-value">{escape(str(valor))}</div>'
        f'<div class="kpi-detail">{escape(str(detalle))}</div>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def tarjeta_insight(titulo, textos):

    parrafos = "".join(
        f"<p>{texto}</p>"
        for texto in textos
    )

    html = (
        '<div class="insight-card">'
        f'<div class="insight-title">{titulo}</div>'
        f'<div class="insight-text">{parrafos}</div>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def estilo_plotly(fig, altura=420):

    fig.update_layout(
        height=altura,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color=WHITE,
            size=11
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=45
        ),
        hoverlabel=dict(
            bgcolor="#1B1E26",
            font_color="white",
            bordercolor=RED
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5
        )
    )

    fig.update_xaxes(
        gridcolor="#252932",
        zerolinecolor="#3B3F49"
    )

    fig.update_yaxes(
        gridcolor="#252932",
        zerolinecolor="#3B3F49"
    )

    return fig


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data
def cargar_datos():

    ruta = (
        CARPETA_BASE
        / "Marketing_Regresion_Limpia.csv"
    )

    datos = pd.read_csv(ruta)

    datos.columns = (
        datos.columns
        .str.strip()
    )

    requeridas = (
        ["Fecha"]
        + PREDICTORES
        + [OBJETIVO]
    )

    faltantes = [
        columna
        for columna in requeridas
        if columna not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas: "
            + ", ".join(faltantes)
        )

    datos["Fecha"] = pd.to_datetime(
        datos["Fecha"],
        errors="coerce"
    )

    for columna in (
        PREDICTORES
        + [OBJETIVO]
    ):

        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        )

    datos = (
        datos
        .sort_values("Fecha")
        .reset_index(drop=True)
    )

    return datos


try:

    df = cargar_datos()

except Exception as error:

    st.error(
        f"No se pudo cargar la base: {error}"
    )

    st.stop()


if len(df) <= 6:

    st.warning(
        "Se necesitan más de seis meses "
        "para separar entrenamiento y prueba."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 📣 GAC | Marketing"
)

st.sidebar.caption(
    "Modelado de leads y ventas"
)

st.sidebar.markdown("---")


tipo_analisis = st.sidebar.radio(
    "Tipo de análisis",
    [
        "Vista general",
        "Regresión Lineal Simple",
        "Regresión Lineal Múltiple",
        "Comparación de Modelos",
        "Valores Atípicos",
        "Simulador"
    ],
    index=0
)


st.sidebar.markdown("---")


predictor_simple = st.sidebar.selectbox(
    "Variable X · Regresión Simple",
    options=PREDICTORES,
    index=0
)


st.sidebar.caption(
    "Los modelos se entrenan con los primeros "
    "meses y se evalúan con los últimos seis."
)


# ============================================================
# ENTRENAMIENTO Y PRUEBA
# ============================================================

entrenamiento = (
    df.iloc[:-6]
    .copy()
)

prueba = (
    df.iloc[-6:]
    .copy()
)

y_train = entrenamiento[OBJETIVO]
y_test = prueba[OBJETIVO]


# ============================================================
# REGRESIÓN SIMPLE
# ============================================================

modelo_simple = LinearRegression()

modelo_simple.fit(
    entrenamiento[
        [predictor_simple]
    ],
    y_train
)


pred_train_simple = (
    modelo_simple.predict(
        entrenamiento[
            [predictor_simple]
        ]
    )
)


pred_test_simple = (
    modelo_simple.predict(
        prueba[
            [predictor_simple]
        ]
    )
)


# ============================================================
# REGRESIÓN MÚLTIPLE
# ============================================================

modelo_multiple = LinearRegression()

modelo_multiple.fit(
    entrenamiento[PREDICTORES],
    y_train
)


pred_train_multiple = (
    modelo_multiple.predict(
        entrenamiento[PREDICTORES]
    )
)


pred_test_multiple = (
    modelo_multiple.predict(
        prueba[PREDICTORES]
    )
)


# ============================================================
# REFERENCIA
# ============================================================

ultima_venta = float(
    y_train.iloc[-1]
)

pred_referencia = np.full(
    len(prueba),
    ultima_venta
)


# ============================================================
# MÉTRICAS
# ============================================================

def evaluar_modelo(
    nombre,
    reales,
    predicciones
):

    return {
        "Modelo": nombre,

        "R²":
            r2_score(
                reales,
                predicciones
            ),

        "MAE":
            mean_absolute_error(
                reales,
                predicciones
            ),

        "RMSE":
            np.sqrt(
                mean_squared_error(
                    reales,
                    predicciones
                )
            )
    }


metricas_simple_train = evaluar_modelo(
    "Entrenamiento",
    y_train,
    pred_train_simple
)

metricas_simple_test = evaluar_modelo(
    "Prueba",
    y_test,
    pred_test_simple
)


metricas_multiple_train = evaluar_modelo(
    "Entrenamiento",
    y_train,
    pred_train_multiple
)

metricas_multiple_test = evaluar_modelo(
    "Prueba",
    y_test,
    pred_test_multiple
)


comparacion = pd.DataFrame([

    evaluar_modelo(
        "Lineal simple",
        y_test,
        pred_test_simple
    ),

    evaluar_modelo(
        "Lineal múltiple",
        y_test,
        pred_test_multiple
    ),

    evaluar_modelo(
        "Último mes conocido",
        y_test,
        pred_referencia
    )

])


# ============================================================
# CORRELACIONES
# ============================================================

correlaciones = (
    df[
        PREDICTORES
        + [OBJETIVO]
    ]
    .corr(method="pearson")
)


correlacion_ventas = (
    correlaciones[OBJETIVO]
    .drop(OBJETIVO)
)


canal_principal = (
    correlacion_ventas
    .abs()
    .idxmax()
)


valor_correlacion = (
    correlacion_ventas[
        canal_principal
    ]
)


mejor_modelo = (
    comparacion.loc[
        comparacion["MAE"].idxmin()
    ]
)


# ============================================================
# HEADER
# ============================================================

col_logo, col_titulo = st.columns(
    [1.2, 5],
    vertical_alignment="center"
)


with col_logo:

    ruta_logo = (
        CARPETA_BASE
        / "logo_gac.png"
    )

    if ruta_logo.exists():

        st.image(
            str(ruta_logo),
            width=125
        )


with col_titulo:

    st.markdown(
        (
            '<h1 style="'
            'margin-bottom:4px;'
            'font-size:48px;'
            'font-weight:700;'
            '">'
            'Marketing'
            '</h1>'
        ),
        unsafe_allow_html=True
    )

    st.markdown(
        (
            '<div class="dashboard-subtitle">'
            'Regresión lineal aplicada a leads por canal y ventas mensuales'
            '</div>'
        ),
        unsafe_allow_html=True
    )


st.markdown(
    '<div class="red-line"></div>',
    unsafe_allow_html=True
)


if tipo_analisis == "Vista general":
    # ============================================================
    # KPIs GENERALES
    # ============================================================

    k1, k2, k3 = st.columns(
        3,
        gap="small"
    )


    with k1:

        tarjeta_kpi(
            "Meses analizados",
            len(df),
            (
                f"{len(entrenamiento)} entrenamiento · "
                f"{len(prueba)} prueba"
            )
        )


    with k2:

        tarjeta_kpi(
            "Mayor asociación con ventas",
            canal_principal,
            f"Pearson: {valor_correlacion:.3f}"
        )


    with k3:

        tarjeta_kpi(
            "Menor error en prueba",
            mejor_modelo["Modelo"],
            f"MAE: {mejor_modelo['MAE']:.2f} ventas"
        )


    st.markdown(
        '<div class="soft-divider"></div>',
        unsafe_allow_html=True
    )


# ============================================================
# VISTA GENERAL
# ============================================================

if tipo_analisis == "Vista general":

    st.markdown(
        (
            '<div class="section-title">'
            'Comparación de modelos'
            '</div>'
            '<div class="section-subtitle">'
            'Desempeño de las regresiones durante los seis meses de prueba'
            '</div>'
        ),
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # GRÁFICA ESTRELLA
    # --------------------------------------------------------

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=prueba["Fecha"],
            y=y_test,
            mode="lines+markers",
            name="Ventas reales",
            line=dict(
                color=WHITE,
                width=3
            ),
            marker=dict(
                size=8
            )
        )
    )


    fig.add_trace(
        go.Scatter(
            x=prueba["Fecha"],
            y=pred_test_simple,
            mode="lines+markers",
            name="Regresión simple",
            line=dict(
                color=RED_LIGHT,
                width=2,
                dash="dash"
            )
        )
    )


    fig.add_trace(
        go.Scatter(
            x=prueba["Fecha"],
            y=pred_test_multiple,
            mode="lines+markers",
            name="Regresión múltiple",
            line=dict(
                color=RED,
                width=3
            )
        )
    )


    fig.add_trace(
        go.Scatter(
            x=prueba["Fecha"],
            y=pred_referencia,
            mode="lines",
            name="Último mes conocido",
            line=dict(
                color=GRAY,
                width=2,
                dash="dot"
            )
        )
    )


    fig.update_layout(
        title="Ventas reales vs. estimaciones",
        xaxis_title="Mes de prueba",
        yaxis_title="Ventas"
    )


    fig.update_xaxes(
        tickformat="%m/%Y"
    )


    st.plotly_chart(
        estilo_plotly(
            fig,
            460
        ),
        use_container_width=True
    )


    st.dataframe(
        comparacion.round(3),
        hide_index=True,
        use_container_width=True
    )


    tarjeta_insight(
        "Lectura ejecutiva",
        [
            (
                f'El menor MAE durante el periodo de prueba '
                f'corresponde a <b>{mejor_modelo["Modelo"]}</b>.'
            ),
            (
                'La comparación se realiza sobre exactamente '
                'los mismos seis meses para los tres enfoques.'
            )
        ]
    )


# ============================================================
# REGRESIÓN SIMPLE
# ============================================================

elif tipo_analisis == "Regresión Lineal Simple":

    st.markdown(
        (
            '<div class="section-title">'
            'Regresión Lineal Simple'
            '</div>'
            '<div class="section-subtitle">'
            f'Ventas Totales explicadas por {predictor_simple}'
            '</div>'
        ),
        unsafe_allow_html=True
    )


    m1, m2, m3 = st.columns(3)


    with m1:

        tarjeta_kpi(
            "R² · Prueba",
            f"{metricas_simple_test['R²']:.3f}",
            "Capacidad explicativa"
        )


    with m2:

        tarjeta_kpi(
            "MAE · Prueba",
            f"{metricas_simple_test['MAE']:.2f}",
            "Error promedio"
        )


    with m3:

        tarjeta_kpi(
            "RMSE · Prueba",
            f"{metricas_simple_test['RMSE']:.2f}",
            "Error típico"
        )


    # --------------------------------------------------------
    # GRÁFICA SIMPLE
    # --------------------------------------------------------

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=entrenamiento[
                predictor_simple
            ],
            y=y_train,
            mode="markers",
            name="Real · Entrenamiento",
            marker=dict(
                color=GRAY,
                size=8
            )
        )
    )


    fig.add_trace(
        go.Scatter(
            x=prueba[
                predictor_simple
            ],
            y=y_test,
            mode="markers",
            name="Real · Prueba",
            marker=dict(
                color=RED_LIGHT,
                size=11
            )
        )
    )


    fig.add_trace(
        go.Scatter(
            x=prueba[
                predictor_simple
            ],
            y=pred_test_simple,
            mode="markers",
            name="Predicción · Prueba",
            marker=dict(
                color=RED,
                size=12,
                symbol="x"
            )
        )
    )


    valores_x = pd.DataFrame({

        predictor_simple:
            np.linspace(
                df[predictor_simple].min(),
                df[predictor_simple].max(),
                100
            )
    })


    fig.add_trace(
        go.Scatter(
            x=valores_x[
                predictor_simple
            ],
            y=modelo_simple.predict(
                valores_x
            ),
            mode="lines",
            name="Recta estimada",
            line=dict(
                color=RED_MEDIUM,
                width=3
            )
        )
    )


    fig.update_layout(
        title=(
            f"{predictor_simple} vs. Ventas Totales"
        ),
        xaxis_title=predictor_simple,
        yaxis_title="Ventas Totales"
    )


    st.plotly_chart(
        estilo_plotly(
            fig,
            470
        ),
        use_container_width=True
    )


    beta = (
        modelo_simple.coef_[0]
    )


    intercepto = (
        modelo_simple.intercept_
    )


    tarjeta_insight(
        "Ecuación estimada",
        [
            (
                f'Ventas estimadas = '
                f'<b>{intercepto:.3f}</b> '
                f'{beta:+.4f} × {predictor_simple}'
            ),
            (
                f'Un lead adicional en <b>{predictor_simple}</b> '
                f'se asocia con un cambio estimado de '
                f'<b>{beta:.4f} ventas</b>.'
            )
        ]
    )


    evaluacion_simple = pd.DataFrame([
        metricas_simple_train,
        metricas_simple_test
    ])


    st.dataframe(
        evaluacion_simple.round(3),
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# REGRESIÓN MÚLTIPLE
# ============================================================

elif tipo_analisis == "Regresión Lineal Múltiple":

    st.markdown(
        (
            '<div class="section-title">'
            'Regresión Lineal Múltiple'
            '</div>'
            '<div class="section-subtitle">'
            'Estimación simultánea utilizando los tres canales de leads'
            '</div>'
        ),
        unsafe_allow_html=True
    )


    m1, m2, m3 = st.columns(3)


    with m1:

        tarjeta_kpi(
            "R² · Prueba",
            f"{metricas_multiple_test['R²']:.3f}",
            "Capacidad explicativa"
        )


    with m2:

        tarjeta_kpi(
            "MAE · Prueba",
            f"{metricas_multiple_test['MAE']:.2f}",
            "Error promedio"
        )


    with m3:

        tarjeta_kpi(
            "RMSE · Prueba",
            f"{metricas_multiple_test['RMSE']:.2f}",
            "Error típico"
        )


    canal_visual = st.selectbox(
        "Canal para visualizar",
        options=PREDICTORES,
        index=0
    )


    # --------------------------------------------------------
    # GRÁFICA MÚLTIPLE
    # --------------------------------------------------------

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=prueba[
                canal_visual
            ],
            y=y_test,
            mode="markers",
            name="Ventas reales",
            marker=dict(
                color=RED_LIGHT,
                size=12
            ),
            text=(
                prueba["Fecha"]
                .dt.strftime("%m/%Y")
            ),
            hovertemplate=(
                "Mes: %{text}<br>"
                "Leads: %{x}<br>"
                "Ventas reales: %{y}"
                "<extra></extra>"
            )
        )
    )


    fig.add_trace(
        go.Scatter(
            x=prueba[
                canal_visual
            ],
            y=pred_test_multiple,
            mode="markers",
            name="Ventas predichas",
            marker=dict(
                color=RED,
                size=13,
                symbol="x"
            ),
            text=(
                prueba["Fecha"]
                .dt.strftime("%m/%Y")
            ),
            hovertemplate=(
                "Mes: %{text}<br>"
                "Leads: %{x}<br>"
                "Ventas predichas: %{y:.2f}"
                "<extra></extra>"
            )
        )
    )


    fig.update_layout(
        title="Ventas reales y estimadas por el modelo múltiple",
        xaxis_title=canal_visual,
        yaxis_title="Ventas Totales"
    )


    st.plotly_chart(
        estilo_plotly(
            fig,
            450
        ),
        use_container_width=True
    )


    st.caption(
        "Cada predicción utiliza simultáneamente Leads Plaza, "
        "Leads Digitales y Leads Piso. El eje X muestra únicamente "
        "el canal seleccionado para facilitar la visualización."
    )


    # --------------------------------------------------------
    # COEFICIENTES
    # --------------------------------------------------------

    coeficientes = pd.DataFrame({

        "Variable":
            PREDICTORES,

        "Coeficiente":
            modelo_multiple.coef_
    })


    st.markdown("#### Coeficientes del modelo")


    st.dataframe(
        coeficientes.round(4),
        hide_index=True,
        use_container_width=True
    )


    ecuacion = (
        f"Ventas estimadas = "
        f"{modelo_multiple.intercept_:.3f}"
    )


    for variable, coeficiente in zip(
        PREDICTORES,
        modelo_multiple.coef_
    ):

        ecuacion += (
            f" {coeficiente:+.4f} × {variable}"
        )


    tarjeta_insight(
        "Ecuación del modelo",
        [
            ecuacion,
            (
                'Cada coeficiente representa el cambio estimado '
                'en ventas manteniendo constantes los otros canales.'
            )
        ]
    )


    evaluacion_multiple = pd.DataFrame([
        metricas_multiple_train,
        metricas_multiple_test
    ])


    st.dataframe(
        evaluacion_multiple.round(3),
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# COMPARACIÓN DE MODELOS
# ============================================================

elif tipo_analisis == "Comparación de Modelos":

    st.markdown(
        (
            '<div class="section-title">'
            'Comparación de Modelos'
            '</div>'
            '<div class="section-subtitle">'
            'Comparación temporal durante los seis meses reservados para prueba'
            '</div>'
        ),
        unsafe_allow_html=True
    )


    st.dataframe(
        comparacion.round(3),
        hide_index=True,
        use_container_width=True
    )


    fig = go.Figure()


    series = [

        (
            "Ventas reales",
            y_test,
            WHITE,
            "solid"
        ),

        (
            "Regresión simple",
            pred_test_simple,
            RED_LIGHT,
            "dash"
        ),

        (
            "Regresión múltiple",
            pred_test_multiple,
            RED,
            "solid"
        ),

        (
            "Último mes conocido",
            pred_referencia,
            GRAY,
            "dot"
        )
    ]


    for (
        nombre,
        valores,
        color,
        dash
    ) in series:

        fig.add_trace(
            go.Scatter(
                x=prueba["Fecha"],
                y=valores,
                mode="lines+markers",
                name=nombre,
                line=dict(
                    color=color,
                    width=(
                        3
                        if nombre
                        in [
                            "Ventas reales",
                            "Regresión múltiple"
                        ]
                        else 2
                    ),
                    dash=dash
                )
            )
        )


    fig.update_layout(
        title="¿Qué tan cerca están las estimaciones de las ventas reales?",
        xaxis_title="Mes de prueba",
        yaxis_title="Ventas"
    )


    fig.update_xaxes(
        tickformat="%m/%Y"
    )


    st.plotly_chart(
        estilo_plotly(
            fig,
            480
        ),
        use_container_width=True
    )


    tarjeta_insight(
        "Resultado",
        [
            (
                f'El menor MAE corresponde a '
                f'<b>{mejor_modelo["Modelo"]}</b>, '
                f'con <b>{mejor_modelo["MAE"]:.2f} ventas</b>.'
            ),
            (
                'MAE y RMSE: mientras menor sea el valor, '
                'menor es el error de predicción.'
            )
        ]
    )


# ============================================================
# VALORES ATÍPICOS
# ============================================================

elif tipo_analisis == "Valores Atípicos":

    st.markdown(
        (
            '<div class="section-title">'
            'Valores Atípicos'
            '</div>'
            '<div class="section-subtitle">'
            'Revisión exploratoria mediante boxplot y análisis IQR'
            '</div>'
        ),
        unsafe_allow_html=True
    )


    variable_box = st.selectbox(
        "Variable para revisar",
        options=(
            PREDICTORES
            + [OBJETIVO]
        )
    )


    fig = px.box(
        df,
        y=variable_box,
        points="all",
        hover_data=["Fecha"]
    )


    fig.update_traces(
        marker_color=RED,
        line_color=RED_LIGHT
    )


    fig.update_layout(
        title=f"Distribución mensual de {variable_box}",
        yaxis_title=variable_box
    )


    st.plotly_chart(
        estilo_plotly(
            fig,
            420
        ),
        use_container_width=True
    )


    # --------------------------------------------------------
    # IQR SOBRE LEADS PISO
    # --------------------------------------------------------

    q1 = (
        entrenamiento[
            "Leads Piso"
        ]
        .quantile(0.25)
    )

    q3 = (
        entrenamiento[
            "Leads Piso"
        ]
        .quantile(0.75)
    )

    iqr = q3 - q1

    inferior = (
        q1
        - 1.5 * iqr
    )

    superior = (
        q3
        + 1.5 * iqr
    )


    atipicos = (

        (
            entrenamiento[
                "Leads Piso"
            ]
            < inferior
        )

        |

        (
            entrenamiento[
                "Leads Piso"
            ]
            > superior
        )
    )


    tabla_atipicos = (
        entrenamiento.loc[
            atipicos,
            [
                "Fecha",
                "Leads Piso",
                OBJETIVO
            ]
        ]
    )


    st.markdown(
        "#### Meses señalados en Leads Piso"
    )


    st.dataframe(
        tabla_atipicos,
        hide_index=True,
        use_container_width=True
    )


    entrenamiento_sin = (
        entrenamiento.loc[
            ~atipicos
        ]
        .copy()
    )


    if len(
        entrenamiento_sin
    ) > len(PREDICTORES):

        modelo_sensibilidad = (
            LinearRegression()
        )

        modelo_sensibilidad.fit(
            entrenamiento_sin[
                PREDICTORES
            ],
            entrenamiento_sin[
                OBJETIVO
            ]
        )


        pred_sensibilidad = (
            modelo_sensibilidad.predict(
                prueba[PREDICTORES]
            )
        )


        sensibilidad = pd.DataFrame([

            evaluar_modelo(
                "Múltiple con todos los meses",
                y_test,
                pred_test_multiple
            ),

            evaluar_modelo(
                "Múltiple sin atípicos",
                y_test,
                pred_sensibilidad
            )

        ])


        st.markdown(
            "#### Sensibilidad del modelo"
        )


        st.dataframe(
            sensibilidad.round(3),
            hide_index=True,
            use_container_width=True
        )


    tarjeta_insight(
        "Importante",
        [
            (
                'Los registros señalados por IQR se conservan '
                'en la base original.'
            ),
            (
                'Ser identificado como valor atípico no significa '
                'automáticamente que exista un error de captura.'
            )
        ]
    )


# ============================================================
# SIMULADOR
# ============================================================

elif tipo_analisis == "Simulador":

    st.markdown(
        (
            '<div class="section-title">'
            'Simulador de Ventas'
            '</div>'
            '<div class="section-subtitle">'
            'Explora escenarios utilizando la regresión múltiple'
            '</div>'
        ),
        unsafe_allow_html=True
    )


    st.caption(
        "Introduce un escenario de leads para los tres canales."
    )


    entradas = {}


    with st.form(
        "simulador_marketing"
    ):

        c1, c2, c3 = st.columns(3)


        for columna, variable in zip(
            [c1, c2, c3],
            PREDICTORES
        ):

            with columna:

                entradas[
                    variable
                ] = st.number_input(

                    variable,

                    min_value=0,

                    value=int(
                        entrenamiento[
                            variable
                        ]
                        .median()
                    ),

                    step=1
                )


        calcular = (
            st.form_submit_button(
                "Calcular ventas estimadas"
            )
        )


    if calcular:

        escenario = pd.DataFrame(
            [entradas],
            columns=PREDICTORES
        )


        estimacion = float(
            modelo_multiple.predict(
                escenario
            )[0]
        )


        c_resultado, c_info = (
            st.columns(
                [1, 2]
            )
        )


        with c_resultado:

            tarjeta_kpi(
                "Ventas estimadas",
                f"{estimacion:.1f}",
                "Regresión múltiple"
            )


        with c_info:

            fuera_rango = [

                variable

                for variable
                in PREDICTORES

                if (
                    entradas[variable]
                    <
                    entrenamiento[
                        variable
                    ].min()

                    or

                    entradas[variable]
                    >
                    entrenamiento[
                        variable
                    ].max()
                )
            ]


            if fuera_rango:

                st.warning(
                    "El escenario contiene valores "
                    "fuera del rango de entrenamiento en: "
                    + ", ".join(
                        fuera_rango
                    )
                    + "."
                )

            else:

                st.success(
                    "El escenario se encuentra dentro "
                    "de los rangos observados durante "
                    "el entrenamiento."
                )


        tarjeta_insight(
            "Interpretación",
            [
                (
                    'La cifra es una estimación exploratoria '
                    'generada a partir de los tres canales.'
                ),
                (
                    'No representa una garantía de ventas futuras '
                    'ni demuestra causalidad.'
                )
            ]
        )


# ============================================================
# INFORMACIÓN DE LA BASE
# ============================================================

st.markdown(
    '<div class="soft-divider"></div>',
    unsafe_allow_html=True
)


with st.expander(
    "📋 Consultar base utilizada"
):

    st.write(
        "El modelo utiliza Leads Plaza, Leads Digitales "
        "y Leads Piso para analizar las Ventas Totales."
    )


    st.write(
        "Los últimos seis meses se reservan como conjunto "
        "de prueba y no participan en el entrenamiento."
    )


    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )
