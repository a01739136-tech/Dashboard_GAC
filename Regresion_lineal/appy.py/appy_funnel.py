
# ============================================================
# LIBRERÍAS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================

st.set_page_config(
    page_title="GAC | Funnel de Ventas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PALETA GAC
# ============================================================

BG = "#0E1117"
SIDEBAR = "#262730"

RED = "#EF3340"
RED_DARK = "#8F1D2C"
RED_MEDIUM = "#B52A3B"
RED_LIGHT = "#E88D98"

WHITE = "#FFFFFF"
GRAY = "#9B9CA3"
GRID = "#30323A"


# ============================================================
# CSS GENERAL
# ============================================================

st.markdown("""
<style>

/* =========================================================
   FONDO GENERAL
========================================================= */

.stApp {
    background-color: #0E1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================================================
   SIDEBAR
========================================================= */

[data-testid="stSidebar"] {
    background-color: #262730;
    border-right: 1px solid #383A45;
}

[data-testid="stSidebar"] * {
    color: white;
}


/* =========================================================
   SELECTBOX
========================================================= */

div[data-baseweb="select"] > div {
    background-color: #8F1D2C !important;
    border: 1px solid #B52A3B !important;
    border-radius: 9px !important;
}

div[data-baseweb="select"] span {
    color: white !important;
}

div[data-baseweb="select"] svg {
    fill: white !important;
}

div[data-baseweb="select"] > div:hover {
    background-color: #A62638 !important;
    border-color: #D13A4B !important;
}


/* =========================================================
   MULTISELECT
========================================================= */

[data-baseweb="tag"] {
    background-color: #EF3340 !important;
    color: white !important;
    border-radius: 5px !important;
}


/* =========================================================
   TARJETAS KPI
========================================================= */

.metric-card {
    background: linear-gradient(
        145deg,
        #2B0D16,
        #210A11
    );

    border: 1px solid #501421;
    border-radius: 12px;

    padding: 13px 16px;
    min-height: 88px;

    box-shadow:
        0px 4px 14px rgba(0,0,0,0.16);
}

.metric-title {
    color: #C8C8CC;
    font-size: 12px;
    margin-bottom: 4px;
}

.metric-value {
    color: #FFFFFF;
    font-size: 24px;
    font-weight: 700;
    line-height: 1.15;
}

.metric-detail {
    color: #9B9CA3;
    font-size: 11px;
    margin-top: 4px;
}


/* =========================================================
   TARJETAS DE INTERPRETACIÓN
========================================================= */

.insight-card {
    background-color: #171A21;

    border-left: 4px solid #EF3340;
    border-radius: 10px;

    padding: 17px 19px;

    margin-top: 10px;
    margin-bottom: 10px;

    box-shadow:
        0px 4px 14px rgba(0,0,0,0.15);
}

.insight-title {
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 10px;
}

.insight-text {
    color: #B8B8BD;
    font-size: 14px;
    line-height: 1.55;
}

.insight-text p {
    margin-top: 0px;
    margin-bottom: 10px;
}

.insight-text p:last-child {
    margin-bottom: 0px;
}


/* =========================================================
   HEADER
========================================================= */

.red-line {
    height: 3px;
    background: #E51B2A;
    border-radius: 10px;

    margin-top: 12px;
    margin-bottom: 26px;
}

.dashboard-subtitle {
    color: #9B9CA3;
    font-size: 14px;
}


/* =========================================================
   SECCIONES
========================================================= */

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: white;

    margin-bottom: 2px;
}

.section-subtitle {
    color: #9B9CA3;
    font-size: 13px;
    margin-bottom: 15px;
}


/* =========================================================
   DIVISORES
========================================================= */

.soft-divider {
    border-top: 1px solid #30323A;

    margin-top: 25px;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Funnel_limpia.csv")

    return df


df = load_data()


# ============================================================
# VARIABLE DEPENDIENTE
# ============================================================

VARIABLE_Y = "ventas"


# ============================================================
# VARIABLES EXPLICATIVAS
# ============================================================

VARIABLES_X = {

    "Leads generados":
        "leads_generados",

    "Leads efectivos":
        "leads_efectivos",

    "Cita programada":
        "cita_programada",

    "Cita efectiva":
        "cita_efectiva",

    "Prueba de manejo":
        "prueba_de_manejo",

    "Solicitud de crédito generada":
        "solicitud_de_credito_generada",

    "Solicitud de crédito aprobada":
        "solicitud_de_credito_aprobada"
}


# ============================================================
# MODELO SIMPLE
# ============================================================

def crear_modelo_simple(data, x):

    datos = data[
        [VARIABLE_Y, x]
    ].dropna()

    X = sm.add_constant(
        datos[[x]].astype(float)
    )

    y = datos[
        VARIABLE_Y
    ].astype(float)

    modelo = sm.OLS(
        y,
        X
    ).fit()

    pred = modelo.predict(X)

    rmse = np.sqrt(
        mean_squared_error(
            y,
            pred
        )
    )

    return modelo, datos, pred, rmse


# ============================================================
# MODELO MÚLTIPLE
# ============================================================

def crear_modelo_multiple(data, variables_x):

    columnas = [
        VARIABLE_Y
    ] + variables_x

    datos = data[
        columnas
    ].dropna()

    X = datos[
        variables_x
    ].astype(float)

    X_const = sm.add_constant(X)

    y = datos[
        VARIABLE_Y
    ].astype(float)

    modelo = sm.OLS(
        y,
        X_const
    ).fit()

    pred = modelo.predict(
        X_const
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            pred
        )
    )

    return modelo, datos, pred, rmse


# ============================================================
# VIF
# ============================================================

def calcular_vif(data, variables):

    if len(variables) < 2:

        return pd.DataFrame({
            "Variable": variables,
            "VIF": [1.0] * len(variables)
        })

    X = data[
        variables
    ].dropna().astype(float)

    resultado = []

    for i, variable in enumerate(variables):

        try:

            vif = variance_inflation_factor(
                X.values,
                i
            )

        except:

            vif = np.nan

        resultado.append({
            "Variable": variable,
            "VIF": vif
        })

    return pd.DataFrame(
        resultado
    )


# ============================================================
# TARJETA KPI
# ============================================================

def tarjeta_kpi(titulo, valor, detalle=""):

    html = (
        f'<div class="metric-card">'
        f'<div class="metric-title">{titulo}</div>'
        f'<div class="metric-value">{valor}</div>'
        f'<div class="metric-detail">{detalle}</div>'
        f'</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# TARJETA DE INSIGHT
# ============================================================

def tarjeta_insight(titulo, parrafos):

    # Si llega solamente un texto,
    # lo convertimos automáticamente en lista
    if isinstance(parrafos, str):

        parrafos = [parrafos]

    # Construimos cada párrafo de forma independiente
    contenido = "".join(

        f'<p>{parrafo}</p>'

        for parrafo in parrafos
    )

    # IMPORTANTE:
    # todo el HTML se construye sin indentaciones internas
    # para evitar que Streamlit lo interprete como código.
    html = (
        f'<div class="insight-card">'
        f'<div class="insight-title">{titulo}</div>'
        f'<div class="insight-text">{contenido}</div>'
        f'</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# ESTILO GENERAL DE PLOTLY
# ============================================================

def estilo_plotly(fig):

    fig.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color=WHITE,
            size=12
        ),

        margin=dict(
            l=20,
            r=20,
            t=45,
            b=20
        ),

        hoverlabel=dict(
            bgcolor="#1B1E26",
            font_color="white",
            bordercolor=RED
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
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 📊 GAC | Funnel"
)

st.sidebar.caption(
    "Modelado explicativo de ventas"
)

st.sidebar.markdown("---")


# ============================================================
# TIPO DE ANÁLISIS
# ============================================================

tipo_modelo = st.sidebar.radio(

    "Tipo de análisis",

    [
        "Ambos",
        "Regresión Lineal Simple",
        "Regresión Lineal Múltiple"
    ],

    index=0
)


# ============================================================
# FILTRO DE AÑO
# ============================================================

anios = sorted(
    df["anio"]
    .dropna()
    .unique()
    .tolist()
)


anio_seleccionado = st.sidebar.selectbox(

    "Año",

    options=[
        "Todos"
    ] + anios,

    index=0
)


# ============================================================
# FILTRO DE CANAL
# ============================================================

canales = sorted(
    df["canal"]
    .dropna()
    .unique()
    .tolist()
)


canal_seleccionado = st.sidebar.selectbox(

    "Canal",

    options=[
        "Todos"
    ] + canales,

    index=0
)


# ============================================================
# APLICACIÓN DE FILTROS
# ============================================================

df_filtrado = df.copy()


if anio_seleccionado != "Todos":

    df_filtrado = df_filtrado[
        df_filtrado["anio"]
        == anio_seleccionado
    ].copy()


if canal_seleccionado != "Todos":

    df_filtrado = df_filtrado[
        df_filtrado["canal"]
        == canal_seleccionado
    ].copy()


# ============================================================
# CONFIGURACIÓN DE VARIABLES
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Configuración del modelo"
)


# ============================================================
# VARIABLE X SIMPLE
# ============================================================

if tipo_modelo in [
    "Ambos",
    "Regresión Lineal Simple"
]:

    x_simple_nombre = st.sidebar.selectbox(

        "Variable X · Regresión Simple",

        options=list(
            VARIABLES_X.keys()
        ),

        index=1
    )

    x_simple = VARIABLES_X[
        x_simple_nombre
    ]


# ============================================================
# VARIABLES X MÚLTIPLE
# ============================================================

if tipo_modelo in [
    "Ambos",
    "Regresión Lineal Múltiple"
]:

    default_multiple = [
        "Leads efectivos",
        "Cita efectiva",
        "Prueba de manejo"
    ]

    x_multiple_nombres = st.sidebar.multiselect(

        "Variables X · Regresión Múltiple",

        options=list(
            VARIABLES_X.keys()
        ),

        default=default_multiple
    )

    x_multiple = [

        VARIABLES_X[nombre]

        for nombre
        in x_multiple_nombres
    ]


# ============================================================
# HEADER
# ============================================================

col_logo, col_titulo = st.columns(
    [1.2, 5],
    vertical_alignment="center"
)


with col_logo:

    st.image(
        "logo_gac.png",
        width=125
    )


with col_titulo:

    st.markdown(
        """
        <h1 style="
            margin-bottom:4px;
            font-size:48px;
            font-weight:700;
        ">
        Funnel de Ventas
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-subtitle">
        Análisis explicativo mediante regresión lineal simple y múltiple
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    '<div class="red-line"></div>',
    unsafe_allow_html=True
)


# ============================================================
# VALIDACIÓN GENERAL
# ============================================================

if len(df_filtrado) < 3:

    st.warning(
        "La selección actual contiene muy pocas observaciones para estimar una regresión."
    )

    st.stop()


# ============================================================
# GRÁFICA DE REGRESIÓN SIMPLE
# ============================================================

def grafica_simple(
    data,
    modelo,
    variable_x,
    nombre_x
):

    datos_grafica = data.copy()

    datos_grafica["Predicción"] = (
        modelo.predict(
            sm.add_constant(
                datos_grafica[
                    [variable_x]
                ].astype(float)
            )
        )
    )


    # --------------------------------------------------------
    # PUNTOS
    # --------------------------------------------------------

    fig = px.scatter(

        datos_grafica,

        x=variable_x,

        y="ventas",

        hover_data={
            "ventas": True,
            variable_x: True
        },

        labels={
            variable_x: nombre_x,
            "ventas": "Ventas"
        }
    )


    fig.update_traces(

        marker=dict(
            size=11,
            color=RED_MEDIUM,
            opacity=0.78,

            line=dict(
                width=1,
                color="#F4A6AF"
            )
        )
    )


    # --------------------------------------------------------
    # LÍNEA ESTIMADA
    # --------------------------------------------------------

    linea = (
        datos_grafica[
            [variable_x, "Predicción"]
        ]
        .sort_values(variable_x)
    )


    fig.add_trace(

        go.Scatter(

            x=linea[
                variable_x
            ],

            y=linea[
                "Predicción"
            ],

            mode="lines",

            name="Ajuste estimado",

            line=dict(
                color=RED,
                width=3
            )
        )
    )


    fig.update_layout(

        title=dict(
            text=f"{nombre_x} vs Ventas",
            font=dict(
                size=19
            )
        ),

        height=430,

        showlegend=True,

        legend=dict(
            orientation="h",
            y=1.08,
            x=0
        )
    )


    return estilo_plotly(
        fig
    )


# ============================================================
# MEDIDOR DE R²
# ============================================================

def gauge_r2(
    r2,
    titulo="R² del modelo"
):

    valor = max(
        0,
        min(
            1,
            r2
        )
    ) * 100


    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=valor,

            number={
                "suffix": "%",
                "font": {
                    "size": 30,
                    "color": "white"
                }
            },

            title={
                "text": titulo,
                "font": {
                    "size": 15,
                    "color": "#C8C8CC"
                }
            },

            gauge={

                "axis": {
                    "range": [
                        0,
                        100
                    ],

                    "tickcolor":
                        "#888"
                },

                "bar": {
                    "color":
                        RED
                },

                "bgcolor":
                    "#171A21",

                "borderwidth":
                    0,

                "steps": [

                    {
                        "range": [
                            0,
                            25
                        ],

                        "color":
                            "#321018"
                    },

                    {
                        "range": [
                            25,
                            50
                        ],

                        "color":
                            "#501421"
                    },

                    {
                        "range": [
                            50,
                            75
                        ],

                        "color":
                            "#70202D"
                    },

                    {
                        "range": [
                            75,
                            100
                        ],

                        "color":
                            "#8F1D2C"
                    }
                ]
            }
        )
    )


    fig.update_layout(

        height=300,

        paper_bgcolor=
            "rgba(0,0,0,0)",

        font=dict(
            color="white"
        ),

        margin=dict(
            l=30,
            r=30,
            t=60,
            b=20
        )
    )


    return fig


# ============================================================
# GRÁFICA DE COEFICIENTES
# ============================================================

def grafica_coeficientes(
    modelo
):

    coeficientes = (
        modelo.params
        .drop(
            "const",
            errors="ignore"
        )
    )


    conf = (
        modelo.conf_int()
        .drop(
            "const",
            errors="ignore"
        )
    )


    tabla = pd.DataFrame({

        "Variable":
            coeficientes.index,

        "Coeficiente":
            coeficientes.values,

        "Inferior":
            conf[0].values,

        "Superior":
            conf[1].values
    })


    mapa_nombres = {

        valor: clave

        for clave, valor
        in VARIABLES_X.items()
    }


    tabla["Nombre"] = (

        tabla["Variable"]
        .map(
            mapa_nombres
        )
        .fillna(
            tabla["Variable"]
        )
    )


    error_plus = (

        tabla["Superior"]

        - tabla["Coeficiente"]
    )


    error_minus = (

        tabla["Coeficiente"]

        - tabla["Inferior"]
    )


    fig = go.Figure()


    fig.add_trace(

        go.Scatter(

            x=tabla[
                "Coeficiente"
            ],

            y=tabla[
                "Nombre"
            ],

            mode="markers",

            marker=dict(

                size=13,

                color=["#EF3340", "#FFFFFF", "#9B9CA3", "#E88D98", "#B52A3B", "#D4D4D8", "#F36B75"][:len(tabla)], # GAC_MULTICOLOR_FUNNEL

                line=dict(
                    color="#F3A4AD",
                    width=1
                )
            ),

            error_x=dict(

                type="data",

                symmetric=False,

                array=
                    error_plus,

                arrayminus=
                    error_minus,

                color=
                    RED_LIGHT,

                thickness=2,

                width=6
            ),

            hovertemplate=(
                "<b>%{y}</b><br>"
                "Coeficiente: %{x:.4f}"
                "<extra></extra>"
            )
        )
    )


    fig.add_vline(

        x=0,

        line_width=1,

        line_dash="dash",

        line_color="#A5A5AA"
    )


    fig.update_layout(

        title=
            "Impacto estimado de las variables",

        xaxis_title=
            "Coeficiente estimado",

        yaxis_title="",

        height=430,

        showlegend=False
    )


    return estilo_plotly(
        fig
    )


# ============================================================
# VISTA 1: AMBOS MODELOS
# ============================================================

if tipo_modelo == "Ambos":


    # ========================================================
    # REGRESIÓN SIMPLE
    # ========================================================

    (
        modelo_simple,
        datos_simple,
        pred_simple,
        rmse_simple
    ) = crear_modelo_simple(

        df_filtrado,

        x_simple
    )


    # ========================================================
    # REGRESIÓN MÚLTIPLE
    # ========================================================

    modelo_multiple_valido = (
        len(x_multiple) >= 2
    )


    if modelo_multiple_valido:

        (
            modelo_multiple,
            datos_multiple,
            pred_multiple,
            rmse_multiple
        ) = crear_modelo_multiple(

            df_filtrado,

            x_multiple
        )


    # ========================================================
    # ENCABEZADO
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
        Comparación de modelos
        </div>

        <div class="section-subtitle">
        Vista ejecutiva de la regresión lineal simple y múltiple
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # KPIs
    # ========================================================

    k1, k2, k3, k4 = st.columns(
        4,
        gap="small"
    )


    with k1:

        tarjeta_kpi(
            "Observaciones",
            f"{len(df_filtrado):,}",
            "Registros de la selección"
        )


    with k2:

        tarjeta_kpi(
            "Variable dependiente",
            "Ventas",
            "Y permanece fija"
        )


    with k3:

        tarjeta_kpi(
            "R² · Simple",
            f"{modelo_simple.rsquared:.1%}",
            x_simple_nombre
        )


    with k4:

        if modelo_multiple_valido:

            tarjeta_kpi(
                "R² ajustado · Múltiple",
                f"{modelo_multiple.rsquared_adj:.1%}",
                f"{len(x_multiple)} variables X"
            )

        else:

            tarjeta_kpi(
                "R² ajustado · Múltiple",
                "—",
                "Selecciona al menos 2 variables"
            )


    st.markdown(
        '<div class="soft-divider"></div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # COMPARACIÓN VISUAL
    # ========================================================

    izquierda, derecha = st.columns(
        2,
        gap="large"
    )


    # ========================================================
    # MODELO SIMPLE
    # ========================================================

    with izquierda:

        st.markdown(
            """
            <div class="section-title">
            Regresión Lineal Simple
            </div>
            """,
            unsafe_allow_html=True
        )


        st.caption(
            f"Ventas explicadas por {x_simple_nombre}"
        )


        fig_simple = grafica_simple(

            datos_simple,

            modelo_simple,

            x_simple,

            x_simple_nombre
        )


        st.plotly_chart(
            fig_simple,
            use_container_width=True
        )


        beta = (
            modelo_simple.params[
                x_simple
            ]
        )


        tarjeta_insight(
            "Lectura del modelo",
            [
                (
                    f'El coeficiente estimado de '
                    f'<b>{x_simple_nombre}</b> es '
                    f'<b>{beta:.3f}</b>.'
                ),

                (
                    f'Un incremento de una unidad en esta variable '
                    f'se asocia con un cambio promedio de '
                    f'<b>{beta:.3f} ventas</b>.'
                )
            ]
        )


    # ========================================================
    # MODELO MÚLTIPLE
    # ========================================================

    with derecha:

        st.markdown(
            """
            <div class="section-title">
            Regresión Lineal Múltiple
            </div>
            """,
            unsafe_allow_html=True
        )


        if modelo_multiple_valido:

            st.caption(
                "Efecto estimado de cada variable manteniendo las demás constantes"
            )


            fig_coef = (
                grafica_coeficientes(
                    modelo_multiple
                )
            )


            st.plotly_chart(
                fig_coef,
                use_container_width=True
            )


            tarjeta_insight(
                "Lectura del modelo",
                [
                    (
                        f'El modelo utiliza '
                        f'<b>{len(x_multiple)} variables explicativas</b> '
                        f'y alcanza un R² ajustado de '
                        f'<b>{modelo_multiple.rsquared_adj:.1%}</b>.'
                    ),

                    (
                        'El gráfico permite comparar la dirección '
                        'y magnitud de los coeficientes junto con '
                        'sus intervalos de confianza.'
                    )
                ]
            )


        else:

            st.warning(
                "Selecciona al menos dos variables X para construir la regresión múltiple."
            )


# ============================================================
# VISTA 2: REGRESIÓN LINEAL SIMPLE
# ============================================================

elif tipo_modelo == "Regresión Lineal Simple":


    (
        modelo_simple,
        datos_simple,
        pred_simple,
        rmse_simple
    ) = crear_modelo_simple(

        df_filtrado,

        x_simple
    )


    beta = (
        modelo_simple.params[
            x_simple
        ]
    )


    intercepto = (
        modelo_simple.params[
            "const"
        ]
    )


    p_value = (
        modelo_simple.pvalues[
            x_simple
        ]
    )


    # ========================================================
    # TÍTULO
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
        Regresión Lineal Simple
        </div>
        """,
        unsafe_allow_html=True
    )


    st.caption(
        f"Relación entre {x_simple_nombre} y Ventas"
    )


    # ========================================================
    # KPIs
    # ========================================================

    k1, k2, k3, k4 = st.columns(
        4,
        gap="small"
    )


    with k1:

        tarjeta_kpi(
            "Observaciones",
            f"{len(datos_simple):,}",
            "Registros utilizados"
        )


    with k2:

        tarjeta_kpi(
            "R²",
            f"{modelo_simple.rsquared:.1%}",
            "Variabilidad explicada"
        )


    with k3:

        tarjeta_kpi(
            "RMSE",
            f"{rmse_simple:.2f}",
            "Error típico de predicción"
        )


    with k4:

        tarjeta_kpi(
            "p-value",
            f"{p_value:.4f}",
            "Variable X"
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # GRÁFICA Y MEDIDOR
    # ========================================================

    grafica_col, gauge_col = st.columns(
        [2.2, 1],
        gap="large"
    )


    with grafica_col:

        fig_simple = grafica_simple(

            datos_simple,

            modelo_simple,

            x_simple,

            x_simple_nombre
        )


        st.plotly_chart(
            fig_simple,
            use_container_width=True
        )


    with gauge_col:

        st.plotly_chart(

            gauge_r2(
                modelo_simple.rsquared
            ),

            use_container_width=True
        )


        tarjeta_insight(
            "Ecuación estimada",
            [
                (
                    f'Ventas = '
                    f'<b>{intercepto:.3f}</b> + '
                    f'<b>{beta:.3f}</b> × '
                    f'{x_simple_nombre}'
                )
            ]
        )


    # ========================================================
    # SIGNIFICANCIA
    # ========================================================

    if p_value < 0.05:

        significancia = (
            "estadísticamente significativa"
        )

    else:

        significancia = (
            "no estadísticamente significativa"
        )


    # ========================================================
    # INTERPRETACIÓN AUTOMÁTICA
    # ========================================================

    tarjeta_insight(
        "Interpretación automática",
        [
            (
                f'El coeficiente de '
                f'<b>{x_simple_nombre}</b> '
                f'es <b>{beta:.3f}</b>.'
            ),

            (
                f'Esto indica que, en promedio, '
                f'una unidad adicional de esta variable '
                f'se asocia con un cambio de '
                f'<b>{beta:.3f} ventas</b>.'
            ),

            (
                f'Con un p-value de '
                f'<b>{p_value:.4f}</b>, '
                f'la relación es '
                f'<b>{significancia}</b> '
                f'al nivel de significancia del 5%.'
            )
        ]
    )


# ============================================================
# VISTA 3: REGRESIÓN LINEAL MÚLTIPLE
# ============================================================

elif tipo_modelo == "Regresión Lineal Múltiple":


    # ========================================================
    # VALIDACIÓN
    # ========================================================

    if len(x_multiple) < 2:

        st.warning(
            "Selecciona al menos dos variables explicativas para estimar una regresión múltiple."
        )

        st.stop()


    # ========================================================
    # MODELO
    # ========================================================

    (
        modelo_multiple,
        datos_multiple,
        pred_multiple,
        rmse_multiple
    ) = crear_modelo_multiple(

        df_filtrado,

        x_multiple
    )


    # ========================================================
    # TÍTULO
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
        Regresión Lineal Múltiple
        </div>
        """,
        unsafe_allow_html=True
    )


    st.caption(
        "Análisis simultáneo de los factores asociados con las ventas"
    )


    # ========================================================
    # KPIs
    # ========================================================

    k1, k2, k3, k4 = st.columns(
        4,
        gap="small"
    )


    with k1:

        tarjeta_kpi(
            "Observaciones",
            f"{len(datos_multiple):,}",
            "Registros utilizados"
        )


    with k2:

        tarjeta_kpi(
            "R²",
            f"{modelo_multiple.rsquared:.1%}",
            "Variabilidad explicada"
        )


    with k3:

        tarjeta_kpi(
            "R² ajustado",
            f"{modelo_multiple.rsquared_adj:.1%}",
            "Ajustado por número de X"
        )


    with k4:

        tarjeta_kpi(
            "RMSE",
            f"{rmse_multiple:.2f}",
            "Error típico de predicción"
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # COEFFICIENT PLOT + R²
    # ========================================================

    grafica_col, gauge_col = st.columns(
        [2.2, 1],
        gap="large"
    )


    with grafica_col:

        fig_coef = (
            grafica_coeficientes(
                modelo_multiple
            )
        )


        st.plotly_chart(
            fig_coef,
            use_container_width=True
        )


    with gauge_col:

        st.plotly_chart(

            gauge_r2(
                modelo_multiple.rsquared_adj,
                "R² ajustado"
            ),

            use_container_width=True
        )


        tarjeta_insight(
            "Resumen del modelo",
            [
                (
                    f'El modelo utiliza '
                    f'<b>{len(x_multiple)} variables explicativas</b>.'
                ),

                (
                    f'El R² ajustado es de '
                    f'<b>{modelo_multiple.rsquared_adj:.1%}</b>, '
                    f'considerando el número de predictores incluidos.'
                )
            ]
        )


    # ========================================================
    # DIVISOR
    # ========================================================

    st.markdown(
        '<div class="soft-divider"></div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # RESULTADOS
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
        Resultados del modelo
        </div>
        """,
        unsafe_allow_html=True
    )


    st.caption(
        "Coeficientes, significancia estadística y multicolinealidad"
    )


    # ========================================================
    # CALCULAR VIF
    # ========================================================

    vif_df = calcular_vif(
        datos_multiple,
        x_multiple
    )


    # ========================================================
    # TABLA DE RESULTADOS
    # ========================================================

    resultados = pd.DataFrame({

        "Variable":
            modelo_multiple.params.index,

        "Coeficiente":
            modelo_multiple.params.values,

        "p-value":
            modelo_multiple.pvalues.values
    })


    resultados = resultados[
        resultados["Variable"]
        != "const"
    ].copy()


    resultados = resultados.merge(

        vif_df,

        on="Variable",

        how="left"
    )


    # ========================================================
    # NOMBRES BONITOS
    # ========================================================

    mapa_nombres = {

        valor: clave

        for clave, valor
        in VARIABLES_X.items()
    }


    resultados["Variable"] = (

        resultados["Variable"]
        .map(
            mapa_nombres
        )
        .fillna(
            resultados["Variable"]
        )
    )


    # ========================================================
    # SIGNIFICANCIA
    # ========================================================

    resultados["Significativa"] = np.where(

        resultados["p-value"] < 0.05,

        "Sí",

        "No"
    )


    # ========================================================
    # REDONDEO
    # ========================================================

    resultados["Coeficiente"] = (

        resultados["Coeficiente"]
        .round(4)
    )


    resultados["p-value"] = (

        resultados["p-value"]
        .round(4)
    )


    resultados["VIF"] = (

        resultados["VIF"]
        .round(2)
    )


    # ========================================================
    # ORDEN DE COLUMNAS
    # ========================================================

    resultados = resultados[
        [
            "Variable",
            "Coeficiente",
            "p-value",
            "Significativa",
            "VIF"
        ]
    ]


    st.dataframe(

        resultados,

        use_container_width=True,

        hide_index=True
    )


    # ========================================================
    # DIAGNÓSTICO DE MULTICOLINEALIDAD
    # ========================================================

    max_vif = (

        resultados[
            "VIF"
        ]

        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )

        .max()
    )


    if pd.notna(
        max_vif
    ):

        if max_vif >= 10:

            st.error(
                f"⚠️ Se detecta multicolinealidad alta. "
                f"El VIF máximo es {max_vif:.2f}. "
                f"Conviene revisar la combinación de variables."
            )


        elif max_vif >= 5:

            st.warning(
                f"⚠️ Existe evidencia de multicolinealidad moderada. "
                f"El VIF máximo es {max_vif:.2f}."
            )


        else:

            st.success(
                f"✓ No se observa multicolinealidad importante "
                f"con la selección actual. "
                f"VIF máximo: {max_vif:.2f}."
            )
