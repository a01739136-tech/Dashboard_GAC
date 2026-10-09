


import streamlit as st

import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

from pathlib import Path



st.set_page_config(

    page_title="GAC · Análisis Univariado",

    page_icon="📊",

    layout="wide"

)



CSV = "Plazas_sin_nulos.csv"



DIAS = {

    0: "Lunes",

    1: "Martes",

    2: "Miércoles",

    3: "Jueves",

    4: "Viernes",

    5: "Sábado",

    6: "Domingo"

}



# ============================================================

# PALETA GAC — ÚNICO CAMBIO VISUAL

# ============================================================



ROJO = "#EF3340"

VINO = "#8F1D2C"

ROSA = "#E88D98"

GRIS = "#9B9CA3"

BLANCO = "#FFFFFF"

ROJO_OSCURO = "#641322"



COLORES_GAC = [

    ROJO,

    VINO,

    ROSA,

    GRIS,

    "#C94354",

    "#D4D4D8",

    "#B52A3B",

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



ESCALA_CORRELACION_GAC = [
    [0.00, "#641322"],  # -1: vino oscuro
    [0.25, "#B52A3B"],  # -0.5: rojo vino
    [0.50, "#F6E9EA"],  # 0: neutral claro
    [0.75, "#EF3340"],  # +0.5: rojo GAC
    [1.00, "#8F1D2C"]  # +1: rojo profundo
]





def arreglar_texto(s):

    """Corrige texto UTF-8 mal leído y espacios repetidos."""

    if not isinstance(s, str):

        return s



    for enc in ("cp1252", "latin-1"):

        try:

            s = s.encode(enc).decode("utf-8")

            break

        except (UnicodeEncodeError, UnicodeDecodeError):

            continue



    return " ".join(s.split())





@st.cache_data

def load_data():

    base = Path(__file__).parent



    ruta = next(

        p for p in (

            base / CSV,

            base / "data" / CSV

        )

        if p.exists()

    )



    df = pd.read_csv(

        ruta,

        encoding="utf-8-sig"

    )



    # Transformaciones originales

    for c in ["Asesor", "Producto", "Origen", "Estatus"]:

        df[c] = df[c].map(arreglar_texto)



    df["Producto"] = df["Producto"].str.upper()

    df["Asesor"] = df["Asesor"].str.title()

    df["Origen"] = df["Origen"].str.title()



    df["Estatus"] = df["Estatus"].replace({

        "Contactato": "Contactado",

        "False": "Sin estatus",

        "FALSE": "Sin estatus"

    })



    for c in [

        "Seguimiento",

        "Cita efetiva",

        "PDM",

        "SDC"

    ]:

        df[c] = df[c].map({

            1.0: "Sí",

            0.0: "No"

        })



    # Variables derivadas de fecha

    df["Fecha Origen"] = pd.to_datetime(

        df["Fecha Origen"],

        errors="coerce"

    )



    df["Hora"] = df["Fecha Origen"].dt.hour



    df["Día de la semana"] = (

        df["Fecha Origen"]

        .dt.dayofweek

        .map(DIAS)

    )



    df["Franja horaria"] = pd.cut(

        df["Hora"],

        [-1, 11, 14, 17, 23],

        labels=[

            "Mañana (<12h)",

            "Mediodía (12-14h)",

            "Tarde (15-17h)",

            "Noche (18h+)"

        ]

    ).astype(str)



    df = df.dropna(subset=["Hora"])



    Lista = [

        "Producto",

        "Asesor",

        "Estatus"

    ]



    Cruce = [

        "Origen",

        "Franja horaria",

        "Día de la semana",

        "Seguimiento",

        "Cita efetiva",

        "PDM",

        "SDC"

    ] + Lista



    return df, Lista, Cruce





# GAC_ESTETICA_PLAZAS_FINAL
px.defaults.template = "plotly_dark"

st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FFFFFF; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background-color: #262730; }
[data-testid="stSidebar"] * { color: #FFFFFF; }
[data-baseweb="tag"] { background-color: #EF3340 !important; color: white !important; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #2B0D16, #210A11);
    border: 1px solid #501421;
    border-radius: 14px;
    padding: 18px 20px;
    min-height: 125px;
}
[data-testid="stMetricValue"] { color: #FFFFFF; font-weight: 700; }
[data-testid="stAlert"] {
    background-color: #171A22 !important;
    border: 1px solid #30333D !important;
    border-left: 5px solid #EF3340 !important;
    border-radius: 12px !important;
}
[data-testid="stAlert"] * { color: #FFFFFF !important; }
.gac-title { font-size: 48px; font-weight: 750; color: #FFFFFF; }
.gac-subtitle { color: #9B9CA3; font-size: 15px; margin-top: 6px; }
.gac-red-line { height: 4px; background: #EF3340; border-radius: 6px; margin: 18px 0 30px; }
</style>
""", unsafe_allow_html=True)

df, Lista, Cruce = load_data()



# ============================================================

# BARRA LATERAL ORIGINAL

# ============================================================



st.sidebar.title("🚗 GAC · Leads Plazas")



View = st.sidebar.selectbox(

    "Tipo de Análisis",

    [

        "Extracción de Características",

        "Regresión Lineal Simple",

        "Regresión Lineal Múltiple",

        "Base de datos limpia"

    ]

)



origenes = st.sidebar.multiselect(

    "📍 Origen (plaza)",

    sorted(df["Origen"].unique()),

    default=sorted(df["Origen"].unique())

)



h_min, h_max = st.sidebar.slider(

    "⏰ Hora de origen del lead",

    0,

    23,

    (0, 23)

)



dff = df[

    df["Origen"].isin(origenes)

    & df["Hora"].between(h_min, h_max)

]



if dff.empty:

    st.warning(

        "No hay datos con los filtros seleccionados."

    )

    st.stop()







# Encabezado compartido entre las vistas (solo presentación)
col_logo_gac, col_texto_gac = st.columns([1, 5], vertical_alignment="center")
with col_logo_gac:
    ruta_logo_gac = Path(__file__).resolve().parent / "logo_gac.png"
    if ruta_logo_gac.exists():
        st.image(str(ruta_logo_gac), width=125)
with col_texto_gac:
    st.markdown(
        '<div class="gac-title">Leads Plazas</div>'
        '<div class="gac-subtitle">Análisis de leads por plaza, producto y desempeño comercial</div>',
        unsafe_allow_html=True
    )
st.markdown('<div class="gac-red-line"></div>', unsafe_allow_html=True)


# ============================================================

# FUNCIONES ORIGINALES DE REGRESIÓN

# ============================================================



def tabla_diaria(d):

    """Agrupa los leads por día para obtener variables numéricas."""



    t = d.assign(

        Fecha=d["Fecha Origen"].dt.normalize(),

        _seg=(d["Seguimiento"] == "Sí").astype(int),

        _cita=(d["Cita efetiva"] == "Sí").astype(int),

        _pdm=(d["PDM"] == "Sí").astype(int),

        _sdc=(d["SDC"] == "Sí").astype(int),

        _emz=(d["Producto"] == "EMZOOM").astype(int),

        _gs8=(d["Producto"] == "GS8").astype(int)

    )



    g = t.groupby("Fecha").agg(

        **{

            "Leads": ("Hora", "size"),

            "Hora promedio": ("Hora", "mean"),

            "Seguimientos": ("_seg", "sum"),

            "Citas efectivas": ("_cita", "sum"),

            "Pruebas de manejo": ("_pdm", "sum"),

            "Solicitudes de crédito": ("_sdc", "sum"),

            "Leads EMZOOM": ("_emz", "sum"),

            "Leads GS8": ("_gs8", "sum")

        }

    ).reset_index()



    g["Día de la semana (0=Lun)"] = (

        g["Fecha"].dt.dayofweek

    )



    g["Día del periodo"] = (

        g["Fecha"] - g["Fecha"].min()

    ).dt.days



    return g





def regresion(X, y):

    """Mínimos cuadrados: coeficientes, predicciones y métricas."""



    A = np.column_stack([

        np.ones(len(X)),

        X

    ])



    b = np.linalg.lstsq(

        A,

        y,

        rcond=None

    )[0]



    pred = A @ b



    sse = ((y - pred) ** 2).sum()

    sst = ((y - y.mean()) ** 2).sum()



    r2 = (

        1 - sse / sst

        if sst > 0 else 0.0

    )



    n, k = len(y), X.shape[1]



    r2a = (

        1 - (1 - r2) * (n - 1) / (n - k - 1)

        if n - k - 1 > 0

        else np.nan

    )



    return (

        b,

        pred,

        r2,

        r2a,

        np.sqrt(sse / n),

        np.abs(y - pred).mean()

    )





def lectura_r2(r2):

    return (

        "débil"

        if r2 < 0.3

        else (

            "moderada"

            if r2 < 0.6

            else "fuerte"

        )

    )





def ajustar(fig, h=450):

    """Conserva márgenes y posición de la leyenda."""



    fig.update_layout(

        height=h,

        margin=dict(

            t=70,

            b=70,

            l=60,

            r=20

        ),

        legend=dict(

            orientation="h",

            yanchor="bottom",

            y=1.02,

            xanchor="left",

            x=0

        )

    )



    fig.update_xaxes(

        automargin=True,

        title_standoff=15

    )



    fig.update_yaxes(

        automargin=True,

        title_standoff=15

    )



    return fig





# ============================================================

# VISTA 1: EXTRACCIÓN DE CARACTERÍSTICAS

# ============================================================




# GAC_HALLAZGO_GRIS_V2
st.markdown("""
<style>
div[data-testid="stAlert"] {
    background: #171A22 !important;
    background-color: #171A22 !important;
    border: 1px solid #343741 !important;
    border-left: 5px solid #EF3340 !important;
    border-radius: 12px !important;
}
div[data-testid="stAlert"] * {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

if View == "Extracción de Características":



    Variable_Cat = st.sidebar.selectbox(

        "📊 Variable a analizar",

        Lista

    )



    Variable_Cruce = st.sidebar.selectbox(

        "🔥 Variable de cruce (heatmap)",

        [

            c for c in Cruce

            if c != Variable_Cat

        ]

    )



    Tabla_frecuencias = (

        dff[Variable_Cat]

        .value_counts()

        .reset_index()

    )



    Tabla_frecuencias.columns = [

        "categorias",

        "frecuencia"

    ]



    Tabla_frecuencias["porcentaje (%)"] = (

        100

        * Tabla_frecuencias["frecuencia"]

        / Tabla_frecuencias["frecuencia"].sum()

    ).round(1)



    st.title(

        f"Extracción de Características · {Variable_Cat}"

    )



    # KPI ORIGINALES

    top = Tabla_frecuencias.iloc[0]



    k1, k2, k3, k4 = st.columns(4)



    k1.metric(

        "👥 Leads",

        len(dff)

    )



    k2.metric(

        "🏷️ Categorías",

        len(Tabla_frecuencias)

    )



    k3.metric(

        "🥇 Categoría más frecuente",

        str(top["categorias"])

    )



    k4.metric(

        "📞 Con seguimiento",

        f"{100 * (dff['Seguimiento'] == 'Sí').mean():.0f}%"

    )



    texto_hallazgo = (
    f"**Hallazgo:** «{top['categorias']}» concentra "
    
            f"{top['porcentaje (%)']}% de los leads "
    
            f"({top['frecuencia']} de {len(dff)})."
    )
    import html
    st.markdown(
        f'''<div style="background-color:#171A22;color:#FFFFFF;border-left:6px solid #EF3340;border-radius:12px;padding:20px 22px;margin:12px 0 24px 0;font-size:17px;line-height:1.5;">{html.escape(texto_hallazgo.replace("**Hallazgo:**", "Hallazgo:"))}</div>''',
        unsafe_allow_html=True
    )


    # Heatmap y boxplot en la primera fila
    col_heatmap, col_boxplot = st.columns(2, gap="large")

    with col_heatmap:



        st.write("🔥 Heatmap (cruce de variables)")



        ct = pd.crosstab(

            dff[Variable_Cat],

            dff[Variable_Cruce]

        )



        figure5 = px.imshow(

            ct,

            text_auto=True,

            aspect="auto",

            color_continuous_scale=ESCALA_GAC,

            title=f"{Variable_Cat} × {Variable_Cruce}"

        )



        figure5.update_layout(

            height=420

        )



        st.plotly_chart(

            figure5,

            use_container_width=True

        )

    with col_boxplot:



        st.write("📦 Boxplot (hora de origen del lead)")



        figure6 = px.box(

            dff,

            x=Variable_Cat,

            y="Hora",

            color=Variable_Cat,

            points="outliers",

            title=f"Hora de origen por {Variable_Cat}",

            color_discrete_sequence=COLORES_GAC

        )



        figure6.update_xaxes(

            automargin=True,

            tickangle=-35

        )



        figure6.update_layout(

            height=420,

            showlegend=False

        )



        st.plotly_chart(

            figure6,

            use_container_width=True

        )

    # Gráfico de barras original a ancho completo



    st.write("Gráfico de Barras")



    figure1 = px.bar(

        Tabla_frecuencias,

        x="categorias",

        y="frecuencia",

        text="frecuencia",

        color="categorias",

        title="Frecuencia por categoría",

        color_discrete_sequence=COLORES_GAC

    )



    figure1.update_xaxes(

        automargin=True,

        tickangle=-35

    )



    figure1.update_layout(

        height=380,

        showlegend=False

    )



    st.plotly_chart(

        figure1,

        use_container_width=True

    )

    # TABLA ORIGINAL

    st.write("📋 Tabla de frecuencias")



    st.dataframe(

        Tabla_frecuencias,

        use_container_width=True,

        hide_index=True

    )





# ============================================================

# VISTA 2: BASE DE DATOS LIMPIA

# ============================================================



elif View == "Base de datos limpia":



    st.title("Base de datos limpia")



    st.write(

        f"{len(dff)} registros · {dff.shape[1]} columnas"

    )



    st.dataframe(

        dff,

        use_container_width=True

    )



    st.download_button(

        "⬇️ Descargar CSV filtrado",

        dff.to_csv(index=False).encode("utf-8-sig"),

        "leads_filtrado.csv",

        "text/csv"

    )





# ============================================================

# VISTA 3: REGRESIÓN LINEAL SIMPLE

# ============================================================



elif View == "Regresión Lineal Simple":



    st.title("📈 Regresión Lineal Simple")



    datos = tabla_diaria(dff)



    num = [

        c for c in datos.columns

        if c != "Fecha"

    ]



    if len(datos) < 3:

        st.warning(

            "Hay muy pocos días con los filtros actuales "

            "para ajustar una regresión."

        )

        st.stop()



    c1, c2 = st.columns(2)



    X_var = c1.selectbox(

        "Variable independiente (X)",

        num,

        index=num.index("Leads EMZOOM")

    )



    opciones_y = [

        c for c in num

        if c != X_var

    ]



    Y_var = c2.selectbox(

        "Variable dependiente (Y)",

        opciones_y,

        index=(

            opciones_y.index("Leads")

            if "Leads" in opciones_y

            else 0

        )

    )



    x = datos[X_var].to_numpy(float)

    y = datos[Y_var].to_numpy(float)



    if x.std() == 0 or y.std() == 0:

        st.warning(

            "Una de las variables no varía con estos filtros; "

            "elige otra."

        )

        st.stop()



    b, pred, r2, r2a, rmse, mae = regresion(

        x.reshape(-1, 1),

        y

    )



    m1, m2, m3, m4 = st.columns(4)



    m1.metric(

        "🔗 Correlación (r)",

        f"{np.corrcoef(x, y)[0, 1]:.2f}"

    )



    m2.metric(

        "🎯 R²",

        f"{r2:.3f}"

    )



    m3.metric(

        "📏 RMSE",

        f"{rmse:.2f}"

    )



    m4.metric(

        "📐 MAE",

        f"{mae:.2f}"

    )



    texto_ecuacion = (
    f"**Ecuación:** {Y_var} = {b[0]:.3f} "
    
            f"+ {b[1]:.3f} × ({X_var}) · "
    
            f"Relación {lectura_r2(r2)}: "
    
            f"{X_var} explica {100 * r2:.1f}% "
    
            f"de la variación de {Y_var} "
    
            f"(n = {len(datos)} días)."
    )
    import html
    st.markdown(
        f'''<div style="background-color:#171A22;color:#FFFFFF;border-left:6px solid #EF3340;border-radius:12px;padding:20px 22px;margin:12px 0 24px 0;font-size:17px;line-height:1.5;">{html.escape(texto_ecuacion.replace("**Ecuación:**", "Ecuación:"))}</div>''',
        unsafe_allow_html=True
    )


    Contenedor_A, Contenedor_B = st.columns(2)



    with Contenedor_A:



        st.write(

            "Dispersión de datos reales vs. predichos"

        )



        o = np.argsort(x)



        fig = go.Figure()



        fig.add_trace(

            go.Scatter(

                x=x,

                y=y,

                mode="markers",

                name="Datos reales",

                marker=dict(

                    size=9,

                    color=VINO

                )

            )

        )



        fig.add_trace(

            go.Scatter(

                x=x,

                y=pred,

                mode="markers",

                name="Datos predichos",

                marker=dict(

                    size=8,

                    symbol="diamond",

                    color=ROSA

                )

            )

        )



        fig.add_trace(

            go.Scatter(

                x=x[o],

                y=pred[o],

                mode="lines",

                name="Recta de regresión",

                line=dict(

                    color=ROJO

                )

            )

        )



        fig.update_layout(

            xaxis_title=X_var,

            yaxis_title=Y_var

        )



        ajustar(fig)



        st.plotly_chart(

            fig,

            use_container_width=True

        )



    with Contenedor_B:



        st.write(

            "Residuos (real − predicho)"

        )



        fig_r = go.Figure(

            go.Scatter(

                x=pred,

                y=y - pred,

                mode="markers",

                marker=dict(

                    size=9,

                    color=ROSA

                )

            )

        )



        fig_r.add_hline(

            y=0,

            line_dash="dash",

            line_color=GRIS

        )



        fig_r.update_layout(

            xaxis_title="Valor predicho",

            yaxis_title="Residuo"

        )



        ajustar(fig_r)



        st.plotly_chart(

            fig_r,

            use_container_width=True

        )



    with st.expander(

        "📋 Ver tabla diaria utilizada"

    ):



        st.dataframe(

            datos,

            use_container_width=True,

            hide_index=True

        )





# ============================================================

# VISTA 4: REGRESIÓN LINEAL MÚLTIPLE

# ============================================================



elif View == "Regresión Lineal Múltiple":



    st.title("📊 Regresión Lineal Múltiple")



    datos = tabla_diaria(dff)



    num = [

        c for c in datos.columns

        if c != "Fecha"

    ]



    if len(datos) < 5:

        st.warning(

            "Hay muy pocos días con los filtros actuales "

            "para ajustar una regresión."

        )

        st.stop()



    Y_var = st.selectbox(

        "Variable dependiente (Y)",

        num,

        index=num.index("Seguimientos")

    )



    opciones_x = [

        c for c in num

        if c != Y_var

    ]



    pref = [

        "Leads",

        "Hora promedio",

        "Día de la semana (0=Lun)"

    ]



    X_vars = st.multiselect(

        "Variables independientes (X)",

        opciones_x,

        default=[

            c for c in pref

            if c in opciones_x

        ]

    )



    if not X_vars:

        st.info(

            "Selecciona al menos una variable independiente."

        )

        st.stop()



    if len(datos) - len(X_vars) - 1 <= 0:

        st.warning(

            "Hay más variables X que datos; quita alguna."

        )

        st.stop()



    X = datos[X_vars].to_numpy(float)

    y = datos[Y_var].to_numpy(float)



    b, pred, r2, r2a, rmse, mae = regresion(

        X,

        y

    )



    m1, m2, m3, m4 = st.columns(4)



    m1.metric(

        "🎯 R²",

        f"{r2:.3f}"

    )



    m2.metric(

        "🎯 R² ajustada",

        f"{r2a:.3f}"

    )



    m3.metric(

        "📏 RMSE",

        f"{rmse:.2f}"

    )



    m4.metric(

        "📐 MAE",

        f"{mae:.2f}"

    )



    ecuacion = " + ".join(

        f"{c:.3f}×({v})"

        for c, v in zip(

            b[1:],

            X_vars

        )

    )



    texto_ecuacion = (
    f"**Ecuación:** {Y_var} = {b[0]:.3f} "
    
            f"+ {ecuacion} · "
    
            f"Ajuste {lectura_r2(r2)} "
    
            f"(R² = {r2:.2f})."
    )
    import html
    st.markdown(
        f'''<div style="background-color:#171A22;color:#FFFFFF;border-left:6px solid #EF3340;border-radius:12px;padding:20px 22px;margin:12px 0 24px 0;font-size:17px;line-height:1.5;">{html.escape(texto_ecuacion.replace("**Ecuación:**", "Ecuación:"))}</div>''',
        unsafe_allow_html=True
    )


    # ========================================================

    # FILA 1: GRÁFICAS ORIGINALES

    # ========================================================



    Contenedor_A, Contenedor_B = st.columns(2)



    with Contenedor_A:



        st.write(

            "Valores reales vs. predichos"

        )



        lim = [

            min(y.min(), pred.min()),

            max(y.max(), pred.max())

        ]



        fig = go.Figure()



        fig.add_trace(

            go.Scatter(

                x=y,

                y=pred,

                mode="markers",

                name="Predicho vs. real",

                marker=dict(

                    size=9,

                    color=ROJO

                )

            )

        )



        fig.add_trace(

            go.Scatter(

                x=lim,

                y=lim,

                mode="lines",

                name="Ajuste perfecto",

                line=dict(

                    dash="dash",

                    color=GRIS

                )

            )

        )



        fig.update_layout(

            xaxis_title=f"{Y_var} real",

            yaxis_title=f"{Y_var} predicho"

        )



        ajustar(fig)



        st.plotly_chart(

            fig,

            use_container_width=True

        )



    with Contenedor_B:



        st.write(

            "Serie diaria: datos reales y predichos superpuestos"

        )



        fig2 = go.Figure()



        fig2.add_trace(

            go.Scatter(

                x=datos["Fecha"],

                y=y,

                mode="lines+markers",

                name="Datos reales",

                line=dict(

                    color=ROJO

                )

            )

        )



        fig2.add_trace(

            go.Scatter(

                x=datos["Fecha"],

                y=pred,

                mode="markers",

                name="Datos predichos",

                marker=dict(

                    size=8,

                    symbol="diamond",

                    color=GRIS

                )

            )

        )



        fig2.update_layout(

            xaxis_title="Fecha",

            yaxis_title=Y_var

        )



        ajustar(fig2)



        fig2.update_xaxes(

            tickformat="%d %b"

        )



        st.plotly_chart(

            fig2,

            use_container_width=True

        )



    # ========================================================

    # FILA 2: MATRIZ Y COEFICIENTES ORIGINALES

    # ========================================================



    Contenedor_C, Contenedor_D = st.columns(2)



    with Contenedor_C:



        st.write(

            "Matriz de correlaciones"

        )



        corr = datos[

            [Y_var] + X_vars

        ].corr()



        fig3 = px.imshow(

            corr,

            text_auto=".2f",

            aspect="auto",

            color_continuous_scale=ESCALA_CORRELACION_GAC,

            zmin=-1,

            zmax=1

        )



        fig3.update_layout(

            height=430

        )



        st.plotly_chart(

            fig3,

            use_container_width=True

        )



    with Contenedor_D:



        st.write(

            "📋 Coeficientes del modelo"

        )



        coef = pd.DataFrame({

            "Variable": ["Intercepto"] + X_vars,

            "Coeficiente": b.round(4)

        })



        st.dataframe(

            coef,

            use_container_width=True,

            hide_index=True

        )



        st.caption(

            "Con pocos días de datos, un R² bajo es normal: "

            "indica que estas variables explican poco "

            "la variable Y."

        )
