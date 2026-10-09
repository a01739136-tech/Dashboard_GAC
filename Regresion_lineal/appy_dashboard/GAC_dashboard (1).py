import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

st.set_page_config(page_title='GAC | Business Intelligence', page_icon='🚘', layout='wide', initial_sidebar_state='expanded')
BASE = Path(__file__).resolve().parent
RED, RED_DARK, RED_LIGHT, WHITE, GRAY = '#EF3340', '#8F1D2C', '#E88D98', '#FFFFFF', '#9B9CA3'
st.markdown('''<style>
.stApp{background:#0E1117;color:white}.block-container{padding-top:1.8rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:#262730;border-right:1px solid #383A45}
[data-testid="stSidebarNav"] a{border-radius:8px}
.red-line{height:3px;background:#EF3340;border-radius:8px;margin:12px 0 28px}
.preview-card{background:linear-gradient(145deg,#171A21,#11141A);border:1px solid #30323A;border-radius:14px;padding:17px 20px;min-height:123px;margin-bottom:8px}
.preview-category{font-size:11px;font-weight:700;letter-spacing:1px;color:#EF3340;margin-bottom:8px}
.preview-title{font-size:22px;font-weight:700;color:#fff;margin-bottom:6px}
.preview-description{font-size:13px;color:#9B9CA3;line-height:1.5}
</style>''', unsafe_allow_html=True)


def preview_style(fig, title, xaxis='', yaxis='', height=325):
    fig.update_layout(title=title, height=height, xaxis_title=xaxis, yaxis_title=yaxis,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color=WHITE, size=11),
                      margin=dict(l=20, r=20, t=55, b=55),
                      legend=dict(orientation='h', y=-0.25, x=0),
                      hoverlabel=dict(bgcolor='#1B1E26', font_color=WHITE))
    fig.update_xaxes(gridcolor='#30323A', zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor='#30323A', zeroline=False, automargin=True)
    return fig


def card(category, title, description):
    st.markdown(('<div class="preview-card">'
                 f'<div class="preview-category">{category}</div>'
                 f'<div class="preview-title">{title}</div>'
                 f'<div class="preview-description">{description}</div>'
                 '</div>'), unsafe_allow_html=True)


def render_preview(make_chart, page, label, icon):
    try:
        fig = make_chart()
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning('No se pudo generar esta vista previa. Revisa el nombre y contenido del CSV.')
        with st.expander('Detalle técnico'):
            st.code(str(exc))
    st.page_link(page, label=label, icon=icon)


def funnel_chart():
    df = pd.read_csv(BASE / 'Funnel_limpia.csv')
    df.columns = df.columns.str.strip()
    xs = ['leads_efectivos', 'cita_efectiva', 'prueba_de_manejo']
    data = df[['ventas'] + xs].apply(pd.to_numeric, errors='coerce').dropna()
    # No incluir filas de total, si están identificadas por canal.
    if 'canal' in df.columns:
        mask = ~df['canal'].astype(str).str.strip().str.lower().eq('gac angelopolis')
        data = df.loc[mask, ['ventas'] + xs].apply(pd.to_numeric, errors='coerce').dropna()
    model = sm.OLS(data['ventas'], sm.add_constant(data[xs])).fit()
    names = {'leads_efectivos':'Leads efectivos', 'cita_efectiva':'Cita efectiva', 'prueba_de_manejo':'Prueba de manejo'}
    fig = go.Figure()
    ci = model.conf_int()
    for var in xs:
        val = model.params[var]
        fig.add_trace(go.Scatter(x=[val], y=[names[var]], mode='markers', showlegend=False,
            marker=dict(color=RED, size=11),
            error_x=dict(type='data', symmetric=False, array=[ci.loc[var,1]-val],
                         arrayminus=[val-ci.loc[var,0]], color=RED_LIGHT, thickness=2, width=5)))
    fig.add_vline(x=0, line_dash='dash', line_color=GRAY)
    return preview_style(fig, 'Impacto estimado sobre ventas', 'Coeficiente estimado', '')


def redes_chart():
    df = pd.read_csv(BASE / '03_Redes_Sociales.csv')
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how='all')
    label_col = df.columns[0]
    labels = df[label_col].astype(str).str.strip()
    a = df.loc[labels.eq('2025 Visualizaciones')].iloc[0]
    b = df.loc[labels.eq('Real visualizaciones')].iloc[0]
    months = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
    data = pd.DataFrame({'x':[pd.to_numeric(a.get(m), errors='coerce') for m in months],
                         'y':[pd.to_numeric(b.get(m), errors='coerce') for m in months]}).dropna()
    model = LinearRegression().fit(data[['x']], data['y'])
    line = data.sort_values('x')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.x, y=data.y, mode='markers', name='Observaciones', marker=dict(color=RED_DARK,size=9)))
    fig.add_trace(go.Scatter(x=line.x, y=model.predict(line[['x']]), mode='lines', name='Ajuste', line=dict(color=RED,width=3)))
    return preview_style(fig, 'Visualizaciones vs. Real', 'Visualizaciones', 'Real')


def leads_chart():
    df = pd.read_csv(BASE / 'base leads.csv')
    df.columns = df.columns.str.strip()
    for col in ['Total','Efectivos','Abiertos','Ventas']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    data = df.loc[df.Total > 0, ['Efectivos','Abiertos','Ventas']].dropna()
    model = LinearRegression().fit(data[['Efectivos','Abiertos']], data['Ventas'])
    actual, predicted = data['Ventas'].to_numpy(), model.predict(data[['Efectivos','Abiertos']])
    low, high = min(actual.min(),predicted.min()), max(actual.max(),predicted.max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=predicted, mode='markers', name='Observaciones', marker=dict(color=RED,size=9)))
    fig.add_trace(go.Scatter(x=[low,high], y=[low,high], mode='lines', name='Predicción perfecta', line=dict(color=RED_LIGHT,dash='dash')))
    return preview_style(fig, 'Real vs. Predicho', 'Ventas reales', 'Ventas predichas')


def marketing_chart():
    df = pd.read_csv(BASE / 'Marketing_Regresion_Limpia.csv')
    df.columns = df.columns.str.strip()
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    xs = ['Leads Plaza','Leads Digitales','Leads Piso']
    for col in xs + ['Ventas Totales']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Fecha','Ventas Totales']+xs).sort_values('Fecha')
    if len(df) <= 6:
        raise ValueError('Se requieren más de seis meses para separar entrenamiento y prueba.')
    train, test = df.iloc[:-6], df.iloc[-6:]
    simple = LinearRegression().fit(train[['Leads Plaza']], train['Ventas Totales'])
    multiple = LinearRegression().fit(train[xs], train['Ventas Totales'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.Fecha,y=test['Ventas Totales'],mode='lines+markers',name='Ventas reales',line=dict(color=WHITE,width=3)))
    fig.add_trace(go.Scatter(x=test.Fecha,y=simple.predict(test[['Leads Plaza']]),mode='lines+markers',name='Regresión simple',line=dict(color=RED_LIGHT,dash='dash')))
    fig.add_trace(go.Scatter(x=test.Fecha,y=multiple.predict(test[xs]),mode='lines+markers',name='Regresión múltiple',line=dict(color=RED,width=3)))
    fig.add_trace(go.Scatter(x=test.Fecha,y=np.full(len(test),train['Ventas Totales'].iloc[-1]),mode='lines',name='Último mes conocido',line=dict(color=GRAY,dash='dot')))
    fig = preview_style(fig,'Comparación mensual de modelos','Mes de prueba','Ventas',height=350)
    fig.update_xaxes(tickformat='%m/%Y')
    return fig


def plazas_chart():
    df = pd.read_csv(BASE / 'Plazas_sin_nulos.csv',encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    if 'Fecha Origen' in df.columns:
        df['Fecha Origen'] = pd.to_datetime(df['Fecha Origen'],errors='coerce')
        df = df.dropna(subset=['Fecha Origen'])
    def repair(s):
        if not isinstance(s,str):
            return s
        for encoding in ('cp1252','latin-1'):
            try:
                return ' '.join(s.encode(encoding).decode('utf-8').split())
            except (UnicodeError, UnicodeEncodeError):
                continue
        return ' '.join(s.split())
    df['Origen'] = df['Origen'].map(repair).str.title()
    counts = df['Origen'].dropna().value_counts().sort_values()
    fig = go.Figure(go.Bar(x=counts.values,y=counts.index,orientation='h',marker_color=RED,
                           text=counts.values,textposition='outside'))
    fig.update_layout(showlegend=False)
    return preview_style(fig,'Distribución de leads por plaza','Número de leads','')


def inicio():
    left,right = st.columns([1.2,5],vertical_alignment='center')
    with left:
        if (BASE/'logo_gac.png').exists():
            st.image(str(BASE/'logo_gac.png'),width=125)
    with right:
        st.title('Business Intelligence')
        st.caption('Plataforma integral de análisis comercial y desempeño')
    st.markdown('<div class="red-line"></div>',unsafe_allow_html=True)
    st.subheader('Paneles de análisis')
    st.caption('Selecciona un dashboard para explorar sus indicadores, modelos y visualizaciones.')
    a,b = st.columns(2,gap='large')
    with a:
        card('ANÁLISIS COMERCIAL','Funnel de Ventas','Modelado explicativo de las ventas mediante regresión lineal simple y múltiple.')
        render_preview(funnel_chart,'appy_funnel.py','Abrir Funnel →','📊')
    with b:
        card('MARKETING DIGITAL','Redes Sociales','Análisis de métricas digitales mediante regresión lineal simple y múltiple.')
        render_preview(redes_chart,'app_redes.py','Abrir Redes Sociales →','📱')
    st.divider()
    a,b = st.columns(2,gap='large')
    with a:
        card('GESTIÓN DE LEADS','Análisis de Leads','Evaluación del desempeño mediante comparación real vs. predicho.')
        render_preview(leads_chart,'app_leads.py','Abrir Leads →','📈')
    with b:
        card('MARKETING','Marketing | Regresión Lineal','Impacto de los leads por canal sobre las ventas mensuales.')
        render_preview(marketing_chart,'app_marketing.py','Abrir Marketing →','📣')
    st.divider()
    a,b = st.columns(2,gap='large')
    with a:
        card('ANÁLISIS OPERATIVO','Leads Plazas','Distribución de leads por plaza y comportamiento operativo.')
        render_preview(plazas_chart,'app_plazas.py','Abrir Leads Plazas →','🚗')

pages = [st.Page('appy_funnel.py',title='Funnel de Ventas',icon='📊'),
         st.Page('app_redes.py',title='Redes Sociales',icon='📱'),
         st.Page('app_leads.py',title='Análisis de Leads',icon='📈'),
         st.Page('app_marketing.py',title='Marketing',icon='📣'),
         st.Page('app_plazas.py',title='Leads Plazas',icon='🚗')]
pg = st.navigation({'GAC Business Intelligence':[st.Page(inicio,title='Inicio',icon='🏠',default=True)],'Dashboards':pages})
pg.run()
