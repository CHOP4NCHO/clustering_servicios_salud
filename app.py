import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Clustering de Servicios de Salud",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Mapeo de nombres amigables para profesiones
# ---------------------------------------------------------
TIPO_MAP = {
    'tabla03_matrona.pdf': 'Matrona',
    'tabla04_nutricionista.pdf': 'Nutricionista',
    'tabla05_psicologo.pdf': 'Psicólogo',
    'tabla06_fonoaudiologo.pdf': 'Fonoaudiólogo',
    'tabla07_tocupacional.pdf': 'Terapeuta Ocupacional',
    'tabla08_tecnomedico.pdf': 'Tecnólogo Médico',
    'tabla09_asisocial.pdf': 'Trabajador Social',
    'tabla10_kine.pdf': 'Kinesiólogo'
}

# ---------------------------------------------------------
# Carga y preprocesamiento de datos
# ---------------------------------------------------------
@st.cache_data
def load_data(filepath: str = "atenciones_por_tipo.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['tipo_nom'] = df['tipo'].map(TIPO_MAP).fillna(df['tipo'])
    df['variable'] = df['tipo_nom'] + ' - ' + df['nivel']
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error al cargar el archivo de datos: {e}")
    st.stop()

# ---------------------------------------------------------
# Barra Lateral (Controles)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuración del Análisis")

# 1. Filtro Temporal
anios_disponibles = sorted(df_raw['anio'].unique(), reverse=True)
opciones_temporales = ["Promedio Período Completo (2020-2024)"] + [str(a) for a in anios_disponibles]

filtro_tiempo = st.sidebar.selectbox(
    "📅 Dimensión Temporal:",
    options=opciones_temporales,
    index=0,
    help="Elige si deseas analizar el perfil medio consolidado del quinquenio o focalizarte en un año específico."
)

# 2. Número de Clusters
k_clusters = st.sidebar.slider(
    "🎯 Número de Clusters (N):",
    min_value=2,
    max_value=10,
    value=3,
    step=1,
    help="Define la cantidad de grupos en los que se segmentarán los 29 Servicios de Salud."
)

# 3. Semilla aleatoria
random_seed = st.sidebar.number_input("🎲 Semilla aleatoria (K-Means):", min_value=0, max_value=9999, value=42, step=1)

# ---------------------------------------------------------
# Preparación de la Matriz de Variables
# ---------------------------------------------------------
# Excluir el total nacional agregado 'SNSS'
df_servicios = df_raw[df_raw['servicio_salud'] != 'SNSS'].copy()

# Filtrar según temporalidad
if filtro_tiempo == "Promedio Período Completo (2020-2024)":
    df_agg = df_servicios.groupby(['servicio_salud', 'variable'])['valor'].sum().reset_index()
else:
    anio_sel = int(filtro_tiempo)
    df_agg = df_servicios[df_servicios['anio'] == anio_sel].groupby(['servicio_salud', 'variable'])['valor'].sum().reset_index()

# Pivotear: Filas = Servicios de Salud, Columnas = Profesión x Nivel
pivot_val = df_agg.pivot_table(
    index='servicio_salud',
    columns='variable',
    values='valor',
    aggfunc='sum',
    fill_value=0
)

# Normalización por fila a proporciones relativas (%)
row_sums = pivot_val.sum(axis=1)
proporciones = pivot_val.div(row_sums, axis=0).fillna(0) * 100

# Estandarización de las variables para el clustering (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(proporciones)

# ---------------------------------------------------------
# Cálculo de Métricas Diagnósticas (Codo y Silueta)
# ---------------------------------------------------------
k_range = range(2, 11)
inertias = []
silhouettes = []

for k in k_range:
    km_test = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    km_test.fit(X_scaled)
    inertias.append(km_test.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km_test.labels_))

# ---------------------------------------------------------
# Modelo K-Means con el N seleccionado
# ---------------------------------------------------------
kmeans_model = KMeans(n_clusters=k_clusters, random_state=random_seed, n_init=15)
cluster_labels = kmeans_model.fit_predict(X_scaled)
current_silhouette = silhouette_score(X_scaled, cluster_labels)

# Reducción de dimensionalidad con PCA para visualización 2D
pca = PCA(n_components=2, random_state=random_seed)
coords_pca = pca.fit_transform(X_scaled)
var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100

# DataFrame con resultados
df_plot = pd.DataFrame({
    'Servicio de Salud': proporciones.index,
    'PC1': coords_pca[:, 0],
    'PC2': coords_pca[:, 1],
    'Cluster': [f"Cluster {c+1}" for c in cluster_labels]
}).sort_values(by='Cluster')

# ---------------------------------------------------------
# Renderizado de la Interfaz Principal
# ---------------------------------------------------------
st.title("🏥 Clustering de Servicios de Salud en Chile")
st.markdown(
    """
    Esta herramienta agrupa de forma no supervisada a los **29 Servicios de Salud del país**
    según su **perfil relativo de atenciones** (distribución porcentual de prestaciones por profesión y nivel asistencial).
    Permite identificar qué servicios tienen una estructura asistencial más parecida entre sí.
    """
)

# Métricas destacadas
col1, col2, col3, col4 = st.columns(4)
col1.metric("🏥 Servicios Analizados", len(proporciones))
col2.metric("🎯 Clusters Activos", k_clusters)
col3.metric("📊 Coef. Silueta Actual", f"{current_silhouette:.3f}")
col4.metric("📈 Varianza PCA 2D", f"{(var_pc1 + var_pc2):.1f}%")

st.divider()

# Gráfico de Dispersión PCA 2D interactivo
st.subheader("🗺️ Mapa de Similitud (Proyección PCA 2D)")
st.caption(
    f"Cada punto representa un Servicio de Salud. Cuanto más cercanos estén dos puntos, más similar es la distribución de su oferta de atenciones. "
    f"PC1 explica {var_pc1:.1f}% de la varianza y PC2 explica {var_pc2:.1f}%."
)

fig_pca = px.scatter(
    df_plot,
    x='PC1',
    y='PC2',
    color='Cluster',
    text='Servicio de Salud',
    hover_name='Servicio de Salud',
    template='plotly_white',
    height=600,
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig_pca.update_traces(
    textposition='top center',
    marker=dict(size=12, opacity=0.85, line=dict(width=1, color='DarkSlateGrey')),
    textfont=dict(size=11)
)

fig_pca.update_layout(
    xaxis_title=f"Componente Principal 1 ({var_pc1:.1f}% varianza)",
    yaxis_title=f"Componente Principal 2 ({var_pc2:.1f}% varianza)",
    legend_title="Grupos",
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig_pca, width='stretch')

st.divider()

# Tablas Resumen de Grupos
st.subheader("📋 Resumen de Grupos Formados")

tab_miembros, tab_perfil, tab_evaluacion = st.tabs([
    "👥 Servicios por Cluster",
    "📊 Perfil Promedio del Cluster (%)",
    "📈 Evaluación de N (Codo y Silueta)"
])

with tab_miembros:
    st.markdown("A continuación se lista qué Servicios de Salud quedaron agrupados en cada cluster:")
    grupos_resumen = []
    for c in sorted(df_plot['Cluster'].unique()):
        miembros = df_plot[df_plot['Cluster'] == c]['Servicio de Salud'].tolist()
        grupos_resumen.append({
            "Cluster": c,
            "Cantidad de Servicios": len(miembros),
            "Servicios de Salud que lo integran": ", ".join(miembros)
        })
    st.dataframe(pd.DataFrame(grupos_resumen), width='stretch', hide_index=True)

with tab_perfil:
    st.markdown("Distribución promedio de atenciones (%) dentro de cada cluster para identificar la especialización de cada grupo:")
    df_con_cluster = proporciones.copy()
    df_con_cluster['Cluster'] = [f"Cluster {c+1}" for c in cluster_labels]
    perfil_clusters = df_con_cluster.groupby('Cluster').mean().round(2)
    
    st.dataframe(
        perfil_clusters.style.format("{:.2f}%").background_gradient(cmap="Blues", axis=1),
        width='stretch'
    )

with tab_evaluacion:
    st.markdown("Guía analítica para respaldar la elección del número óptimo de clusters $N$:")
    col_codo, col_sil = st.columns(2)
    
    with col_codo:
        fig_codo = go.Figure()
        fig_codo.add_trace(go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            marker=dict(size=8, color='#1f77b4'),
            line=dict(width=2)
        ))
        # Marcar K actual
        fig_codo.add_trace(go.Scatter(
            x=[k_clusters],
            y=[inertias[k_clusters - 2]],
            mode='markers',
            marker=dict(size=14, color='red', symbol='star'),
            name=f'N Actual ({k_clusters})'
        ))
        fig_codo.update_layout(
            title="Método del Codo (Inercia)",
            xaxis_title="Número de clusters (k)",
            yaxis_title="Inercia (Suma de distancias al cuadrado)",
            template="plotly_white",
            showlegend=False,
            height=380
        )
        st.plotly_chart(fig_codo, width='stretch')
    
    with col_sil:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=list(k_range),
            y=silhouettes,
            mode='lines+markers',
            marker=dict(size=8, color='#2ca02c'),
            line=dict(width=2)
        ))
        # Marcar K actual
        fig_sil.add_trace(go.Scatter(
            x=[k_clusters],
            y=[silhouettes[k_clusters - 2]],
            mode='markers',
            marker=dict(size=14, color='red', symbol='star'),
            name=f'N Actual ({k_clusters})'
        ))
        fig_sil.update_layout(
            title="Coeficiente de Silueta Promedio",
            xaxis_title="Número de clusters (k)",
            yaxis_title="Silhouette Score (-1 a 1)",
            template="plotly_white",
            showlegend=False,
            height=380
        )
        st.plotly_chart(fig_sil, width='stretch')
    
    mejor_k = list(k_range)[np.argmax(silhouettes)]
    st.info(f"💡 **Sugerencia técnica:** Según el coeficiente de silueta, la mayor separación neta entre grupos ocurre con **N = {mejor_k}** (score: {max(silhouettes):.3f}).")

with st.expander("🔍 Ver tabla de datos base normalizada (% por servicio)"):
    st.dataframe(proporciones.style.format("{:.2f}%"), width='stretch')
