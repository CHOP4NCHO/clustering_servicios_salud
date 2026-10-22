# 🏥 Dashboard de Clustering de Servicios de Salud en Chile

Aplicación interactiva desarrollada en **Streamlit** para realizar aprendizaje no supervisado (**K-Means**) sobre los datos de atenciones de salud de los 29 Servicios de Salud de Chile.

---

## 📌 Características Implementadas

1. **Estructura de Variables (Perfil Relativo):**
   - Las atenciones se desglosan por la combinación de **Profesión** (Matrona, Nutricionista, Kinesiólogo, etc.) y **Nivel asistencial** (Nivel Primario, Secundario, Urgencia).
   - Se calculan las **proporciones relativas (%)** de cada servicio, neutralizando las diferencias por tamaño poblacional entre regiones.
   - El registro agregado nacional (`SNSS`) se excluye automáticamente para no distorsionar las escalas.

2. **Dimensión Temporal Interactiva:**
   - Permite analizar el **promedio consolidado del período (2020-2024)** o seleccionar un **año específico** (2020 a 2024).

3. **Clustering y Diagnóstico de $N$:**
   - Algoritmo **K-Means** configurable mediante slider interactivo ($N$ entre 2 y 10).
   - Control de semilla aleatoria para garantizar reproducibilidad.
   - Pestaña de evaluación diagnóstica con curvas del **Método del Codo (Inercia)** y **Coeficiente de Silueta**, indicando visualmente el $N$ activo y sugiriendo el óptimo técnico.

4. **Visualización y Resumen de Grupos:**
   - **Mapa de Similitud 2D (PCA):** Proyección interactiva bidimensional con etiquetas y colores por cluster, permitiendo ver a simple vista qué servicios son más cercanos entre sí.
   - **Listado de miembros por cluster:** Tabla que detalla qué servicios componen cada grupo y cuántos son.
   - **Perfil Promedio (%):** Matriz coloreada con gradiente para identificar las especialidades y niveles preponderantes en cada cluster.
   - **Vista de datos brutos:** Tabla desplegable con la matriz normalizada.

---

## 🚀 Cómo Ejecutar la Aplicación

Desde la terminal en el directorio del proyecto, ejecuta:

```bash
streamlit run app.py
```

Se abrirá automáticamente en tu navegador predeterminado (por defecto en `http://localhost:8501`).
