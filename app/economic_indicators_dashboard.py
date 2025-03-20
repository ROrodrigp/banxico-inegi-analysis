# Economic Indicators Dashboard
# Esta aplicación muestra los resultados del análisis de indicadores económicos de México
# y los modelos de regresión entre tipo de cambio, tasas de interés e inflación

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import io
import statsmodels.api as sm
from PIL import Image
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Indicadores Económicos de México",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Variables globales
BUCKET_NAME = "itam-analytics-ragp"
DATABASE_NAME = "econ"
PROFILE_NAME = "datascientist"

# Función para conectar con AWS
@st.cache_resource
def get_aws_clients():
    try:
        session = boto3.Session(profile_name=PROFILE_NAME)
        s3_client = session.client('s3')
        athena_client = session.client('athena')
        return s3_client, athena_client
    except Exception as e:
        st.error(f"Error al conectar con AWS: {e}")
        return None, None

# Función para cargar datos desde Athena
@st.cache_data
def load_data_from_athena(_athena_client):
    try:
        # Ubicación de resultados de Athena
        s3_output = "s3://arquitectura-athena-queries-ragp/"
        
        # Consulta para obtener datos consolidados
        query = f"""
        SELECT 
            date, 
            tasa_de_interes, 
            inflacion, 
            tipo_de_cambio
        FROM 
            {DATABASE_NAME}.indicadores_economicos_consolidados
        ORDER BY 
            date
        """
        
        # Ejecutar consulta
        response = _athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": DATABASE_NAME},
            ResultConfiguration={"OutputLocation": s3_output}
        )
        
        query_id = response["QueryExecutionId"]
        
        # Esperar a que termine la consulta
        import time
        status = "RUNNING"
        while status in ["RUNNING", "QUEUED"]:
            response = _athena_client.get_query_execution(QueryExecutionId=query_id)
            status = response["QueryExecution"]["Status"]["State"]
            if status in ["RUNNING", "QUEUED"]:
                time.sleep(1)
        
        if status == "SUCCEEDED":
            # Obtener resultados
            result = _athena_client.get_query_results(QueryExecutionId=query_id)
            
            # Extraer nombres de columnas
            columns = [col['Label'] for col in result['ResultSet']['ResultSetMetadata']['ColumnInfo']]
            
            # Extraer datos (omitir primera fila que son encabezados)
            rows = []
            for row in result['ResultSet']['Rows'][1:]:
                values = [cell.get('VarCharValue', None) for cell in row['Data']]
                rows.append(values)
            
            # Crear DataFrame
            df = pd.DataFrame(rows, columns=columns)
            
            # Convertir tipos de datos
            df['date'] = pd.to_datetime(df['date'])
            numeric_cols = ['tasa_de_interes', 'inflacion', 'tipo_de_cambio']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
            
            return df
        else:
            error = response["QueryExecution"]["Status"].get("StateChangeReason", "Unknown error")
            st.error(f"La consulta Athena falló: {error}")
            return None
    
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Función alternativa para cargar datos desde CSV local (si no hay conexión a AWS)
@st.cache_data
def load_data_from_csv():
    try:
        return pd.read_csv('indicadores_economicos_consolidados.csv', 
                          parse_dates=['date'])
    except Exception as e:
        st.error(f"Error al cargar datos locales: {e}")
        # Crear datos de ejemplo si no hay archivo
        dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='M')
        df = pd.DataFrame({
            'date': dates,
            'tasa_de_interes': np.random.uniform(4, 11, len(dates)),
            'inflacion': np.random.uniform(0, 6, len(dates)),
            'tipo_de_cambio': np.random.uniform(18, 22, len(dates))
        })
        return df

# Función para generar estadísticas descriptivas
def generate_descriptive_stats(df):
    return df[['tasa_de_interes', 'inflacion', 'tipo_de_cambio']].describe()

# Función para generar matriz de correlación
def generate_correlation_matrix(df):
    return df[['tasa_de_interes', 'inflacion', 'tipo_de_cambio']].corr()

# Función para realizar regresión lineal
def run_regression(df, x_var, y_var):
    X = df[x_var]
    X = sm.add_constant(X)
    y = df[y_var]
    model = sm.OLS(y, X).fit()
    return model

# Función para graficar series temporales
def plot_time_series(df):
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Tipo de cambio
    ax[0].plot(df['date'], df['tipo_de_cambio'], 'b-')
    ax[0].set_title('Evolución del Tipo de Cambio')
    ax[0].set_ylabel('MXN por USD')
    ax[0].grid(True)
    
    # Tasa de interés
    ax[1].plot(df['date'], df['tasa_de_interes'], 'r-')
    ax[1].set_title('Evolución de la Tasa de Interés (CETES 28 días)')
    ax[1].set_ylabel('Porcentaje (%)')
    ax[1].grid(True)
    
    # Inflación
    ax[2].plot(df['date'], df['inflacion'], 'g-')
    ax[2].set_title('Evolución de la Inflación Mensual')
    ax[2].set_ylabel('Porcentaje (%)')
    ax[2].set_xlabel('Fecha')
    ax[2].grid(True)
    
    plt.tight_layout()
    return fig

# Función para graficar regresión
def plot_regression(df, x_var, y_var, model):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot de datos originales
    ax.scatter(df[x_var], df[y_var], alpha=0.5)
    
    # Línea de regresión
    x_range = np.linspace(df[x_var].min(), df[x_var].max(), 100)
    X_range = sm.add_constant(x_range)
    y_pred = model.predict(X_range)
    ax.plot(x_range, y_pred, 'r-', linewidth=2)
    
    # Etiquetas y título
    ax.set_xlabel(x_var.replace('_', ' ').title())
    ax.set_ylabel(y_var.replace('_', ' ').title())
    ax.set_title(f'Regresión Lineal: {y_var.replace("_", " ").title()} vs {x_var.replace("_", " ").title()}')
    ax.grid(True)
    
    # Añadir ecuación y R² al gráfico
    equation = f"y = {model.params[0]:.4f} + {model.params[1]:.4f}x"
    r_squared = f"R² = {model.rsquared:.4f}"
    p_value = f"p-valor = {model.pvalues[1]:.4f}"
    ax.annotate(f"{equation}\n{r_squared}\n{p_value}", 
                xy=(0.05, 0.95), 
                xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
    
    return fig

# Función para crear heatmap de correlación
def plot_correlation_heatmap(corr_matrix):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title('Matriz de Correlación entre Indicadores Económicos')
    return fig

# Función principal de la aplicación
def main():
    # Título principal
    st.title("📊 Dashboard de Indicadores Económicos de México")
    st.write("Esta aplicación muestra los resultados del análisis de indicadores económicos y los modelos de regresión.")
    
    # Barra lateral con información y controles
    st.sidebar.header("Información")
    st.sidebar.info("""
    Este dashboard muestra los resultados del análisis de indicadores económicos de México:
    - Tipo de cambio (MXN/USD)
    - Tasa de interés (CETES 28 días)
    - Inflación mensual
    
    Los datos provienen de Banco de México e INEGI, procesados a través de un pipeline ELT en AWS.
    """)
    
    # Selector de modo de carga de datos
    data_source = st.sidebar.radio("Fuente de datos:", ["AWS Athena", "Archivo local"])
    
    # Cargar datos
    if data_source == "AWS Athena":
        s3_client, athena_client = get_aws_clients()
        if athena_client:
            with st.spinner("Cargando datos desde AWS Athena..."):
                df = load_data_from_athena(athena_client)
        else:
            st.warning("No se pudo conectar con AWS. Cambiando a datos locales.")
            df = load_data_from_csv()
    else:
        df = load_data_from_csv()
    
    if df is None or len(df) == 0:
        st.error("No se pudieron cargar los datos. Revise la conexión o el archivo local.")
        return
    
    # Mostrar rango de fechas disponible
    st.sidebar.subheader("Rango de fechas")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    st.sidebar.text(f"Datos disponibles desde: {min_date}")
    st.sidebar.text(f"Datos disponibles hasta: {max_date}")
    
    # Filtro de fechas
    date_range = st.sidebar.date_input(
        "Seleccione rango de fechas:",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    else:
        filtered_df = df
    
    # Pestañas para diferentes secciones del dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["Series Temporales", "Estadísticas", "Correlaciones", "Modelos de Regresión"])
    
    # Pestaña 1: Series Temporales
    with tab1:
        st.header("Evolución de Indicadores Económicos")
        st.write("Las siguientes gráficas muestran la evolución temporal de los principales indicadores económicos de México.")
        
        fig_series = plot_time_series(filtered_df)
        st.pyplot(fig_series)
        
        # Botón para descargar datos
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar datos filtrados como CSV",
            data=csv,
            file_name="indicadores_economicos_filtrados.csv",
            mime="text/csv"
        )
    
    # Pestaña 2: Estadísticas
    with tab2:
        st.header("Estadísticas Descriptivas")
        st.write("Resumen estadístico de los indicadores económicos en el período seleccionado.")
        
        stats = generate_descriptive_stats(filtered_df)
        st.dataframe(stats.style.highlight_max(axis=0, color='lightgreen'))
        
        # Gráficos de distribución
        st.subheader("Distribución de Indicadores")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(filtered_df['tipo_de_cambio'], kde=True, ax=ax)
            ax.set_title("Distribución del Tipo de Cambio")
            ax.set_xlabel("Pesos por Dólar")
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(filtered_df['tasa_de_interes'], kde=True, ax=ax)
            ax.set_title("Distribución de Tasas de Interés")
            ax.set_xlabel("Porcentaje (%)")
            st.pyplot(fig)
        
        with col3:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(filtered_df['inflacion'], kde=True, ax=ax)
            ax.set_title("Distribución de Inflación")
            ax.set_xlabel("Porcentaje (%)")
            st.pyplot(fig)
    
    # Pestaña 3: Correlaciones
    with tab3:
        st.header("Análisis de Correlaciones")
        st.write("Este análisis muestra las relaciones lineales entre los diferentes indicadores económicos.")
        
        # Calcular matriz de correlación
        corr_matrix = generate_correlation_matrix(filtered_df)
        
        # Mostrar matriz de correlación
        st.subheader("Matriz de Correlación")
        fig_corr = plot_correlation_heatmap(corr_matrix)
        st.pyplot(fig_corr)
        
        # Scatter plots para visualizar relaciones
        st.subheader("Diagramas de Dispersión")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(filtered_df['tasa_de_interes'], filtered_df['tipo_de_cambio'], alpha=0.6)
            ax.set_xlabel("Tasa de Interés (%)")
            ax.set_ylabel("Tipo de Cambio (MXN/USD)")
            ax.set_title("Tipo de Cambio vs Tasa de Interés")
            ax.grid(True)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(filtered_df['inflacion'], filtered_df['tasa_de_interes'], alpha=0.6, color='green')
            ax.set_xlabel("Inflación (%)")
            ax.set_ylabel("Tasa de Interés (%)")
            ax.set_title("Tasa de Interés vs Inflación")
            ax.grid(True)
            st.pyplot(fig)
        
        with col3:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(filtered_df['inflacion'], filtered_df['tipo_de_cambio'], alpha=0.6, color='red')
            ax.set_xlabel("Inflación (%)")
            ax.set_ylabel("Tipo de Cambio (MXN/USD)")
            ax.set_title("Tipo de Cambio vs Inflación")
            ax.grid(True)
            st.pyplot(fig)
    
    # Pestaña 4: Modelos de Regresión
    with tab4:
        st.header("Modelos de Regresión Lineal")
        st.write("""
        Esta sección presenta los resultados de los modelos de regresión lineal entre los indicadores económicos.
        Estos modelos exploran las relaciones directas entre pares de variables.
        """)
        
        # Seleccionar modelo a visualizar
        model_option = st.selectbox(
            "Seleccione el modelo a visualizar:",
            ["Tipo de Cambio ~ Tasa de Interés", 
             "Tasa de Interés ~ Inflación", 
             "Tipo de Cambio ~ Inflación"]
        )
        
        # Configurar variables según el modelo seleccionado
        if model_option == "Tipo de Cambio ~ Tasa de Interés":
            y_var = "tipo_de_cambio"
            x_var = "tasa_de_interes"
            title_y = "Tipo de Cambio"
            title_x = "Tasa de Interés"
        elif model_option == "Tasa de Interés ~ Inflación":
            y_var = "tasa_de_interes"
            x_var = "inflacion"
            title_y = "Tasa de Interés"
            title_x = "Inflación"
        else:  # "Tipo de Cambio ~ Inflación"
            y_var = "tipo_de_cambio"
            x_var = "inflacion"
            title_y = "Tipo de Cambio"
            title_x = "Inflación"
        
        # Ejecutar regresión
        model = run_regression(filtered_df, x_var, y_var)
        
        # Mostrar resultados en dos columnas
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Gráfico de regresión
            fig_reg = plot_regression(filtered_df, x_var, y_var, model)
            st.pyplot(fig_reg)
        
        with col2:
            # Resumen del modelo
            st.subheader("Resumen del Modelo")
            
            # Mostrar ecuación
            st.markdown(f"**Ecuación de la recta:**")
            st.latex(f"{title_y} = {model.params[0]:.4f} + {model.params[1]:.4f} \\times {title_x}")
            
            # Mostrar métricas clave
            st.markdown("**Métricas del modelo:**")
            metrics = pd.DataFrame({
                'Métrica': ['R²', 'R² Ajustado', 'Valor F', 'P-valor (F)', 'AIC', 'BIC'],
                'Valor': [
                    f"{model.rsquared:.4f}",
                    f"{model.rsquared_adj:.4f}",
                    f"{model.fvalue:.4f}",
                    f"{model.f_pvalue:.4f}",
                    f"{model.aic:.2f}",
                    f"{model.bic:.2f}"
                ]
            })
            st.dataframe(metrics)
            
            # Interpretación de los resultados
            st.markdown("**Interpretación:**")
            
            # Significancia
            if model.f_pvalue < 0.05:
                st.success(f"✅ El modelo es estadísticamente significativo (p-valor = {model.f_pvalue:.4f} < 0.05)")
            else:
                st.warning(f"⚠️ El modelo no es estadísticamente significativo (p-valor = {model.f_pvalue:.4f} > 0.05)")
            
            # Coeficiente
            if model.params[1] > 0:
                relationship = "positiva"
            else:
                relationship = "negativa"
                
            st.write(f"Existe una relación {relationship} entre {title_x} y {title_y}. Por cada unidad de aumento en {title_x}, {title_y} cambia en {model.params[1]:.4f} unidades.")
            
            # Bondad de ajuste
            if model.rsquared < 0.3:
                st.write(f"Sin embargo, el valor R² de {model.rsquared:.4f} indica que el modelo solo explica {model.rsquared*100:.2f}% de la variabilidad en {title_y}, lo que sugiere que la relación lineal es débil.")
            elif model.rsquared < 0.7:
                st.write(f"El valor R² de {model.rsquared:.4f} indica que el modelo explica {model.rsquared*100:.2f}% de la variabilidad en {title_y}, lo que sugiere una relación lineal moderada.")
            else:
                st.write(f"El valor R² de {model.rsquared:.4f} indica que el modelo explica {model.rsquared*100:.2f}% de la variabilidad en {title_y}, lo que sugiere una relación lineal fuerte.")

    # Sección de información adicional al final de la página
    st.markdown("---")
    st.subheader("Sobre este Dashboard")
    st.write("""
    Este dashboard muestra los resultados de tres fases de análisis económico:
    1. **Pipeline ETL/ELT**: Extracción de datos de Banxico e INEGI, carga en S3 y transformación en AWS Glue/Athena
    2. **Análisis Exploratorio**: Estadísticas descriptivas y visualización de tendencias temporales
    3. **Modelos de Regresión**: Análisis de relaciones entre indicadores económicos clave
    
    Desarrollado como parte del proyecto de análisis de indicadores económicos de México.
    """)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()