# Análisis de Indicadores Económicos de México

Este proyecto analiza tres indicadores económicos clave de México: tipo de cambio, inflación y tasas de interés. El análisis incluye la extracción de datos de APIs oficiales (Banxico e INEGI), su procesamiento, almacenamiento en AWS, análisis estadístico y visualización interactiva.

## Estructura del Proyecto

```
analisis-economico/
├── app/                          # Aplicación Streamlit
│   └── economic_indicators_dashboard.py
├── config/                       # Archivos de configuración
│   └── tokens.yaml               # Tokens de API (no incluidos en el repositorio)
├── data/                         # Datos del proyecto
│   └── raw/                      # Datos sin procesar
│       ├── inflacion.csv         # Datos de inflación mensual (INEGI)
│       ├── tasas_interes.csv     # Datos de tasas CETES a 28 días (Banxico)
│       └── tipo_cambio.csv       # Datos de tipo de cambio FIX (Banxico)
├── environment.yml               # Entorno Conda para reproducibilidad
├── graphs/                       # Visualizaciones generadas
│   ├── regresion_tasa_de_interes_vs_inflacion.png
│   ├── regresion_tipo_de_cambio_vs_inflacion.png
│   └── regresion_tipo_de_cambio_vs_tasa_de_interes.png
└── notebooks/                    # Jupyter notebooks con análisis
    ├── economic_indicators_etl.ipynb         # Extracción inicial de datos
    ├── economic_indicators_elt_pipeline.ipynb # Carga de datos a AWS
    └── economic_indicators_regression_analysis.ipynb # Análisis de regresión
```

## Resumen del Proyecto

Este proyecto implementa un pipeline completo de análisis de datos económicos:

1. **Extracción de datos** de las APIs oficiales de Banxico e INEGI
2. **Transformación y limpieza** de los datos para análisis
3. **Carga de datos** en un bucket de S3 y creación de tablas en AWS Glue/Athena
4. **Análisis estadístico** incluyendo modelos de regresión lineal
5. **Visualización interactiva** a través de una aplicación Streamlit

## Instalación y Configuración

### Requisitos Previos

- Python 3.11 o superior
- Conda (para gestión de entornos)
- Cuenta de AWS con permisos para S3, Glue y Athena (opcional para ELT)
- Tokens de API para Banxico e INEGI

### Configuración del Entorno

1. Clona este repositorio:
   ```bash
   git clone https://github.com/yourusername/analisis-economico.git
   cd analisis-economico
   ```

2. Crea el entorno Conda:
   ```bash
   conda env create -f environment.yml
   conda activate analisis-economico
   ```

3. Configura los tokens de API:
   - Crea un archivo `config/tokens.yaml` con la siguiente estructura:
     ```yaml
     banxico:
       token: "tu_token_banxico_aquí"
     inegi:
       token: "tu_token_inegi_aquí"
     ```

   - Para obtener los tokens:
     - Banxico: [https://www.banxico.org.mx/SieAPIRest/service/v1/](https://www.banxico.org.mx/SieAPIRest/service/v1/)
     - INEGI: [https://www.inegi.org.mx/servicios/api_indicadores.html](https://www.inegi.org.mx/servicios/api_indicadores.html)

4. Configura las credenciales de AWS (opcional para ELT):
   ```bash
   aws configure --profile datascientist
   ```

## Ejecución del Proyecto

### 1. Extracción de Datos (ETL)

Para ejecutar la extracción inicial de datos desde Banxico e INEGI:

```bash
cd notebooks
jupyter notebook economic_indicators_etl.ipynb
```

Ejecuta todas las celdas del notebook para obtener los datos actualizados de tipo de cambio, inflación y tasas de interés. Los datos se guardarán en `data/raw/`.

### 2. Pipeline ELT en AWS (Opcional)

Si deseas cargar los datos en AWS para análisis en la nube:

```bash
cd notebooks
jupyter notebook economic_indicators_elt_pipeline.ipynb
```

Este notebook:
1. Sube los archivos CSV a un bucket de S3
2. Crea un catálogo en AWS Glue
3. Define tablas en Athena para consultas SQL

### 3. Análisis de Regresión

Para ejecutar los modelos de regresión y visualizaciones:

```bash
cd notebooks
jupyter notebook economic_indicators_regression_analysis.ipynb
```

Este análisis incluye:
- Exploración de datos y estadísticas descriptivas
- Correlaciones entre indicadores económicos
- Modelos de regresión lineal entre pares de variables
- Visualizaciones diagnósticas de los modelos

### 4. Dashboard Interactivo

Para ejecutar la aplicación Streamlit:

```bash
cd app
streamlit run economic_indicators_dashboard.py
```

El dashboard incluye:
- Visualización de series temporales
- Estadísticas descriptivas y distribuciones
- Matriz de correlación interactiva
- Modelos de regresión con interpretaciones automáticas

Accede al dashboard en tu navegador: [http://localhost:8501](http://localhost:8501)

## Indicadores Económicos Analizados

1. **Tipo de Cambio FIX (SF43718)**: Tipo de cambio peso-dólar publicado por Banxico para solventar obligaciones denominadas en dólares.

2. **Inflación Mensual (910399)**: Índice Nacional de Precios al Consumidor (INPC), medido como variación porcentual mensual.

3. **Tasa de Interés CETES a 28 días (SF43783)**: Rendimiento promedio mensual de los Certificados de la Tesorería de la Federación a 28 días.

## Modelos de Regresión

Se analizaron tres relaciones lineales entre los indicadores:

1. **Tipo de Cambio ~ Tasa de Interés**: Exploración de la relación entre política monetaria y mercado cambiario.

2. **Tasa de Interés ~ Inflación**: Análisis de la respuesta de la política monetaria a la inflación.

3. **Tipo de Cambio ~ Inflación**: Evaluación del impacto de la inflación en el tipo de cambio.

Los resultados muestran relaciones estadísticamente significativas aunque con valores R² relativamente bajos, sugiriendo que estos indicadores tienen relaciones más complejas que lo que un modelo lineal simple puede capturar.

## Resolución de Problemas

### Error de Caché en Streamlit

Si encuentras un error relacionado con "UnhashableParamError" al ejecutar la aplicación Streamlit, asegúrate de que los parámetros que no se pueden hashear (como clientes de AWS) tengan un guion bajo al inicio:

```python
@st.cache_data
def load_data_from_athena(_athena_client):
    # Código...
```

### Error con Módulos Itertools

Si encuentras errores relacionados con `groupby` o `count`, asegúrate de importar estas funciones:

```python
from itertools import groupby, count
```

## Contribuciones y Desarrollo Futuro

Áreas para expandir el proyecto:

- Incluir más indicadores económicos (PIB, balanza comercial, etc.)
- Implementar modelos más sofisticados para series temporales (ARIMA, VAR)
- Añadir predicciones y proyecciones de indicadores
- Mejorar la visualización con mapas de calor temporales

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.

## Reconocimientos

- Datos proporcionados por [Banco de México](https://www.banxico.org.mx/) e [INEGI](https://www.inegi.org.mx/)
- Inspirado en análisis económicos del sector financiero mexicano

---

© 2025 Análisis Económico México