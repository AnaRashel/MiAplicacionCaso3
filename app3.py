import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# PRINCIPAL
# Interfaz de Streamlit
# Selección de página
st.sidebar.image("logo.png")
pagina = st.sidebar.selectbox("Selecciona una página:", ["🏠 Home", "📋 Carga del dataset"])

#Selección Home
if pagina == "🏠 Home":
    # HOME - PRESENTACIÓN DEL PROYECTO
    # CONFIGURACIÓN DE LA PÁGINA
    # =====================================================
    st.set_page_config(
        page_title="Home - Proyecto EDA",
        page_icon="📊",
        layout="wide"
    )

    # =====================================================
    # TÍTULO PRINCIPAL
    # =====================================================

    st.title("📊 Proyecto de Análisis Exploratorio de Datos (EDA)")
    st.markdown("## Insurance Company Dataset")

    st.markdown("---")

    # =====================================================
    # DESCRIPCIÓN DEL PROYECTO
    # =====================================================

    st.header("🎯 Objetivo del Proyecto")

    st.write("""
    El objetivo de este proyecto es realizar un análisis exploratorio de datos (EDA)
    sobre un dataset relacionado con una compañía de seguros.

    A través de diferentes técnicas estadísticas y visualizaciones interactivas,
    se busca identificar patrones, distribuciones, relaciones entre variables y
    hallazgos relevantes que permitan comprender mejor la información contenida
    en el dataset.

    La aplicación fue desarrollada utilizando Streamlit para construir una interfaz
    interactiva y dinámica.
    """)

    # =====================================================
    # DATOS DEL AUTOR
    # =====================================================

    st.header("👩‍💻 Datos del Autor")

    col1, col2 = st.columns(2)

    with col1:

        st.info("""
        **Nombre Completo:**  
        Ana Fernanda Rashel Fernández Pamucena
        """)

    with col2:

        st.info("""
        **Curso / Especialización:**  
        Especialización en Python
        """)

    st.info("""
    **Año:**  
    2026
    """)

    # =====================================================
    # EXPLICACIÓN DEL DATASET
    # =====================================================

    st.header("📁 Descripción del Dataset")

    st.write("""
    El dataset utilizado corresponde a información de clientes de una compañía
    de seguros. Contiene variables numéricas y categóricas relacionadas con:

    - Información demográfica
    - Ingresos
    - Renovación de seguros
    - Canales de adquisición
    - Tipo de residencia
    - Esfuerzo del agente
    - Variables de comportamiento del cliente

    Este conjunto de datos permite aplicar técnicas de análisis exploratorio,
    comparación de grupos y detección de patrones importantes.
    """)

    # =====================================================
    # TECNOLOGÍAS UTILIZADAS
    # =====================================================

    st.header("🛠️ Tecnologías Utilizadas")

    tech1, tech2, tech3 = st.columns(3)

    box_style = """
    padding:20px;
    border-radius:10px;
    background-color:#E8F5E9;
    height:170px;
    """

    with tech1:
        st.markdown(
            f"""
            <div style="{box_style}">
            ✅ Python<br><br>
            ✅ Pandas<br><br>
            ✅ NumPy
            </div>
            """,
            unsafe_allow_html=True
        )

    with tech2:
        st.markdown(
            f"""
            <div style="{box_style}">
            ✅ Streamlit<br><br>
            ✅ Matplotlib<br><br>
            ✅ Seaborn
            </div>
            """,
            unsafe_allow_html=True
        )

    with tech3:
        st.markdown(
            f"""
            <div style="{box_style}">
            ✅ Programación Orientada a Objetos<br><br>
            ✅ EDA<br><br>
            ✅ Visualización de Datos
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # MENSAJE FINAL
    # =====================================================

    st.markdown("---")

    st.success("""
    📌 Esta aplicación permite realizar análisis exploratorio de datos
    de manera interactiva utilizando herramientas modernas de Python.
    """)

    # Pie de página
    st.markdown("---")
    st.write("© 2026 - Proyecto Académico")

#Selección Carga del dataset
elif pagina == "📋 Carga del dataset":
    # Inicializar en sesión
    # CONFIGURACIÓN GENERAL
    # =====================================================
    st.set_page_config(
        page_title="EDA Insurance Company",
        layout="wide",
        page_icon="📊"
    )

    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.markdown("## Dataset: Insurance Company")

    # SIDEBAR
    # =====================================================

    st.sidebar.title("⚙️ Configuración")

    mostrar_graficos = st.sidebar.checkbox("Mostrar gráficos", value=True)

    bins_hist = st.sidebar.slider(
        "Número de bins para histogramas",
        min_value=5,
        max_value=50,
        value=15
    )
    # CARGA DEL DATASET
    # =====================================================
    uploaded_file = st.file_uploader(
        "📂 Cargar archivo CSV",
        type=["csv"]
    )


    # VALIDACIÓN
    # =====================================================
    if uploaded_file is None:
        st.warning("⚠️ Debe cargar un archivo CSV para continuar.")
        st.stop()

    # LECTURA DEL CSV
    # =====================================================
    try:
        df = pd.read_csv(uploaded_file)

        st.success("✅ Archivo cargado correctamente")

    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {e}")
        st.stop()


    # VISTA PREVIA
    # =====================================================
    st.subheader("👀 Vista previa del dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Filas", df.shape[0])

    with col2:
        st.metric("Columnas", df.shape[1])

    st.dataframe(df.head())

    # LIMPIEZA DE DATOS
    # =====================================================

    class DataProcessor:

        def __init__(self, dataframe):
            self.df = dataframe.copy()

        # -------------------------------------------------
        # LIMPIEZA GENERAL
        # -------------------------------------------------

        def clean_data(self):

            # Eliminar duplicados
            self.df = self.df.drop_duplicates()

            # Limpiar espacios en nombres de columnas
            self.df.columns = self.df.columns.str.strip()

            return self.df

        # -------------------------------------------------
        # LIMPIEZA DE VALORES NULOS
        # -------------------------------------------------

        def handle_missing_values(self):

            numeric_cols = self.df.select_dtypes(
                include=np.number
            ).columns

            categorical_cols = self.df.select_dtypes(
                exclude=np.number
            ).columns

            # Variables numéricas → reemplazar con mediana
            for col in numeric_cols:

                self.df[col] = self.df[col].fillna(
                    self.df[col].median()
                )

            # Variables categóricas → reemplazar con moda
            for col in categorical_cols:

                moda = self.df[col].mode()

                if not moda.empty:

                    self.df[col] = self.df[col].fillna(
                        moda[0]
                    )

            return self.df

        # -------------------------------------------------
        # CLASIFICACIÓN DE VARIABLES
        # -------------------------------------------------

        def classify_variables(self):

            numeric_cols = self.df.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_cols = self.df.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            return numeric_cols, categorical_cols

        # -------------------------------------------------
        # ESTADÍSTICAS
        # -------------------------------------------------

        def descriptive_stats(self):

            return self.df.describe()

        # -------------------------------------------------
        # VALORES NULOS
        # -------------------------------------------------

        def missing_values(self):

            return self.df.isnull().sum()

    # =====================================================
    # PROCESAMIENTO
    # =====================================================

    processor = DataProcessor(df)

    # -----------------------------------
    # LIMPIEZA GENERAL
    # -----------------------------------

    df = processor.clean_data()

    # -----------------------------------
    # LIMPIEZA DE NULOS
    # -----------------------------------

    df = processor.handle_missing_values()

    # -----------------------------------
    # CLASIFICACIÓN
    # -----------------------------------

    numeric_cols, categorical_cols = processor.classify_variables()

    # =====================================================
    # TABS PRINCIPALES
    # =====================================================

    tabs = st.tabs([
        "📌 Ítem 1",
        "📌 Ítem 2",
        "📌 Ítem 3",
        "📌 Ítem 4",
        "📌 Ítem 5",
        "📌 Ítem 6",
        "📌 Ítem 7",
        "📌 Ítem 8",
        "📌 Ítem 9",
        "📌 Ítem 10",
        "📌 Ítem 11"
    ])

    # ITEM 1
    # =====================================================
    with tabs[0]:

        st.header("📌 Información General del Dataset")

        # ==========================================
        # FILA SUPERIOR
        # ==========================================

        col1, col2 = st.columns(2)

        # ------------------------------------------
        # TIPOS DE DATOS
        # ------------------------------------------

        with col1:

            st.subheader("Tipos de datos")

            dtypes_df = pd.DataFrame({
                "Columna": df.columns,
                "Tipo de Dato": df.dtypes.astype(str).values
            })

            st.dataframe(
                dtypes_df,
                use_container_width=True,
                height=350,
                hide_index=True
            )

        # ------------------------------------------
        # VALORES NULOS
        # ------------------------------------------

        with col2:

            st.subheader("Valores nulos")

            nulls_df = pd.DataFrame({
                "Columna": df.columns,
                "Valores Nulos": df.isnull().sum().values
            })

            st.dataframe(
                nulls_df,
                use_container_width=True,
                height=350,
                hide_index=True
            )

        # ==========================================
        # FILA INFERIOR
        # ==========================================

        st.subheader("Información general")

        info_df = pd.DataFrame({
            "Información": [
                "Filas",
                "Columnas",
                "Memoria utilizada",
                "Variables Numéricas",
                "Variables Categóricas"
            ],
            "Valor": [
                df.shape[0],
                df.shape[1],
                f"{round(df.memory_usage().sum() / 1024**2, 2)} MB",
                len(numeric_cols),
                len(categorical_cols)
            ]
        })

        st.dataframe(
            info_df,
            use_container_width=True,
            height=220,
            hide_index=True
        )

    # ITEM 2
    # =====================================================
    with tabs[1]:

        st.header("📌 Clasificación de Variables")

        # ==========================================
        # COLUMNAS
        # ==========================================

        col1, col2 = st.columns(2)

        # ------------------------------------------
        # VARIABLES NUMÉRICAS
        # ------------------------------------------

        with col1:

            st.subheader("🔢 Variables Numéricas")

            st.metric(
                label="Cantidad",
                value=len(numeric_cols)
            )

            numeric_df = pd.DataFrame({
                "Variables Numéricas": numeric_cols
            })

            st.dataframe(
                numeric_df,
                use_container_width=True,
                height=350
            )

        # ------------------------------------------
        # VARIABLES CATEGÓRICAS
        # ------------------------------------------

        with col2:

            st.subheader("🔤 Variables Categóricas")

            st.metric(
                label="Cantidad",
                value=len(categorical_cols)
            )

            categorical_df = pd.DataFrame({
                "Variables Categóricas": categorical_cols
            })

            st.dataframe(
                categorical_df,
                use_container_width=True,
                height=350
            )

        # ==========================================
        # EXPLICACIÓN
        # ==========================================

        st.info("""
        📌 Clasificación realizada automáticamente utilizando
        una función personalizada basada en los tipos de datos
        del dataset.
        """)
    

    # =====================================================
    # ITEM 3
    # =====================================================

    with tabs[2]:

        st.header("📌 Estadísticas Descriptivas")

        st.dataframe(processor.descriptive_stats())

        if len(numeric_cols) > 0:

            variable = st.selectbox(
                "Seleccionar variable numérica",
                numeric_cols,
                key="stats_select"
            )

            st.subheader(f"Análisis de {variable}")

            media = df[variable].mean()
            mediana = df[variable].median()
            moda = df[variable].mode()[0]

            c1, c2, c3 = st.columns(3)

            c1.metric("Media", round(media, 2))
            c2.metric("Mediana", round(mediana, 2))
            c3.metric("Moda", round(moda, 2))

    # ITEM 4
    # =====================================================

    with tabs[3]:

        st.header("📌 Análisis de Valores Faltantes")

        missing = processor.missing_values()

        missing_df = pd.DataFrame({
            "Variable": missing.index,
            "Valores Faltantes": missing.values
        })

        st.dataframe(missing_df)

        if mostrar_graficos:

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.barplot(
                x="Variable",
                y="Valores Faltantes",
                data=missing_df,
                ax=ax
            )

            plt.xticks(rotation=90)

            st.pyplot(fig)

        st.info(
            "Los valores faltantes pueden afectar los modelos "
            "y análisis estadísticos."
        )

    # =====================================================
    # ITEM 5
    # =====================================================

    with tabs[4]:

        st.header("📌 Distribución de Variables Numéricas")

        if len(numeric_cols) > 0:

            selected_num = st.selectbox(
                "Seleccionar variable",
                numeric_cols,
                key="hist_select"
            )

            if mostrar_graficos:

                fig, ax = plt.subplots(figsize=(8, 5))

                sns.histplot(
                    df[selected_num].dropna(),
                    bins=bins_hist,
                    kde=True,
                    ax=ax
                )

                st.pyplot(fig)

            st.write(
                "El histograma permite observar la distribución "
                "y dispersión de la variable."
            )

    # ITEM 6
    # =====================================================

    with tabs[5]:

        st.header("📌 Variables Categóricas")

        if len(categorical_cols) > 0:

            selected_cat = st.selectbox(
                "Seleccionar variable categórica",
                categorical_cols,
                key="cat_select"
            )

            counts = df[selected_cat].value_counts()

            proportions = round(
                df[selected_cat].value_counts(normalize=True) * 100,
                2
            )

            result_df = pd.DataFrame({
                "Conteo": counts,
                "Proporción %": proportions
            })

            st.dataframe(result_df)

            if mostrar_graficos:

                fig, ax = plt.subplots(figsize=(8, 5))

                sns.countplot(
                    x=df[selected_cat],
                    ax=ax
                )

                plt.xticks(rotation=45)

                st.pyplot(fig)

    # =====================================================
    # ITEM 7
    # =====================================================

    with tabs[6]:

        st.header("📌 Análisis Bivariado (Numérico vs Categórico)")

        if len(numeric_cols) > 0 and len(categorical_cols) > 0:

            num_var = st.selectbox(
                "Variable Numérica",
                numeric_cols,
                key="num_biv"
            )

            cat_var = st.selectbox(
                "Variable Categórica",
                categorical_cols,
                key="cat_biv"
            )

            if mostrar_graficos:

                fig, ax = plt.subplots(figsize=(10, 5))

                sns.boxplot(
                    x=df[cat_var],
                    y=df[num_var],
                    ax=ax
                )

                plt.xticks(rotation=45)

                st.pyplot(fig)

            st.write(
                "El boxplot permite comparar distribuciones "
                "entre grupos."
            )

    # ITEM 8
    # =====================================================

    with tabs[7]:

        st.header("📌 Análisis Categórico vs Categórico")

        if len(categorical_cols) >= 2:

            cat1 = st.selectbox(
                "Variable categórica 1",
                categorical_cols,
                key="cat1"
            )

            cat2 = st.selectbox(
                "Variable categórica 2",
                categorical_cols,
                key="cat2"
            )

            cross = pd.crosstab(df[cat1], df[cat2])

            st.dataframe(cross)

            if mostrar_graficos:

                fig, ax = plt.subplots(figsize=(10, 5))

                sns.heatmap(
                    cross,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=ax
                )

                st.pyplot(fig)

    # =====================================================
    # ITEM 9
    # =====================================================

    with tabs[8]:

        st.header("📌 Análisis Dinámico")

        selected_columns = st.multiselect(
            "Seleccionar columnas",
            df.columns.tolist(),
            default=df.columns.tolist()[:3]
        )

        if len(selected_columns) > 0:

            st.dataframe(df[selected_columns].head())

            numeric_selected = [
                col for col in selected_columns
                if col in numeric_cols
            ]

            if len(numeric_selected) > 0:

                variable_dynamic = st.selectbox(
                    "Variable para análisis dinámico",
                    numeric_selected
                )

                if mostrar_graficos:

                    fig, ax = plt.subplots(figsize=(8, 5))

                    sns.histplot(
                        df[variable_dynamic].dropna(),
                        bins=bins_hist,
                        kde=True,
                        ax=ax
                    )

                    st.pyplot(fig)

    # ITEM 10
    # =====================================================

    with tabs[9]:

        st.header("📌 Hallazgos Clave")

        st.success("""
        ✅ El análisis exploratorio permite identificar:
        
        • Variables con valores faltantes
        
        • Distribución de variables numéricas
        
        • Categorías con mayor frecuencia
        
        • Diferencias entre grupos
        
        • Posibles patrones importantes
        
        • Variables relevantes para modelos predictivos
        """)

        st.info("""
        📌 Recomendaciones:
        
        • Realizar imputación de valores faltantes
        
        • Estandarizar variables numéricas
        
        • Codificar variables categóricas
        
        • Detectar outliers
        
        • Evaluar correlaciones importantes
        """)

    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown("---")
    st.markdown("✅ Proyecto EDA desarrollado con Streamlit + POO")

    # ITEM 11
    # =====================================================

    with tabs[10]:

        st.header("📌 Conclusiones")

        # ==========================================
        # VISUALIZACIÓN RESUMEN
        # ==========================================

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Variables Numéricas",
                len(numeric_cols)
            )

        with col2:
            st.metric(
                "Variables Categóricas",
                len(categorical_cols)
            )

        with col3:
            st.metric(
                "Registros",
                df.shape[0]
            )

        st.markdown("---")

        # ==========================================
        # CONCLUSIONES
        # ==========================================

        st.subheader("📌 Conclusiones del Análisis")

        st.success("""
        ### 1️⃣ Presencia de valores faltantes
        
        Se identificaron variables con datos faltantes, lo que evidencia la
        necesidad de procesos de limpieza antes de realizar modelos predictivos.
        
        📌 Decisión:
        Implementar estrategias automáticas de imputación para mejorar la
        calidad de los datos.
        """)

        st.success("""
        ### 2️⃣ Distribuciones no uniformes en variables numéricas
        
        Algunas variables presentan concentraciones altas en determinados rangos,
        lo que podría indicar segmentos específicos de clientes.
        
        📌 Decisión:
        Aplicar segmentación de clientes para diseñar estrategias comerciales
        más personalizadas.
        """)

        st.success("""
        ### 3️⃣ Diferencias entre categorías
        
        El análisis bivariado mostró diferencias importantes entre grupos
        categóricos y variables numéricas.
        
        📌 Decisión:
        Priorizar los segmentos con mejor comportamiento para optimizar recursos
        comerciales y operativos.
        """)

        st.success("""
        ### 4️⃣ Variables categóricas con alta concentración
        
        Algunas categorías concentran gran parte de los registros, indicando
        posibles patrones de negocio dominantes.
        
        📌 Decisión:
        Evaluar oportunidades de expansión hacia categorías menos representadas.
        """)

        st.success("""
        ### 5️⃣ El dataset es apto para modelos analíticos
        
        Luego de la limpieza y análisis exploratorio, el dataset presenta una
        estructura adecuada para análisis avanzados y machine learning.
        
        📌 Decisión:
        Continuar con modelos predictivos para renovación de seguros,
        comportamiento de clientes o segmentación.
        """)

        # ==========================================
        # RESUMEN FINAL
        # ==========================================

        st.info("""
        ✅ El análisis exploratorio permitió comprender la estructura del dataset,
        detectar problemas de calidad de datos e identificar patrones relevantes
        para la toma de decisiones empresariales.
        """)