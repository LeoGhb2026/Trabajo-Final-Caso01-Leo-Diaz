import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analyzer import DataAnalyzer

st.set_page_config(page_title="Bank Marketing EDA", layout="wide")

# SIDEBAR
st.sidebar.title("Menú")

menu = st.sidebar.radio(
    "Navegación",
    ["Home","Carga de Dataset","EDA"]
)

# ---------------------------------------------------
# HOME
# ---------------------------------------------------

if menu == "Home":

    st.title("Análisis Exploratorio - Bank Marketing")

    st.write("""
    Esta aplicación analiza el dataset **BankMarketing** utilizando técnicas de
    **Análisis Exploratorio de Datos (EDA)**.
    """)

    st.subheader("Autor")

    st.write("""
    Nombre: Leonardo Díaz Vargas
    
    Curso: Python for Analytics  
    
    Año: 2026
    """)

    st.subheader("Dataset")

    st.write("""
    Dataset de campañas de marketing de una institución financiera
    que busca entender qué factores influyen en la aceptación
    de sus campañas.
    """)

    st.subheader("Tecnologías")

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Streamlit
    - Matplotlib
    - Seaborn
    """)

# ---------------------------------------------------
# CARGA DE DATASET
# ---------------------------------------------------

elif menu == "Carga de Dataset":

    st.title("Carga del Dataset")

    uploaded_file = st.file_uploader("Sube el archivo CSV")

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file, sep=";")

        st.success("Dataset cargado correctamente")

        st.subheader("Vista previa")

        st.dataframe(df)

        st.subheader("Dimensiones")

        st.write(f"Filas: {df.shape[0]}")
        st.write(f"Columnas: {df.shape[1]}")

        st.session_state["data"] = df

    else:

        st.warning("Debes cargar un dataset para continuar")

# ---------------------------------------------------
# EDA
# ---------------------------------------------------

elif menu == "EDA":

    if "data" not in st.session_state:

        st.warning("Primero debes cargar el dataset")

    else:

        df = st.session_state["data"]

        analyzer = DataAnalyzer(df)

        st.title("Análisis Exploratorio")

        tab1,tab2,tab3,tab4,tab5 = st.tabs([
            "Información",
            "Variables",
            "Estadísticas",
            "Distribuciones",
            "Análisis Avanzado"
        ])

        # ------------------------------------------------
        # ITEM 1
        # ------------------------------------------------

        with tab1:

            st.subheader("Información del Dataset")

            st.write("Dimensiones del dataset:")
            st.write(df.shape)

            st.write("Tipos de datos:")
            st.write(df.dtypes)

            st.subheader("Valores nulos")

            st.dataframe(analyzer.missing_values())

        # ------------------------------------------------
        # ITEM 2
        # ------------------------------------------------

        with tab2:

            st.subheader("Clasificación de variables")

            numeric, categorical = analyzer.classify_variables()

            col1,col2 = st.columns(2)

            with col1:
                st.write("Variables numéricas")
                st.write(numeric)

            with col2:
                st.write("Variables categóricas")
                st.write(categorical)

        # ------------------------------------------------
        # ITEM 3
        # ------------------------------------------------

        with tab3:

            st.subheader("Estadísticas descriptivas")

            st.dataframe(analyzer.descriptive_stats())

        # ------------------------------------------------
        # ITEM 5
        # ------------------------------------------------

        with tab4:

            st.subheader("Distribución de variables numéricas")

            numeric, categorical = analyzer.classify_variables()

            if len(numeric) > 0:

                variable = st.selectbox(
                    "Selecciona variable numérica",
                    numeric
                )

                fig,ax = plt.subplots()

                sns.histplot(df[variable], kde=True, ax=ax)

                st.pyplot(fig)

            else:

                st.warning("No hay variables numéricas en el dataset")

        # ------------------------------------------------
        # ITEMS 6-10
        # ------------------------------------------------

        with tab5:

            st.subheader("Análisis de variables categóricas")

            numeric, categorical = analyzer.classify_variables()

            if len(categorical) > 0:

                cat_var = st.selectbox(
                    "Variable categórica",
                    categorical
                )

                counts = df[cat_var].value_counts()

                fig,ax = plt.subplots()

                sns.barplot(
                    x=counts.index,
                    y=counts.values,
                    ax=ax
                )

                plt.xticks(rotation=45)

                st.pyplot(fig)

            st.subheader("Análisis bivariado")

            if len(numeric) > 0 and len(categorical) > 0:

                num_var = st.selectbox("Variable numérica", numeric)

                cat_var2 = st.selectbox("Categoría", categorical)

                fig,ax = plt.subplots()

                sns.boxplot(
                    x=df[cat_var2],
                    y=df[num_var],
                    ax=ax
                )

                plt.xticks(rotation=45)

                st.pyplot(fig)

            st.subheader("Análisis dinámico")

            selected_cols = st.multiselect(
                "Selecciona columnas",
                df.columns
            )

            if selected_cols:
                st.dataframe(df[selected_cols].describe())

            # ------------------------------------------------
            # CONCLUSIONES
            # ------------------------------------------------

            st.subheader("Conclusiones Finales")

            st.write("""
            **1. Importancia de la duración del contacto**

            El análisis muestra que la duración de la llamada (duration) tiene una relación importante 
            con la aceptación de la campaña. Los clientes que permanecen más tiempo en la conversación 
            presentan mayor probabilidad de aceptar la oferta. Esto sugiere que mejorar las habilidades 
            de comunicación de los agentes comerciales podría aumentar la efectividad de las campañas.

            **2. Valor de los contactos previos**

            Los clientes que han tenido contactos previos (previous) o resultados positivos en campañas 
            anteriores (poutcome) muestran una mayor predisposición a aceptar nuevas ofertas. Esto indica 
            que las estrategias de seguimiento y fidelización pueden mejorar significativamente los 
            resultados comerciales.

            **3. Importancia de la segmentación de clientes**

            Variables como edad (age), tipo de trabajo (job) y nivel educativo (education) presentan 
            diferencias en los patrones de aceptación. Esto sugiere que segmentar a los clientes según 
            características demográficas puede ayudar a diseñar campañas de marketing más efectivas.

            **4. Frecuencia de contacto en la campaña**

            El número de contactos realizados durante la campaña (campaign) puede influir en la respuesta 
            del cliente. Un exceso de intentos de contacto podría generar saturación o rechazo, por lo 
            que es recomendable optimizar la frecuencia de comunicación.

            **5. Influencia del contexto económico**

            Las variables económicas del dataset, como euribor3m, emp.var.rate y cons.conf.idx, reflejan 
            el contexto económico en el que se desarrollan las campañas. Estas condiciones pueden influir 
            en la disposición de los clientes a aceptar productos financieros.
            """)
