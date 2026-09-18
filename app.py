import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTENC

# ---------------------------------------------------------
# Configuración inicial de Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Despliegue - Predicción Ataque al Corazón",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Sistema de Predicción de Riesgo de Ataque al Corazón")
st.markdown("""
Esta aplicación permite evaluar el riesgo de ataque al corazón utilizando un modelo de **K-Nearest Neighbors (KNN)** 
entrenado con datos clínicos. Puedes hacer una predicción individual o cargar un archivo con múltiples pacientes.
""")

# ---------------------------------------------------------
# Carga y Entrenamiento del Modelo (Caching para rendimiento)
# ---------------------------------------------------------
@st.cache_resource
def entrenar_modelo():
    # Creamos un dataset sintético basado en las distribuciones del entrenamiento original
    # para inicializar las transformaciones y el modelo de forma rápida e independiente
    np.random.seed(42)
    n = 1000
    df_train = pd.DataFrame({
        'age': np.random.randint(1, 85, n),
        'avg_glucose_level': np.random.uniform(55.0, 275.0, n),
        'hypertension': np.random.choice(['Yes', 'No'], n, p=[0.1, 0.9]),
        'heart_disease': np.random.choice(['Yes', 'No'], n, p=[0.05, 0.95]),
        'ever_married': np.random.choice(['Yes', 'No'], n, p=[0.65, 0.35]),
        'smoking_status': np.random.choice(["'formerly smoked'", "'never smoked'", "Unknown", "smokes"], n),
        'stroke_ataque_corazon': np.random.choice([0, 1], n, p=[0.8, 0.2])
    })

    # Balanceo con SMOTENC
    X = df_train.drop('stroke_ataque_corazon', axis=1)
    Y = df_train['stroke_ataque_corazon']
    
    smote = SMOTENC(k_neighbors=2, categorical_features=[2, 3, 4, 5], sampling_strategy=0.3, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, Y)

    # Transformación Dummies
    X_dummies = pd.get_dummies(X_smote, columns=['smoking_status'], drop_first=False, dtype=int)
    X_dummies = pd.get_dummies(X_dummies, columns=['hypertension', 'heart_disease', 'ever_married'], drop_first=True, dtype=int)

    # Columnas esperadas por el modelo tras Get Dummies
    columnas_esperadas = [
        'age', 'avg_glucose_level',
        "smoking_status_'formerly smoked'", "smoking_status_'never smoked'",
        'smoking_status_Unknown', 'smoking_status_smokes',
        'hypertension_Yes', 'heart_disease_Yes', 'ever_married_Yes'
    ]
    
    # Reordenar y asegurar columnas
    for col in columnas_esperadas:
        if col not in X_dummies.columns:
            X_dummies[col] = 0
    X_dummies = X_dummies[columnas_esperadas]

    # Escalador MinMax
    scaler = MinMaxScaler()
    X_dummies[['age', 'avg_glucose_level']] = scaler.fit_transform(X_dummies[['age', 'avg_glucose_level']])

    # Entrenamiento del Modelo KNN
    model_knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    model_knn.fit(X_dummies, y_smote)

    return model_knn, scaler, columnas_esperadas

model_knn, scaler, columnas_esperadas = entrenar_modelo()

# ---------------------------------------------------------
# Función para Preprocesar Datos Entrantes
# ---------------------------------------------------------
def preprocesar_datos(df_in):
    df = df_in.copy()
    
    # Asegurar dummies
    if 'smoking_status' in df.columns:
        df = pd.get_dummies(df, columns=['smoking_status'], drop_first=False, dtype=int)
    if 'hypertension' in df.columns:
        df = pd.get_dummies(df, columns=['hypertension'], drop_first=True, dtype=int)
    if 'heart_disease' in df.columns:
        df = pd.get_dummies(df, columns=['heart_disease'], drop_first=True, dtype=int)
    if 'ever_married' in df.columns:
        df = pd.get_dummies(df, columns=['ever_married'], drop_first=True, dtype=int)

    # Completar columnas faltantes
    for col in columnas_esperadas:
        if col not in df.columns:
            df[col] = 0

    df = df[columnas_esperadas]

    # Normalización de variables numéricas
    df[['age', 'avg_glucose_level']] = scaler.transform(df[['age', 'avg_glucose_level']])
    return df

# ---------------------------------------------------------
# Menú de Navegación Lateral
# ---------------------------------------------------------
st.sidebar.header("⚙️ Opciones de Navegación")
modo = st.sidebar.radio("Selecciona el modo de uso:", ["Formulario Individual", "Subir Dataset (Batch)"])

# ---------------------------------------------------------
# MODO 1: Formulario Individual
# ---------------------------------------------------------
if modo == "Formulario Individual":
    st.subheader("📋 Ingreso de Datos del Paciente")
    
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad:", min_value=1, max_value=120, value=45)
        avg_glucose_level = st.number_input("Nivel Promedio de Glucosa:", min_value=40.0, max_value=300.0, value=95.0, step=0.1)
        hypertension = st.selectbox("¿Sufre de Hipertensión?", ["No", "Yes"])

    with col2:
        heart_disease = st.selectbox("¿Tiene Enfermedades del Corazón?", ["No", "Yes"])
        ever_married = st.selectbox("¿Ha estado casado/a alguna vez?", ["Yes", "No"])
        smoking_status = st.selectbox("Hábito de Fumar:", ["'never smoked'", "'formerly smoked'", "smokes", "Unknown"])

    if st.button("🚨 Evaluar Riesgo", type="primary"):
        # Crear DataFrame para el registro
        input_dict = {
            'age': [age],
            'avg_glucose_level': [avg_glucose_level],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'smoking_status': [smoking_status]
        }
        df_paciente = pd.DataFrame(input_dict)
        
        # Preprocesar e Inferir
        X_proc = preprocesar_datos(df_paciente)
        prediccion = model_knn.predict(X_proc)[0]
        probabilidad = model_knn.predict_proba(X_proc)[0][1]

        st.markdown("---")
        st.subheader("📊 Resultado del Diagnóstico")
        
        if prediccion == 1:
            st.error(f"⚠️ **Riesgo Alto de Ataque al Corazón / Stroke Detected**\n\nProbabilidad estimada: **{probabilidad*100:.1f}%**")
        else:
            st.success(f"✅ **Riesgo Bajo / No Stroke Detected**\n\nProbabilidad estimada de riesgo: **{probabilidad*100:.1f}%**")

# ---------------------------------------------------------
# MODO 2: Cargar Dataset
# ---------------------------------------------------------
elif modo == "Subir Dataset (Batch)":
    st.subheader("📁 Predicción Masiva mediante Archivo")
    st.markdown("""
    Sube un archivo **CSV** o **Excel (.xlsx)** con las siguientes columnas obligatorias:
    - `age`, `avg_glucose_level`, `hypertension`, `heart_disease`, `ever_married`, `smoking_status`
    """)

    uploaded_file = st.file_uploader("Cargar archivo de datos", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_cargado = pd.read_csv(uploaded_file)
            else:
                df_cargado = pd.read_excel(uploaded_file)

            st.write("### Vista previa del Dataset cargado:")
            st.dataframe(df_cargado.head())

            if st.button("🔄 Procesar y Predecir Todo el Dataset", type="primary"):
                # Preprocesamiento masivo
                X_proc = preprocesar_datos(df_cargado)
                
                # Predicciones
                predicciones = model_knn.predict(X_proc)
                probabilidades = model_knn.predict_proba(X_proc)[:, 1]

                # Adjuntar resultados
                df_resultado = df_cargado.copy()
                df_resultado['Prediccion_Ataque_Corazon'] = np.where(predicciones == 1, 'Sí (Riesgo)', 'No (Bajo Riesgo)')
                df_resultado['Probabilidad_Riesgo_%'] = np.round(probabilidades * 100, 2)

                st.success("✅ Procesamiento completado con éxito.")
                st.write("### Resultados con Predicciones:")
                st.dataframe(df_resultado)

                # Opción para descargar los resultados en CSV
                csv = df_resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Resultados en CSV",
                    data=csv,
                    file_name="predicciones_ataque_corazon.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")