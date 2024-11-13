import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Título de la aplicación
st.title("Predicción de Desembolsos")

# Cargar archivo
uploaded_file = st.file_uploader("Cargar un archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    # Leer el archivo Excel
    producto = pd.read_excel(uploaded_file)

    # Mostrar los primeros registros del DataFrame
    st.write("Datos cargados:")
    st.write(producto.head())

    # Mostrar estadística descriptiva de las variables
    st.subheader("Estadísticas Descriptivas")
    st.write(producto.describe())

    # Convertir columnas a numéricas (ajusta según tus datos)
    for col in producto.columns:
        producto[col] = pd.to_numeric(producto[col], errors='coerce')

    # Mostrar la matriz de correlación
    st.subheader("Matriz de Correlación")
    correlation_matrix = producto.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    st.pyplot(plt)

    # Obtener todas las columnas excepto la de salida
    all_columns = producto.columns.tolist()
    if 'MTO_DESEMBOLSADO' in all_columns:
        all_columns.remove('MTO_DESEMBOLSADO')  # Suponiendo que esta es la variable a predecir

    # Seleccionar variables independientes
    selected_columns = st.multiselect("Selecciona variables independientes", options=all_columns)

    # Verificar si se han seleccionado columnas
    if selected_columns:
        # Sumar desembolsos por gestión
        suma_dese = producto.groupby('GESTION')['MTO_DESEMBOLSADO'].sum().reset_index()
        suma_dese.columns = ['GESTION', 'MTO_DESEMBOLSADO_TOTAL']

        # Mostrar la suma de desembolsos por gestión
        st.write("Suma de desembolsos por gestión:")
        st.write(suma_dese)

        # Combinar con las variables independientes
        combined_data = producto.merge(suma_dese, on='GESTION')

        # Crear X y y
        X = combined_data[selected_columns]
        y = combined_data['MTO_DESEMBOLSADO_TOTAL']  # Usar el total de desembolsos

        # Limpieza de datos (eliminar filas con valores nulos)
        X = X.dropna()
        y = y[X.index]

        # Verificar que haya suficientes datos para dividir
        if len(y) < 2:
            st.warning("No hay suficientes datos para dividir en conjuntos de entrenamiento y prueba.")
        else:
            # Normalización de los datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Dividir el conjunto de datos
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Añadir características polinómicas
            poly = PolynomialFeatures(degree=2)  # O probar con un grado más alto
            X_poly_train = poly.fit_transform(X_train)
            X_poly_test = poly.transform(X_test)

            # Crear el modelo de Regresión Lineal
            model = LinearRegression()

            # Entrenar el modelo
            model.fit(X_poly_train, y_train)

            # Predecir en el conjunto de prueba
            y_pred = model.predict(X_poly_test)

            # Calcular métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"MSE del modelo de Regresión Lineal: {mse:.2f}")
            st.write(f"R^2 del modelo de Regresión Lineal: {r2:.2f}")

            # Validación cruzada
            scores = cross_val_score(model, X_poly_train, y_train, cv=5, scoring='neg_mean_squared_error')
            st.write(f"MSE medio de validación cruzada: {-scores.mean():.2f}")

            # Opcional: Usar regularización (Ridge o Lasso) si el modelo es sobreajustado
            st.subheader("Probar con Regularización")
            regularization_model = st.selectbox("Seleccionar Regularización", ["Ninguna", "Ridge", "Lasso"])

            if regularization_model == "Ridge":
                reg_model = Ridge(alpha=1.0)
                reg_model.fit(X_poly_train, y_train)
                y_pred_reg = reg_model.predict(X_poly_test)
                reg_r2 = r2_score(y_test, y_pred_reg)
                st.write(f"R^2 con Ridge: {reg_r2:.2f}")

            elif regularization_model == "Lasso":
                reg_model = Lasso(alpha=0.1)
                reg_model.fit(X_poly_train, y_train)
                y_pred_reg = reg_model.predict(X_poly_test)
                reg_r2 = r2_score(y_test, y_pred_reg)
                st.write(f"R^2 con Lasso: {reg_r2:.2f}")

            # Interfaz para hacer predicciones
            st.subheader("Predicciones de las gestiones 2024 - 2033 en adelante")

            # Crear entradas para las gestiones futuras
            gestiones = [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033]
            input_data = []

            for gestion in gestiones:
                row = {'GESTION': gestion}
                for col in selected_columns:
                    # Usar un valor aleatorio cercano a la suma de desembolsos de gestiones pasadas
                    random_variation = (combined_data[col].std() * 0.1) * np.random.randn()
                    row[col] = combined_data[col].mean() + random_variation
                input_data.append(row)

            # Convertir a DataFrame
            input_df = pd.DataFrame(input_data)

            # Normalizar los datos de entrada
            input_scaled = scaler.transform(input_df[selected_columns])
            input_poly_scaled = poly.transform(input_scaled)

            # Hacer la predicción
            if st.button("Predecir"):
                predictions = model.predict(input_poly_scaled)

                # Mostrar predicciones
                prediction_df = pd.DataFrame({
                    'GESTION': gestiones,
                    'Predicción de MTO_DESEMBOLSADO': predictions
                })

                # Mostrar datos reales para comparación
                real_data = suma_dese.copy()

                # Graficar predicciones y datos reales
                plt.figure(figsize=(12, 6))
                plt.bar(real_data['GESTION'].astype(str), real_data['MTO_DESEMBOLSADO_TOTAL'], color='orange', label='Datos Reales')
                plt.bar(prediction_df['GESTION'].astype(str), prediction_df['Predicción de MTO_DESEMBOLSADO'], color='blue', label='Predicciones', alpha=0.7)
                plt.xlabel('Gestión')
                plt.ylabel('MTO_DESEMBOLSADO')
                plt.title('Comparación de Desembolsos Reales y Predicciones')
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot(plt)

                # Mostrar resultados
                st.write(prediction_df)

    else:
        st.warning("Por favor, selecciona al menos una variable independiente.")
