# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:16:57 2024

@author: jperezr
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from itertools import combinations
import time
import plotly.express as px

# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

class ModelAnalysis:
    def __init__(self, input_df, model_type=None, target_column='Class', seed=None):
        self.input_df = input_df
        self.target_column = target_column
        self.X = self.input_df.drop(columns=[self.target_column])
        self.y = self.input_df[self.target_column]
        self.feature_names = self.X.columns
        self.seed = seed

        self.models = {
            'SVM': SVC(),
            'DecisionTree': DecisionTreeClassifier(),
            'RandomForest': RandomForestClassifier(),
            'LogisticRegression': LogisticRegression(max_iter=1000)
        }

        self.model_type = model_type if model_type else 'SVM'

    def run_all_combinations(self, train_test_split_ratio=0.7):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=1-train_test_split_ratio, random_state=self.seed)

        results = []

        for n in range(1, len(self.feature_names) + 1):
            for comb in combinations(self.feature_names, n):
                for model_name, model in self.models.items():
                    start_time = time.time()
                    X_train_comb = X_train[list(comb)]
                    X_test_comb = X_test[list(comb)]

                    model.fit(X_train_comb, y_train)
                    predictions = model.predict(X_test_comb)

                    elapsed_time = time.time() - start_time

                    accuracy = accuracy_score(y_test, predictions)
                    conf_matrix = confusion_matrix(y_test, predictions)

                    if conf_matrix.shape == (2, 2):
                        tn, fp, fn, tp = conf_matrix.ravel()
                    else:
                        tn, fp, fn, tp = [np.nan]*4

                    report = classification_report(y_test, predictions, output_dict=True)

                    results.append({
                        'Model': model_name,
                        'Combination': comb,
                        'Accuracy': accuracy,
                        'True Positive (TP)': tp,
                        'False Positive (FP)': fp,
                        'True Negative (TN)': tn,
                        'False Negative (FN)': fn,
                        'Precision': report['weighted avg']['precision'],
                        'Recall': report['weighted avg']['recall'],
                        'F1-Score': report['weighted avg']['f1-score'],
                        'Time': elapsed_time,
                        'Confusion Matrix': f"[{conf_matrix.tolist()}]"
                    })

        results_df = pd.DataFrame(results)

        # Save results
        self.save_results_to_csv(results_df)

        return results_df

    def save_results_to_csv(self, results_df):
        results_df.to_csv("all_combinations_results.csv", index=False)

    def best_models_by_num_attributes(self, results_df):
        best_models = {}

        for num_attributes in [1, 2, 3, 4]:
            subset = results_df[results_df['Combination'].apply(lambda x: len(x) == num_attributes)]
            if not subset.empty:
                best_model_row = subset.loc[subset['Accuracy'].idxmax()]
                best_models[num_attributes] = best_model_row

        best_models_df = pd.DataFrame.from_dict(best_models, orient='index')
        return best_models_df

def highlight_best_models(df):
    def calculate_score(row):
        metrics = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
        return np.mean(metrics)

    df['Score'] = df.apply(calculate_score, axis=1)
    return df

# Streamlit code for visualization
st.sidebar.title("Ayuda")
st.sidebar.write("""
Este código realiza un análisis exhaustivo de modelos de clasificación utilizando diferentes combinaciones de atributos. Sus funcionalidades principales incluyen:

1. **Base de Datos:** Utiliza el conjunto de datos Iris, aunque se puede adaptar a otros conjuntos de datos.
2. **Modelos Utilizados:** Soporte Vectorial (SVM), Árbol de Decisión, Bosques Aleatorios y Regresión Logística.
3. **Combinaciones por Atributos:** Prueba todas las combinaciones posibles de atributos del conjunto de datos para identificar las más efectivas.
4. **Tiempo de Ejecución:** Calcula el tiempo que cada modelo tarda en resolver cada combinación de atributos.
5. **Métricas Clave:** Evalúa precisión, matriz de confusión, precisión ponderada, recall, F1-score, y tiempo de ejecución.
6. **Dashboard Interactivo:** Permite analizar gráficamente la distribución de métricas clave y comparar modelos.
7. **Evolución Temporal:** Visualiza cómo la precisión cambia a lo largo del tiempo con animaciones.
8. **Mejores Modelos:** Identifica y muestra los mejores modelos para combinaciones de 1, 2, 3 y 4 atributos.
9. **Exportación de Resultados:** Los resultados detallados se guardan automáticamente en un archivo CSV para análisis adicional.

Autor: Javier Horacio Pérez Ricárdez
""")

st.title('Análisis de modelos para todas las combinaciones')

# Load data (modify as needed)
input_df = pd.read_csv("iris.csv")
st.write("### Iris Dataset", input_df)
target_column = 'Class'

# Initialize Model Analysis
analysis = ModelAnalysis(input_df=input_df, target_column=target_column, seed=1271673)
results_df = analysis.run_all_combinations()

# Highlight the best models and apply colors (now without styling)
results_df = highlight_best_models(results_df)

# Display the results without colors
st.write("### Resultados de rendimiento del modelo", results_df)

# Comparison of models
fig = px.bar(
    results_df, x='Model', y='Accuracy', color='Combination',
    title="Comparación de precisión entre modelos",
    labels={'Accuracy': 'Precisión', 'Combination': 'Combinación de Atributos'}
)
st.plotly_chart(fig)

# Animated plot of accuracy over time
fig_anim = px.scatter(
    results_df.sort_values('Time'),
    x='Time', y='Accuracy', color='Model',
    animation_frame='Combination',
    title="Evolución de la precisión en función del tiempo",
    labels={'Time': 'Tiempo (s)', 'Accuracy': 'Precisión'}
)
fig_anim.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2000  # Ajustar la duración del frame
st.plotly_chart(fig_anim)

# Get the best models for 1, 2, 3, and 4 attributes
best_models_df = analysis.best_models_by_num_attributes(results_df)

# Display the best models by attributes
st.write("### Mejores modelos por número de atributos", best_models_df)

# Key metrics dashboard
st.header("Dashboard de Métricas Clave")

# Distribution of accuracy
st.subheader("Distribución de Precisión")
fig_accuracy = px.histogram(results_df, x='Accuracy', title='Distribución de Precisión')
st.plotly_chart(fig_accuracy)

# Execution time distribution
st.subheader("Distribución del Tiempo de Ejecución")
fig_time = px.histogram(results_df, x='Time', title='Distribución del Tiempo de Ejecución')
st.plotly_chart(fig_time)

# Average metrics by model
avg_metrics = results_df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean().reset_index()
st.subheader("Promedios por Modelo")
fig_avg_metrics = px.bar(avg_metrics, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                         title='Promedios de Métricas por Modelo', barmode='group')
st.plotly_chart(fig_avg_metrics)
