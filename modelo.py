import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el modelo guardado
with open('modelo_svm.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Función para predecir emociones
def predecir_emocion(comentarios):
    comentarios_tfidf = loaded_model.named_steps['vectorizer'].transform(comentarios)
    predicciones = loaded_model.named_steps['svm'].predict(comentarios_tfidf)
    probabilidades = loaded_model.named_steps['svm'].predict_proba(comentarios_tfidf)

    resultados = pd.DataFrame({
        'Comentario': comentarios,
        'Emocion_Predicha': predicciones,
        'Confianza': probabilidades.max(axis=1)
    })

    return resultados

# Cargar el dataset desde el archivo CSV
dataset = pd.read_csv('dataset.csv')

# Asegúrate de que la columna de comentarios en el CSV sea la correcta (aquí supongo que es 'Comentario')
comentarios = dataset['Comment'].tolist()

# Realizar predicciones con los comentarios del dataset
resultados = predecir_emocion(comentarios)

# Guardar los resultados en un nuevo archivo CSV
resultados.to_csv('dataset_predicha.csv', index=False)

# 1️⃣ Gráfico de barras - Distribución de Sentimientos
plt.figure(figsize=(8, 5))
sns.countplot(x='Emocion_Predicha', data=resultados, palette='coolwarm', order=sorted(resultados['Emocion_Predicha'].unique()))
plt.xlabel("Categoría de Sentimiento")
plt.ylabel("Número de Comentarios")
plt.title("Distribución de Sentimientos en Comentarios")
plt.show()

# 2️⃣ Histograma - Distribución de Confianza del Modelo
plt.figure(figsize=(8, 5))
sns.histplot(resultados['Confianza'], bins=10, kde=True, color='skyblue')
plt.xlabel("Nivel de Confianza")
plt.ylabel("Frecuencia")
plt.title("Distribución de Confianza en Predicciones")
plt.show()
