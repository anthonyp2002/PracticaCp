import csv
import time
import re
import os
import logging
import threading
import asyncio
import aiohttp
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from multiprocessing import Pool, Process, Manager
from flask import Flask, jsonify, request
from flask_cors import CORS
import praw
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import emoji
from unicodedata import normalize
import string
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from langdetect import detect, LangDetectException

# Cargar variables de entorno
load_dotenv()

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Inicializar Flask
app = Flask(__name__)
CORS(app)

# Configuración de Reddit
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# Configuración de Selenium
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=options)

# Función para eliminar tildes
def eliminar_tildes(texto):
    # Normalizar el texto y eliminar tildes
    texto = normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    return texto

# Función para limpiar el texto
def clean_text(texto):
    if pd.isna(texto) or texto.strip().lower() == "nan":  # Manejar NaN o la palabra "nan"
        return None
    
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar tildes
    texto = eliminar_tildes(texto)
    
    # Eliminar números
    texto = re.sub(r'\d+', '', texto)
    
    # Eliminar puntuación
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    
    # Eliminar emojis y caracteres especiales
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Emojis de caras
        u"\U0001F300-\U0001F5FF"  # Símbolos y pictogramas
        u"\U0001F680-\U0001F6FF"  # Transporte y mapas
        u"\U0001F700-\U0001F77F"  # Alquimia
        u"\U0001F780-\U0001F7FF"  # Formas geométricas
        u"\U0001F800-\U0001F8FF"  # Símbolos suplementarios
        u"\U0001F900-\U0001F9FF"  # Símbolos suplementarios 2
        u"\U0001FA00-\U0001FA6F"  # Ajedrez
        u"\U0001FA70-\U0001FAFF"  # Símbolos suplementarios 3
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Símbolos de letras
        "]+", flags=re.UNICODE
    )
    texto = emoji_pattern.sub(r'', texto)
    
    # Eliminar URLs
    texto = re.sub(r"http\S+|www\S+|https\S+", "", texto, flags=re.MULTILINE)
    
    # Eliminar espacios extra
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def limpiezaDatos():
    # Leer los datasets
    dataset1 = pd.read_csv("limpiarMPI.csv")
    dataset2 = pd.read_csv("comments.csv")
    dataset3 = pd.read_csv("reddit_comments.csv")

    # Combinar los datasets
    dataset = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)

    # Asegurarse de que la columna 'Comment' sea de tipo string
    dataset['Comment'] = dataset['Comment'].astype(str)

    # Aplicar la limpieza de texto
    dataset['Comment'] = dataset['Comment'].apply(clean_text)

    # Eliminar filas con comentarios vacíos, NaN o la palabra "nan"
    dataset = dataset[dataset['Comment'].notna() & (dataset['Comment'] != "")]

    # Guardar el dataset limpio
    dataset.to_csv("dataset.csv", index=False, encoding="utf-8")

async def fetch_comments(session, url, headers):
    async with session.get(url, headers=headers) as response:
        return await response.json()

async def tiktok_extract_comments_async(video_url):
    if "vm.tiktok.com" in video_url or "vt.tiktok.com" in video_url:
        try:
            video_url = requests.head(video_url, allow_redirects=True, timeout=5).url
        except Exception as e:
            logging.error(f"❌ Error siguiendo redirección de URL: {e}")
            return 0

    try:
        videoid = video_url.split("/")[5].split("?", 1)[0]
    except IndexError:
        logging.error("❌ Error al extraer el videoid de la URL.")
        return 0

    logging.info(f"📢 Extrayendo comentarios del video {videoid}...")

    cursor = 0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        "Referer": "https://www.tiktok.com/",
    }
    total_comments = 0
    all_comments = []

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                url = f"https://www.tiktok.com/api/comment/list/?aid=1988&aweme_id={videoid}&count=20&cursor={cursor}"
                data = await fetch_comments(session, url, headers)

                comments = data.get("comments")
                if not comments:
                    logging.info("✅ No hay más comentarios disponibles.")
                    break

                for comment in comments:
                    parent_comment = comment.get('text', '')
                    if parent_comment:
                        logging.info(f"💬 Comentario: {parent_comment}")
                        all_comments.append(parent_comment)
                        total_comments += 1

                cursor += len(comments)

            except Exception as e:
                logging.error(f"❌ Error durante la extracción del video {videoid}: {e}")
                break

    logging.info(f"✅ Extracción completada. Total de comentarios: {total_comments}")
    return all_comments

def search_tiktok(search_term):
    driver = setup_driver()
    driver.get("https://www.tiktok.com")

    try:
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "TUXButton--capsule"))
        )
        search_button.click()
        logging.info("✅ Botón de búsqueda clickeado!")

        search_input = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "css-1rl3x5e-InputElement"))
        )
        search_input.send_keys(search_term)
        search_input.send_keys(Keys.RETURN)
        logging.info("✅ Búsqueda enviada!")

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div[data-e2e='search_top-item'] a.css-1mdo0pl-AVideoContainer.e19c29qe4")
            )
        )
        logging.info("✅ Resultados cargados!")

        video_elements = driver.find_elements(
            By.CSS_SELECTOR,
            "div[data-e2e='search_top-item'] a.css-1mdo0pl-AVideoContainer.e19c29qe4"
        )
        video_urls = [elem.get_attribute('href') for elem in video_elements[:1]]

        async def main():
            tasks = [tiktok_extract_comments_async(url) for url in video_urls]
            return await asyncio.gather(*tasks)

        all_comments = asyncio.run(main())

        with open("comments.csv", "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Comment"])
            for comments in all_comments:
                for comment in comments:
                    writer.writerow([comment])
        logging.info("✅ Comentarios guardados en 'comments.csv'")

    except Exception as e:
        logging.error(f"❌ Error en la búsqueda o extracción: {e}")

    finally:
        driver.quit()

# Extracción de comentarios de Reddit
def get_reddit_comments(busqueda):
    comments_data = []

    for submission in reddit.subreddit('all').search(busqueda, limit=10):
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            soup = BeautifulSoup(comment.body, 'html.parser')
            clean_comment = soup.get_text()
            cleaned_comment = clean_text(clean_comment)

            if cleaned_comment:
                comments_data.append({'Comment': cleaned_comment})

    df = pd.DataFrame(comments_data)
    df.to_csv('reddit_comments.csv', index=False)
    return jsonify('Extracción completa')

# Extracción de comentarios de YouTube
def extract_comments(video_url, max_comments=10):
    driver = setup_driver()
    driver.get(video_url)
    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    
    comments_data = []
    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    
    while len(comments_data) < max_comments:
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(2)
        
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        
        comments_elements = driver.find_elements(By.ID, "content-text")
        
        for i in range(len(comments_elements)):
            if len(comments_data) >= max_comments:
                break
            comment_text = comments_elements[i].text.strip()
            comments_data.append(clean_text(comment_text))
    
    driver.quit()
    return comments_data

def search_youtube(search_query, max_results=5):
    driver = setup_driver()
    driver.get("https://www.youtube.com/")
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "search_query")))

    search_box = driver.find_element(By.NAME, "search_query")
    search_box.send_keys(search_query)
    search_box.send_keys(Keys.RETURN)

    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "video-title")))

    video_elements = driver.find_elements(By.ID, "video-title")[:max_results]
    video_urls = [video.get_attribute("href") for video in video_elements]

    driver.quit()
    return video_urls

def scrape(search_query):
    urls = search_youtube(search_query)
    video_urls = [url for url in urls if url is not None]
    
    with Pool(processes=4) as pool:
        results = pool.map(extract_comments, video_urls)
    
    all_comments = [comment for sublist in results for comment in sublist]
    
    output_file = "limpiarMPI.csv"
    with open(output_file, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Comment"])
        for comment in all_comments:
            writer.writerow([comment])
    
    logging.info(f"🎉 Extracción completada. Todos los comentarios guardados en {output_file}")

def process_entry(i, busqueda, result_list):
    try:
        if i == 'Youtube':
            print('Entro a Youtube')
            result = scrape(busqueda)
            result_list.append(result)
        elif i == 'Tiktok':
            print('Entro a Tiktok')
            result = search_tiktok(busqueda)
            result_list.append("Procesando TikTok")
        elif i == 'Reddit':
            print('Entro a Reddit')
            result = get_reddit_comments(busqueda)
            result_list.append(result)
    except Exception as e:
        result_list.append(f"Error: {str(e)}")

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

def detectar_idioma(texto):
    """
    Detecta el idioma de un texto.
    """
    try:
        return detect(texto)
    except LangDetectException:
        return None  # En caso de que no se pueda detectar el idioma

def stopwor():
    dataset = pd.read_csv("dataset_procesado.csv")

    # Asegurarnos de que los tokens sean listas
    if 'tokens_lemmatized' in dataset.columns:
        dataset['tokens_lemmatized'] = dataset['tokens_lemmatized'].apply(eval)  # Convertir de string a lista
        
        # Convertir las listas de tokens en una única cadena por comentario
        corpus = [" ".join(tokens) for tokens in dataset['tokens_lemmatized']]
        
        # Crear el modelo de CountVectorizer
        vectorizer = CountVectorizer()
        
        # Ajustar el modelo y transformar el corpus en una matriz de términos
        X = vectorizer.fit_transform(corpus)
        
        # Obtener las palabras y sus frecuencias
        word_freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1))
        
        # Ordenar las palabras por frecuencia
        word_freq = {k: v for k, v in sorted(word_freq.items(), key=lambda item: item[1], reverse=True)}
        
        # Convertir las palabras y frecuencias en un DataFrame
        word_freq_df = pd.DataFrame(list(word_freq.items()), columns=['Palabra', 'Frecuencia'])
        
        # Guardar las palabras y frecuencias en un archivo CSV
        word_freq_df.to_csv('palabras_frecuentes.csv', index=False, encoding='utf-8')

        # Mostrar las 10 palabras más frecuentes
        print("Las 10 palabras más frecuentes:")
        print(list(word_freq.items())[:10])

        # Generar la nube de palabras
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        
        # Visualizar la nube de palabras
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig("static/bili.png")  # Guarda la imagen


        # Crear el diagrama de barras para las 10 palabras más frecuentes
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Frecuencia', y='Palabra', data=word_freq_df.head(10), palette='viridis')
        plt.title('Las 10 Palabras Más Frecuentes')
        plt.xlabel('Frecuencia')
        plt.ylabel('Palabra')
        plt.savefig("static/topPala.png")  # Guarda la imagen
        
        print("Nube de palabras generada.")
    else:
        print("No se encontró la columna 'tokens_lemmatized' en el dataset.")
   

def procesar_comentarios(dataset_path, columna_comentarios='Comment'):
    """
    Procesa un dataset con comentarios en español e inglés: tokeniza, elimina stopwords y aplica lematización.
    
    Parámetros:
    - dataset_path: Ruta al archivo CSV que contiene los comentarios.
    - columna_comentarios: Nombre de la columna que contiene los comentarios (por defecto 'Comment').
    
    Retorna:
    - Guarda el dataset procesado en un archivo CSV y lo retorna como DataFrame.
    """
    # Descargar stopwords si no están disponibles
    nltk.download('stopwords')
    nltk.download('punkt')
    
    # Cargar modelos de spaCy para español e inglés
    nlp_es = spacy.load("es_core_news_sm")
    nlp_en = spacy.load("en_core_web_sm")
    
    # Cargar el dataset
    dataset = pd.read_csv(dataset_path)
    
    # Verificar que la columna de comentarios existe
    if columna_comentarios not in dataset.columns:
        raise ValueError(f"No se encontró la columna '{columna_comentarios}' en el dataset.")
    
    # Definir el patrón de tokenización avanzado
    pattern = r'''(?xi)                   # Modo verbose e insensible a mayúsculas y minúsculas
                  (?:[A-Z]\.)+             # Coincide con abreviaciones como U.S.A.
                  | \w+(?:-\w+)*           # Coincide con palabras que pueden tener un guión interno
                  | \$?\d+(?:\.\d+)?%?     # Coincide con dinero o porcentajes como $15.5 o 100%
                  | \.\.\.                 # Coincide con puntos suspensivos
                  | [.,;'"?():_\-]         # Coincide con signos de puntuación específicos
                  | \w+(?:'\w+)?           # Coincide con palabras que pueden contener apóstrofes
    '''
    
    # Tokenización con el patrón avanzado
    dataset['tokens'] = dataset[columna_comentarios].apply(
        lambda x: nltk.regexp_tokenize(x, pattern) if isinstance(x, str) else []
    )
    
    # Detectar idioma y eliminar stopwords según el idioma
    stopwords_es = set(stopwords.words('spanish'))
    stopwords_en = set(stopwords.words('english'))
    
    dataset['idioma'] = dataset[columna_comentarios].apply(
        lambda x: detectar_idioma(x) if isinstance(x, str) else None
    )
    
    dataset['tokens_sin_stopwords'] = dataset.apply(
        lambda row: [word for word in row['tokens'] if word.lower() not in (stopwords_es if row['idioma'] == 'es' else stopwords_en)]
        if row['idioma'] in ['es', 'en'] else row['tokens'], axis=1
    )
    
    # Aplicar lematización según el idioma
    def lemmatize_tokens(tokens, idioma):
        if idioma == 'es':
            doc = nlp_es(" ".join(tokens))  # Crear un documento spaCy en español
        elif idioma == 'en':
            doc = nlp_en(" ".join(tokens))  # Crear un documento spaCy en inglés
        else:
            return tokens  # Si no se detecta el idioma, no se lematiza
        return [token.lemma_ for token in doc]  # Extraer los lemas
    
    dataset['tokens_lemmatized'] = dataset.apply(
        lambda row: lemmatize_tokens(row['tokens_sin_stopwords'], row['idioma']), axis=1
    )
    
    # Guardar el dataset procesado
    output_path = "dataset_procesado.csv"
    dataset.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"El dataset procesado ha sido guardado como '{output_path}'.")
    print(dataset[[columna_comentarios, 'tokens_lemmatized']].head(5))  # Mostrar ejemplo
    
    return dataset

# Rutas de Flask
@app.route('/procesar', methods=['POST'])
def procesar():
    datos = request.json

    if 'array' not in datos or 'string' not in datos:
        return jsonify({'error': 'Faltan datos'}), 400

    array = datos['array']
    busqueda = datos['string']

    with Manager() as manager:
        result_list = manager.list() 
        processes = []

        for i in array:
            p = Process(target=process_entry, args=(i, busqueda, result_list))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return jsonify({
            'resultados': list(result_list),
            'array_recibido': array,
            'string_recibido': busqueda
        }), 200

@app.route('/analizar', methods=['GET'])
def Analizar():
    limpiezaDatos()
    dataset = pd.read_csv('dataset.csv')
    comentarios = dataset['Comment'].tolist()
    resultados = predecir_emocion(comentarios)
    resultados.to_csv('dataset_predicha.csv', index=False)

    plt.figure(figsize=(8, 5))
    sns.countplot(x='Emocion_Predicha', data=resultados, palette='coolwarm', order=sorted(resultados['Emocion_Predicha'].unique()))
    plt.xlabel("Categoría de Sentimiento")
    plt.ylabel("Número de Comentarios")
    plt.title("Distribución de Sentimientos en Comentarios")
    plt.savefig("static/distribucion_sentimientos.png")  # Guarda la imagen

    plt.figure(figsize=(8, 5))
    sns.histplot(resultados['Confianza'], bins=10, kde=True, color='skyblue')
    plt.xlabel("Nivel de Confianza")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Confianza en Predicciones")
    plt.savefig("static/distribucion_confianza.png")  # Guarda la imagen

    procesar_comentarios("dataset.csv", columna_comentarios='Comment')
    stopwor()
    bigramas()
    return jsonify(message="Análisis Completo")

def bigramas():
    # Cargar el dataset procesado
    dataset = pd.read_csv("dataset_procesado.csv")

    # Verificar que la columna de texto exista
    if 'tokens_lemmatized' in dataset.columns:
        # Crear una lista de comentarios
        corpus = dataset['tokens_lemmatized'].dropna().tolist()  # Eliminar valores NaN y convertir a lista

        # Crear el modelo de bi-gramas
        vectorizer = CountVectorizer(ngram_range=(2, 2))  # Para bi-gramas
        X = vectorizer.fit_transform(corpus)

        # Obtener los bi-gramas más frecuentes
        bigram_freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1))

        # Ordenar los bi-gramas por frecuencia
        bigram_freq = {k: v for k, v in sorted(bigram_freq.items(), key=lambda item: item[1], reverse=True)}

        # Crear un DataFrame de los bi-gramas más frecuentes
        bigram_df = pd.DataFrame(list(bigram_freq.items()), columns=['bigram', 'frequency'])
        
        # Seleccionar los 10 bi-gramas más frecuentes
        top_bigrams = bigram_df.head(10)
        
        # Crear el gráfico de barras
        plt.figure(figsize=(12, 6))
        plt.barh(top_bigrams['bigram'], top_bigrams['frequency'], color='skyblue')
        plt.xlabel('Frecuencia')
        plt.ylabel('Bi-gramas')
        plt.title('Top 10 Bi-gramas Más Frecuentes')
        plt.gca().invert_yaxis()  # Invertir el eje Y para mostrar el más frecuente arriba
        plt.savefig("static/bigramasTwo.png")  # Guarda la imagen

        # Guardar los bi-gramas y sus frecuencias en un archivo CSV
        bigram_df.to_csv("bigram_frequencies.csv", index=False, encoding='utf-8')
        print("Frequencias de bi-gramas guardadas en 'bigram_frequencies.csv'.")
    else:
        print("La columna 'comentario' no se encuentra en el dataset.")


@app.route('/')
def home():
    return jsonify(message="¡Hola, Flask!")

if __name__ == '__main__':
    app.run(debug=True)