import csv
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from multiprocessing import Pool

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")  # Modo sin cabeza (nuevo headless)
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--window-size=1920,1080")  # Simular pantalla completa
    return webdriver.Chrome(options=options)

def clean_text(text):
    """
    Normaliza y limpia un texto: elimina caracteres especiales, 
    convierte a minúsculas y elimina espacios extras.
    """
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ0-9\s]", "", text)  # Mantiene letras, números y espacios
    text = re.sub(r"\s+", " ", text).strip()  # Elimina espacios extra
    return text

def extract_comments(video_url, max_comments=10):
    driver = setup_driver()
    driver.get(video_url)
    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    comments_data = []
    last_height = driver.execute_script("return document.documentElement.scrollHeight")

    while len(comments_data) < max_comments:
        print(f"🔄 Scroll para cargar más comentarios... ({len(comments_data)}/{max_comments})")

        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(2)

        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            print("✅ Fin de los comentarios cargados.")
            break
        last_height = new_height

        comments_elements = driver.find_elements(By.ID, "content-text")

        for i in range(len(comments_elements)):
            if len(comments_data) >= max_comments:
                break

            comment_text = comments_elements[i].text.strip()

            # Normalizar texto del comentario
            normalized_comment = clean_text(comment_text)

            comments_data.append(normalized_comment)

    driver.quit()
    return comments_data

def scrape_video_comments(video_url):
    max_comments = 1000
    print(f"📥 Extrayendo hasta {max_comments} comentarios de: {video_url}")
    return extract_comments(video_url, max_comments)

def main():
    search_query = input("Introduce el término de búsqueda para YouTube: ").strip()
    max_results = 5  # Número de videos a analizar
    all_comments = []

    driver = setup_driver()

    try:
        # Buscar videos en YouTube con el término de búsqueda
        print(f"🔍 Buscando videos con el término: {search_query}")
        driver.get("https://www.youtube.com/")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "search_query")))

        search_box = driver.find_element(By.NAME, "search_query")
        search_box.send_keys(search_query)
        search_box.send_keys(Keys.RETURN)

        # Esperar a que se carguen los resultados de búsqueda
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "video-title")))

        video_elements = driver.find_elements(By.ID, "video-title")[:max_results]
        video_urls = [video.get_attribute("href") for video in video_elements]
        print(f"📝 Videos encontrados: {video_urls}")

    finally:
        driver.quit()

    # Usar multiprocessing para extraer comentarios en paralelo
    with Pool(processes=4) as pool:
        results = pool.map(scrape_video_comments, video_urls)

    # Combinar los resultados de todos los procesos
    all_comments = [comment for sublist in results for comment in sublist]

    # CREACIÓN Y GUARDADO DE DATASET
    output_file = "limpiarMPI.csv"
    with open(output_file, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["comment"])  # Cabecera
        for comment in all_comments:
            writer.writerow([comment])

    print(f"🎉 Extracción completada. Todos los comentarios guardados en {output_file}")

if __name__ == "__main__":
    main()
