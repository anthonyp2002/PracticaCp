import os
import csv
import time
import requests
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Minimizar mensajes de TensorFlow Lite
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configurar logging detallado
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Ejecutar en segundo plano
options.add_argument("--use-gl=desktop")
options.add_argument("--enable-gpu-rasterization")
options.add_argument("--enable-webgl")
options.add_argument("--enable-zero-copy")

# Variable global para almacenar comentarios
all_comments = []

def set_headers(videoid):
    """
    Configura y devuelve los encabezados necesarios para las solicitudes a la API de TikTok.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/112.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.tiktok.com/",
    }
    return headers

def extract_replies(videoid, comment_id, headers, cursor):
    try:
        logging.info(f"📢 Extrayendo respuestas del comentario {comment_id}...")
        response = requests.get(
            f"https://www.tiktok.com/api/comment/list/reply/?aweme_id={videoid}&comment_id={comment_id}&count=5&cursor={cursor}",
            headers=headers,
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        logging.info(f"✅ {len(data.get('comments', []))} respuestas extraídas.")
        return data
    except requests.exceptions.Timeout:
        logging.error("❌ Timeout al extraer respuestas.")
        return {"comments": []}
    except Exception as e:
        logging.error(f"❌ Error al extraer respuestas: {e}")
        return {"comments": []}

def tiktok_extract_comments(video_url):
    global all_comments

    # Resolver redirecciones
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
    headers = set_headers(videoid)
    total_comments = 0

    while True:
        try:
            logging.info(f"🔄 Solicitando comentarios... Cursor: {cursor}")
            response = requests.get(
                f"https://www.tiktok.com/api/comment/list/?aid=1988&aweme_id={videoid}&count=5&cursor={cursor}",
                headers=headers,
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

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

                reply_count = comment.get('reply_comment_total', 0)
                if reply_count > 0:
                    comment_id = comment.get("cid")
                    replies = extract_replies(videoid, comment_id, headers, cursor)
                    for reply in replies.get("comments", []):
                        reply_text = reply.get('text', '')
                        if reply_text:
                            logging.info(f"↪️ Respuesta: {reply_text}")
                            all_comments.append(reply_text)
                            total_comments += 1

            cursor += len(comments)

        except requests.exceptions.Timeout:
            logging.error("❌ Timeout durante la extracción del video.")
            break
        except Exception as e:
            logging.error(f"❌ Error durante la extracción del video {videoid}: {e}")
            break

    logging.info(f"✅ Extracción completada. Total de comentarios: {total_comments}")
    return total_comments

def search_tiktok(search_term):
    driver = webdriver.Chrome(options=options)
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
        video_urls = [elem.get_attribute('href') for elem in video_elements[:5]]
        logging.info("✅ URLs de los primeros 5 videos:")
        for url in video_urls:
            logging.info(url)

        for url in video_urls:
            logging.info(f"\n📢 Extrayendo comentarios de: {url}")
            tiktok_extract_comments(url)
            time.sleep(2)

        logging.info("\n✅ Comentarios extraídos de todos los videos.")

        with open("comments.csv", "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Comments"])
            for comment in all_comments:
                writer.writerow([comment])
        logging.info("✅ Comentarios guardados en 'comments.csv'")

    except Exception as e:
        logging.error(f"❌ Error en la búsqueda o extracción: {e}")

    finally:
        driver.quit()

if __name__ == "__main__":
    search_term = input("Ingrese el término de búsqueda en TikTok: ")
    search_tiktok(search_term)
