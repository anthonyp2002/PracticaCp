import asyncio
import aiohttp
import csv
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Ejecutar en segundo plano
options.add_argument("--use-gl=desktop")
options.add_argument("--enable-gpu-rasterization")
options.add_argument("--enable-webgl")
options.add_argument("--enable-zero-copy")
driver = webdriver.Chrome(options=options)

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

async def fetch_comments(session, url, headers):
    async with session.get(url, headers=headers) as response:
        return await response.json()

async def tiktok_extract_comments_async(video_url):
    global all_comments

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

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                url = f"https://www.tiktok.com/api/comment/list/?aid=1988&aweme_id={videoid}&count=20&cursor={cursor}"
                data = await fetch_comments(session, url, headers)

                comments = data.get("comments")
                if not comments:
                    logging.info("✅ No hay más comentarios disponibles.")
                    break

                # Procesar solo los comentarios principales (sin respuestas)
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
    return total_comments

def search_tiktok(search_term):
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

        async def main():
            tasks = [tiktok_extract_comments_async(url) for url in video_urls]
            await asyncio.gather(*tasks)

        asyncio.run(main())

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