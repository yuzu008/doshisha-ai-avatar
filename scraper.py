import time
import re
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
# 【修正点】 'Exception' ではなく、より具体的な 'WebDriverException' をインポートします
from selenium.common.exceptions import WebDriverException
from bs4 import BeautifulSoup

# --- 設定項目 ---
START_URL = "https://www.doshisha.ac.jp/"
ALLOWED_DOMAIN = "doshisha.ac.jp"
OUTPUT_FILE = "doshisha_data.txt"
MAX_PAGES = 100 
WAIT_TIME = 2

def crawl_doshisha_for_rag():
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    options.add_argument('--log-level=3')
    options.add_argument('--headless')

    print("クローラーを起動します...")
    driver = webdriver.Chrome(options=options)

    crawled_urls = set()
    queue = [START_URL]
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        print(f"データ保存先: {OUTPUT_FILE}")

        while queue and len(crawled_urls) < MAX_PAGES:
            current_url = queue.pop(0)

            if current_url in crawled_urls:
                continue

            try:
                print(f"[{len(crawled_urls) + 1}/{MAX_PAGES}] クローリング中: {current_url}")
                driver.get(current_url)
                
                time.sleep(WAIT_TIME)

                soup = BeautifulSoup(driver.page_source, 'html.parser')

                main_content = soup.find('main')
                if not main_content:
                    main_content = soup.find('body')
                
                if main_content:
                    for tag in main_content(['script', 'style']):
                        tag.decompose()
                    
                    text = re.sub(r'\s{2,}', '\n', main_content.get_text(separator='\n')).strip()

                    f.write(f"--- URL: {current_url} ---\n")
                    f.write(text + "\n\n")

                crawled_urls.add(current_url)

                for a_tag in soup.find_all('a', href=True):
                    link = a_tag['href']
                    full_url = urljoin(current_url, link)
                    full_url = full_url.split('#')[0]

                    if urlparse(full_url).netloc.endswith(ALLOWED_DOMAIN) and \
                       full_url not in queue and \
                       full_url not in crawled_urls:
                        queue.append(full_url)
            
            # 【修正点】 ここも 'WebDriverException' に合わせます
            except WebDriverException as e:
                print(f"エラー発生、スキップします: {current_url} - {str(e).splitlines()[0]}") # エラーメッセージを1行に
            except Exception as e:
                print(f"予期せぬエラー: {e}")

    driver.quit()
    print(f"\nクローリング完了。合計 {len(crawled_urls)} ページを収集し、{OUTPUT_FILE} に保存しました。")

if __name__ == "__main__":
    crawl_doshisha_for_rag()