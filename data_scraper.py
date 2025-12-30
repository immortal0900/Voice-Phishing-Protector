import os
import re
import time
import traceback
from urllib.parse import urljoin, unquote
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- 1. 설정 ---
# 대상 게시판 (BOARD_ID, MENU_NO)
TARGETS = [
    ("B0000206", "200690"),
    ("B0000207", "200691"),
]

# 기본 URL 및 셀렉터
BASE_URL = "https://www.fss.or.kr/"
CONTENT_SELECTOR = "#content > div.bd-view > div > div"
VIDEO_SELECTOR = '#content div.bd-view video'

# 출력 디렉토리
AUDIO_OUT_DIR = "fss_paired_audio"
TEXT_OUT_DIR = "fss_paired_text"

# 다운로드할 사운드 파일 확장자
SOUND_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.ogg')


def sanitize_filename(name: str) -> str:
    """파일 이름으로 사용할 수 없는 문자를 제거하고 길이를 제한합니다."""
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = name.strip().replace("..", "_")
    return name[:160]


def extract_total_pages(page_html: str) -> int:
    """페이지 HTML에서 전체 페이지 수를 추출합니다."""
    nums = re.findall(r"fnSearch\((\d+)\)", page_html)
    return max(int(n) for n in nums) if nums else 1


def extract_post_links_from_html(soup: BeautifulSoup):
    """페이지 HTML에서 게시물 링크 목록을 추출합니다."""
    post_links = soup.select('td.title > a, td.td_left a, td.subject a')
    if post_links:
        return post_links
    
    # Fallback: onclick 속성에서 nttSn 추출하여 가상 링크 생성
    onclick_raw = re.findall(r"fn_view\(([^)]+)\)", str(soup))
    ids = [m.group(1) for raw in onclick_raw if (m := re.search(r"['\"](\d+)['\"]", raw))]
    return [
        BeautifulSoup(f'<a data-nttSn="{pid}">게시물 {pid}</a>', 'html.parser').a
        for pid in ids
    ]


def resolve_ntt_sn_from_link(a) -> str | None:
    """<a> 태그에서 게시물 고유 ID(nttSn)를 추출합니다."""
    # 다양한 속성에서 nttSn 또는 nttId를 찾습니다.
    for attr in ['onclick', 'href']:
        attr_val = a.get(attr, '')
        m = re.search(r"fn_view\(['\"]?(\d+)|nttSn=(\d+)|nttId=(\d+)", attr_val)
        if m:
            return next((g for g in m.groups() if g), None)
    return a.get('data-nttSn')


def build_detail_url(board_id: str, menu_no: str, link, ntt_id: str) -> str:
    """게시물 상세 페이지 URL을 생성합니다."""
    href = link.get('href', '')
    if href:
        # href에 'nttId'가 명시적으로 사용된 경우, 이를 최우선으로 존중
        if 'nttId=' in href:
            return urljoin(BASE_URL, href)
        if href.startswith('/'):
            return urljoin(BASE_URL, href)
        if href.startswith('http'):
            return href

    # Fallback: nttId 또는 nttSn을 사용하여 URL 직접 구성
    # 제공된 ID가 nttId를 사용하는 최신 형식일 가능성을 먼저 고려
    if len(ntt_id) <= 5: # 경험적으로 nttId가 nttSn보다 짧은 경향이 있음
        return urljoin(BASE_URL, f"/fss/bbs/{board_id}/view.do?menuNo={menu_no}&nttId={ntt_id}")
    return urljoin(BASE_URL, f"/fss/bbs/{board_id}/view.do?menuNo={menu_no}&nttSn={ntt_id}")


def process_text(raw: str) -> str:
    """추출한 원본 텍스트에서 역할(예: '피해자:') 표시를 제거하고 한 줄로 합칩니다."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        m = re.match(r"^\s*([^:]{1,15})\s*:\s*(.*)$", ln)
        cleaned.append(m.group(2).strip() if m else ln)
    return re.sub(r"\s+", " ", " ".join(cleaned)).strip()


def robust_download_with_cookies(driver, url, folder, filename, referer_url=None):
    """
    scrap.py의 안정적인 다운로드 로직을 통합한 함수.
    Content-Disposition, 파일 시그니처 분석, 이름 충돌 처리 등을 수행합니다.
    """
    cookies = {c['name']: c['value'] for c in driver.get_cookies()}
    with requests.Session() as session:
        session.cookies.update(cookies)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Referer': referer_url,
            'Accept': '*/*'
        }
        session.headers.update(headers)
        return robust_download_with_session(session, url, folder, filename, referer_url)


def process_post(post_info: dict):
    """개별 게시물 하나를 처리하는 함수 (스레드에서 실행됨)"""
    board_id = post_info["board_id"]
    menu_no = post_info["menu_no"]
    ntt_sn = post_info["ntt_sn"]
    title_text = post_info["title_text"]
    detail_url = post_info["detail_url"]
    cookies = post_info["cookies"]

    try:
        # 스레드별로 독립적인 requests 세션 사용
        with requests.Session() as session:
            session.cookies.update(cookies)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                'Referer': detail_url,
            }
            session.headers.update(headers)

            response = session.get(detail_url, timeout=30)
            response.raise_for_status()
            detail_soup = BeautifulSoup(response.text, 'html.parser')

            # 1. 텍스트 추출 및 길이 검사
            content_node = detail_soup.select_one(CONTENT_SELECTOR)
            if not content_node:
                print(f"    - (SKIP) nttSn={ntt_sn}: 본문 내용을 찾을 수 없음")
                return

            raw_text = content_node.get_text("\n", strip=True)
            processed_text = process_text(raw_text)

            if len(processed_text) < 100:
                print(f"    - (SKIP) nttSn={ntt_sn}: 텍스트 길이 미달 ({len(processed_text)}자)")
                return

            print(f"  > 처리 중: {title_text} (ID: {ntt_sn})")

            # 2. 오디오 파일 탐색 및 다운로드
            found_audio = False

            # Case A: <video> 태그
            for video_tag in detail_soup.select(VIDEO_SELECTOR):
                src = video_tag.get('src') or (video_tag.find('source') and video_tag.find('source').get('src'))
                if src:
                    audio_url = urljoin(BASE_URL, src)
                    ext = os.path.splitext(audio_url.split('?')[0])[1] or ".mp3"
                    base_filename = f"{ntt_sn}_{sanitize_filename(title_text)}"

                    download_ok, final_audio_path = robust_download_with_cookies(driver, audio_url, AUDIO_OUT_DIR, f"{base_filename}{ext}", detail_url)
                    if download_ok:
                        found_audio = True
                        # 오디오 파일명과 동일한 이름으로 텍스트 파일 저장
                        text_filename = os.path.splitext(os.path.basename(final_audio_path))[0] + ".txt"
                        with open(os.path.join(TEXT_OUT_DIR, text_filename), 'w', encoding='utf-8') as f:
                            f.write(processed_text)
                        print(f"    ✔ (Text) 저장: {text_filename}")

            # Case B: 첨부파일 목록
            for link in detail_soup.select('.file-list li a, .file-list a, ul.attach a'):
                filename_text = link.get_text(strip=True)
                if filename_text.lower().endswith(SOUND_EXTENSIONS):
                    m = re.search(r"fn_cmmFileDown\(['\"](\w+)['\"],\s*['\"](\w+)['\"]\)", link.get('href', ''))
                    if m:
                        atch_id, file_sn = m.groups()
                        audio_url = urljoin(BASE_URL, f"/fss/cmm/file/fileDown.do?atchFileNo={atch_id}&fileSn={file_sn}")
                        
                        base_filename = f"{ntt_sn}_{sanitize_filename(title_text)}"
                        ext = os.path.splitext(filename_text)[1]

                        download_ok, final_audio_path = robust_download_with_cookies(driver, audio_url, AUDIO_OUT_DIR, f"{base_filename}{ext}", detail_url)
                        if download_ok:
                            found_audio = True
                            # 오디오 파일명과 동일한 이름으로 텍스트 파일 저장
                            text_filename = os.path.splitext(os.path.basename(final_audio_path))[0] + ".txt"
                            with open(os.path.join(TEXT_OUT_DIR, text_filename), 'w', encoding='utf-8') as f:
                                f.write(processed_text)
                            print(f"    ✔ (Text) 저장: {text_filename}")

            if not found_audio:
                print("    - (SKIP) 텍스트 조건은 만족했으나, 음성 파일을 찾지 못함")

    except Exception as e:
        print(f"    ✖ nttSn={ntt_sn} 처리 중 오류: {e}")
        # traceback.print_exc()


def main():
    print("금융감독원 게시물 음성/텍스트 데이터 스크래핑을 시작합니다.")
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    os.makedirs(TEXT_OUT_DIR, exist_ok=True)

    # --- Selenium WebDriver 설정 ---
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
    wait = WebDriverWait(driver, 20)

    try:
        for board_id, menu_no in TARGETS:
            print(f"\n[게시판 시작] BOARD_ID: {board_id}, MENU_NO: {menu_no}")
            list_url_base = f"https://www.fss.or.kr/fss/bbs/{board_id}/list.do?menuNo={menu_no}"

            # 첫 페이지 로드하여 전체 페이지 수 확인
            driver.get(f"{list_url_base}&pageIndex=1")
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table')))
            total_pages = extract_total_pages(driver.page_source)
            print(f"  총 {total_pages} 페이지 발견")

            for page in range(1, total_pages + 1):
                print(f"\n  [페이지 {page}/{total_pages}]")
                if page > 1:
                    driver.get(f"{list_url_base}&pageIndex={page}")
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table')))
                
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                links = extract_post_links_from_html(soup)
                print(f"  게시물 {len(links)}건 발견. 병렬 처리 시작...")

                post_infos = []
                for a in links:
                    ntt_sn = resolve_ntt_sn_from_link(a)
                    if not ntt_sn:
                        continue
                    
                    post_infos.append({
                        "board_id": board_id,
                        "menu_no": menu_no,
                        "ntt_sn": ntt_sn,
                        "title_text": a.get_text(strip=True) or f"게시물_{ntt_sn}",
                        "detail_url": build_detail_url(board_id, menu_no, a, ntt_sn),
                        "cookies": {c['name']: c['value'] for c in driver.get_cookies()}
                    })

                with ThreadPoolExecutor(max_workers=10) as executor:
                    # list()를 사용하여 모든 스레드가 완료될 때까지 기다립니다.
                    list(executor.map(process_post, post_infos))
    
    finally:
        if 'driver' in locals() and driver:
            driver.quit()
        print("\nWebDriver 종료. 스크래핑 완료.")


if __name__ == "__main__":
    main()
