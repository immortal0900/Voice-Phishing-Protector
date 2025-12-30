import os
import re
import time
import traceback
from urllib.parse import urljoin, unquote

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# 대상 게시판(BOARD_ID, MENU_NO)
TARGETS = [
    ("B0000206", "200690"),
    ("B0000207", "200691"),
]

BASE_URL = "https://www.fss.or.kr/"
CONTENT_SELECTOR = "#content > div.bd-view > div > div"
OUT_ROOT = "fss_text_outputs"


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = name.strip().replace("..", "_")
    return name[:160]


def extract_total_pages(page_html: str) -> int:
    nums = re.findall(r"fnSearch\((\d+)\)", page_html)
    if not nums:
        return 1
    try:
        return max(int(n) for n in nums)
    except ValueError:
        return 1


def extract_post_links_from_html(page_html: str, soup: BeautifulSoup):
    # 1차: 일반적인 제목 셀
    post_links = soup.select('td.title > a')
    if post_links:
        return post_links
    # 2차: 대체 셀렉터들
    post_links = soup.select('td.td_left a, td.subject a')
    if post_links:
        return post_links
    # 3차: fn_view(...) 정규식 복구
    onclick_raw = re.findall(r"fn_view\(([^)]+)\)", page_html)
    ids = []
    for raw in onclick_raw:
        m = re.search(r"['\"](\d+)['\"]", raw)
        if m:
            ids.append(m.group(1))
    forged = []
    for pid in ids:
        fake = soup.new_tag('a')
        fake['data-nttSn'] = pid
        fake.string = f'게시물 {pid}'
        forged.append(fake)
    return forged


def resolve_ntt_sn_from_link(a) -> str | None:
    onclick_attr = a.get('onclick', '') or ''
    href_attr = a.get('href', '') or ''
    data_attr = a.get('data-nttSn') or ''
    # onclick 내 fn_view
    m = re.search(r"fn_view\(['\"]?(\d+)", onclick_attr)
    if m:
        return m.group(1)
    # data-nttSn
    if data_attr.isdigit():
        return data_attr
    # href 내 nttSn= 또는 nttId=
    m2 = re.search(r"nttSn=(\d+)", href_attr)
    if m2:
        return m2.group(1)
    m3 = re.search(r"nttId=(\d+)", href_attr)
    if m3:
        return m3.group(1)
    # 마지막 시도: href 전체에서 fn_view
    m4 = re.search(r"fn_view\(['\"]?(\d+)", href_attr)
    if m4:
        return m4.group(1)
    return None


def build_detail_url(board_id: str, menu_no: str, link, ntt_sn: str) -> str:
    href = link.get('href', '') or ''
    if href.startswith('/'):
        return urljoin(BASE_URL, href)
    if href.startswith('http'):
        return href
    return urljoin(BASE_URL, f"/fss/bbs/{board_id}/view.do?menuNo={menu_no}&nttSn={ntt_sn}")


def process_text(raw: str) -> str:
    """라인별 '역할: 내용' 형태가 섞인 텍스트에서 역할 prefix 제거 후 공백 단일화"""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        # 시작부의 'XXXX :' 패턴 제거(라벨 길이 1~15글자 내)
        m = re.match(r"^\s*([^:]{1,15})\s*:\s*(.*)$", ln)
        if m:
            cleaned.append(m.group(2).strip())
        else:
            cleaned.append(ln)
    merged = " ".join(cleaned)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def save_text(out_dir: str, filename: str, original_text: str, processed_text: str) -> None:
    os.makedirs(out_dir, exist_ok = True)
    path = os.path.join(out_dir, filename)
    with open(path, 'w', encoding = 'utf-8') as f:
        # 요청사항: 가공 데이터(merged single line)만 저장
        f.write(processed_text.strip() + "\n")


def main():
    print("금융감독원 게시물 텍스트 추출을 시작합니다.")
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    for arg in [
        '--headless=new',
        '--disable-gpu',
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--log-level=3',
        '--window-size=1400,1000'
    ]:
        options.add_argument(arg)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service = service, options = options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
    )
    driver.set_page_load_timeout(20)

    wait = WebDriverWait(driver, 25)

    try:
        for board_id, menu_no in TARGETS:
            # 페이지 네비게이션을 pageIndex 쿼리로 직접 이동
            list_url_base = (
                f"https://www.fss.or.kr/fss/bbs/{board_id}/list.do?menuNo={menu_no}"
                f"&bbsId=&cl1Cd=&sdate=&edate=&searchCnd=1&searchWrd="
            )
            out_dir = os.path.join(OUT_ROOT, f"{board_id}_{menu_no}")
            os.makedirs(out_dir, exist_ok = True)
            print(f"\n[BOARD] {board_id} menuNo={menu_no}")

            # 1페이지 로드
            driver.get(f"{list_url_base}&pageIndex=1")
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table')))
            page_html = driver.page_source
            soup = BeautifulSoup(page_html, 'html.parser')
            total_pages = extract_total_pages(page_html)
            print(f"  페이지 수: {total_pages}")

            for page in range(1, total_pages + 1):
                try:
                    driver.get(f"{list_url_base}&pageIndex={page}")
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table')))
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  페이지 {page} 이동 실패: {e}")
                    break

                page_html = driver.page_source
                soup = BeautifulSoup(page_html, 'html.parser')
                links = extract_post_links_from_html(page_html, soup)
                print(f"  페이지 {page}: 게시물 {len(links)}건")

                for a in links:
                    try:
                        ntt_sn = resolve_ntt_sn_from_link(a)
                        if not ntt_sn:
                            continue
                        title_text = a.get_text(strip = True) or f"게시물 {ntt_sn}"
                        detail_url = build_detail_url(board_id, menu_no, a, ntt_sn)

                        driver.get(detail_url)
                        # 상세 로딩 대기(본문 컨테이너)
                        end_time = time.time() + 10
                        content_html = None
                        while time.time() < end_time:
                            html_now = driver.page_source
                            sp = BeautifulSoup(html_now, 'html.parser')
                            node = sp.select_one(CONTENT_SELECTOR)
                            if node:
                                content_html = node
                                break
                            time.sleep(0.3)
                        if not content_html:
                            # 바로 한 번 더 시도
                            sp = BeautifulSoup(driver.page_source, 'html.parser')
                            content_html = sp.select_one(CONTENT_SELECTOR)
                        if not content_html:
                            # 건너뜀
                            continue

                        raw_text = content_html.get_text("\n", strip = True)
                        if not raw_text or len(raw_text) < 50:
                            continue
                        processed = process_text(raw_text)

                        safe_title = sanitize_filename(title_text)[:60]
                        filename = f"{ntt_sn}_{safe_title}.txt"
                        save_text(out_dir, filename, raw_text, processed)
                        print(f"    ✔ 저장: {filename} (길이 {len(raw_text)}자)")
                    except Exception as e:
                        print(f"    ✖ 게시물 처리 실패: {e}")
                        traceback.print_exc()

    finally:
        driver.quit()
        print("\nWebDriver 종료")

    print("\n모든 텍스트 추출 종료")


if __name__ == "__main__":
    main()
