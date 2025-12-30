import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from urllib.parse import unquote
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
import traceback

# --- 설정 ---
# 게시판 식별자 / 메뉴번호 (다른 게시판 크롤링 시 여기만 변경)
BOARD_ID = "B0000207"
MENU_NO = "200691"
# 게시판 목록 URL (동적 구성)
LIST_URL = f"https://www.fss.or.kr/fss/bbs/{BOARD_ID}/list.do?menuNo={MENU_NO}"
# 기본 URL (상대 경로를 절대 경로로 변환하기 위함)
BASE_URL = "https://www.fss.or.kr/"
# 파일 저장 경로 (게시판별 구분)
DOWNLOAD_DIR = f"fss_voice_files_{MENU_NO}"
# 다운로드할 사운드 파일 확장자
SOUND_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.ogg')
VIDEO_SELECTOR = '#content div.bd-view video'
DEBUG_SINGLE_MODE = False  # A 모드: 첫 게시물 한 개만 정밀 진단
# 최대 페이지 제한 (None이면 전체 페이지 크롤링)
MAX_PAGE_LIMIT = None  # 예: 10 으로 설정하면 10페이지만, None이면 전체

def debug_print(title, content):
    print(f"[DEBUG] {title}: {content}")

def probe_url(driver, url, referer):
    """URL 사전 진단: HEAD 시도 후 실패 시 소량 GET.
    반환: (status, content_type, size_hint, note)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Referer': referer,
        'Accept': '*/*'
    }
    cookies = build_cookie_header(driver)
    try:
        resp = requests.head(url, headers=headers, cookies=cookies, timeout=15, allow_redirects=True)
        ct = resp.headers.get('Content-Type', '')
        cl = resp.headers.get('Content-Length', '')
        return resp.status_code, ct, cl or 'unknown', 'HEAD'
    except Exception as he:
        # fallback GET (range-like small read)
        try:
            resp = requests.get(url, headers=headers, cookies=cookies, stream=True, timeout=20)
            ct = resp.headers.get('Content-Type', '')
            first = b''
            size = 0
            for i, chunk in enumerate(resp.iter_content(chunk_size=4096)):
                if not chunk:
                    break
                first += chunk
                size += len(chunk)
                if i >= 1:
                    break
            resp.close()
            note = 'GET-first-chunks'
            if ct.lower().startswith('text/html'):
                note += ' HTML-snippet=' + first[:120].decode(errors='ignore').replace('\n', ' ')
            return getattr(resp, 'status_code', 'NA'), ct, f">={size} bytes (partial)", note
        except Exception as ge:
            return 'ERR', 'N/A', '0', f'probe-failed {he} / {ge}'

# --- 다운로드 디렉토리 생성 ---
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
    print(f"'{DOWNLOAD_DIR}' 폴더를 생성했습니다.")

def download_file(url, folder, filename):
    """지정된 URL에서 파일을 다운로드하여 로컬 폴더에 저장합니다."""
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        print(f"이미 파일이 존재합니다: {filename}")
        return
    try:
        # 다운로드 시 User-Agent를 설정하여 403 Forbidden 오류 방지
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        with requests.get(url, stream=True, headers=headers) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"다운로드 완료: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"'{filename}' 다운로드 중 오류 발생: {e}")

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    name = name.strip().replace('..', '_')
    return name[:180]  # 너무 긴 이름 컷

def build_cookie_header(driver):
    cookies = driver.get_cookies()
    return {c['name']: c['value'] for c in cookies}

def robust_download_with_cookies(driver, url, folder, filename, referer_url=None):
    filename = sanitize_filename(filename)
    filepath = os.path.join(folder, filename)
    # 이미 존재하지만 .do / .bin 이면 재판별 시도
    if os.path.exists(filepath) and not filename.lower().endswith(('.do', '.bin')):
        return
    # 임시 파일 경로 (완료 후 rename)
    temp_filepath = filepath + '.part'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Referer': referer_url or LIST_URL,
        'Accept': '*/*'
    }
    try:
        with requests.get(url, headers=headers, cookies=build_cookie_header(driver), stream=True, timeout=40) as r:
            r.raise_for_status()
            ctype = r.headers.get('Content-Type', '')
            if 'text/html' in ctype.lower() and len(url) < 500:
                # 의도치 않은 HTML (권한/오류) 반환 시 로그
                snippet = r.text[:200].replace('\n', ' ')
                print(f"  ! 예상과 다른 응답(HTML) -> {ctype} / {snippet}...")
            # 서버가 보내는 실제 파일명(Content-Disposition)
            disp = r.headers.get('Content-Disposition', '')
            header_filename = None
            if 'filename=' in disp:
                # filename*= or filename="..."
                m_fn = re.search(r'filename\*=UTF-8\'\'?([^";]+)', disp)
                if m_fn:
                    header_filename = m_fn.group(1)
                else:
                    m_fn2 = re.search(r'filename="?([^";]+)"?', disp)
                    if m_fn2:
                        header_filename = m_fn2.group(1)
            if header_filename:
                # 퍼센트 인코딩(UTF-8) 디코딩 후 정리
                try:
                    header_filename = unquote(header_filename)
                except Exception:
                    pass
                header_filename = sanitize_filename(header_filename)
            # 최초 파일명 확장자 추출
            base_orig, ext_orig = os.path.splitext(filename)
            # 스트림 다운로드 + 시그니처 버퍼
            sig = b''
            total = 0
            with open(temp_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=16384):
                    if not chunk:
                        continue
                    if len(sig) < 65536:  # 64KB 이내만 헤더 판독용 보관
                        need = 65536 - len(sig)
                        sig += chunk[:need]
                    f.write(chunk)
                    total += len(chunk)

            def detect_ext(sig_bytes: bytes, content_type: str) -> str:
                ct = (content_type or '').lower()
                # Content-Type 우선
                if 'audio/mpeg' in ct or 'mpeg' in ct:
                    return '.mp3'
                if 'audio/wav' in ct or 'x-wav' in ct or 'wave' in ct:
                    return '.wav'
                if 'audio/ogg' in ct or 'application/ogg' in ct:
                    return '.ogg'
                if 'aac' in ct:
                    return '.aac'
                if 'mp4' in ct or 'm4a' in ct or 'video/mp4' in ct:
                    return '.mp4'
                # 시그니처 판별
                if sig_bytes.startswith(b'ID3'):
                    return '.mp3'
                if len(sig_bytes) > 2 and sig_bytes[0] == 0xFF and (sig_bytes[1] & 0xE0) == 0xE0:
                    return '.mp3'  # MP3 frame sync
                if sig_bytes.startswith(b'RIFF') and sig_bytes[8:12] == b'WAVE':
                    return '.wav'
                if sig_bytes.startswith(b'OggS'):
                    return '.ogg'
                # MP4/ISOBMFF: offset 4~8 'ftyp'
                if len(sig_bytes) >= 12 and sig_bytes[4:8] == b'ftyp':
                    # 브랜드로 m4a 가능
                    brand = sig_bytes[8:12]
                    if brand.lower().startswith(b'm4a'):
                        return '.m4a'
                    return '.mp4'
                return ''

            # 최종 확장자 결정 우선순위: Content-Disposition -> 기존 확장자(유효) -> 시그니처 -> 기존(.do 등) fallback
            final_ext = None
            # 1) header filename
            if header_filename:
                _, header_ext = os.path.splitext(header_filename)
                if header_ext:
                    final_ext = header_ext
            # 2) 기존 확장자 사용 (단, 의미 없는 .do/.bin/.part 제외)
            if not final_ext and ext_orig and ext_orig.lower() not in ('.do', '.bin', '.part'):
                final_ext = ext_orig
            # 3) 시그니처 기반
            if not final_ext:
                detected = detect_ext(sig, ctype)
                if detected:
                    final_ext = detected
            # 4) fallback
            if not final_ext:
                final_ext = ext_orig if ext_orig else ''

            # 파일명 재구성: header_filename이 있다면 그것 사용, 없으면 기존 base + final_ext
            if header_filename:
                new_base, new_ext = os.path.splitext(header_filename)
                new_ext = new_ext or final_ext
                new_name = sanitize_filename(header_filename if new_ext else header_filename + final_ext)
            else:
                new_name = base_orig + final_ext
            # 퍼센트 인코딩 잔재 제거 재시도
            if '%' in new_name:
                try:
                    new_name = sanitize_filename(unquote(new_name))
                except Exception:
                    pass
            # ID prefix 보존: base_orig이 이미 ID_* 형태이면 그대로, 아니면 기존 filename 유지
            # 기존 filename이  nnnn_video_* 형태면 그것 유지
            if filename.startswith(base_orig) and not filename.endswith(final_ext):
                # rename target
                pass
            # 최종 경로
            final_path = os.path.join(folder, new_name)
            # 이름 충돌 시 접미어 숫자 부여
            if os.path.exists(final_path) and temp_filepath != final_path:
                root, ext2 = os.path.splitext(new_name)
                cnt = 1
                while os.path.exists(os.path.join(folder, f"{root}_{cnt}{ext2}")):
                    cnt += 1
                final_path = os.path.join(folder, f"{root}_{cnt}{ext2}")
            os.replace(temp_filepath, final_path)
            if DEBUG_SINGLE_MODE:
                debug_print('saved_file_info', {
                    'orig_req_name': filename,
                    'header_filename': header_filename,
                    'chosen_name': os.path.basename(final_path),
                    'content_type': ctype,
                    'size_bytes': total,
                    'url_tail': url[-60:]
                })
        print(f"  ✔ 저장됨: {os.path.basename(final_path)} (요청명: {filename})")
    except Exception as e:
        print(f"  ✖ 다운로드 실패({filename}): {e}  URL={url}")
        # 실패 시 임시파일 제거
        try:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
        except Exception:
            pass

def extract_post_links_from_html(page_html: str, soup: BeautifulSoup):
    # 1차 기본
    post_links = soup.select('td.title > a')
    if post_links:
        return post_links
    # 2차 대체
    post_links = soup.select('td.td_left a, td.subject a')
    if post_links:
        return post_links
    # 3차 정규식 기반 추출
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

def extract_total_pages(page_html: str) -> int:
    nums = re.findall(r"fnSearch\((\d+)\)", page_html)
    if not nums:
        return 1
    try:
        return max(int(n) for n in nums)
    except ValueError:
        return 1

def main():
    """메인 실행 함수 (게시물 목록 + 상세 모두 Selenium 처리)"""
    print("금융감독원 '그놈 목소리' 음성파일 다운로드를 시작합니다.")

    # --- Selenium WebDriver 설정 ---
    print("WebDriver 초기화 중...")
    try:
        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        # 안정성 옵션
        for arg in [
            '--headless=new',
            '--disable-gpu',
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--log-level=3',
            '--window-size=1400,1000'
        ]:
            options.add_argument(arg)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])  # bot 차단 회피 일부
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
        driver = webdriver.Chrome(service=service, options=options)
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
        )
        driver.set_page_load_timeout(20)
    except Exception as e:
        print(f"WebDriver 설정 실패: {e}")
        return

    wait = WebDriverWait(driver, 25)

    try:
        print("목록 페이지 로딩...")
        driver.get(LIST_URL)
        # 목록 로드 대기 (제목 셀 혹은 페이징 영역)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table')))  # 테이블 존재
        page_html = driver.page_source
        soup = BeautifulSoup(page_html, 'html.parser')
        total_pages = extract_total_pages(page_html)
        print(f"탐지된 페이지 수: {total_pages}")

        collected = []
        first_page_links = extract_post_links_from_html(page_html, soup)
        collected.append((1, first_page_links))
        if not first_page_links:
            print('첫 페이지에서 게시물 링크를 찾지 못했습니다.')
            return

        # 추가 페이지 순회: 제한값이 있으면 적용, 없으면 전체
        if MAX_PAGE_LIMIT is not None:
            effective_limit = min(total_pages, MAX_PAGE_LIMIT)
            print(f"페이지 제한 적용: {effective_limit}/{total_pages}")
        else:
            effective_limit = total_pages
        max_pages_to_fetch = 1 if DEBUG_SINGLE_MODE else effective_limit
        for p in range(2, max_pages_to_fetch + 1):
            try:
                driver.execute_script(f'fnSearch({p})')
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table')))  # 테이블 재로딩
                time.sleep(0.6)
                ph = driver.page_source
                sp = BeautifulSoup(ph, 'html.parser')
                links = extract_post_links_from_html(ph, sp)
                print(f"페이지 {p}: {len(links)}건")
                collected.append((p, links))
            except Exception as e:
                print(f"페이지 {p} 로딩 실패: {e}")
                break

        all_links = []
        for _p, links in collected:
            all_links.extend(links)
        print(f"총 수집 게시물: {len(all_links)} -> 상세 페이지 순회 시작", flush=True)

        # 링크들에서 ID, 제목만 먼저 평탄화 (href 패턴 추가)
        id_title_list = []
        for i, link in enumerate(all_links, start=1):
            onclick_attr = link.get('onclick', '') or ''
            href_attr = link.get('href', '') or ''
            data_attr = link.get('data-nttSn') or ''
            ntt_sn = None
            # 1) onclick 내 fn_view
            m = re.search(r"fn_view\(['\"]?(\d+)", onclick_attr)
            if m:
                ntt_sn = m.group(1)
                param_type = 'fn_view_onclick'
            # 2) data-nttSn
            if not ntt_sn and data_attr.isdigit():
                ntt_sn = data_attr
                param_type = 'data-nttSn'
            # 3) href 내 nttSn= 파라미터
            if not ntt_sn:
                m2 = re.search(r"nttSn=(\d+)", href_attr)
                if m2:
                    ntt_sn = m2.group(1)
                    param_type = 'nttSn'
            # 3b) href 내 nttId= 파라미터 (새 구조)
            if not ntt_sn:
                m2b = re.search(r"nttId=(\d+)", href_attr)
                if m2b:
                    ntt_sn = m2b.group(1)
                    param_type = 'nttId'
            # 4) href/onlick 자바스크립트 전체에서 fn_view 패턴 다시 탐색
            if not ntt_sn:
                m3 = re.search(r"fn_view\(['\"]?(\d+)", href_attr)
                if m3:
                    ntt_sn = m3.group(1)
            # 5) 마지막 fallback: 숫자만으로 된 href 끝부분 (조심스럽게)
            if not ntt_sn:
                tail_digits = re.search(r"(\d{4,})$", href_attr.split('?')[-1])
                if tail_digits:
                    ntt_sn = tail_digits.group(1)
            if not ntt_sn:
                if DEBUG_SINGLE_MODE:
                    debug_print('skip_link_no_id', {
                        'index': i,
                        'text': link.get_text(strip=True)[:80],
                        'href': href_attr[:120],
                        'onclick': onclick_attr[:120]
                    })
                continue
            title_text = link.get_text(strip=True) or f'게시물 {ntt_sn}'
            id_title_list.append((ntt_sn, title_text))
        if DEBUG_SINGLE_MODE:
            debug_print('link_extraction_summary', {'total_links': len(all_links), 'extracted': len(id_title_list)})

        if DEBUG_SINGLE_MODE and len(id_title_list) > 1:
            id_title_list = id_title_list[:1]
        print(f"세부 처리 대상 수: {len(id_title_list)} (DEBUG_SINGLE_MODE={DEBUG_SINGLE_MODE})", flush=True)

        for idx, (ntt_sn, title_text) in enumerate(id_title_list, start=1):
            print(f"\n[{idx}/{len(id_title_list)}] {title_text} (ID={ntt_sn})", flush=True)
            # href 안에 nttId 혹은 nttSn 포함된 원본을 가능한 사용 (구조 변화 대응)
            # all_links 와 동일 순서를 유지했으므로 ntt_sn 매칭되는 첫 href 재추출
            orig_href = None
            for link in all_links:
                h = link.get('href','') or ''
                if ntt_sn and ntt_sn in h:
                    orig_href = h
                    break
            if orig_href and orig_href.startswith('/'):
                post_url = urljoin(BASE_URL, orig_href)
            elif orig_href and orig_href.startswith('http'):
                post_url = orig_href
            else:
                post_url = urljoin(BASE_URL, f"/fss/bbs/{BOARD_ID}/view.do?menuNo={MENU_NO}&nttSn={ntt_sn}")
            if DEBUG_SINGLE_MODE:
                debug_print('detail_url', post_url)
            try:
                driver.get(post_url)
                # 본문/파일 리스트 대기
                # 파일리스트 혹은 비디오 태그 명시적 대기 (폴링 포함)
                end_time = time.time() + 8
                video_ready = False
                list_ready = False
                while time.time() < end_time:
                    html_now = driver.page_source
                    if ('fileDown.do' in html_now) and ('.file-list' in html_now):
                        list_ready = True
                    if '<video' in html_now:
                        video_ready = True
                    if video_ready or list_ready:
                        break
                    time.sleep(0.4)
                if not (video_ready or list_ready):
                    print('  (경고) 비디오/첨부 로딩 신호 없음 - 강제 진행')
                page_source_now = driver.page_source
                detail_soup = BeautifulSoup(page_source_now, 'html.parser')
                if DEBUG_SINGLE_MODE:
                    # 핵심 단서 출력
                    debug_print('video_tag_count', len(detail_soup.select(VIDEO_SELECTOR)))
                    debug_print('contains_fileDown_string', 'fileDown.do' in page_source_now)
                    # a 태그 중 fileDown 문구를 가진 것
                    anchors = detail_soup.find_all('a')
                    hit = [a for a in anchors if 'fileDown' in (a.get('href','') + a.get('onclick',''))]
                    debug_print('anchor_fileDown_count', len(hit))
                    if hit:
                        sample = hit[0]
                        debug_print('sample_anchor_html', str(sample)[:200])
                    # hidden inputs 조사
                    hidden_inputs = detail_soup.select('input[type=hidden]')
                    candidate_hidden = []
                    for hi in hidden_inputs:
                        name = (hi.get('name') or '') + (hi.get('id') or '')
                        val = hi.get('value') or ''
                        if any(k in name.lower() for k in ['atch', 'file']):
                            candidate_hidden.append((name, val))
                    debug_print('hidden_fields_match', candidate_hidden[:5])
            except Exception as e:
                print(f"  상세 진입 실패: {e}")
                traceback.print_exc()
                continue

            sound_found = False

            # 1) video 태그 직접 소스 추출 (요청 받은 셀렉터)
            try:
                video_tags = detail_soup.select(VIDEO_SELECTOR)
                media_sources = []
                for v in video_tags:
                    vs = v.get('src')
                    if vs:
                        media_sources.append(vs)
                    for s in v.find_all('source'):
                        ss = s.get('src')
                        if ss:
                            media_sources.append(ss)
                media_sources = list(dict.fromkeys(media_sources))
                for ms_idx, media_rel in enumerate(media_sources, start=1):
                    media_url = urljoin(BASE_URL, media_rel)
                    basename = os.path.basename(media_url.split('?')[0]) or f"media_{ms_idx}.mp4"
                    # 확장자 추론 (Content-Type까지 보려면 사전 HEAD 시도 가능 - 단순화)
                    if not os.path.splitext(basename)[1]:
                        # audio/video MIME 추론이 없어 확장자 미포함이면 .mp4 기본
                        basename += '.mp4'
                    # 게시물 제목 기반 가독성 향상 (제목은 너무 길 수 있어 60자 컷)
                    safe_title = sanitize_filename(title_text)[:60]
                    final_name = f"{ntt_sn}_{safe_title}_video{ms_idx}{os.path.splitext(basename)[1]}"
                    if DEBUG_SINGLE_MODE:
                        status, ctype, size_hint, note = probe_url(driver, media_url, post_url)
                        debug_print('probe_video', {'url': media_url, 'status': status, 'ctype': ctype, 'size': size_hint, 'note': note})
                    robust_download_with_cookies(driver, media_url, DOWNLOAD_DIR, final_name, referer_url=post_url)
                    sound_found = True
                if media_sources:
                    print(f"  video 태그 소스 {len(media_sources)}건 처리")
            except Exception as ve:
                print(f"  video 처리 오류: {ve}")

            # 2) 첨부파일 리스트 (기존 방식)
            file_links = detail_soup.select('.file-list li a, .file-list a, ul.attach a')
            if not file_links and not sound_found:
                print('  첨부파일/비디오 없음')
                continue

            for a in file_links:
                try:
                    fname = a.get_text(strip=True)
                    if not fname or not fname.lower().endswith(SOUND_EXTENSIONS):
                        continue
                    sound_found = True
                    raw = a.get('href', '') or a.get('onclick', '')
                    # atchFileNo / atchFileId 모두 지원 정규식
                    mfile = re.search(r"fn_cmmFileDown\(['\"](\w+)['\"],\s*['\"](\w+)['\"]\)", raw)
                    atch_id = None
                    file_sn = None
                    if mfile:
                        atch_id, file_sn = mfile.group(1), mfile.group(2)
                    else:
                        data_attrs = ' '.join([f"{k}={v}" for k, v in a.attrs.items()])
                        mfile = re.search(r"(atchFile(?:No|Id))=([A-Za-z0-9]+).*?fileSn=([0-9]+)", data_attrs)
                        if mfile:
                            atch_id = mfile.group(2)
                            file_sn = mfile.group(3)
                    if not (atch_id and file_sn):
                        print(f"  식별자 추출 실패 -> {fname}  raw={raw[:80]}")
                        continue
                    # 엔드포인트: 관측된 video는 cmmn, 기존 첨부는 cmm 사용 → 둘 다 시도
                    candidate_paths = [
                        f"/fss/cmm/file/fileDown.do?atchFileNo={atch_id}&fileSn={file_sn}",
                        f"/fss/cmmn/file/fileDown.do?atchFileId={atch_id}&fileSn={file_sn}&menuNo={MENU_NO}"
                    ]
                    download_url = None
                    # 첫 시도: cmmn (더 일반화 추정) → 실패시 fallback
                    for path in candidate_paths:
                        test_url = urljoin(BASE_URL, path)
                        download_url = test_url
                        break
                    base_without_ext, ext = os.path.splitext(fname)
                    final_name = f"{ntt_sn}_{base_without_ext}{ext}" if not fname.startswith(f"{ntt_sn}_") else fname
                    if DEBUG_SINGLE_MODE:
                        status, ctype, size_hint, note = probe_url(driver, download_url, post_url)
                        debug_print('probe_attach', {'url': download_url, 'status': status, 'ctype': ctype, 'size': size_hint, 'note': note, 'atch': atch_id, 'fileSn': file_sn})
                    robust_download_with_cookies(driver, download_url, DOWNLOAD_DIR, final_name, referer_url=post_url)
                except Exception as fe:
                    print(f"  파일 처리 중 오류: {fe}")
                    traceback.print_exc()
            print(f"  처리 완료 (음성파일 {'있음' if sound_found else '없음'})", flush=True)
            if not sound_found:
                print('  음성 파일 확장자(.mp3/.wav/.m4a/.ogg) 첨부 없음')

    finally:
        driver.quit()
        print('\nWebDriver 종료')
    print('\n모든 다운로드 작업 종료')

if __name__ == "__main__":
    main()