import os
import re
import time
import json
import sqlite3
import urllib.parse
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Any, Literal
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

# 對接成員 A 的東華 OPAC 爬蟲模組
from opac_crawler import search_ndhu_opac, init_storage, DB_PATH

load_dotenv()

MAX_TOOL_TURNS = 3
# 使用官方穩定模型，避免 404 找不到模型的錯誤
MODEL = "gemini-3.5-flash-lite"

# ==========================================
# 外部工具區 (保留您原本寫好的摘要抓取與快取邏輯)
# ==========================================
def init_summary_db():
    init_storage()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS summary_cache (
            book_key TEXT PRIMARY KEY,
            summary TEXT,
            source_type TEXT,
            updated_at REAL
        )
    """)
    conn.commit()
    conn.close()

def _save_summary_cache(book_key: str, summary: str, source_type: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO summary_cache (book_key, summary, source_type, updated_at) VALUES (?, ?, ?, ?)",
        (book_key, summary, source_type, time.time())
    )
    conn.commit()
    conn.close()

def _search_books_com_tw(query_str: str) -> Optional[str]:
    try:
        encoded_q = urllib.parse.quote(query_str)
        url = f"https://search.books.com.tw/search/query/key/{encoded_q}/cat/all"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            item = soup.select_one(".table-td, .box_1, .item")
            if item:
                desc_el = item.select_one("p, .txt_box")
                if desc_el:
                    clean_desc = re.sub(r"\s+", " ", desc_el.get_text(" ", strip=True))
                    if len(clean_desc) > 25: return clean_desc[:350]
    except:
        pass
    return None

def fetch_book_summary(isbn: str = "", title: str = "") -> Dict[str, str]:
    init_summary_db()
    book_key = isbn.strip() if isbn.strip() else title.strip()
    if not book_key:
        return {"summary": "無足夠書目識別資訊", "source_type": "inferred_from_subject"}

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT summary, source_type, updated_at FROM summary_cache WHERE book_key = ?", (book_key,))
    row = c.fetchone()
    conn.close()
    if row and (time.time() - row[2] < 604800):
        return {"summary": row[0], "source_type": row[1]}

    query = f"isbn:{isbn.strip()}" if isbn.strip() else f"intitle:{title.strip()}"
    try:
        resp = requests.get("https://www.googleapis.com/books/v1/volumes", params={"q": query, "maxResults": 1}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("totalItems", 0) > 0:
                vol_info = data["items"][0].get("volumeInfo", {})
                desc = vol_info.get("description", "")
                if desc:
                    summary_text = desc[:400]
                    _save_summary_cache(book_key, summary_text, "google_books_api")
                    return {"summary": summary_text, "source_type": "google_books_api"}
    except:
        pass

    time.sleep(0.4)
    books_tw_desc = _search_books_com_tw(book_key)
    if books_tw_desc:
        _save_summary_cache(book_key, books_tw_desc, "books_com_tw")
        return {"summary": books_tw_desc, "source_type": "books_com_tw"}

    fallback_msg = "此書在外部公開資料庫查無詳細目錄與簡介，請依書名推估主題。"
    _save_summary_cache(book_key, fallback_msg, "inferred_from_subject")
    return {"summary": fallback_msg, "source_type": "inferred_from_subject"}

# ==========================================
# 【教材重點】定義 Action 容器 (取代容易報錯的 Gemini 原生 Tools)
# ==========================================
class AgentAction(BaseModel):
    tool: Literal['search_opac', 'fetch_summary', 'finish'] = Field(description="選擇要執行的工具")
    keyword: str = Field(default="", description="當 tool 為 search_opac 時，填寫 OPAC 關鍵字")
    book_title: str = Field(default="", description="當 tool 為 fetch_summary 時，填寫要查簡介的書名")
    answer: str = Field(default="", description="當 tool 為 finish 時，輸出的最終結構化拿書路線報告")

SYSTEM_INSTRUCTION = """
你是一位「東華大學圖書館 AI 館員」。
1. 請根據任務與 observations 決定下一步 tool。
2. 找到書後，請使用 fetch_summary 抓簡介。
3. 資料齊全後選 finish，列出書單並依據 floor_num 由高樓層到低樓層規劃拿書路線。
"""

class LibraryAgent:
    def __init__(self, use_mock_opac: bool = False):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.use_mock_opac = use_mock_opac

    #@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
    def _ask(self, contents: list, schema: BaseModel = None):
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1
        )
        # 強制使用 JSON Schema 確保模型輸出穩定
        if schema:
            config.response_mime_type = "application/json"
            config.response_schema = schema
            
        resp = self.client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config
        )
        return schema.model_validate_json(resp.text) if schema else resp.text

    def run(self, user_query: str, image: Optional[Image.Image] = None, pdf_bytes: Optional[bytes] = None, status_callback=None) -> Dict[str, Any]:
        def log_step(msg: str):
            print(f"[AGENT STEP] {msg}", flush=True)
            if status_callback: status_callback(msg)

        observations = []
        turn_count = 0

        # 組合多模態輸入 (圖片、PDF、純文字)
        base_parts = [types.Part.from_text(text=user_query)]
        if image:
            base_parts.append(image)
            log_step("📸 偵測到圖片...")
        if pdf_bytes:
            base_parts.append(types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))
            log_step("📄 偵測到 PDF...")

        while turn_count < MAX_TOOL_TURNS:
            turn_count += 1
            log_step(f"🧠 第 {turn_count}/{MAX_TOOL_TURNS} 輪思考...")

            prompt_text = json.dumps({
                "task": "請分析任務與觀察紀錄。先用 search_opac 查書，接著用 fetch_summary 抓簡介。資料夠了就選 finish 產出最終路線。",
                "observations": observations
            }, ensure_ascii=False)
            
            current_contents = base_parts + [types.Part.from_text(text=prompt_text)]
            
            try:
                action: AgentAction = self._ask(current_contents, schema=AgentAction)
            except Exception as e:
                log_step(f"⚠️ 解析錯誤：{e}，強制安全煞車。")
                break

            # 程式負責執行分支邏輯 (完全符合教材 ReAct 規範)
            if action.tool == 'finish':
                log_step("✅ 資訊整合完畢！")
                return {"answer": action.answer, "trace_logs": observations, "turns_used": turn_count}
                
            elif action.tool == 'search_opac':
                log_step(f"🔧 調用 OPAC 搜尋: {action.keyword}")
                books = search_ndhu_opac(keyword=action.keyword, use_mock=self.use_mock_opac)
                observations.append({'tool': 'search_opac', 'keyword': action.keyword, 'result': books[:3] if books else "查無此書"})
                
            elif action.tool == 'fetch_summary':
                log_step(f"🔧 抓取摘要: {action.book_title}")
                summary = fetch_book_summary(title=action.book_title)
                observations.append({'tool': 'fetch_summary', 'title': action.book_title, 'result': summary})
            
            time.sleep(0.8)

        # 達到迴圈上限時的安全煞車
        log_step("⚠️ 已達工具調用上限，強制生成最終結果...")
        final_prompt = json.dumps({
            "task": "已達次數上限，請立刻使用 finish 工具，根據現有 observations 產出最終路線。",
            "observations": observations
        }, ensure_ascii=False)
        
        final_contents = base_parts + [types.Part.from_text(text=final_prompt)]
        final_action: AgentAction = self._ask(final_contents, schema=AgentAction)
        
        return {
            "answer": final_action.answer,
            "trace_logs": observations,
            "turns_used": turn_count
        }