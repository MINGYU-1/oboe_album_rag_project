# my_agent/utils/tools.py
from __future__ import annotations

from typing import List, Dict
from bs4 import BeautifulSoup
import httpx
from duckduckgo_search import DDGS

def web_search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    DuckDuckGo로 간단한 웹 검색을 수행하고, title/url/snippet을 반환.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            # ddgs 결과 키: title, href(url), body(snippet) 등
            results.append({
                "title": r.get("title") or "",
                "url": r.get("href") or "",
                "snippet": r.get("body") or "",
            })
    return results

def fetch_text(url: str, timeout: float = 15.0) -> str:
    """
    HTML 본문에서 텍스트만 추출하여 반환.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; my-agent/0.1; +https://example.local)"
    }
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    # 불필요한 스크립트/스타일 제거
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # 여분 공백 정리
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln])
