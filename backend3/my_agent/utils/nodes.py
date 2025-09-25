# my_agent/utils/nodes.py
from __future__ import annotations

from typing import Dict, Any, List

from my_agent.utils.tools import web_search_duckduckgo, fetch_text
from my_agent.utils.state import AgentState

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def _build_web_context(pages: List[Dict[str, str]], per_page_limit: int = 1800) -> str:
    """LLM 컨텍스트 문자열 생성(페이지당 내용 길이 제한)"""
    blocks: List[str] = []
    for i, p in enumerate(pages, 1):
        text = (p.get("text") or "")[:per_page_limit]
        blocks.append(f"[{i}] {p.get('title','')} — {p.get('url','')}\n{text}")
    return "\n\n".join(blocks)


def web_node(state: AgentState) -> AgentState:
    """
    웹 검색/스크래핑 → 컨텍스트 조립 → LLM 요약 답변 생성.
    입력: state['question'] 또는 state['search_query']
    출력: state에 answer / citations / error 병합
    """
    new_state: AgentState = dict(state)  # 원본 보존

    # 1) 질의 확보
    q = (state.get("search_query") or state.get("question") or "").strip()
    if not q:
        new_state["error"] = "질문이 비어 있습니다. 'question' 또는 'search_query' 필드를 채워주세요."
        return new_state

    try:
        # 2) DDG 검색 (tools.py 시그니처에 맞춤: max_results, 반환키 title/url/snippet)
        results = web_search_duckduckgo(q, max_results=5)

        # 3) 본문 수집(최대 3페이지)
        pages: List[Dict[str, str]] = []
        for r in results:
            url = r.get("url") or ""
            if not url:
                continue
            try:
                body = fetch_text(url)
            except Exception:
                continue
            if len(body) < 400:  # 본문이 너무 짧으면 제외
                continue
            pages.append({"title": r.get("title", ""), "url": url, "text": body})
            if len(pages) >= 3:
                break

        # 4) 본문이 없으면 링크만 안내
        if not pages:
            new_state["answer"] = (
                "문서에서 직접 근거를 찾지 못해 관련 웹 자료를 확인했습니다.\n\n" +
                "\n".join(f"- {r.get('title','')} — {r.get('url','')}" for r in results if r.get("url"))
            )
            new_state["citations"] = new_state.get("citations", []) + [
                {"web": r.get("url", ""), "title": r.get("title", "")}
                for r in results if r.get("url")
            ]
            return new_state

        # 5) LLM 컨텍스트 조립
        web_context = _build_web_context(pages)

        # 6) LLM 호출
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 한국어 전문 요약가입니다. 아래 웹 자료만 근거로 간결하고 정확한 답을 만듭니다. "
             "추측을 금지하고, 모호하면 모른다고 말합니다. 반드시 존댓말을 사용합니다."),
            ("user",
             "질문: {question}\n\n"
             "웹 자료(인용가능):\n{web_context}\n\n"
             "요구사항:\n"
             "1) 핵심 답변을 5문장 이내로 한국어로 제시\n"
             "2) 마지막에 '출처' 섹션을 번호와 URL로 나열")
        ])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": q, "web_context": web_context})

        # 7) 상태 병합
        new_state["answer"] = answer
        new_state["citations"] = new_state.get("citations", []) + [
            {"web": p["url"], "title": p["title"]} for p in pages
        ]
        return new_state

    except Exception as e:
        new_state["error"] = f"[웹 검색 실패] {e}"
        return new_state
