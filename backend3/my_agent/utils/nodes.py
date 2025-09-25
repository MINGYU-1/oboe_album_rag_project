# my_agent/utils/nodes.py
from __future__ import annotations

import os
import operator
from functools import lru_cache
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
load_dotenv()

from my_agent.utils.tools import web_search_duckduckgo, fetch_text
from my_agent.utils.state import AgentState

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser


# ========= 환경설정 =========
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # backend3/
PDF_PATH = os.getenv("OBOE_PDF_PATH", os.path.join(ROOT_DIR, "data", "oboe.pdf"))
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(ROOT_DIR, "chroma_store"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
ENABLE_WEB_FALLBACK = os.getenv("ENABLE_WEB_FALLBACK", "1") not in ("0", "false", "False")


# ========= 유틸리티 =========
def _clean_api_key(value: Optional[str]) -> str:
    if not value:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣어주세요.")
    v = str(value).strip()
    quotes = ['"', "'", "“", "”", "‘", "’"]
    if len(v) >= 2 and v[0] in quotes and v[-1] in quotes:
        v = v[1:-1].strip()
    invisibles = {"\u200b", "\u200c", "\u200d", "\ufeff"}
    v = "".join(ch for ch in v if ch.isprintable() and ch not in invisibles)
    os.environ["OPENAI_API_KEY"] = v
    return v

def _ensure_paths():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"분석 문서를 찾을 수 없습니다: {PDF_PATH}")
    os.makedirs(CHROMA_DIR, exist_ok=True)

@lru_cache(maxsize=1)
def _build_components() -> Dict[str, Any]:
    _clean_api_key(os.getenv("OPENAI_API_KEY"))
    _ensure_paths()
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR, collection_name="oboe-rag")
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 양홍원의 콘셉트 앨범 『오보에』 상세 분석 문서에만 근거해 답하는 전문가 비평가입니다. 규칙: (1) 제공 문서 내용만 사용, (2) 문서에 근거가 없으면 그렇게 명시, (3) 답변 말미에 '출처' 섹션으로 핵심 근거를 한국어로 간단히 요약해 나열, (4) 과장·추측 금지, (5) 존댓말 사용."),
        ("user", "질문: {question}\n\n이전 대화(있다면): {chat_history}\n\n아래는 검색으로 찾은 관련 문서 조각들입니다.\n문서 조각:\n{context}\n\n요구사항: 질문에 정확하고 간결하게 답하시고, 마지막에 '출처' 섹션을 추가하십시오."),
    ])
    return {"retriever": retriever, "llm": llm, "prompt": prompt}

def _format_docs(docs) -> tuple[str, List[Dict[str, Any]]]:
    formatted_blocks: List[str] = []
    cites: List[Dict[str, Any]] = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        p = meta.get("page", "N/A")
        snippet = d.page_content.strip()
        formatted_blocks.append(f"[{i}] p.{p} — {snippet[:500]}")
        cites.append({"page": p, "snippet": snippet[:180]})
    return "\n\n".join(formatted_blocks), cites


# ========= 노드들 =========
def validate_node(state: AgentState) -> AgentState:
    try:
        _clean_api_key(os.getenv("OPENAI_API_KEY"))
        _ensure_paths()
    except Exception as e:
        return {"error": f"[환경 점검 실패] {e}"}
    q = (state.get("question") or "").strip()
    if not q:
        return {"error": "질문이 비어 있습니다. 질문을 입력해 주십시오."}
    return {"question": q, "chat_history": state.get("chat_history", [])}

def retrieve_node(state: AgentState) -> AgentState:
    try:
        comps = _build_components()
        retriever = comps["retriever"]
        docs = retriever.invoke(state["question"])
        if not docs:
            return {"context": "", "citations": []}
        context, citations = _format_docs(docs)
        return {"context": context, "citations": citations}
    except Exception as e:
        return {"context": "", "citations": [], "fallback": "web", "error": f"[RAG 검색 경고] {e}"}

def _build_web_context(pages: List[Dict[str, str]], per_page_limit: int = 1800) -> str:
    """LLM 컨텍스트 문자열 생성(페이지당 내용 길이 제한)"""
    blocks: List[str] = []
    for i, p in enumerate(pages, 1):
        text = (p.get("text") or "")[:per_page_limit]
        blocks.append(f"[{i}] {p.get('title','')} — {p.get('url','')}\n{text}")
    return "\n\n".join(blocks)

def generate_node(state: AgentState) -> AgentState:
    if state.get("error") and not ENABLE_WEB_FALLBACK:
        return state
    if not state.get("context"):
        return {"answer": "", "fallback": "web"}
    try:
        comps = _build_components()
        llm = comps["llm"]
        prompt = comps["prompt"]
        chain: RunnableSerializable = (
            {"question": RunnablePassthrough(), "chat_history": RunnablePassthrough(), "context": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        output = chain.invoke({
            "question": state["question"],
            "chat_history": state.get("chat_history", []),
            "context": state["context"],
        })
        if ("근거가 없" in output or "문서에" in output and "없" in output) and ENABLE_WEB_FALLBACK:
            return {"answer": "", "fallback": "web"}
        return {"answer": output}
    except Exception as e:
        return {"error": f"[생성 실패] {e}", "fallback": "web" if ENABLE_WEB_FALLBACK else ""}

def web_node(state: AgentState) -> AgentState:
    """
    웹 검색/스크래핑 → 컨텍스트 조립 → LLM 요약 답변 생성.
    입력: state['question'] 또는 state['search_query']
    출력: state에 answer / citations / error 병합
    """
    if not ENABLE_WEB_FALLBACK:
        return state

    new_state: AgentState = dict(state)  # 원본 보존

    # 1) 질의 확보
    q = (state.get("search_query") or state.get("question") or "").strip()
    if not q:
        new_state["error"] = "질문이 비어 있습니다. 'question' 또는 'search_query' 필드를 채워주세요."
        return new_state

    try:
        # 2) DDG 검색
        results = web_search_duckduckgo(q, max_results=3)

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
                "문서에서 직접적인 근거를 찾지 못해 웹에서 보조 정보를 찾았습니다.\n\n" +
                "\n".join(f"- {r.get('title','')} — {r.get('url','')}" for r in results if r.get("url"))
                + "\n\n※ 웹 정보는 2차 보조 자료이며, 공식 분석 문서와 다를 수 있습니다."
            )
            web_cites = [{"web": r.get("url", ""), "title": r.get("title", "")} for r in results if r.get("url")]
            new_state["citations"] = [
                new_state.get("citations", []) + web_cites
                if web_cites else new_state.get("citations", [])
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

def finalize_node(state: AgentState) -> AgentState:
    history = state.get("chat_history", [])
    q = state.get("question", "")
    a = state.get("answer", "")
    err = state.get("error", "")

    new_history = history
    if q:
        new_history = new_history + [{"role": "user", "content": q}]
    if a:
        new_history = new_history + [{"role": "assistant", "content": a}]

    return {
        "question": q,
        "answer": a if a else (err or "알 수 없는 오류가 발생했습니다."),
        "chat_history": new_history,
        "context": state.get("context", ""),
        "citations": state.get("citations", []),
        "error": err,
    }
