from __future__ import annotations

import os
import re
import html
import operator
from functools import lru_cache
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()  # backend2/.env 자동 로드

from langgraph.graph import StateGraph, END

# LangChain / RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
# 파일 상단 import에 추가
from langchain.schema import Document



# ========= 환경설정 =========
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # backend2/
PDF_PATH = os.getenv("OBOE_PDF_PATH", os.path.join(ROOT_DIR, "data", "oboe.pdf"))
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(ROOT_DIR, "chroma_store"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
ENABLE_WEB_FALLBACK = os.getenv("ENABLE_WEB_FALLBACK", "1") not in ("0", "false", "False")


# ========= 상태 정의 =========
class AgentState(TypedDict, total=False):
    question: str
    chat_history: Annotated[List[Dict[str, str]], operator.add]
    # RAG
    context: str
    citations: Annotated[List[Dict[str, Any]], operator.add]
    # 생성 결과
    answer: str
    # 분기/에러
    fallback: str
    error: str

def _to_documents(docs_any) -> List[Document]:
    """
    retriever가 반환하는 결과가 Document 또는 dict 혼재해도
    안전하게 LangChain Document 리스트로 변환.
    """
    if docs_any is None:
        return []
    docs_list = list(docs_any) if not isinstance(docs_any, list) else docs_any
    norm: List[Document] = []
    for d in docs_list:
        if isinstance(d, Document):
            norm.append(d)
        elif isinstance(d, dict):
            # 직렬화된 형태 지원: {"page_content": "...", "metadata": {...}} 또는 {"_type":"document",...}
            pc = d.get("page_content") or d.get("content") or ""
            md = d.get("metadata") or {}
            norm.append(Document(page_content=str(pc), metadata=dict(md)))
        else:
            # 알 수 없는 타입은 문자열로 강제
            norm.append(Document(page_content=str(d), metadata={}))
    return norm
# ========= 유틸리티 =========
def _clean_api_key(value: Optional[str]) -> str:
    """
    OPENAI_API_KEY 정규화(느슨하게):
    - 앞뒤 공백 제거
    - 양끝 따옴표/스마트따옴표 제거
    - 보이지 않는 문자(ZWSP/BOM 등) 제거
    - 'sk-'로 시작 안 해도 일단 사용(최종 검증은 OpenAI SDK에서 수행)
    """
    if not value:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. backend2/.env에 키를 넣어주세요.")
    v = str(value).strip()

    quotes = ['"', "'", "“", "”", "‘", "’"]
    if len(v) >= 2 and v[0] in quotes and v[-1] in quotes:
        v = v[1:-1].strip()

    invisibles = {"\u200b", "\u200c", "\u200d", "\ufeff"}
    v = "".join(ch for ch in v if ch.isprintable() and ch not in invisibles)

    # 환경변수에 정제 값 반영
    os.environ["OPENAI_API_KEY"] = v
    return v


def _ensure_paths():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"분석 문서를 찾을 수 없습니다: {PDF_PATH}")
    os.makedirs(CHROMA_DIR, exist_ok=True)


@lru_cache(maxsize=1)
def _build_components() -> Dict[str, Any]:
    """
    최초 호출 시 1회만 준비(지연 초기화).
    """
    # API 키/경로 정리
    _clean_api_key(os.getenv("OPENAI_API_KEY"))
    _ensure_paths()

    # 1) 문서 로드
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    # 2) 청크 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # 3) 임베딩 + 벡터 스토어(Chroma)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="oboe-rag",
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # 4) LLM
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)

    # 5) 프롬프트: 문서 근거만 허용, ‘출처’ 섹션 필수
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 양홍원의 콘셉트 앨범 『오보에』 상세 분석 문서에만 근거해 답하는 전문가 비평가입니다. "
                "규칙: (1) 제공 문서 내용만 사용, (2) 문서에 근거가 없으면 그렇게 명시, "
                "(3) 답변 말미에 '출처' 섹션으로 핵심 근거를 한국어로 간단히 요약해 나열, "
                "(4) 과장·추측 금지, (5) 존댓말 사용."
            ),
            (
                "user",
                "질문: {question}\n\n"
                "이전 대화(있다면): {chat_history}\n\n"
                "아래는 검색으로 찾은 관련 문서 조각들입니다.\n"
                "문서 조각:\n{context}\n\n"
                "요구사항: 질문에 정확하고 간결하게 답하시고, 마지막에 '출처' 섹션을 추가하십시오."
            ),
        ]
    )

    return {"retriever": retriever, "llm": llm, "prompt": prompt}


def _format_docs(docs) -> tuple[str, List[Dict[str, Any]]]:
    """LangChain 문서 리스트 → (컨텍스트 텍스트, 인용 메타데이터 리스트)"""
    formatted_blocks: List[str] = []
    cites: List[Dict[str, Any]] = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        p = meta.get("page", "N/A")
        snippet = d.page_content.strip()
        formatted_blocks.append(f"[{i}] p.{p} — {snippet[:500]}")
        cites.append({"page": p, "snippet": snippet[:180]})
    return "\n\n".join(formatted_blocks), cites


# 간단한 DuckDuckGo HTML 검색 (무API, 실패해도 전체가 죽지 않도록)
def _web_search_duckduckgo(query: str, k: int = 3) -> List[Dict[str, str]]:
    import httpx
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://html.duckduckgo.com/html/"

    try:
        r = httpx.post(url, data=params, headers=headers, timeout=15)
        r.raise_for_status()
        html_text = r.text
    except Exception as e:
        return [{"title": "검색 실패", "link": "", "snippet": f"{e}"}]

    # 아주 단순한 파싱(의존성 최소화)
    # 결과 링크/제목
    items: List[Dict[str, str]] = []
    # 패턴 예: <a rel="nofollow" class="result__a" href="https://...">제목</a>
    for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_text, flags=re.I | re.S):
        link = html.unescape(m.group(1))
        title = re.sub("<.*?>", "", html.unescape(m.group(2))).strip()
        items.append({"title": title, "link": link, "snippet": ""})
        if len(items) >= k:
            break

    # 스니펫(간단)
    if items:
        snips = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', html_text, flags=re.I | re.S)
        for i in range(min(k, len(snips), len(items))):
            snippet = re.sub("<.*?>", "", html.unescape(snips[i])).strip()
            items[i]["snippet"] = snippet

    if not items:
        items = [{"title": "검색 결과 없음", "link": "", "snippet": "DuckDuckGo에서 결과를 찾지 못했습니다."}]
    return items


# ========= 노드들 =========
def validate_node(state: AgentState) -> AgentState:
    """환경/입력 검증: 키·경로·질문"""
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
    """RAG 검색 → context, citations 생성(없으면 빈 값)"""
    try:
        comps = _build_components()
        retriever = comps["retriever"]
        docs = retriever.invoke(state["question"])
        if not docs:
            return {"context": "", "citations": []}
        context, citations = _format_docs(docs)
        return {"context": context, "citations": citations}
    except Exception as e:
        # 검색 자체가 실패하면 웹으로 넘길 수 있도록 에러 대신 빈 컨텍스트 반환
        return {"context": "", "citations": [], "fallback": "web", "error": f"[RAG 검색 경고] {e}"}


def generate_node(state: AgentState) -> AgentState:
    """문서 기반 생성. 컨텍스트 없거나 모호하면 웹으로 분기"""
    if state.get("error") and not ENABLE_WEB_FALLBACK:
        return state

    if not state.get("context"):
        return {"answer": "", "fallback": "web"}

    try:
        comps = _build_components()
        llm = comps["llm"]
        prompt = comps["prompt"]

        chain: RunnableSerializable = (
            {
                "question": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
                "context": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        output = chain.invoke(
            {
                "question": state["question"],
                "chat_history": state.get("chat_history", []),
                "context": state["context"],
            }
        )

        # 답변이 문서 근거 부족을 명시하면 웹으로 보조
        if ("근거가 없" in output or "문서에" in output and "없" in output) and ENABLE_WEB_FALLBACK:
            return {"answer": "", "fallback": "web"}

        return {"answer": output}
    except Exception as e:
        return {"error": f"[생성 실패] {e}", "fallback": "web" if ENABLE_WEB_FALLBACK else ""}


def web_node(state: AgentState) -> AgentState:
    """문서 근거가 없을 때 간단 웹 검색 보조"""
    if not ENABLE_WEB_FALLBACK:
        return state

    try:
        q = state["question"]
        results = _web_search_duckduckgo(q, k=3)
        # 간단 요약(그대로 노출)
        lines = []
        cites = []
        for r in results:
            title = r.get("title") or ""
            link = r.get("link") or ""
            snip = r.get("snippet") or ""
            lines.append(f"- {title}\n  {snip}\n  {link}")
            if link:
                cites.append({"web": link, "title": title})

        answer = (
            "문서에서 직접적인 근거를 찾지 못해 웹에서 보조 정보를 찾았습니다.\n\n"
            + "\n\n".join(lines)
            + "\n\n※ 웹 정보는 2차 보조 자료이며, 공식 분석 문서와 다를 수 있습니다."
        )
        # 문서 근거가 없을 때의 보조 답변이므로 answer만 채워 반환
        return {"answer": answer, "citations": (state.get("citations", []) + cites) if cites else state.get("citations", [])}
    except Exception as e:
        return {"error": f"[웹 검색 실패] {e}"}


def finalize_node(state: AgentState) -> AgentState:
    """대화 이력 누적 및 최종 정리"""
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


# ========= 그래프 구성 =========
workflow = StateGraph(AgentState)
workflow.add_node("validate", validate_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("web", web_node)
workflow.add_node("finalize", finalize_node)

workflow.set_entry_point("validate")
workflow.add_edge("validate", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_conditional_edges(
    "generate",
    lambda s: "web" if (s.get("fallback") == "web") else "finalize",
    {"web": "web", "finalize": "finalize"},
)
workflow.add_edge("web", "finalize")
workflow.add_edge("finalize", END)

app = workflow.compile()


# ========= 단독 실행 테스트 =========
if __name__ == "__main__":
    q = "타이틀곡 '사계' 가사의 메타포를 설명해 주세요."
    res = app.invoke({"question": q, "chat_history": []})
    if res.get("error"):
        print("ERROR:", res["error"])
    else:
        print(res["answer"])
        if res.get("citations"):
            print("\n[참고]", res["citations"])
