# backend3/my_agent/utils/tools.py
from __future__ import annotations

import os
import re
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

# ── .env 로드: backend3/.env → my_agent/.env → CWD/.env 순서
def _load_env() -> Path:
    root = Path(__file__).resolve().parents[2]  # backend3/
    for p in (root / ".env", Path(__file__).resolve().parents[1] / ".env", Path.cwd() / ".env"):
        if p.exists():
            load_dotenv(p, override=False)
    return root

ROOT_DIR = _load_env()

# ── 환경값
PDF_PATH      = os.getenv("OBOE_PDF_PATH", str(ROOT_DIR / "data" / "oboe.pdf"))
CHROMA_DIR    = os.getenv("CHROMA_DIR",    str(ROOT_DIR / "chroma_store"))
OPENAI_MODEL  = os.getenv("OPENAI_MODEL",  "gpt-4o-mini")
EMBED_MODEL   = os.getenv("EMBED_MODEL",   "text-embedding-3-small")
ALLOW_WEB_ENV = os.getenv("ALLOW_WEB", "false").strip().lower() in ("1","true","yes","on")

# ── LangChain/RAG 구성요소
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


def normalize_openai_key(value: Optional[str]) -> str:
    """
    OPENAI_API_KEY를 실용적으로 정리/검증한다.
    - 앞뒤 공백/BOM 제거
    - 따옴표/라인 끝 주석 제거
    - 비인쇄 제어문자 제거(ASCII 0x20~0x7E)
    - ASCII 여부 확인
    - 'sk-' 시작 && 길이(>=20)만 확인(언더스코어/하이픈 등 ASCII 허용)
    - 정제 값을 os.environ에 재주입
    """
    if not value:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. backend3/.env에 실제 키를 넣으십시오.")

    v = value.strip().lstrip("\ufeff")  # 공백 + BOM 제거

    # 라인 끝 주석 제거(따옴표로 감싼 경우가 아니라면)
    if "#" in v and not (v.startswith('"') or v.startswith("'")):
        v = v.split("#", 1)[0].strip()

    # 양끝 따옴표 제거
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()

    # 비인쇄 제어문자 제거(ASCII 0x20~0x7E 범위만 유지)
    v = "".join(ch for ch in v if " " <= ch <= "~")

    # ASCII 확인
    try:
        v.encode("ascii")
    except UnicodeEncodeError:
        raise RuntimeError("OPENAI_API_KEY에 비ASCII 문자가 섞였습니다. 따옴표/한글/특수 따옴표(…, “ ”) 제거 후 저장하십시오.")

    # 아주 느슨한 형식 확인
    if not v.startswith("sk-") or len(v) < 20:
        raise RuntimeError("OPENAI_API_KEY 형식이 올바르지 않습니다. 'sk-'로 시작하는 실제 키를 입력하십시오.")

    os.environ["OPENAI_API_KEY"] = v  # 정제 값 재주입
    return v


def ensure_paths() -> None:
    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(f"[RAG] 분석 문서를 찾을 수 없습니다: {PDF_PATH}")
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def build_components() -> Dict[str, Any]:
    """PDF 로드/분할 → 임베딩/Chroma → Retriever, LLM, Prompt, DDG (최초 1회 준비)"""
    # 1) 문서 로드/분할
    docs = PyPDFLoader(PDF_PATH).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(docs)

    # 2) 벡터스토어
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR, collection_name="oboe-rag")
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # 3) LLM/프롬프트
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "당신은 양홍원의 콘셉트 앨범 『오보에』 분석 문서와 (허용된 경우) 공개 웹 자료에만 근거해 답하는 비평가입니다. "
             "규칙: ① 문서/웹 근거만 사용 ② 근거 없으면 명시 ③ 답변 말미에 '출처' 섹션(페이지/URL) ④ 과장·추측 금지 ⑤ 존댓말."),
            ("user",
             "질문: {question}\n\n"
             "이전 대화(있다면): {chat_history}\n\n"
             "문서 조각:\n{context}\n\n"
             "요구사항: 질문에 정확·간결하게 답하고 마지막에 '출처' 섹션을 추가하십시오.")
        ]
    )

    # 4) 웹검색
    ddg = DuckDuckGoSearchAPIWrapper()

    return {"retriever": retriever, "llm": llm, "prompt": prompt, "splitter": splitter, "ddg": ddg}


def format_docs(docs) -> Tuple[str, List[Dict[str, Any]]]:
    blocks, cites = [], []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "")
        page = meta.get("page", None)
        snippet = (d.page_content or "").strip()
        label = f"p.{page}" if page not in (None, "N/A") else (src if src else "N/A")
        blocks.append(f"[{i}] {label} — {snippet[:500]}")
        cites.append({"page": page if page not in (None, "N/A") else "", "source": src, "snippet": snippet[:180]})
    return "\n\n".join(blocks), cites


def web_search_and_load(query: str, topn: int = 3, per_site_limit: int = 2):
    comps = build_components()
    results = comps["ddg"].results(query, max_results=topn) or []
    urls = [r.get("link") for r in results if r.get("link")]
    if not urls:
        return []
    try:
        raw = WebBaseLoader(urls).load()
    except Exception:
        return []
    split_docs = comps["splitter"].split_documents(raw)
    return split_docs[: topn * per_site_limit]


def want_web(state_allow_web: Optional[bool]) -> bool:
    return state_allow_web if state_allow_web is not None else ALLOW_WEB_ENV
