# my_agent/utils/tools.py
import os, re, html, httpx
from functools import lru_cache
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
PDF_PATH = os.getenv("OBOE_PDF_PATH", os.path.join(ROOT_DIR, "data", "oboe.pdf"))
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(ROOT_DIR, "chroma_store"))

def _ensure_paths():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"[RAG] 분석 문서를 찾을 수 없습니다: {PDF_PATH}")
    os.makedirs(CHROMA_DIR, exist_ok=True)

@lru_cache(maxsize=1)
def load_pdf_and_split():
    _ensure_paths()
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    return tuple(splitter.split_documents(docs))  # 캐시 가능하도록 tuple 반환

@lru_cache(maxsize=1)
def build_vectordb():
    chunks = list(load_pdf_and_split())
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="oboe-rag",
    )

@lru_cache(maxsize=1)
def get_retriever():
    vectordb = build_vectordb()
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def web_search_duckduckgo(query: str, k: int = 3):
    url = "https://html.duckduckgo.com/html/"
    try:
        r = httpx.post(url, data={"q": query}, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        html_text = r.text
    except Exception as e:
        return [{"title": "검색 실패", "snippet": str(e), "link": ""}]

    items = []
    for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_text, re.I|re.S):
        link = html.unescape(m.group(1))
        title = re.sub("<.*?>", "", html.unescape(m.group(2))).strip()
        items.append({"title": title, "link": link})
        if len(items) >= k: break
    return items
