# my_agent/utils/nodes.py
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from my_agent.utils.tools import get_retriever, web_search_duckduckgo
from my_agent.utils.state import AgentState

def _to_documents(maybe_docs) -> List[Document]:
    if maybe_docs is None:
        return []
    seq = maybe_docs if isinstance(maybe_docs, list) else list(maybe_docs)
    out: List[Document] = []
    for d in seq:
        if isinstance(d, Document):
            out.append(d)
        elif isinstance(d, dict):
            pc = d.get("page_content") or d.get("content") or ""
            md = d.get("metadata") or {}
            out.append(Document(page_content=str(pc), metadata=dict(md)))
        else:
            out.append(Document(page_content=str(d), metadata={}))
    return out

def retrieve_node(state: AgentState) -> AgentState:
    try:
        retriever = get_retriever()
        try:
            docs_any = retriever.invoke(state["question"])
        except Exception:
            docs_any = retriever.invoke({"query": state["question"]})

        docs = _to_documents(docs_any)
        if not docs:
            return {"context": "", "citations": [], "fallback": "web"}

        context = "\n\n".join([(d.page_content or "")[:500] for d in docs])
        cites = []
        for d in docs:
            meta = d.metadata or {}
            page = meta.get("page", "N/A")
            cites.append({"page": page, "snippet": (d.page_content or "")[:160]})
        return {"context": context, "citations": cites}

    except Exception as e:
        # 여기서 실패해도 전체를 죽이지 말고 웹 폴백으로 넘김
        return {"context": "", "citations": [], "fallback": "web", "error": f"[RAG 검색 실패] {e}"}

def generate_node(state: AgentState) -> AgentState:
    if not state.get("context"):
        return {"fallback": "web"}

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 『오보에』 분석 문서에만 근거해 답하는 전문가입니다. 문서에 없는 내용은 없다고 명시하고, 답변 말미에 '출처' 섹션을 추가하세요."),
            ("user", "질문: {question}\n\n문서: {context}")
        ])
        chain = ({"question": RunnablePassthrough(), "context": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        answer = chain.invoke({"question": state["question"], "context": state["context"]})
        return {"answer": answer}
    except Exception as e:
        return {"fallback": "web", "error": f"[생성 실패] {e}"}

def web_node(state: AgentState) -> AgentState:
    try:
        results = web_search_duckduckgo(state["question"])
        lines = [f"- {r.get('title','')}\n  {r.get('link','')}" for r in results]
        answer = "문서에서 근거를 찾지 못해 웹 검색 결과를 제공합니다:\n\n" + "\n\n".join(lines)
        return {"answer": answer, "citations": [{"web": r.get("link",""), "title": r.get("title","")} for r in results]}
    except Exception as e:
        # 웹 폴백마저 실패하면 에러 반환
        return {"error": f"[웹 검색 실패] {e}"}
