import os
from typing import List, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import StrOutputParser

# --- 1. 그래프의 상태(State) 정의 ---
class GraphState(TypedDict):
    """그래프의 상태를 나타냅니다."""
    question: str
    documents: List[Document]
    generation: str

# --- 2. 그래프에 필요한 핵심 컴포넌트 준비 ---
# 실제 환경에서는 이 부분도 설정 파일 등을 통해 관리하는 것이 좋습니다.
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# PDF 파일 경로 설정 (실제 프로젝트 경로에 맞게 조정 필요)
pdf_filename = os.getenv("PDF_FILENAME", "오보에_앨범_분석.pdf")
pdf_path = os.path.join(os.path.dirname(__file__), pdf_filename)

# 벡터 스토어를 로드하거나 생성하는 로직 (애플리케이션 시작 시 한 번만 실행되도록)
# 이 부분은 실제 서비스에서는 DB 연결처럼 별도 모듈로 관리될 수 있습니다.
if not os.path.exists("faiss_index"):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ FAISS 인덱스 생성 및 저장 완료.")

embeddings = OpenAIEmbeddings()
loaded_vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = loaded_vectorstore.as_retriever(search_kwargs={'k': 5})
print("✅ FAISS 인덱스 로드 완료.")


# --- 3. 그래프의 노드(Node) 함수 정의 ---
def retrieve_documents(state: GraphState) -> GraphState:
    """문서를 검색하는 노드"""
    print("---노드: 문서 검색---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState) -> GraphState:
    """검색된 문서의 관련성을 평가하는 노드"""
    print("---노드: 문서 평가---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = PromptTemplate.from_template(
        "다음 질문에 대해 검색된 문서가 관련성이 있는지 판단하세요. 'yes' 또는 'no'로만 답하세요.\n\n질문: {question}\n문서: {document_text}"
    )
    
    relevant_docs = []
    for d in documents:
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question, "document_text": d.page_content})
        if "yes" in result.lower():
            relevant_docs.append(d)

    print(f"---평가: {len(relevant_docs)}개의 관련성 있는 문서 발견---")
    return {"documents": relevant_docs, "question": question}

def generate_answer(state: GraphState) -> GraphState:
    """답변을 생성하는 노드"""
    print("---노드: 답변 생성---")
    question = state["question"]
    documents = state["documents"]
    
    prompt_template = """
    당신은 사용자와 1:1로 대화하는 친절하고 매력적인 '음악 큐레이터'입니다. 
    당신의 이름은 '오보'이며, 양홍원의 '오보에' 앨범에 대해 모르는 것이 없는 전문가입니다. 
    딱딱한 정보 전달이 아닌, 상대방의 눈높이에 맞춰 설명하고 감상을 공유하는 역할을 합니다.
    
    --- 분석 문서 내용 ---
    {context}
    --------------------
    
    질문: {question}
    음악 큐레이터 '오보'의 답변:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    rag_chain = PROMPT | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}

def fallback_action(state: GraphState) -> GraphState:
    """관련 문서가 없을 때의 대체 행동 노드"""
    print("---노드: 대체 행동---")
    generation = "제가 가진 분석 자료에서는 질문에 대한 정보를 찾기 어렵네요. 혹시 다른 질문이 있으신가요?"
    return {"generation": generation}

def decide_to_generate(state: GraphState) -> str:
    """평가 결과에 따라 분기하는 엣지 로직"""
    print("---엣지: 생성 여부 결정---")
    if not state["documents"]:
        return "fallback"
    else:
        return "generate"

# --- 4. 그래프 정의 및 컴파일 ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate_answer)
workflow.add_node("fallback", fallback_action)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "fallback": "fallback"
    }
)
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

# langgraph dev가 실행할 수 있도록 컴파일된 그래프를 'app' 변수에 할당
app = workflow.compile()
print("✅ LangGraph 컴파일 완료. 'app'을 실행할 수 있습니다.")