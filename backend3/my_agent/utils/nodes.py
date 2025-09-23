# backend3/my_agent/utils/nodes.py

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate  # [개선] 명시적인 import 추가

from .state import AgentState
from .tools import (
    normalize_openai_key,
    ensure_paths,
    build_components,
    format_docs,
    web_search_and_load,
    want_web,
)

# --- 기존 노드 함수 (validate_node, retrieve_node, generate_node, finalize_node) ---
# ... (이 부분은 수정 없이 그대로 둡니다)


# ## --- ✨ 자기 수정(Self-Correction)을 위한 노드들 --- ##

def prepare_for_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """검색 실행 전, 재시도 횟수 초기화 및 검색어 설정"""
    retries = state.get("retries_left", 2)
    search_query = state.get("search_query", state["question"])
    
    # [개선] 변경된 값만 명확하게 반환하여 그래프의 데이터 흐름을 예측하기 쉽게 만듭니다.
    return {
        "retries_left": retries,
        "search_query": search_query
    }

def grade_answer_node(state: AgentState) -> Dict[str, Any]:
    """생성된 답변을 평가(Grade)하여 'sufficient' 또는 'insufficient'로 판단"""
    print("--- 🧐 답변 평가 중... ---")
    try:
        comps = build_components()
        
        grading_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 답변의 품질을 평가하는 엄격한 평가자입니다. 주어진 '문서 내용'만을 근거로 '답변'이 '질문'에 충분하고 정확하게 답했는지 평가해야 합니다. 답변은 'sufficient' 또는 'insufficient' 둘 중 하나로만 해야 합니다."),
            ("user", "질문: {question}\n\n문서 내용:\n{context}\n\n답변:\n{answer}")
        ])
        
        grader_chain = grading_prompt | comps["llm"]
        
        result = grader_chain.invoke({
            "question": state["question"],
            "context": state["context"],
            "answer": state["answer"]
        })
        
        grade = result.content.strip().lower()
        print(f"--- 평가 결과: {grade} ---")
        
        if "insufficient" in grade:
            print("--- 🚨 답변이 불충분하여 질문 재구성을 시도합니다. ---")
            # [개선] 변경된 값만 명확하게 반환합니다.
            return {
                "retries_left": state["retries_left"] - 1,
                "error": "graded_insufficient"
            }
        else:
            # 평가 통과 시, 재시도 오류 상태를 명시적으로 제거
            return {"error": None}
            
    except Exception as e:
        return {"error": f"답변 평가 중 오류: {e}"}

def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
    """더 나은 검색 결과를 위해 질문을 재구성"""
    print("--- ✍️ 질문 재구성 중... ---")
    try:
        comps = build_components()
        
        rewriting_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 사용자의 질문을 벡터 검색에 더 적합하도록 명확한 검색어로 재구성하는 전문가입니다. 원본 질문의 핵심 의도는 유지하되, 문서에서 찾기 쉬운 키워드 중심으로 변환해주세요."),
            ("user", "이전 대화: {chat_history}\n\n원본 질문: {question}\n\n재구성된 검색어:")
        ])
        
        rewriter_chain = rewriting_prompt | comps["llm"]
        
        result = rewriter_chain.invoke({
            "chat_history": state.get("chat_history", []),
            "question": state["question"]
        })
        
        new_query = result.content.strip()
        print(f"--- 재구성된 질문: {new_query} ---")
        
        # [개선] 변경된 값과 초기화할 값을 명확하게 반환합니다.
        return {
            "search_query": new_query,
            "error": None  # 재시도 상태를 초기화
        }
        
    except Exception as e:
        return {"error": f"질문 재구성 중 오류: {e}"}