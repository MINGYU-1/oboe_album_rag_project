# backend3/my_agent/agent.py

from langgraph.graph import StateGraph, END
from .utils.state import AgentState
from .utils.nodes import (
    validate_node, 
    prepare_for_retrieval_node,
    retrieve_node, 
    generate_node, 
    grade_answer_node,
    rewrite_query_node,
    finalize_node
)

def should_retry(state: AgentState) -> str:
    """답변 평가 결과와 남은 재시도 횟수에 따라 분기"""
    error = state.get("error")
    retries_left = state.get("retries_left", 0)

    if error == "graded_insufficient" and retries_left > 0:
        return "retry"  # 재시도
    else:
        return "end"    # 종료

# 그래프 워크플로우 정의
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("validate", validate_node)
workflow.add_node("prepare_for_retrieval", prepare_for_retrieval_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade_answer", grade_answer_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("finalize", finalize_node)

# 엣지(흐름) 연결
workflow.set_entry_point("validate")
workflow.add_edge("validate", "prepare_for_retrieval")
workflow.add_edge("prepare_for_retrieval", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade_answer")

# ✨ 자기 수정 루프를 위한 조건부 엣지
workflow.add_conditional_edges(
    "grade_answer",
    should_retry,
    {
        "retry": "rewrite_query", # 재시도 시, 질문 재구성으로
        "end": "finalize"         # 통과 시, 최종 정리로
    }
)

# 재구성된 질문으로 다시 검색 노드에 연결 (루프 생성)
workflow.add_edge("rewrite_query", "retrieve") 
workflow.add_edge("finalize", END)

# 그래프 컴파일
app = workflow.compile()