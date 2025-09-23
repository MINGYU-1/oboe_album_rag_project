# backend3/my_agent/utils/state.py
from __future__ import annotations
import operator
from typing import TypedDict, Annotated, List, Dict, Any

class AgentState(TypedDict, total=False):
    # --- 기존 필드 ---
    question: str
    chat_history: Annotated[List[Dict[str, str]], operator.add]
    allow_web: bool
    context: str
    citations: Annotated[List[Dict[str, Any]], operator.add]
    answer: str
    error: str
    
    # --- ✨ 추가된 필드 ---
    # 자기 수정 루프를 위한 필드
    search_query: str  # 검색에 실제 사용할 재구성된 질문
    retries_left: int  # 남은 재시도 횟수