from typing import TypedDict, Annotated, List, Dict
import operator

class AgentState(TypedDict, total=False):
    question: str
    chat_history: Annotated[List[Dict[str, str]], operator.add]
    context: str
    citations: Annotated[List[Dict[str, str]], operator.add]
    answer: str
    fallback: str
    error: str
