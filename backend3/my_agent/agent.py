# my_agent/agent.py
from langgraph.graph import StateGraph, END
from my_agent.utils.state import AgentState
from my_agent.utils.nodes import (
    validate_node, retrieve_node, generate_node, web_node, finalize_node
)
from my_agent.utils.nodes import (
    validate_node, retrieve_node, generate_node, web_node, finalize_node
)


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
    lambda s: "web" if s.get("fallback") == "web" else "finalize",
    {"web": "web", "finalize": "finalize"},
)
workflow.add_edge("web", "finalize")
workflow.add_edge("finalize", END)

app = workflow.compile()
