# my_agent/agent.py
from langgraph.graph import StateGraph, END
from my_agent.utils.state import AgentState
from my_agent.utils import nodes

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", nodes.retrieve_node)
workflow.add_node("generate", nodes.generate_node)
workflow.add_node("web", nodes.web_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_conditional_edges(
    "generate",
    lambda s: "web" if s.get("fallback") == "web" else "end",
    {"web": "web", "end": END},
)
workflow.add_edge("web", END)

app = workflow.compile()
