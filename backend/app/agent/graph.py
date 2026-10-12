from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.agent.assistant import Assistant, route_after_agent
from app.agent.confirm import confirm_claim
from app.agent.guard import Guard, ask_user, refuse, route_after_guard
from app.agent.lc_tools import TOOLS
from app.agent.llm import DEFAULT_MODEL
from app.agent.state import AssistantState

# one saver for every compiled graph, so switching the model keeps the conversation
checkpointer = InMemorySaver()


@lru_cache(maxsize=None)
def build_graph(model: str = DEFAULT_MODEL):
    guard = Guard()
    assistant = Assistant(model)

    graph = StateGraph(AssistantState)
    graph.add_node("guard", guard.check)
    graph.add_node("ask_user", ask_user)
    graph.add_node("refuse", refuse)
    graph.add_node("agent", assistant.respond)
    graph.add_node("confirm_claim", confirm_claim)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", route_after_guard, ["agent", "ask_user", "refuse"])
    graph.add_edge("ask_user", "guard")
    graph.add_edge("refuse", END)
    graph.add_conditional_edges("agent", route_after_agent, ["confirm_claim", "tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
