from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AssistantState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    guard_verdict: str
    guard_reason: str
    guard_rounds: int
