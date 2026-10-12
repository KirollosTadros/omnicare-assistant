import json
import logging
from typing import Literal

from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from app.agent.state import AssistantState
from app.tools.claims import ClaimSubmission, format_errors

log = logging.getLogger(__name__)

YES_ANSWERS = ["yes", "y", "confirm", "ok", "okay", "sure", "submit"]


def skip_tools(tool_calls: list[dict], status: str, reason: str) -> Command:
    messages = []
    for call in tool_calls:
        is_claim = call["name"] == "submit_claim"
        messages.append(
            ToolMessage(
                reason if is_claim else "Not run because the claim was not submitted. Call it again if needed.",
                tool_call_id=call["id"],
                name=call["name"],
                additional_kwargs={"call_status": status if is_claim else "cancelled"},
            )
        )
    return Command(goto="agent", update={"messages": messages})


def confirm_claim(state: AssistantState) -> Command[Literal["tools", "agent"]]:
    # this node runs again from the top on resume. Everything before the interrupt is
    tool_calls = state["messages"][-1].tool_calls
    submissions = [call for call in tool_calls if call["name"] == "submit_claim"]

    if len(submissions) > 1:
        log.info("[confirm_claim] rejected: more than one claim in one step")
        return skip_tools(tool_calls, "rejected", "Only one claim can be submitted at a time.")

    try:
        claim = ClaimSubmission(**submissions[0]["args"])
    except ValidationError as error:
        log.info("[confirm_claim] rejected: invalid fields")
        return skip_tools(tool_calls, "rejected", json.dumps(format_errors(error)))

    answer = interrupt(
        {
            "type": "confirm_claim",
            "message": "Please check the claim before I submit it:\n\n"
            f"- Policy number: {claim.policy_number}\n"
            f"- Claim type: {claim.claim_type}\n"
            f"- Amount: ${claim.amount:,.2f}\n"
            f"- Description: {claim.description}\n\n"
            "Shall I submit it?",
            "claim": claim.model_dump(),
        }
    )

    if str(answer).strip().strip(".!").lower() in YES_ANSWERS:
        log.info("[confirm_claim] confirmed")
        return Command(goto="tools")

    log.info("[confirm_claim] cancelled")
    return skip_tools(tool_calls, "cancelled", "User cancelled the submission")
