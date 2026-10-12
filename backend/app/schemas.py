from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator

from app.agent.llm import AVAILABLE_MODELS


class ChatRequest(BaseModel):
    user_id: str = Field(pattern=r"^usr_[A-Za-z0-9_-]{1,64}$")
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    model: str | None = None

    @field_validator("model")
    @classmethod
    def known_model(cls, value: str | None) -> str | None:
        if value is not None and value not in AVAILABLE_MODELS:
            raise ValueError(f"model must be one of: {', '.join(AVAILABLE_MODELS)}")
        return value


class ToolCallInfo(BaseModel):
    name: str
    args: dict
    status: Literal["executed", "awaiting_confirmation", "cancelled", "rejected"]
    result: dict | None = None


class ChatResponse(BaseModel):
    response: str
    sources: list[str]
    tool_calls: list[ToolCallInfo]
