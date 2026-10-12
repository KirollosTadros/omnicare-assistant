from langchain_core.tools import tool

from app.tools import claims, policy


@tool(response_format="content_and_artifact")
def search_policy(query: str) -> tuple[str, list[dict]]:
    """Search the OmniCare insurance policy document for sections relevant to a coverage question.
    Returns the matching sections, each one starting with its [Section heading]."""
    chunks = policy.search_policy(query)
    content = "\n\n".join(f"[{chunk['section']}]\n{chunk['text']}" for chunk in chunks)
    sources = [{"section": chunk["section"], "source": chunk["source"]} for chunk in chunks]
    return content or "No policy sections found.", sources


@tool
def get_claim_status(claim_id: str) -> dict:
    """Look up an existing claim by its claim ID (for example CLM-8821).
    Returns the claim record with its status and amount, or found=False if there is no such claim."""
    return claims.get_claim_status(claim_id)


@tool
def submit_claim(policy_number: str, claim_type: str, amount: float, description: str) -> dict:
    """Submit a new insurance claim. Only call this when the user has given all four values.
    policy_number looks like POL-1234. claim_type is "Water Damage" or "Personal Property".
    amount is the claimed amount in dollars. description is what happened, in the user's words.
    The user is asked to confirm before anything is saved. Returns a confirmation_id."""
    return claims.submit_claim(policy_number, claim_type, amount, description)


TOOLS = [search_policy, get_claim_status, submit_claim]
