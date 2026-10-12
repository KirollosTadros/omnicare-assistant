import os
import uuid

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

@st.cache_data(ttl=300)
def fetch_models() -> dict:
    response = requests.get(f"{BACKEND_URL}/api/v1/models", timeout=15)
    response.raise_for_status()
    return response.json()


def available_models() -> tuple[list[str], str]:
    # a failed request raises inside fetch_models, so it is not cached
    try:
        data = fetch_models()
        return data["models"], data["default"]
    except (requests.RequestException, ValueError, KeyError):
        return [], ""


def backend_online() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/api/v1/health", timeout=3).ok
    except requests.RequestException:
        return False


def error_detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):
        detail = "; ".join(item["msg"] for item in detail)
    return detail or f"The backend returned an error (status {response.status_code})."


def show_message(message: dict, autoplay: bool = False):
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            return

        # an unescaped pair of $ signs is rendered as LaTeX
        st.markdown(message["content"].replace("$", "\\$"))
        if message.get("audio"):
            st.audio(message["audio"], format="audio/mpeg", autoplay=autoplay)
        if message.get("sources"):
            with st.expander("Sources"):
                st.markdown("\n".join(f"- {source}" for source in message["sources"]))
        if message.get("tool_calls"):
            with st.expander("Tool calls"):
                for call in message["tool_calls"]:
                    st.markdown(f"**{call['name']}** ({call['status'].replace('_', ' ')})")
                    st.json({"args": call["args"], "result": call["result"]})


def ask_backend(text: str) -> dict:
    payload = {"user_id": st.session_state.user_id, "message": text, "model": st.session_state.model}
    try:
        response = requests.post(f"{BACKEND_URL}/api/v1/chat", json=payload, timeout=120)
    except requests.RequestException:
        return {"role": "assistant", "content": "The backend is not reachable. Is it running?", "error": True}

    if not response.ok:
        return {"role": "assistant", "content": error_detail(response), "error": True}

    data = response.json()
    return {
        "role": "assistant",
        "content": data["response"],
        "sources": data["sources"],
        "tool_calls": data["tool_calls"],
    }


def transcribe(recording) -> str:
    files = {"file": (recording.name, recording.getvalue(), recording.type)}
    try:
        response = requests.post(f"{BACKEND_URL}/api/v1/voice/transcribe", files=files, timeout=120)
    except requests.RequestException:
        st.error("The backend is not reachable. Is it running?")
        return ""

    if not response.ok:
        st.error(error_detail(response))
        return ""
    return response.json()["text"]


def speak(text: str) -> bytes | None:
    try:
        response = requests.post(f"{BACKEND_URL}/api/v1/voice/speak", json={"text": text}, timeout=60)
    except requests.RequestException:
        return None
    return response.content if response.ok else None


def send(text: str):
    user_message = {"role": "user", "content": text}
    st.session_state.messages.append(user_message)
    show_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            reply = ask_backend(text)
            if st.session_state.mode == "Voice" and not reply.get("error"):
                reply["audio"] = speak(reply["content"][:5000])

    st.session_state.messages.append(reply)
    st.session_state.autoplay = bool(reply.get("audio"))
    st.rerun()


st.set_page_config(page_title="OmniCare Assistant")
st.title("OmniCare Assistant")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("thread_suffix", "")
st.session_state.setdefault("base_user_id", "usr_123")
st.session_state.setdefault("recording_key", 0)
st.session_state.setdefault("autoplay", False)

with st.sidebar:
    models, default_model = available_models()
    default_index = models.index(default_model) if default_model in models else 0
    # with no list (backend offline) nothing is selected and the backend uses its own default
    st.session_state.model = st.selectbox("Model", models, index=default_index if models else None)
    st.session_state.mode = st.radio("Mode", ["Chat", "Voice"], horizontal=True)
    base_user_id = st.text_input("User ID", value="usr_123").strip()

    if st.button("New chat"):
        # a new suffix is a new thread on the backend, so the old conversation doesn't leak in
        st.session_state.messages = []
        st.session_state.thread_suffix = f"_{uuid.uuid4().hex[:4]}"
        st.rerun()

    if base_user_id != st.session_state.base_user_id:
        st.session_state.base_user_id = base_user_id
        st.session_state.messages = []
        st.session_state.thread_suffix = ""

    st.session_state.user_id = base_user_id + st.session_state.thread_suffix
    st.caption(f"Conversation: {st.session_state.user_id}")

    if backend_online():
        st.success("Backend online")
    else:
        st.error("Backend offline")

messages = st.session_state.messages
autoplay = st.session_state.autoplay
st.session_state.autoplay = False

for index, message in enumerate(messages):
    show_message(message, autoplay=autoplay and index == len(messages) - 1)

waiting_for_answer = bool(messages) and any(
    call["status"] == "awaiting_confirmation" for call in messages[-1].get("tool_calls", [])
)

if waiting_for_answer:
    confirm, cancel, _ = st.columns([1, 1, 4])
    if confirm.button("Confirm", type="primary"):
        send("yes")
    if cancel.button("Cancel"):
        send("no")

if st.session_state.mode == "Voice":
    recording = st.audio_input("Record your question", key=f"recording_{st.session_state.recording_key}")
    if recording:
        # a new key clears the recorder, so the same audio isn't sent again on the next rerun
        st.session_state.recording_key += 1
        with st.spinner("Transcribing..."):
            transcript = transcribe(recording)
        if transcript:
            send(transcript)

placeholder = "Type yes to submit the claim, or no to cancel" if waiting_for_answer else "Ask about your policy or a claim"

if prompt := st.chat_input(placeholder):
    send(prompt)
