import os
import uuid
import streamlit as st

st.set_page_config(
    page_title="Infinity Skincare Assistant",
    page_icon="🧴",
    layout="centered",
)

st.title("🧴 Infinity Skincare Assistant")
st.caption("LangGraph + RAG demo")

# Secrets are configured in Streamlit Community Cloud > App settings > Secrets.
try:
    os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
    if "OPENROUTER_MODEL" in st.secrets:
        os.environ["OPENROUTER_MODEL"] = st.secrets["OPENROUTER_MODEL"]
except Exception:
    st.error(
        "Missing OPENROUTER_API_KEY. Add it in .streamlit/secrets.toml locally "
        "or in Streamlit Community Cloud → App settings → Secrets."
    )
    st.stop()

try:
    from backend import graph
    from langgraph.types import Command
except Exception as exc:
    st.error("The backend could not start. Make sure the full `ALLURE Docs` folder and all dependencies are present.")
    st.code(str(exc))
    st.stop()


def extract_interrupt_question(result):
    """Return the user-facing LangGraph interrupt question, if the run paused."""
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return None

    interrupt_obj = interrupts[0]
    value = getattr(interrupt_obj, "value", interrupt_obj)

    if isinstance(value, dict):
        return value.get("question") or str(value)
    return str(value)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = []

if "waiting_for_resume" not in st.session_state:
    st.session_state.waiting_for_resume = False

with st.sidebar:
    st.subheader("Session")
    st.caption(f"ID: {st.session_state.thread_id[:8]}")
    if st.button("New conversation", use_container_width=True):
        st.session_state.thread_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.session_state.waiting_for_resume = False
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about skincare or Infinity products...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
        }
    }

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.waiting_for_resume:
                    result = graph.invoke(
                        Command(resume=prompt),
                        config=config,
                    )
                else:
                    result = graph.invoke(
                        {
                            "session_id": st.session_state.thread_id,
                            "all_questions": prompt,
                            "answer": "",
                        },
                        config=config,
                    )

                interrupt_question = extract_interrupt_question(result)

                if interrupt_question:
                    reply = interrupt_question
                    st.session_state.waiting_for_resume = True
                else:
                    reply = result.get("answer", "") if isinstance(result, dict) else ""
                    if not reply:
                        reply = "I couldn't produce a final answer for that request."
                    st.session_state.waiting_for_resume = False

                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

    except Exception as exc:
        st.session_state.waiting_for_resume = False
        st.error("The graph run failed.")
        st.code(str(exc))
