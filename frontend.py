import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import os
from backend import add_new_notices


def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    from backend import chatbot  # deferred import to avoid circular import

    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    from backend import retrieve_all_threads

    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

st.sidebar.title("Campus Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Upload New Notices")
uploaded_files = st.sidebar.file_uploader(
    "Select PDF notice(s) to upload",
    type=["pdf"],
    accept_multiple_files=True,
)
if uploaded_files:
    saved_paths = []
    for uploaded_file in uploaded_files:
        save_path = os.path.join("brochures", uploaded_file.name)
        # Save uploaded file to 'brochures' folder
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(save_path)

    # Add new PDFs to vectorstore dynamically
    added_count = add_new_notices(saved_paths)
    st.sidebar.success(f"Added {added_count} new notice(s) to vector database.")

st.sidebar.header("Chat Threads")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages


st.header("Ask About Courses, Admissions, and More")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type your question about campus info here")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    from backend import chatbot  # deferred import

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if message_chunk.type == "tool":
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                if message_chunk.type == "ai":
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
