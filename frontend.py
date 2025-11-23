import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import uuid
import os
from dotenv import load_dotenv

# Import from backend
from backend import (
    add_new_notices,
    chatbot,
    retrieve_all_threads,
    load_brochures,
    initialize_vector_store,
)

load_dotenv()


def generate_thread_id():
    """Generate a unique thread ID."""
    return str(uuid.uuid4())


def reset_chat():
    """Reset the chat and create a new thread."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    """Add a thread to the list if it doesn't exist."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id: str) -> list:
    """Load message history from a thread."""
    try:
        state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
        if state and state.values:
            messages = state.values.get("messages", [])
            return [m for m in messages if isinstance(m, BaseMessage)]
    except Exception as e:
        st.warning(f"Could not load conversation: {e}")
    return []


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "vector_store_initialized" not in st.session_state:
    st.session_state["vector_store_initialized"] = False

add_thread(st.session_state["thread_id"])


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("Campus Chatbot")

if st.sidebar.button("🆕 New Chat"):
    reset_chat()
    st.rerun()

# Initialize vector store (only once)
if not st.session_state["vector_store_initialized"]:
    with st.spinner("🔄 Loading campus documents..."):
        docs = load_brochures()
        if docs:
            initialize_vector_store(docs)
    st.session_state["vector_store_initialized"] = True

# Upload new notices
st.sidebar.header("📄 Upload New Notices")
uploaded_files = st.sidebar.file_uploader(
    "Select PDF notice(s) to upload",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    saved_paths = []
    os.makedirs("brochures", exist_ok=True)

    progress_bar = st.sidebar.progress(0)

    for idx, uploaded_file in enumerate(uploaded_files):
        save_path = os.path.join("brochures", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(save_path)
        progress_bar.progress((idx + 1) / len(uploaded_files))

    # Add to vector store
    if saved_paths:
        with st.sidebar.spinner("Processing PDFs..."):
            try:
                added_count = add_new_notices(saved_paths)
                if added_count > 0:
                    st.sidebar.success(
                        f"✅ Added {added_count} new page(s) to the database!"
                    )
                else:
                    st.sidebar.warning(
                        "⚠️ No new documents were added. Check file format."
                    )
            except Exception as e:
                st.sidebar.error(f"❌ Error adding notices: {str(e)}")

# Thread selector
st.sidebar.header("💬 Chat History")
for thread_id in st.session_state["chat_threads"][::-1]:
    thread_display = str(thread_id)[:8]
    if st.sidebar.button(
        thread_display, key=f"thread_{thread_id}", use_container_width=True
    ):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        st.session_state["message_history"] = messages
        st.rerun()


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

st.header("🎓 Campus Information Chatbot")
st.write("Ask questions about courses, admissions, campus facilities, and more!")

# Display message history
for message in st.session_state["message_history"]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to history
    user_message = HumanMessage(content=user_input)
    st.session_state["message_history"].append(user_message)

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Stream response from chatbot
    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

        ai_content = ""
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        try:
            # Stream from chatbot
            for chunk in chatbot.stream(
                {"messages": st.session_state["message_history"]},
                config=config,
                stream_mode="values",
            ):
                latest_msg = chunk["messages"][-1]

                # Display AI message content
                if isinstance(latest_msg, AIMessage):
                    ai_content = latest_msg.content
                    response_placeholder.write(ai_content)

                # Show tool status if tools are being used
                elif hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                    tool_names = ", ".join(
                        [tc.get("name", "tool") for tc in latest_msg.tool_calls]
                    )
                    status_placeholder.info(f"🔧 Using: {tool_names}")

            # Clear status after completion
            status_placeholder.empty()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            ai_content = ""

    # Save assistant response to history
    if ai_content:
        assistant_message = AIMessage(content=ai_content)
        st.session_state["message_history"].append(assistant_message)
