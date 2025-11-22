from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_chroma import Chroma
from typing import TypedDict, Annotated
import os
import sqlite3
import glob
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


def load_brochures(folder_path: str = "./brochures"):
    docs = []
    for pdf_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    return docs


embeddings = HuggingFaceEmbeddings()

PERSIST_DIRECTORY = "./chroma_persist"
docs = load_brochures()

# CORRECT Chroma initialization
vector_store = Chroma(
    collection_name="campus_docs",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY,
)
vector_store.add_documents(documents=docs)

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY"),
    model="gemini-2.5-flash",
)


# Modern tool-based approach instead of RetrievalQA
@tool
def campus_qa(query: str) -> str:
    """Query campus documents using RAG."""
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
    return response.content


tools = [campus_qa]
tool_node = ToolNode(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM processes messages and may invoke tools."""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def add_new_notices(pdf_paths):
    new_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        new_docs.extend(loader.load())
    if new_docs:
        vector_store.add_documents(documents=new_docs)
    return len(new_docs)
