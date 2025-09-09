from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
import os
import sqlite3
import glob
from dotenv import load_dotenv
from langchain.vectorstores import Chroma

load_dotenv()


def load_brochures(folder_path: str = "./brochures"):
    """
    Load PDF documents from the given folder path.

    Args:
        folder_path (str): Path to the directory containing PDF brochures.

    Returns:
        list: List of LangChain Documents loaded from PDFs.
    """
    docs = []
    for pdf_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    return docs


embeddings = HuggingFaceEmbeddings()

PERSIST_DIRECTORY = "./chroma_persist"
docs = load_brochures()
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY,
    collection_name="campus_docs",
)
vector_store.persist()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_retries=2,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
)


@tool
def campus_qa(query: str) -> str:
    """
    Query the campus brochure documents and return relevant information.

    Args:
        query (str): User query as text.

    Returns:
        str: Response generated from the retrieved documents.
    """
    return qa_chain.run(query)


tools = [campus_qa]
tool_node = ToolNode(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """
    LangGraph chat node to process human messages and generate assistant responses.

    Args:
        state (ChatState): Current chat state containing message history.

    Returns:
        dict: Dictionary containing a list of AIMessage responses.
    """
    messages = state["messages"]
    user_msg = next(m for m in reversed(messages) if m.type == "human")
    response = campus_qa(user_msg.content)
    return {"messages": [AIMessage(content=response)]}


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
    """
    Retrieve all unique conversation thread IDs stored in the database.

    Returns:
        list: List of thread IDs.
    """
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def add_new_notices(pdf_paths):
    """
    Add new notice PDFs to the vector store at runtime.
    
    Args:
        pdf_paths (list): List of PDF file paths.
        
    Returns:
        int: Number of added documents.
    """
    
    new_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        new_docs.extend(loader.load())
    
    if new_docs:
        vector_store.add_documents(documents=new_docs)
        vector_store.persist()
    return len(new_docs)


