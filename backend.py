import os
import sqlite3
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import PromptTemplate
from langgraph_checkpoint_sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_chroma import Chroma
from typing import TypedDict, Annotated

load_dotenv()


def load_brochures(folder_path: str = "./brochures") -> list:
    """Load PDF documents from the given folder path."""
    docs = []

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Creating {folder_path} folder...")
        os.makedirs(folder_path, exist_ok=True)
        return docs

    # Get all PDF files
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return docs

    print(f"Found {len(pdf_files)} PDF file(s) to load")

    for pdf_path in pdf_files:
        try:
            print(f"Loading: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            loaded_docs = loader.load()
            print(f"  ✓ Loaded {len(loaded_docs)} pages")
            docs.extend(loaded_docs)
        except Exception as e:
            print(f"  ✗ Error loading {pdf_path}: {e}")

    print(f"Total documents loaded: {len(docs)}")
    return docs


# Get API key from environment
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize embeddings (lazy loading)
embeddings = None


def get_embeddings():
    """Lazy load HuggingFace embeddings to avoid timeout on startup."""
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


# Initialize vector store
PERSIST_DIRECTORY = "./chroma_persist"


def get_vector_store():
    """Get or create the vector store."""
    return Chroma(
        collection_name="campus_docs",
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIRECTORY,
    )


# Initialize LLM
llm = ChatGoogleGenerativeAI(
    api_key=google_api_key,
    model="gemini-2.5-flash",
    temperature=0.7,
)

# System prompt for the chatbot
SYSTEM_PROMPT = """You are a helpful Campus Information Assistant for a university.
Your job is to answer questions about courses, admissions, facilities, notices, and other campus-related topics.

Guidelines:
1. Use the campus_qa tool to retrieve relevant information from campus documents
2. Provide accurate, concise answers based on the retrieved documents
3. If information is not found in the documents, clearly state that
4. Be professional and helpful
5. Provide specific details when available (course codes, dates, contact information, etc.)"""


# Define campus QA tool
@tool
def campus_qa(query: str) -> str:
    """
    Query the campus brochure documents and return relevant information.
    Use this tool to answer any questions about courses, admissions, campus facilities, notices, etc.
    Always use this tool when users ask questions.
    """
    try:
        vector_store = get_vector_store()

        # Search for relevant documents
        docs = vector_store.similarity_search(query, k=5)

        if not docs:
            return "I couldn't find relevant information in the campus documents about your question."

        # Combine retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following campus information, answer the question concisely and accurately.
If the information is not in the provided context, say so clearly.

Context:
{context}

Question: {question}

Answer:""",
        )

        # Format and invoke LLM
        formatted_prompt = prompt_template.format(context=context, question=query)
        response = llm.invoke(formatted_prompt)

        return response.content
    except Exception as e:
        return f"Error retrieving information: {str(e)}"


tools = [campus_qa]
tool_node = ToolNode(tools)


# Define chat state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define chat node with system prompt
def chat_node(state: ChatState):
    """LLM processes messages and decides whether to use tools."""
    messages = state["messages"]

    # Invoke LLM with messages - it will decide to use campus_qa tool
    response = llm.invoke(messages)
    return {"messages": [response]}


# Initialize SQLite checkpointer
def get_checkpointer():
    """Get or create SQLite checkpointer for conversation persistence."""
    conn = sqlite3.connect("chatbot.db", check_same_thread=False)
    return SqliteSaver(conn=conn)


# Build the graph
def create_chatbot_graph():
    """Create and compile the chatbot graph."""
    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # Add edges
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    # Compile with checkpointer
    return graph.compile(checkpointer=get_checkpointer())


# Create the chatbot
chatbot = create_chatbot_graph()


# Utility functions for frontend
def retrieve_all_threads() -> list:
    """Retrieve all unique conversation thread IDs from the database."""
    try:
        checkpointer = get_checkpointer()
        all_threads = set()

        for checkpoint in checkpointer.list(None):
            if checkpoint and checkpoint.config:
                thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
                if thread_id:
                    all_threads.add(thread_id)

        return sorted(list(all_threads), reverse=True)
    except Exception as e:
        print(f"Error retrieving threads: {e}")
        return []


def add_new_notices(pdf_paths: list) -> int:
    """Add new notice PDFs to the vector store at runtime."""
    try:
        new_docs = []

        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"File not found: {pdf_path}")
                continue

            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                print(f"Loaded {len(docs)} pages from {pdf_path}")
                new_docs.extend(docs)
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")

        # Add documents to vector store
        if new_docs:
            vector_store = get_vector_store()
            print(f"Adding {len(new_docs)} documents to vector store...")
            vector_store.add_documents(documents=new_docs)
            print(f"Successfully added {len(new_docs)} documents")
            return len(new_docs)
        else:
            print("No documents loaded from PDFs")
            return 0

    except Exception as e:
        print(f"Error in add_new_notices: {e}")
        return 0


def initialize_vector_store(docs: list) -> int:
    """Initialize vector store with documents."""
    try:
        if docs:
            vector_store = get_vector_store()
            vector_store.add_documents(documents=docs)
            return len(docs)
        return 0
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return 0
