import os
from typing import Annotated
from typing_extensions import TypedDict

import streamlit as st
from dotenv import load_dotenv

# --- Loaders / splitting / embeddings / vector store
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore

# --- LLM & prompt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Chat history & messages utils (LangChain)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AnyMessage

# --- LangGraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from operator import itemgetter


# ======================
# Streamlit setup
# ======================
st.set_page_config(page_title="LangGraph RAG Chat", page_icon="🤖", layout="wide")
st.title("RAG Chat with LangGraph 🤖")

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set")
    st.stop()

# ---------------- Session state ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # UI-only rendering

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

if "history_sidebar" not in st.session_state:
    st.session_state.history_sidebar = ChatMessageHistory()


# ======================
# Build / rebuild RAG
# ======================
url = st.text_input(
    "Enter a URL to load documents from:",
    value="https://www.govinfo.gov/content/pkg/CDOC-110hdoc50/html/CDOC-110hdoc50.htm",
)

c1, c2 = st.columns(2)
with c1:
    if st.button("Initialize / Rebuild RAG", type="primary"):
        with st.spinner("Loading and indexing…"):
            loader = WebBaseLoader(url)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
        st.success("RAG initialized!")

with c2:
    if st.button("Reset Conversation"):
        st.session_state.chat_messages = []
        # Reset the LangGraph thread by swapping the session_id (simple trick) or clearing memory via a new checkpointer instance
        st.session_state.session_id = "default"  # keep same id; we'll create a fresh checkpointer below
        st.toast("Conversation memory cleared for this session.")


if st.session_state.vectorstore is None:
    st.info("Initialize the RAG system first.")
    st.stop()


# ======================
# LangGraph RAG app
# ======================

# ---- State for the graph
class RAGState(TypedDict, total=False):
    # running list of chat messages (LangChain message objects)
    messages: Annotated[list[AnyMessage], add_messages]
    # retrieved context we add during the retrieve node
    context: str

# ---- Shared components
retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant that answers strictly from the provided documents. "
         "Use the retrieved context and the prior conversation. "
         "If the answer is not in the docs, say \"I don't know.\""),
        MessagesPlaceholder("messages"),
        ("human", "Context:\n{context}")
    ]
)

# Build a simple LC chain for generation
gen_chain = (
    {
        "messages": itemgetter("messages"),  # <- just the list of BaseMessage
        "context": itemgetter("context"),    # <- just the context string
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ---- Graph nodes
def retrieve_node(state: RAGState) -> dict:
    """Use the latest user message to retrieve context."""
    # The newest user turn is the last message with role 'human' or simply the last message content
    last_user_text = state["messages"][-1].content if state.get("messages") else ""
    docs = retriever.invoke(last_user_text)
    return {"context": format_docs(docs)}

def generate_node(state: RAGState) -> dict:
    """Call the LLM with conversation messages + retrieved context; append the answer."""
    answer = gen_chain.invoke(state)   # <-- pass the state dict directly
    return {"messages": [("assistant", answer)]}

# ---- Build the graph
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# ---- Checkpointer for memory
checkpointer = MemorySaver()  # in-memory, per-thread conversation memory

app = graph.compile(checkpointer=checkpointer)


# ======================
# Streamlit Chat UI
# ======================
left, right = st.columns([2, 1])

with left:
    # render prior UI messages (separate from the graph's stored memory)
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_text = st.chat_input("Ask about the loaded documents…")
    if user_text:
        # UI echo
        st.session_state.chat_messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        # Run one step of the graph with memory tied to thread_id=session_id
        # Input is a new user message; LangGraph will stitch with prior turns via the checkpointer.
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                events = app.invoke(
                    {"messages": [("user", user_text)]},
                    config={"configurable": {"thread_id": st.session_state.session_id}},
                )
                # `events` is the final state. Get the last assistant message text:
                assistant_msg = ""
                for m in events["messages"][::-1]:
                    if getattr(m, "type", "") == "ai" or getattr(m, "role", "") == "assistant":
                        assistant_msg = m.content
                        break

                st.write(assistant_msg)
                st.session_state.chat_messages.append({"role": "assistant", "content": assistant_msg})

with right:
    st.subheader("Session")
    st.write(f"**Thread ID:** `{st.session_state.session_id}`")
    st.caption("Change this to persist/branch conversation state per user or topic.")

    st.subheader("Notes")
    st.markdown(
        "- The RAG pipeline is a LangGraph: **retrieve → generate**.\n"
        "- **MemorySaver** keeps conversation history per `thread_id`.\n"
        "- The prompt receives both the running **messages** and the retrieved **context**."
    )
