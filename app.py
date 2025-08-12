import nest_asyncio
nest_asyncio.apply()

import os
import asyncio
import streamlit as st
import base64
import gc
import tempfile
import uuid
import time
import io
from contextlib import redirect_stdout
from pathlib import Path

# Import our enhanced RAG components
from src.embeddings.embed_data import EmbedData
from src.indexing.milvus_vdb import MilvusVDB
from src.retrieval.retriever_rerank import Retriever
from src.generation.rag import RAG
from src.workflows.agent_workflow import EnhancedRAGWorkflow
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(page_title="Enhanced RAG Demo", layout="wide")

# Initialize session state variables
if "id" not in st.session_state:
    st.session_state.id = str(uuid.uuid4())[:8]
    st.session_state.file_cache = {}
    
if "workflow" not in st.session_state:
    st.session_state.workflow = None
    
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "workflow_logs" not in st.session_state:
    st.session_state.workflow_logs = []
    
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

session_id = st.session_state.id

def reset_chat():
    """Reset chat history and clear memory."""
    st.session_state.messages = []
    st.session_state.workflow_logs = []
    gc.collect()

def display_pdf(file):
    """Display PDF preview in sidebar."""
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    
    st.markdown(pdf_display, unsafe_allow_html=True)

def load_and_split_pdf(file_path: str, chunk_size: int = 512, chunk_overlap: int = 50):
    """Simple PDF loading and splitting function."""
    try:
        # Load PDF using SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        
        # Initialize text splitter
        text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" "
        )
        
        # Split documents into chunks
        all_chunks = []
        for doc in documents:
            nodes = text_splitter.get_nodes_from_documents([doc])
            chunks = [node.text for node in nodes]
            all_chunks.extend(chunks)
        
        return all_chunks
        
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

def initialize_workflow(file_path: str):
    """Initialize the enhanced RAG workflow with uploaded document."""
    with st.spinner("🔄 Loading document and setting up the workflow..."):
        try:
            # Step 1: Load and split document
            st.info("📄 Loading and processing PDF...")
            text_chunks = load_and_split_pdf(file_path)
            
            if not text_chunks:
                st.error("No text chunks extracted from PDF")
                return None
            
            st.success(f"✅ Created {len(text_chunks)} text chunks")
            
            # Step 2: Create embeddings
            st.info("🧠 Generating embeddings...")
            embed_data = EmbedData(
                embed_model_name="BAAI/bge-large-en-v1.5",
                batch_size=256  # Smaller batch for demo
            )
            embed_data.embed(text_chunks)
            st.success("✅ Embeddings generated with binary quantization")
            
            # Step 3: Setup vector database
            st.info("🗄️ Setting up Milvus vector database...")
            collection_name = f"demo_collection_{session_id}"
            
            vector_db = MilvusVDB(
                collection_name=collection_name,
                vector_dim=1024,
                batch_size=256,
                db_file=f"./data/milvus_demo_{session_id}.db"
            )
            
            vector_db.initialize_client()
            vector_db.create_collection()
            vector_db.ingest_data(embed_data)
            
            # Store in session state for cleanup
            st.session_state.vector_db = vector_db
            st.success("✅ Vector database setup completed")
            
            # Step 4: Setup retrieval (simplified - no reranking for demo)
            st.info("🔍 Setting up retrieval system...")
            retriever = Retriever(
                vector_db=vector_db,
                embed_data=embed_data,
                top_k=5
            )
            st.success("✅ Retrieval system ready")
            
            # Step 5: Setup RAG system
            st.info("🤖 Setting up RAG system...")
            rag_system = RAG(
                retriever=retriever,
                llm_model="gpt-3.5-turbo",
                temperature=0.4,
                max_tokens=1000
            )
            st.success("✅ RAG system initialized")
            
            # Step 6: Setup workflow
            st.info("⚙️ Setting up enhanced workflow...")
            workflow = EnhancedRAGWorkflow(
                retriever=retriever,
                rag_system=rag_system,
                firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            st.success("🎉 Workflow setup completed!")
            return workflow
            
        except Exception as e:
            st.error(f"Error initializing workflow: {e}")
            return None

async def run_workflow(query: str):
    """Run the async workflow and capture logs."""
    f = io.StringIO()
    with redirect_stdout(f):
        result = await st.session_state.workflow.run_workflow(query)
    
    # Get the captured logs and store them
    logs = f.getvalue()
    if logs:
        st.session_state.workflow_logs.append(logs)
    
    return result

def cleanup_resources():
    """Cleanup vector database and other resources."""
    if st.session_state.vector_db:
        try:
            st.session_state.vector_db.close()
        except:
            pass
        st.session_state.vector_db = None

# Sidebar for configuration and document upload
with st.sidebar:
    # Header
    st.header("🔧 Configuration")
    
    # API Key inputs
    st.subheader("API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    firecrawl_key = st.text_input("Firecrawl API Key (Optional)", type="password", value=os.getenv("FIRECRAWL_API_KEY", ""))
    
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        st.success("✅ OpenAI API Key set!")
    
    if firecrawl_key:
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_key
        st.success("✅ Firecrawl API Key set!")
    
    st.markdown("---")
    
    # Document upload section
    st.header("📄 Upload Document")
    st.markdown("Upload a PDF document to get started")
    
    uploaded_file = st.file_uploader("Choose your PDF file", type="pdf")
    
    if uploaded_file and openai_key:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                
                if file_key not in st.session_state.get('file_cache', {}):
                    # Initialize workflow with the uploaded document
                    workflow = initialize_workflow(file_path)
                    if workflow:
                        st.session_state.workflow = workflow
                        st.session_state.file_cache[file_key] = workflow
                        st.balloons()
                else:
                    st.session_state.workflow = st.session_state.file_cache[file_key]
                
                if st.session_state.workflow:
                    st.success("🎉 Ready to Chat!")
                    display_pdf(uploaded_file)
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    elif uploaded_file and not openai_key:
        st.warning("⚠️ Please enter your OpenAI API key first!")
    
    # Cleanup button
    st.markdown("---")
    if st.button("🗑️ Clean Up Resources"):
        cleanup_resources()
        st.success("Resources cleaned up!")

# Main chat interface
col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("<h1 style='color: #2E86AB;'>🚀 Enhanced RAG Pipeline</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #A23B72; font-size: 18px;'>Multi-Agent Workflow with Router & Web Search</p>", unsafe_allow_html=True)

with col2:
    if st.button("Clear Chat ↺", on_click=reset_chat):
        st.rerun()

# System info
if st.session_state.workflow:
    st.success("🟢 System Ready - Workflow initialized successfully!")
else:
    st.info("🔵 Upload a PDF document to get started")

# Display chat messages from history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
    # Display workflow logs for user messages
    if (message["role"] == "user" and 
        "log_index" in message and 
        message["log_index"] < len(st.session_state.workflow_logs)):
        
        with st.expander("🔍 View Workflow Execution Details", expanded=False):
            logs = st.session_state.workflow_logs[message["log_index"]]
            st.code(logs, language="text")

# Accept user input
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.workflow:
        st.error("⚠️ Please upload a document first to initialize the workflow.")
        st.stop()
    
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ Please set your OpenAI API key in the sidebar.")
        st.stop()
    
    # Add user message to chat history
    log_index = len(st.session_state.workflow_logs)
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt, 
        "log_index": log_index
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Run the workflow and get response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("🔄 Processing your query..."):
                # Measure end-to-end workflow time
                t0_workflow = time.perf_counter()
                result = asyncio.run(run_workflow(prompt))
                t1_workflow = time.perf_counter()
                workflow_time = (t1_workflow - t0_workflow)
            
            # Display workflow logs
            if log_index < len(st.session_state.workflow_logs):
                with st.expander("🔍 View Workflow Execution Details", expanded=False):
                    st.code(st.session_state.workflow_logs[log_index], language="text")
            
            # Get the final answer
            if isinstance(result, dict) and "answer" in result:
                full_response = result["answer"]
                
                # Show additional info about the workflow
                if result.get("web_search_used", False):
                    st.info("🌐 This response includes information from web search")
                    # Only show completion time (no retrieval time on web search path)
                    if 'workflow_time' in locals():
                        st.caption(f"Completion time: {workflow_time} s")
                else:
                    st.info("📚 This response is based on your document")
                    # Measure retrieval time while fetching citations (only for RAG path)
                    retrieval_ms = None
                    try:
                        retriever = getattr(st.session_state.workflow, "retriever", None)
                        if retriever:
                            t0_retrieve = time.perf_counter()
                            retriever.search(prompt)
                            t1_retrieve = time.perf_counter()
                            retrieval_time = int((t1_retrieve - t0_retrieve) * 1000)

                            citations = retriever.get_citations(prompt, top_k=3, snippet_chars=200)

                            if citations:
                                with st.expander("📎 Citations (top matches)"):
                                    for c in citations:
                                        score = c.get("score")
                                        try:
                                            score_str = f"{float(score):.3f}"
                                        except Exception:
                                            score_str = str(score)
                                        st.markdown(
                                            f"[{c['rank']}] score={score_str} id={c.get('node_id')}"
                                        )
                                        if c.get("snippet"):
                                            st.code(c["snippet"], language="text")
                    except Exception as e:
                        st.warning(f"Could not fetch citations: {e}")

                    # Show timing caption (retrieval + completion)
                    times = []
                    if retrieval_time is not None:
                        times.append(f"🕒 Retrieval time: {retrieval_time} ms")
                    if 'workflow_time' in locals():
                        times.append(f"🕒 Completion time: {workflow_time:.2f} s")
                    if times:
                        st.caption(" • ".join(times))
                
            else:
                full_response = str(result)
            
            # Stream the response word by word for better UX
            streamed_response = ""
            words = full_response.split()
            
            for i, word in enumerate(words):
                streamed_response += word + " "
                message_placeholder.markdown(streamed_response + "▌")
                
                if i < len(words) - 1:
                    time.sleep(0.05)  # Faster streaming
            
            # Display final response without cursor
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_msg = f"❌ Error processing your question: {str(e)}"
            st.error(error_msg)
            full_response = "I apologize, but I encountered an error while processing your question. Please try again."
            message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response
    })

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 12px;'>"
    "Enhanced RAG Pipeline • Built with Streamlit, LlamaIndex, Milvus, and OpenAI"
    "</p>",
    unsafe_allow_html=True
)
