"""
Enhanced Streamlit Frontend with Knowledge Base Upload

Features:
- Document upload (PDF, TXT, DOCX, MD)
- Collection management
- RAG queries with tool integration
- Real-time indexing status
"""

import sys
from pathlib import Path

# Add parent directory to path for imports FIRST
# This handles both direct execution and when run via streamlit from different directories
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import streamlit as st
import requests
import os
from agent.agent import SimpleAgent
from agent.retriever import get_top_k

# Page configuration
st.set_page_config(
    page_title="RAG Agent with Knowledge Base",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'kb_api_url' not in st.session_state:
    st.session_state.kb_api_url = "http://localhost:8100"
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'selected_collection' not in st.session_state:
    st.session_state.selected_collection = "default"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #D4EDDA;
        border: 1px solid #C3E6CB;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #F8D7DA;
        border: 1px solid #F5C6CB;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #D1ECF1;
        border: 1px solid #BEE5EB;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def format_agent_response(response_dict: dict, query: str) -> str:
    """Format agent response dictionary into natural language."""
    if not isinstance(response_dict, dict):
        return str(response_dict)
    
    tool = response_dict.get('selected_tool', 'none')
    tool_result = response_dict.get('tool_result')
    plan = response_dict.get('plan', '')
    
    # Build natural language response
    output = []
    
    if tool == 'calculator' and tool_result:
        if tool_result.get('status') == 'success':
            result = tool_result.get('result', {})
            expr = result.get('expression', query)
            value = result.get('value', 'N/A')
            output.append(f"**Calculation Result:**\n\n`{expr} = {value}`")
        else:
            result = tool_result.get('result', {})
            error = result.get('error') or tool_result.get('error', 'Unknown error')
            output.append(f"❌ **Calculation Error:** {error}")
    
    elif tool == 'plot' and tool_result:
        if tool_result.get('status') == 'success':
            result = tool_result.get('result', {})
            # Support both 'image' and 'artifact_base64' field names
            img_b64 = result.get('image') or result.get('artifact_base64')
            if img_b64:
                output.append("**Generated Visualization:**\n")
                output.append(f"![Plot](data:image/png;base64,{img_b64})")
            else:
                output.append(f"✅ Plot generated successfully")
        else:
            result = tool_result.get('result', {})
            error = result.get('error') or tool_result.get('error', 'Unknown error')
            output.append(f"❌ **Plot Error:** {error}")
    
    elif tool == 'pdf' and tool_result:
        if tool_result.get('status') == 'success':
            result = tool_result.get('result', {})
            text = result.get('text', '')[:500]
            output.append(f"**PDF Content:**\n\n{text}...")
        else:
            result = tool_result.get('result', {})
            error = result.get('error') or tool_result.get('error', 'Unknown error')
            output.append(f"❌ **PDF Error:** {error}")
    
    elif tool == 'web_search' and tool_result:
        if tool_result.get('status') == 'success':
            result = tool_result.get('result', {})
            
            # Check if we have a generated answer
            answer = result.get('answer')
            sources = result.get('sources', [])
            
            if answer:
                # Display generated answer with sources
                output.append(f"**Answer:**\n\n{answer}\n")
                if sources:
                    output.append(f"\n**Sources:**")
                    for idx, res in enumerate(sources, 1):
                        title = res.get('title', 'No title')
                        url = res.get('url', '')
                        output.append(f"{idx}. [{title}]({url})")
            else:
                # Fallback to raw search results
                results = result.get('results', [])
                if results:
                    output.append(f"**Web Search Results:**\n")
                    for idx, res in enumerate(results[:3], 1):
                        title = res.get('title', 'No title')
                        snippet = res.get('snippet', '')
                        url = res.get('url', '')
                        output.append(f"{idx}. **{title}**")
                        output.append(f"   {snippet}")
                        output.append(f"   [{url}]({url})\n")
                else:
                    output.append("No search results found")
        else:
            result = tool_result.get('result', {})
            error = result.get('error') or tool_result.get('error', 'Unknown error')
            output.append(f"❌ **Search Error:** {error}")
    
    elif tool == 'file_ops' and tool_result:
        if tool_result.get('status') == 'success':
            result = tool_result.get('result', {})
            content = result.get('content', '')[:500]
            path = result.get('path', 'unknown')
            output.append(f"**File Content** (`{path}`):\n\n```\n{content}\n```")
        else:
            result = tool_result.get('result', {})
            error = result.get('error') or tool_result.get('error', 'Unknown error')
            output.append(f"❌ **File Error:** {error}")
    
    else:
        # No tool or tool failed
        if plan:
            output.append(f"**Response:** {plan}")
    
    # Add performance metrics in expander
    metrics = []
    if 'retrieval_time_ms' in response_dict:
        metrics.append(f"Retrieval: {response_dict['retrieval_time_ms']:.0f}ms")
    if 'llm_time_ms' in response_dict:
        metrics.append(f"LLM: {response_dict['llm_time_ms']:.0f}ms")
    if 'end_to_end_ms' in response_dict:
        metrics.append(f"Total: {response_dict['end_to_end_ms']:.0f}ms")
    
    if metrics:
        output.append(f"\n\n<details><summary>⚡ Performance</summary>{' | '.join(metrics)}</details>")
    
    return "\n".join(output)


def generate_answer_from_context(query: str, context: str) -> str:
    """Generate answer from retrieved context using LLM."""
    openai_key = os.getenv('OPENAI_API_KEY', '')
    
    if openai_key:
        # Use OpenAI API (new interface for openai>=1.0.0)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            
            prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"OpenAI API error: {e}. Falling back to context summary.")
    
    # Fallback: Simple extractive summary
    sentences = context.split('. ')[:5]
    summary = '. '.join(sentences) + '.'
    return f"""**Based on the retrieved documents:**

{summary}

*Note: Set OPENAI_API_KEY environment variable for AI-generated responses.*"""


def get_collections():
    """Fetch all collections from KB API."""
    try:
        response = requests.get(f"{st.session_state.kb_api_url}/collections")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Failed to fetch collections: {e}")
        return []


def upload_document(file, collection_name):
    """Upload document to KB API."""
    try:
        files = {'file': (file.name, file, file.type)}
        data = {'collection': collection_name}
        
        response = requests.post(
            f"{st.session_state.kb_api_url}/upload",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'status': 'error', 'message': response.text}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def query_kb(query, collection, top_k=5, min_similarity=0.3):
    """Query knowledge base with similarity threshold.
    
    Args:
        query: Search query
        collection: Collection name
        top_k: Maximum number of documents to retrieve
        min_similarity: Minimum similarity score (0-1). Documents below this are filtered out.
                       0.0 = no filtering, 1.0 = exact match only
                       Typical: 0.3 (loose) to 0.7 (strict)
    """
    try:
        response = requests.post(
            f"{st.session_state.kb_api_url}/query",
            json={'query': query, 'collection': collection, 'top_k': top_k}
        )
        
        if response.status_code == 200:
            result = response.json()
            # Filter documents by similarity score (higher = more similar)
            if result and 'documents' in result:
                filtered_docs = [
                    doc for doc in result['documents']
                    if doc.get('score', 0) >= min_similarity
                ]
                result['documents'] = filtered_docs
                result['filtered_count'] = len(result['documents'])
                result['min_similarity_threshold'] = min_similarity
            return result
        return None
    except Exception as e:
        st.error(f"Query failed: {e}")
        return None


def delete_collection(collection_name):
    """Delete a collection."""
    try:
        response = requests.delete(
            f"{st.session_state.kb_api_url}/collections/{collection_name}"
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # KB API Configuration
    st.subheader("Knowledge Base API")
    kb_api_url = st.text_input(
        "KB API URL",
        value=st.session_state.kb_api_url,
        help="URL of the Knowledge Base API service"
    )
    if kb_api_url != st.session_state.kb_api_url:
        st.session_state.kb_api_url = kb_api_url
        st.rerun()
    
    # Test KB API connection
    try:
        response = requests.get(f"{st.session_state.kb_api_url}/")
        if response.status_code == 200:
            st.success("✅ KB API Connected")
        else:
            st.error("❌ KB API Not Responding")
    except:
        st.error("❌ KB API Unavailable")
    
    st.divider()
    
    # MCP Tool Endpoints
    st.subheader("MCP Tool Endpoints")
    plot_url = st.text_input("Plot Service URL", value="http://127.0.0.1:8000/mcp/plot")
    calc_url = st.text_input("Calculator URL", value="http://127.0.0.1:8001/mcp/calculate")
    pdf_url = st.text_input("PDF Parser URL", value="http://127.0.0.1:8002/mcp/parse")
    web_search_url = st.text_input("Web Search URL", value="http://127.0.0.1:8003/mcp/search")
    file_ops_url = st.text_input("File Ops URL", value="http://127.0.0.1:8004/mcp/read_file")
    
    st.divider()
    
    # Agent Configuration
    st.subheader("Agent Settings")
    use_llm = st.checkbox("Use LLM for tool selection", value=True)
    llm_model = st.selectbox("LLM Model", ["local", "openai"])
    
    if llm_model == "openai":
        openai_key = st.text_input("OpenAI API Key", type="password", help="Required for OpenAI model. Get your key at https://platform.openai.com/api-keys")
        if not openai_key:
            st.warning("⚠️ OpenAI API key required. Agent will fall back to keyword-based tool selection.")
    else:
        openai_key = None
        st.info("💡 Using free local LLM (TinyLlama). No API key needed!")
    
    # Initialize or update agent
    if st.button("Initialize Agent"):
        endpoints = {
            'plot': plot_url,
            'calculator': calc_url,
            'pdf': pdf_url,
            'web_search': web_search_url,
            'file_ops': file_ops_url
        }
        st.session_state.agent = SimpleAgent(
            endpoints=endpoints,
            use_llm=use_llm,
            llm_api_key=openai_key,
            llm_model=llm_model
        )
        st.success("✅ Agent initialized with 5 tools!")
    
    # Always show agent status
    if 'agent' in st.session_state and st.session_state.agent is not None:
        st.success("✅ Agent is ready! MCP tools enabled.")
    else:
        st.warning("⚠️ Agent not initialized. Click 'Initialize Agent' above to enable MCP tools.")


# Main content
st.markdown('<p class="main-header">🤖 RAG Agent with Knowledge Base</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload Documents",
    "💬 Chat with Knowledge Base",
    "📚 Manage Collections",
    "📊 Collection Stats"
])

# Tab 1: Upload Documents
with tab1:
    st.header("Upload Documents to Knowledge Base")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Files")
        
        # Collection selection/creation
        collections = get_collections()
        collection_names = [c['name'] for c in collections]
        
        upload_option = st.radio(
            "Collection Option",
            ["Use existing collection", "Create new collection"]
        )
        
        if upload_option == "Use existing collection":
            if collection_names:
                collection_name = st.selectbox("Select Collection", collection_names)
            else:
                st.warning("No collections exist. Please create a new one.")
                collection_name = st.text_input("Collection Name", value="default")
        else:
            collection_name = st.text_input(
                "New Collection Name",
                help="Letters, numbers, hyphens, and underscores only"
            )
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX, Markdown"
        )
        
        if uploaded_files and collection_name:
            if st.button("Upload & Index Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                success_count = 0
                error_count = 0
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    result = upload_document(file, collection_name)
                    
                    if result['status'] == 'success':
                        success_count += 1
                        st.success(f"✅ {file.name}: {result['chunks_added']} chunks indexed")
                    else:
                        error_count += 1
                        st.error(f"❌ {file.name}: {result.get('message', 'Upload failed')}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("Upload complete!")
                
                if success_count > 0:
                    st.balloons()
                    st.success(f"Successfully uploaded {success_count} file(s) to '{collection_name}'")
    
    with col2:
        st.subheader("📋 Quick Guide")
        st.info("""
        **Steps:**
        1. Select or create a collection
        2. Choose file(s) to upload
        3. Click 'Upload & Index'
        4. Wait for indexing to complete
        
        **Supported Formats:**
        - PDF (.pdf)
        - Text (.txt)
        - Word (.docx)
        - Markdown (.md)
        
        **Tips:**
        - Group related documents in the same collection
        - Use descriptive collection names
        - Smaller files process faster
        """)


# Tab 2: Chat with Knowledge Base
with tab2:
    st.header("Chat with Your Knowledge Base")
    
    # Collection selector
    collections = get_collections()
    if collections:
        collection_names = [c['name'] for c in collections]
        selected_collection = st.selectbox(
            "Select Knowledge Base",
            collection_names,
            key="chat_collection"
        )
        
        # Query input
        query = st.text_area(
            "Ask a question",
            placeholder="e.g., What is machine learning? Plot a sine wave. Calculate 25 * 17.",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            top_k = st.number_input("Max documents", min_value=1, max_value=10, value=5)
        with col2:
            min_similarity = st.slider("Min similarity", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Minimum similarity score (0-1). Higher = stricter. 0.3=loose, 0.5=balanced, 0.7=strict")
        with col3:
            use_tools = st.checkbox("Enable MCP Tools", value=True)
        
        if st.button("Send Query", type="primary"):
            if query:
                with st.spinner("Processing..."):
                    # Query KB with similarity threshold
                    kb_results = query_kb(query, selected_collection, top_k, min_similarity)
                    
                    has_documents = kb_results and kb_results.get('documents')
                    context = ""
                    
                    if has_documents:
                        # Build context from retrieved documents
                        context = "\n\n".join([doc['content'] for doc in kb_results['documents'][:3]])
                    
                    # If agent is initialized and tools are enabled
                    if use_tools and st.session_state.agent:
                        st.subheader("🤖 Agent Response")
                        try:
                            # Agent will use retrieved context + tools (works even without context)
                            raw_response = st.session_state.agent.plan_and_execute(query)
                            formatted_response = format_agent_response(raw_response, query)
                            st.markdown(formatted_response, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Agent error: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                    elif has_documents:
                        # Generate LLM response from retrieved context (only if we have docs)
                        st.subheader("💬 Generated Answer")
                        answer = generate_answer_from_context(query, context)
                        st.markdown(answer)
                    else:
                        # No documents and no agent enabled
                        st.warning(f"⚠️ No relevant documents found above similarity threshold {min_similarity:.2f}. Try lowering the threshold, rephrasing your question, or enable MCP Tools for calculations/visualizations.")
                    
                    # Show source documents if available
                    if has_documents:
                        with st.expander("📄 View Source Documents"):
                            for idx, doc in enumerate(kb_results['documents'], 1):
                                st.markdown(f"**Document {idx}** (Similarity: {doc['score']:.2f})")
                                st.text(doc['content'][:300] + "...")
                                st.caption(f"Source: {doc['metadata'].get('source_file', 'unknown')}")
                                st.divider()
            else:
                st.warning("Please enter a query")
    else:
        st.warning("No collections found. Please upload documents first.")


# Tab 3: Manage Collections
with tab3:
    st.header("Manage Knowledge Base Collections")
    
    collections = get_collections()
    
    if collections:
        for collection in collections:
            with st.expander(f"📚 {collection['name']} ({collection['document_count']} chunks)"):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.metric("Document Count", collection['document_count'])
                with col2:
                    st.metric("Created", collection['created_at'][:10])
                with col3:
                    if st.button("Delete", key=f"delete_{collection['name']}", type="secondary"):
                        if delete_collection(collection['name']):
                            st.success(f"Deleted collection: {collection['name']}")
                            st.rerun()
                        else:
                            st.error("Delete failed")
    else:
        st.info("No collections exist yet. Upload documents to create your first collection.")


# Tab 4: Collection Stats
with tab4:
    st.header("Collection Statistics")
    
    collections = get_collections()
    
    if collections:
        selected_stats_collection = st.selectbox(
            "Select Collection",
            [c['name'] for c in collections],
            key="stats_collection"
        )
        
        if st.button("Load Statistics"):
            try:
                response = requests.get(
                    f"{st.session_state.kb_api_url}/collections/{selected_stats_collection}/stats"
                )
                
                if response.status_code == 200:
                    stats = response.json()
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Chunks", stats['total_chunks'])
                    with col2:
                        st.metric("Source Files", stats['file_count'])
                    with col3:
                        st.metric("Created", stats['created_at'][:10])
                    
                    # Source files breakdown
                    st.subheader("Source Files")
                    for filename, chunk_count in stats['source_files'].items():
                        st.write(f"- **{filename}**: {chunk_count} chunks")
                else:
                    st.error("Failed to load statistics")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("No collections available")


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>RAG Agent Framework v2.0 | Knowledge Base Edition</small>
</div>
""", unsafe_allow_html=True)
