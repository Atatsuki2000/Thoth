# Demo Knowledge Base

This folder contains sample documents for testing the RAG Agent Framework.

## 📚 Available Documents

1. **[machine_learning_basics.md](machine_learning_basics.md)**
   - Introduction to Machine Learning concepts
   - Types of ML (supervised, unsupervised, reinforcement)
   - Common algorithms and workflows
   - Example queries: "What is overfitting?", "Explain supervised learning"

2. **[python_cheatsheet.md](python_cheatsheet.md)**
   - Python syntax quick reference
   - Data types, control flow, functions, classes
   - File operations and common libraries
   - Example queries: "How to read files in Python?", "Show me list comprehension examples"

3. **[fastapi_guide.md](fastapi_guide.md)**
   - FastAPI framework tutorial
   - Request/response models, validation, authentication
   - Database integration and deployment
   - Example queries: "How to create FastAPI endpoints?", "What is Pydantic validation?"

## 🚀 How to Use

### 1. Start the System
```powershell
.\start_kb_system.ps1
```

### 2. Upload Documents (Web UI)
1. Open http://localhost:9001
2. Go to **Upload** tab
3. Select collection name: `demo_collection`
4. Upload all 3 markdown files from this folder
5. Click "Upload to Knowledge Base"

### 3. Test Queries (Chat Tab)

**Machine Learning Questions:**
- "What is machine learning?"
- "Explain the difference between supervised and unsupervised learning"
- "What are common evaluation metrics for classification?"
- "How do I prevent overfitting?"

**Python Questions:**
- "How do I read a file in Python?"
- "Show me examples of list comprehensions"
- "How do I create a virtual environment?"
- "What are lambda functions?"

**FastAPI Questions:**
- "How do I create a FastAPI application?"
- "What is Pydantic and how is it used in FastAPI?"
- "How do I handle file uploads in FastAPI?"
- "Show me an example of dependency injection in FastAPI"

**Combined with MCP Tools:**
- "Calculate the square root of 144" (uses Calculator tool)
- "Plot a sine wave from 0 to 2*pi" (uses Plot tool)
- "What is supervised learning? Then calculate 10 * 5" (uses RAG + Calculator)

## 📊 Expected Results

After uploading these documents, your knowledge base will be able to answer:
- ✅ Machine learning concepts and terminology
- ✅ Python programming syntax and best practices
- ✅ FastAPI framework usage and patterns
- ✅ Combined queries using both RAG and MCP tools

## 🔄 Reset/Clean Up

To start fresh:
1. Go to **Manage** tab in the UI
2. Select `demo_collection`
3. Click "Delete Collection"
4. Re-upload documents

## 💡 Tips

- **Use specific questions** for best results (e.g., "What is overfitting?" instead of "Tell me about ML")
- **Enable MCP Tools** in the Chat tab sidebar to use calculator/plotting features
- **Check Stats tab** to see document counts and collection information
- **Try follow-up questions** to test retrieval accuracy
