from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# import os

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def build_index(texts, persist_directory="db/chroma"):
    docs = []
    for text in texts:
        splits = splitter.split_text(text)
        docs.extend([Document(page_content=s) for s in splits])
    vectordb = Chroma.from_documents(docs, embedding=emb, collection_name="my_index", persist_directory=persist_directory)
    # vectordb.persist()
    return vectordb

def get_top_k(query, k=5, persist_directory="db/chroma"):
    vectordb = Chroma(collection_name="my_index", persist_directory=persist_directory, embedding_function=emb)
    docs = vectordb.similarity_search(query, k=k)
    # Remove duplicate documents
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)
    return unique_docs
