import logging
import faiss
import os
from document.loader import load_all_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from load_env import load_env
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_env()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

FAISS_STORE_PATH = "faiss_store"

def add_all_documents_to_store(vector_store):
    """Add all new documents to the vector store"""
    logger.info("Loading all documents and processing for vector store...")
    
    # Load existing doc_names and append new ones
    if os.path.exists('doc_names.pkl'):
        with open('doc_names.pkl', 'rb') as f:
            existing_doc_names = pickle.load(f)
    else:
        existing_doc_names = set()
    
    # Process new documents
    url_to_title, title_to_chunks, new_docs = load_all_documents()
    logger.info(f"Found {len(new_docs)} new documents to add to vector store")

    # Update doc_names
    all_doc_names = existing_doc_names.union(new_docs)
    
    with open('doc_names.pkl', 'wb') as f:
        pickle.dump(all_doc_names, f)
    
    with open('url_to_title.pkl', 'wb') as u2t:
        pickle.dump(url_to_title, u2t)

    with open('title_to_chunks.pkl', 'wb') as t2c:
        pickle.dump(title_to_chunks, t2c)

    def chunk_documents(docs):
        youtube_chunks = []
        other_chunks = []
        
        for doc in docs:
            if doc.metadata.get('type') == 'youtube':
                youtube_chunks.append(doc)
            else:
                other_chunks.append(doc)
        
        # Re-chunk PDFs and other documents
        if other_chunks:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
            other_chunked = text_splitter.split_documents(other_chunks)
        else:
            other_chunked = []
        
        # YouTube chunks are already properly sized for time-based retrieval
        return youtube_chunks + other_chunked
    
    new_chunks = [chunk for doc in new_docs for chunk in title_to_chunks[doc]]
    logger.info(f"Processing {len(new_chunks)} chunks from new documents")
    
    chunked_docs = chunk_documents(new_chunks) if len(new_chunks) > 0 else []
    
    if len(chunked_docs) > 0:
        logger.info(f"Adding {len(chunked_docs)} chunked documents to vector store")
        vector_store.add_documents(documents=chunked_docs)
        logger.info("Successfully added documents to vector store")
    else:
        logger.info("No new documents to add to vector store")
    
    logger.info("Saving vector store to disk...")
    vector_store.save_local(FAISS_STORE_PATH)
    logger.info("Vector store saved successfully")

    return chunked_docs

def get_vector_store():
    """Get or create the vector store"""
    if os.path.exists(FAISS_STORE_PATH):
        logger.info('Existing vector store found, loading and updating...')
        vector_store = FAISS.load_local(
            FAISS_STORE_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        add_all_documents_to_store(vector_store)
    else:
        logger.info('No existing vector store found, creating new one...')
        vector_store = FAISS(
            embedding_function=embedding_model, 
            index=faiss.IndexFlatIP(3072),  # because we use text-embedding-3-large -> 3072
            docstore=InMemoryDocstore({}), 
            index_to_docstore_id={}
        ) 
        add_all_documents_to_store(vector_store)
    
    logger.info("Vector store initialization complete")
    return vector_store

vector_store = get_vector_store()