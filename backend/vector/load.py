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
    all_documents, url_to_title, title_to_chunks, new_docs = load_all_documents()

    logger.info(f"Found {len(new_docs)} new documents to add to vector store")

    # Keeping track of all document names we have for the future (may not be needed)
    with open('doc_names.pkl', 'wb') as doc_names:
        pickle.dump(all_documents, doc_names)

    with open('url_to_title.pkl', 'wb') as u2t:
        pickle.dump(url_to_title, u2t)

    with open('title_to_chunks.pkl', 'wb') as t2c:
        pickle.dump(title_to_chunks, t2c)

    def chunk_documents(docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
        chunked_docs = text_splitter.split_documents(docs)
        return chunked_docs
    
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