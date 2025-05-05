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
# we need to make it so that when we add new documents, it adds them to the store.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_env()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

FAISS_STORE_PATH = "faiss_store"

# adds all docs and videos to the vector store
def add_all_documents_to_store(vector_store):

    all_documents, url_to_title, title_to_chunks, new_docs = load_all_documents()

    # Keeping track of all document names we have for the future (may not be needed)
    with open('doc_names.pkl', 'wb') as doc_names:
        pickle.dump(all_documents, doc_names)

    with open('url_to_title.pkl', 'wb') as u2t:
        pickle.dump(url_to_title, u2t)

    def chunk_documents(docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
        chunked_docs = text_splitter.split_documents(docs)
        return chunked_docs
    new_chunks=[chunk for doc in new_docs for chunk in title_to_chunks[doc]]
    chunked_docs = chunk_documents(new_chunks) if len(new_chunks)>0 else []
    if len(chunked_docs)>0:
        vector_store.add_documents(documents = chunked_docs)
    vector_store.save_local(FAISS_STORE_PATH)

    return chunked_docs

def get_vector_store():
    if os.path.exists(FAISS_STORE_PATH):
        print('Store found and initialized')
        vector_store = FAISS.load_local(
            FAISS_STORE_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        add_all_documents_to_store(vector_store)
    else:
        print('Store not found')
        vector_store = FAISS(embedding_function=embedding_model, 
                        index = faiss.IndexFlatIP(3072),  # because we use text-embedding-3-large -> 3072
                        docstore = InMemoryDocstore({}), 
                        index_to_docstore_id={}
        ) 
        add_all_documents_to_store(vector_store)
    return vector_store

vector_store = get_vector_store()