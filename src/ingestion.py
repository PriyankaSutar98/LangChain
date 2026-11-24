import os
import requests
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

# Loading Document
def load_document(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    docs = [Document(page_content=response.text, metadata={"source": url})]
    return docs
    #  response.raise_for_status() works same as this code if response.status_code == 200:
     #   return response.text
   # else:
     #   raise Exception(f"Failed to load document from {url}. Status code: {response.status_code}")



#Chunking the document into smaller pieces
def chunking(doc, chunk_size: int = 50) -> list[dict]:
    print("Chunking the document...")
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=0)
    documents = text_splitter.split_documents(doc)
    return documents


#Embedding the chunks not required explicityly from_documents does it


#Storing the chunks and embeddings in Qdrant (Vector Database) 
def store_in_qdrant(document_text):
    COLLECTION_NAME = "hr_docs"
    print("Document loaded successfully.")
    
     # Chunk the document into smaller pieces
    chunks = chunking(document_text, chunk_size=50)
    
    vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedder,
    url="http://localhost:6333",
    collection_name=COLLECTION_NAME,
    )
    return vectorstore
    