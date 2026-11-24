import getpass
import os
from ingestion import store_in_qdrant, load_document, chunking
from retrieval import retrieve
from generation import generation

GITHUB_RAW_URL = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/refs/heads/main/atliqai_hr_policies.txt"

def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        dl_meta  = doc.metadata.get("dl_meta", {})
        parts.append(f"[{i}]: \n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def rag(query: str, top_k: int = 5):
    print("**************Loading**************")
    document = load_document(GITHUB_RAW_URL)
    print("**************Chunking**************")
    chunks = chunking(document)
    print(f"Number of chunks: {len(chunks)} and storing data")
    vector_store = store_in_qdrant(chunks)
    print("**************Retrieval**************")
    docs = retrieve(vector_store, query, top_k)

    print("**************Building Context**************")
    context = format_docs(docs)
    print("**************Generation**************")
    answer = generation(context, query)
    return answer,context

def main() -> None:
    answer, context = rag("What are the main topics covered in this document?")
    print(answer)
    print(f"{250*'='}")
    print(f"\n\nSOURCES:\n {context}")
    

    



if __name__ == "__main__":
    main()
