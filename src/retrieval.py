
def retrieve(vectorStore, query: str, top_k:int =5) :
    retriever = vectorStore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query) 
    return docs