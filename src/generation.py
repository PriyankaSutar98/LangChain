from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv() 

GROQ_MODEL  = "openai/gpt-oss-20b"


def generation(context, question)-> str:
     llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    )
     RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the context below. Cite section names. Say 'I don't know' if unsure."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
     promp_value = RAG_PROMPT.invoke({"context": context, "question": question})
     response = llm.invoke(promp_value)
     return response.content