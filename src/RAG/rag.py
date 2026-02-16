import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

# 1. Setup Database Connection
vector_store = Chroma(
    persist_directory=DB_PATH, 
    embedding_function=OpenAIEmbeddings()
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Get top 5 chunks

# 2. Setup LLM 
llm = ChatOpenAI(model="gpt-4o", temperature=0)


PROMPT_TEMPLATE = """
You are a Research Assistant. Use ONLY the provided context to answer the question.

CONTEXT:
{context}

QUESTION: 
{question}

INSTRUCTIONS:
1. Extract relevant chunks and quotes from the context.
2. If the context does not contain the answer, say "Insufficient Evidence" and give proper reasons behind it.
3. Cite your sources using the [source_id] provided in the metadata.

OUTPUT FORMAT (JSON):
{{
  "answer": "Your answer here...",
  "citations": ["source_01", "source_05"]
}}
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def query_rag(question):
    # A. Retrieve
    docs = retriever.invoke(question)
    
    # B. Format Context
    context_text = "\n\n".join([
        f"[{doc.metadata['source_id']}] {doc.page_content}" 
        for doc in docs
    ])
    
    # C. Generate Answer
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})
    
    return response.content, docs

if __name__ == "__main__":
    q = input("🔎 Enter a research question: ")
    answer, sources = query_rag(q)
    print("\n🤖 AI RESPONSE:\n" + answer)