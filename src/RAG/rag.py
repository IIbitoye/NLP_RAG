import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
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
retriever = vector_store.as_retriever(search_kwargs={"k": 15})

# 2. Setup Primary and Fallback LLMs 
# max_retries=0 ensures it fails fast and swaps to Ollama instantly if the API is down
primary_llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=2)
fallback_llm = ChatOllama(model="llama3.2", temperature=0)

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
 "answer": "Your answer...",
  "citations": ["source_01", "source_02", "source_03", "source_04"]
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
    
    # C. Generate Answer with Fallback Logic
    try:
        # Try OpenAI first
        chain = prompt | primary_llm
        response = chain.invoke({"context": context_text, "question": question})
    except Exception as e:
        # If OpenAI fails, print a warning and hot-swap to Ollama
        print(f"\n⚠️ OpenAI API Error: {e}")
        print("🔄 Hot-swapping to Local Llama 3.2 Backup Model...\n")
        chain = prompt | fallback_llm
        response = chain.invoke({"context": context_text, "question": question})
    
    return response.content, docs

if __name__ == "__main__":
    q = input("🔎 Enter a research question: ")
    answer, sources = query_rag(q)
    print("\n🤖 AI RESPONSE:\n" + answer)