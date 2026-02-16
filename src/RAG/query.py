import sys
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load env
load_dotenv()

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

def query_system_advanced():
    print("Loading RAG with Query Expansion...")
    
    # 1. Setup Standard Components
    # We use the standard vector store and LLM we used in eval.py
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=OpenAIEmbeddings())
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 2. Define the "Brainstorming" Prompt
    # This prompt asks the LLM to act as a search engine expert
    query_gen_prompt = ChatPromptTemplate.from_template("""
    You are a helpful research assistant.
    The user is asking a question about NLP, African Languages, or AI.
    Generate 3 specific search queries to help find the answer in a database of academic papers.
    Focus on technical keywords (e.g., "fine-tuning", "XLM-R", "data augmentation").
    
    User Question: {question}
    
    Output ONLY the 3 queries separated by newlines. No numbering.
    """)


    answer_prompt = ChatPromptTemplate.from_template("""
    You are an expert researcher. Synthesize the provided context to answer.
    
    CONTEXT:
    {context}
    
    QUESTION: 
    {question}
    
    INSTRUCTIONS:
    1. Answer clearly.
    2. If the user asks for "methods", look for specific techniques in the text.
    3. Cite sources like [source_id].
    """)

    print("\n RAG System Ready! Type 'exit' to quit.\n")

    while True:
        question = input("❓ Ask a complex question: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break

        print(f"   🧠 Brainstorming synonyms...")
        
    
        gen_chain = query_gen_prompt | llm
        search_queries_response = gen_chain.invoke({"question": question})
        # Split the response into a list of 3 strings
        search_queries = search_queries_response.content.strip().split('\n')
        
        # Add the original question too, just in case
        search_queries.append(question)
        
        # Clean up list (remove empty strings)
        search_queries = [q.strip() for q in search_queries if q.strip()]
        print(f"      -> Generated Queries: {search_queries}")


        unique_docs = {}
        for query in search_queries:
            # Run the retrieval for EACH query
            docs = retriever.invoke(query)
            for doc in docs:
                # Deduplicate: Don't add the same chunk twice
                # We create a unique key using the Source ID + the first 20 chars of text
                key = doc.metadata.get('source_id', 'unknown') + doc.page_content[:20]
                if key not in unique_docs:
                    unique_docs[key] = doc
        
        # Convert back to a list
        final_docs = list(unique_docs.values())
        print(f"      -> Found {len(final_docs)} unique relevant chunks.")

    
        context_text = "\n\n".join([f"[{doc.metadata.get('source_id', 'Unknown')}] {doc.page_content}" for doc in final_docs])
        
        chain = answer_prompt | llm
        response = chain.invoke({"context": context_text, "question": question})
        
        print(f"\n🟢 Answer: {response.content}\n")
        print("-" * 60)

if __name__ == "__main__":
    query_system_advanced()
