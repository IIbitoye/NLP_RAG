import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime
import sys

# --- IMPORT YOUR EXISTING PHASE 2 LOGIC ---
sys.path.append(os.path.abspath("src/eval"))
from eval import run_query

st.set_page_config(page_title="Personal Research Portal", page_icon="🌍", layout="wide")

HISTORY_FILE = os.path.join("outputs", "chat_history.json")

# Load history from file on startup
if 'history' not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            st.session_state.history = json.load(f)
    else:
        st.session_state.history = []

if 'current_session' not in st.session_state:
    st.session_state.current_session = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌍 Research Portal")
    st.markdown("**Domain:** African Languages & Low-Resource NLP")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🔍 Search & Synthesize", 
        "📚 Research History", 
        "📊 Export Artifacts",
        "📈 Evaluation Metrics" # <-- NEW PAGE
    ])
    
    if st.button("🗑️ Clear History", width='stretch'):
        st.session_state.history = []
        st.success("History cleared!")

    if st.button("💾 Save Thread to File", width='stretch'):
        if st.session_state.history:
            thread_path = os.path.join("outputs", f"research_thread_{datetime.now().strftime('%H%M%S')}.json")
            with open(thread_path, "w") as f:
                json.dump(st.session_state.history, f, indent=4)
            st.success(f"Thread saved to {thread_path}!")
        else:
            st.warning("History is empty.")

# --- MAIN PAGE: SEARCH ---
if page == "🔍 Search & Synthesize":
    st.title("Ask the Corpus")
    st.caption("Chat with academic papers on Low-Resource NLP.")

    # RENDERS PREVIOUS CHAT HISTORY ---
    for item in st.session_state.current_session:
        with st.chat_message("user"):
            st.write(item["query"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            with st.expander("View Citations"):
                st.write(", ".join(item.get("citations", [])))

    query = st.chat_input("e.g. What are the main findings of the AfroBench paper?")
    
    if query:
        with st.chat_message("user"):
            st.write(query)
            
        with st.chat_message("assistant"):
            with st.status("🧠 Consulting the Research Corpus...", expanded=True) as status:
                st.write("🔍 Vectorizing query...")
                st.write("📚 Searching ChromaDB...")
                result = run_query(query)
                status.update(label="✅ Synthesis Complete!", state="complete", expanded=False)
            
            # --- TRUST BEHAVIOR: MISSING EVIDENCE HANDLING ---
            if "Insufficient" in result["answer"] or "Error" in result["answer"]:
                st.warning("⚠️ **Missing Evidence Detected:** The corpus does not contain enough information to fully answer this.")
                st.info("💡 **Suggested Next Retrieval Step:** Try broadening your keywords, or check the `data_manifest.csv` to ensure papers on this specific topic are ingested.")
            else:
                st.markdown("### 📝 Synthesized Answer")
                st.success(result["answer"])
            
            st.divider()
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 📚 Citations")
                if result.get("citations_readable"):
                    for cite in result["citations_readable"]:
                        st.markdown(f"- `{cite}`")
                else:
                    st.warning("No explicit citations.")
                    
            with col2:
                st.markdown("### 🔎 Top Evidence")
                for i, chunk in enumerate(result["retrieved_chunks"][:5]):
                    with st.expander(f"Snippet {i+1}: {chunk.get('citation', 'Source')}"):
                        st.write(chunk['text_snippet'])

            # --- ARTIFACT: SYNTHESIS MEMO EXPORT ---
            memo_content = f"# Synthesis Memo\n**Query:** {query}\n**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n## Answer\n{result['answer']}\n\n## References\n"
            for cite in result.get("citations_readable", []):
                memo_content += f"- {cite}\n"
                
            st.download_button(
                label="📥 Download Synthesis Memo (Markdown)",
                data=memo_content,
                file_name=f"Synthesis_Memo_{datetime.now().strftime('%H%M%S')}.md",
                mime="text/markdown",
                width='stretch'
            )

            new_entry = {
                "query": query,
                "answer": result["answer"],
                "citations": result.get("citations_readable", []),
                "chunks": result["retrieved_chunks"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add to global background memory
            st.session_state.history.append(new_entry)
            
            # Add to the active screen memory
            st.session_state.current_session.append(new_entry)
            
            # Save to physical file immediately
            with open(HISTORY_FILE, "w") as f:
                json.dump(st.session_state.history, f, indent=4)

# --- HISTORY PAGE ---
elif page == "📚 Research History":
    st.title("Research Threads")
    if not st.session_state.history:
        st.info("Your research history is empty.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Query: {item['query']}"):
                st.write(f"**Time:** {item['timestamp']}")
                st.write(f"**Answer:** {item['answer']}")
                st.write("**Sources:**", ", ".join(item.get('citations', [])))

# --- ARTIFACT GENERATOR PAGE ---
elif page == "📊 Export Artifacts":
    st.title("Generate Research Artifacts")
    st.markdown("Convert your search history into structured artifacts.")
    
    if not st.session_state.history:
        st.warning("Ask a question first to generate artifacts.")
    else:
        # Create beautiful UI tabs!
        tab1, tab2, tab3 = st.tabs(["📊 Evidence Table", "📚 Annotated Bibliography", "📝 Synthesis Memo"])
        
        # --- TAB 1: EVIDENCE TABLE ---
        with tab1:
            st.markdown("### Evidence Table")
            artifact_data = []
            
            for item in st.session_state.history:
                if "Insufficient" in item['answer'] or "Error" in item['answer']:
                    continue
                
                # Loop through the top chunk used for this claim
                if item.get('chunks'):
                    chunk = item['chunks'][0]
                    
                    # Grab the raw text and clean out the weird PDF newlines/spaces
                    raw_snippet = chunk.get('text_snippet', 'N/A')
                    clean_snippet = " ".join(raw_snippet.split())
                    
                    # Give it a nice clean cutoff at 250 characters
                    snippet = clean_snippet if len(clean_snippet) <= 250 else clean_snippet[:247].strip() + "..."
                    
                    s_id = chunk.get('source_id', 'unknown_source')

                    # Extract the actual claim (The first sentence of the AI's generated answer)
                    full_answer = item.get('answer', 'No answer generated.')
                    extracted_claim = full_answer.split('.')[0].strip() + "." if '.' in full_answer else full_answer
                    
                    # Append to the table
                    artifact_data.append({
                        "Claim": extracted_claim,  # <--- FIXED: Now it states a fact, not a question!
                        "Evidence snippet": snippet,
                        "Citation (source_id, chunk_id)": f"({s_id}, chunk_0)",
                        "Confidence": "High",
                        "Notes": "Extracted from top MMR retrieval result."
                    })
                
            if artifact_data:
                df = pd.DataFrame(artifact_data)
                st.dataframe(df, width='stretch')
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Evidence Table (CSV)",
                    data=csv,
                    file_name="Evidence_Table_Rubric.csv",
                    mime="text/csv",
                    type="primary",
                    width='stretch'
                )
            else:
                st.info("No valid claims to export yet. Ask some questions on the Search page!")

        # --- TAB 2: ANNOTATED BIBLIOGRAPHY ---
        with tab2:
            st.markdown("### Annotated Bibliography (APA Style)")
            st.markdown("Compiles up to 12 unique sources from your session history into a structured, academic format using LLM synthesis.")
            
            # We add a button here because doing 12 LLM calls takes a few seconds!
            if st.button("🤖 Generate Annotated Bibliography", type="primary", width='stretch'):
                with st.spinner("Synthesizing academic bibliography... this takes about 10-15 seconds..."):
                    
                    # Gather unique chunks and link them to the query they answered
                    unique_chunks = {}
                    for item in st.session_state.history:
                        query_context = item['query']
                        for chunk in item['chunks']:
                            cite = chunk.get('citation', 'Unknown Citation')
                            # Filter out duplicates and ugly raw filenames
                            if cite not in unique_chunks and cite != 'Unknown Citation' and not cite.endswith(".pdf"):
                                unique_chunks[cite] = {
                                    'snippet': chunk.get('text_snippet', ''),
                                    'query': query_context
                                }
                    
                    if unique_chunks:
                        from langchain_openai import ChatOpenAI
                        from langchain_core.prompts import ChatPromptTemplate
                        
                        # Initialize a dedicated LLM chain just for writing the bibliography
                        biblio_llm = ChatOpenAI(model="gpt-4o", temperature=0)
                        biblio_prompt = ChatPromptTemplate.from_template(
                            "You are a PhD-level research assistant. Write an annotated bibliography entry for the following source snippet.\n"
                            "The user's original query was: {query}\n"
                            "The text snippet from the paper is: {snippet}\n\n"
                            "You MUST structure your response EXACTLY with these 4 bullet points. Do not write introductory or concluding sentences. Keep each point to 3-4 concise sentences:\n"
                            "- **Claim:** [The core claim or finding]\n"
                            "- **Method:** [The method used, inferred from context if necessary]\n"
                            "- **Limitations:** [The limitations of the approach]\n"
                            "- **Why it Matters:** [Why this matters to the field of Low-Resource African NLP]\n\n"
                            "Do not hallucinate external facts."
                        )
                        biblio_chain = biblio_prompt | biblio_llm

                        biblio_md = "# Annotated Bibliography\n\n"
                        biblio_md += f"**Topic:** Low-Resource NLP for African Languages\n"
                        biblio_md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                        biblio_md += "---\n\n"
                        
                        # Limit to 12 sources to hit the rubric requirement exactly
                        sources_to_use = list(unique_chunks.items())[:12]
                        
                        for cite, data in sources_to_use:
                            try:
                                # Ask the LLM to write the paragraph
                                response = biblio_chain.invoke({
                                    "query": data['query'],
                                    "snippet": data['snippet']
                                })
                                annotation = response.content.strip()
                                biblio_md += f"**{cite}**\n\n{annotation}\n\n"
                            except Exception as e:
                                biblio_md += f"**{cite}**\n\nError generating annotation: {e}\n\n"
                        
                        st.success("✅ Bibliography generated successfully!")
                        st.download_button(
                            label="📥 Download APA Bibliography (Markdown)",
                            data=biblio_md,
                            file_name=f"Annotated_Bibliography_{datetime.now().strftime('%H%M%S')}.md",
                            mime="text/markdown",
                            width='stretch'
                        )
                        
                        with st.expander("Preview Academic Formatting", expanded=True):
                            st.markdown(biblio_md)
                    else:
                        st.info("You need to ask a few more questions to gather enough unique sources!")
        with tab3:
            st.markdown("### Long-Form Synthesis Memo")
            st.markdown("Synthesizes your entire research thread into an 800–1200 word essay with inline citations and a reference list.")
            
            if st.button("📝 Draft Synthesis Memo", type="primary", width='stretch'):
                with st.spinner("Drafting 800+ word synthesis... this takes about 30-45 seconds..."):
                    
                    # Gather all the findings from the session
                    full_context = ""
                    unique_refs = set()
                    for item in st.session_state.history:
                        if "Insufficient" not in item['answer']:
                            full_context += f"Query: {item['query']}\nFindings: {item['answer']}\n\n"
                            for cite in item.get('citations', []):
                                unique_refs.add(cite)
                                
                    if len(unique_refs) > 0:
                        from langchain_openai import ChatOpenAI
                        from langchain_core.prompts import ChatPromptTemplate
                        
                        # Use GPT-4o to write the long-form essay
                        memo_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
                        memo_prompt = ChatPromptTemplate.from_template(
                            "You are a graduate-level AI researcher writing a formal Synthesis Memo.\n"
                            "Base your memo strictly on the following synthesized notes from a research session:\n"
                            "{notes}\n\n"
                            "References available to cite: {refs}\n\n"
                            "REQUIREMENTS:\n"
                            "1. Your response MUST be between 800 and 1200 words.\n"
                            "2. Synthesize the findings into a cohesive narrative with clear academic headings (e.g., Introduction, Core Findings, Methodological Challenges, Conclusion).\n"
                            "3. You MUST include frequent inline citations using the exact references provided.\n"
                            "4. End the document with a formally formatted 'References' list based on the provided citations.\n"
                            "Do not hallucinate external facts."
                        )
                        
                        try:
                            response = (memo_prompt | memo_llm).invoke({
                                "notes": full_context,
                                "refs": ", ".join(unique_refs)
                            })
                            
                            memo_md = response.content.strip()
                            
                            st.success("✅ Synthesis Memo generated successfully!")
                            st.download_button(
                                label="📥 Download Synthesis Memo (Markdown)",
                                data=memo_md,
                                file_name=f"Synthesis_Memo_{datetime.now().strftime('%H%M%S')}.md",
                                mime="text/markdown",
                                width='stretch'
                            )
                            
                            with st.expander("Preview Memo", expanded=True):
                                st.markdown(memo_md)
                                
                        except Exception as e:
                            st.error(f"Generation failed: {e}")
                    else:
                        st.warning("Your research history is too short. Ask a few more questions to generate an 800-word memo!")

# --- EVALUATION PAGE ---
elif page == "📈 Evaluation Metrics":
    st.title("System Evaluation")
    st.markdown("Metrics generated from the 20-query test set (`eval.py`).")
    
    # Path to your NEW CSV file
    eval_path = os.path.join("outputs", "evaluation_grading_sheet2.csv")
    
    if os.path.exists(eval_path):
        # Use pandas to read the CSV
        df_eval = pd.read_csv(eval_path)
            
        st.success(f"Successfully loaded {len(df_eval)} test queries.")
        
        # --- NEW DASHBOARD LAYOUT (4 Columns!) ---
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="Total Queries", value=len(df_eval))
            
        with col2:
            # Safely calculate average latency
            if "Latency_Seconds" in df_eval.columns and not df_eval["Latency_Seconds"].isnull().all():
                avg_time = df_eval["Latency_Seconds"].mean()
                st.metric(label="Avg Latency", value=f"{avg_time:.2f}s")
            else:
                st.metric(label="Avg Latency", value="N/A")
                
        with col3:
            # Safely calculate Groundedness
            if "Score_1_Groundedness_1_to_4" in df_eval.columns and not df_eval["Score_1_Groundedness_1_to_4"].isnull().all():
                avg_ground = df_eval["Score_1_Groundedness_1_to_4"].mean()
                st.metric(label="Avg Groundedness", value=f"{avg_ground:.2f}/4")
            else:
                st.metric(label="Avg Groundedness", value="Not Graded")
                
        with col4:
            # Safely calculate Citation
            if "Score_2_Citation_1_to_4" in df_eval.columns and not df_eval["Score_2_Citation_1_to_4"].isnull().all():
                avg_cite = df_eval["Score_2_Citation_1_to_4"].mean()
                st.metric(label="Avg Citation", value=f"{avg_cite:.2f}/4")
            else:
                st.metric(label="Avg Citation", value="Not Graded")
            
        st.divider()
        st.markdown("### Detailed Run Logs & Grading Sheet")
        
        # Display the dataframe with the scores!
        st.dataframe(df_eval, use_container_width=True)
    else:
        st.warning(f"Could not find {eval_path}.")
        st.info("Make sure you have run your `eval.py` script to generate the CSV file in the `outputs/` folder.")