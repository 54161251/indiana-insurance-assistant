import streamlit as st
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
PERSIST_DIR = "mydb_insurance"        
COLLECTION_NAME = "idoi_insurance"      
INS_KWS = [
    "insurance", "complaint", "submit", "file", "claim", "appeal", "policy",
    "insurer", "consumer services", "consumer protection", "coverage",
    "denial", "grievance", "department of insurance", "idoi",
    "license", "agent", "producer", "fraud", "mediation"
]
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def get_llm(mode: str = "normal") -> ChatOllama:
    """Return a ChatOllama LLM with different temperature modes."""
    if mode == "strict":
        temp = 0.1
    elif mode == "creative":
        temp = 0.8
    else:
        temp = 0.4

    return ChatOllama(
        model="huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF:latest",  
        temperature=temp,
    )

SYSTEM_PROMPT = """You are an Indiana consumer and insurance assistant.
Your primary job is to help users understand Indiana insurance and consumer rights
(complaints, appeals, fraud, licensing, coverage) using the provided context.

You MUST:
1. First, briefly break the user question into 2–3 subtasks or key issues.
2. Then answer in clearly labeled sections:
   - Overview
   - Step-by-step guidance
   - Relevant Indiana resources (with titles + URLs if available)
   - Disclaimer

3. When context does not contain the answer, say so clearly and suggest contacting
   the Indiana Department of Insurance or relevant agency.

You are NOT a lawyer and do not give legal advice. Be clear, accurate, and kind.
"""

def build_insurance_prompt(question: str, context: str, chat_history: str) -> str:
    """Construct the full prompt for the insurance (RAG) assistant."""
    return f"""{SYSTEM_PROMPT}

Previous conversation:
{chat_history}

Retrieved context from Indiana insurance documents:
{context}

User question:
{question}

Follow the instructions above.
"""

def build_general_prompt(question: str, chat_history: str) -> str:
    """Prompt for general-purpose non-insurance chat."""
    return f"""You are a helpful, general-purpose assistant.
The user may ask about any topic (school, coding, daily life, etc.).
Be clear, friendly, and honest.

Previous conversation:
{chat_history}

User: {question}
Assistant:"""

def is_insurance_question(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in INS_KWS)

st.set_page_config(page_title="Indiana Insurance Assistant")
st.title("Indiana Insurance & General Chat Assistant")
st.caption("Local LLM + RAG over Indiana Department of Insurance documents (which I used a script to collect)")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mode" not in st.session_state:
    st.session_state.mode = "normal"
mode = st.radio(
    "Answer style:",
    options=["strict", "normal", "creative"],
    index=["strict", "normal", "creative"].index(st.session_state.mode),
    horizontal=True,
)
st.session_state.mode = mode
user_input = st.text_area("Ask a question (insurance or anything):", height=120)
if st.button("Send"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        history_str = ""
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            history_str += f"{role.upper()}: {content}\n"
        llm = get_llm(mode)
        sources = []
        if is_insurance_question(user_input):
            docs = retriever.invoke(user_input)   
            context = "\n\n".join(d.page_content for d in docs)
            prompt = build_insurance_prompt(user_input, context, history_str)

            resp = llm.invoke(prompt)
            answer = resp.content if hasattr(resp, "content") else str(resp)
            sources = docs
        else:
            prompt = build_general_prompt(user_input, history_str)
            resp = llm.invoke(prompt)
            answer = resp.content if hasattr(resp, "content") else str(resp)

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
        st.success("Assistant response generated!")

st.subheader("Conversation")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")
        st.markdown("========================================================================================" \
                    "========================================================================================" \
                    "========================================================================================")  

