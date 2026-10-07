
import streamlit as st
import ollama
import chromadb
import json

def load_jsonl(path, min_words=80, limit=None):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = (rec.get("title","") + "\n" + (rec.get("content","") or "")).strip()
            if len(txt.split()) < min_words:
                continue
            items.append(txt)
            if limit and len(items) >= limit:
                break
    return items

DATA_PATH = "C:/Users/19658/Downloads/genai/genai-hw1-document/out/indiana_consumer_corpus.jsonl"
documents = load_jsonl(DATA_PATH)
client = chromadb.PersistentClient(path="mydb_insurance/")
collection = client.get_or_create_collection(name="docs")

if collection.count() == 0:
    for i, doc in enumerate(documents):
        response = ollama.embed(model="nomic-embed-text", input=doc)
        embeddings = response["embeddings"]
        collection.add(ids=[str(i)],
                       embeddings=embeddings,
                       documents=[doc])

def get_relevant_context(prompt):
    prompt_emb = ollama.embed(model="nomic-embed-text", input=prompt)["embeddings"]
    results = collection.query(query_embeddings=prompt_emb, n_results=5)
    return results['documents'][0] if results and 'documents' in results else []

# Streamlit UI
st.title("Indiana Insurance (IDOI) RAG Demo")
user_prompt = st.text_area("Enter a question about Indiana consumer rights:", height=200)

if st.button("Retrieve Context"):
    contexts = get_relevant_context(user_prompt)
    if contexts:
        st.subheader("Relevant Context:")
        for c in contexts:
            st.write(c[:1000])
        st.subheader("Generated Response:")
        response = ollama.generate(
            model="hf.co/unsloth/Qwen3-8B-GGUF:UD-Q4_K_XL",
            prompt=f"Use the following Indiana consumer rights context to answer:\n{contexts}\n\nQuestion: {user_prompt}"
        )
        st.write(response.get('response', 'No response generated'))
    else:
        st.write("No relevant context found.")
