import streamlit as st
import ollama
import chromadb
import json, re
from chromadb.config import Settings

def get_embedding(text: str, model: str = "nomic-embed-text"):
    r = ollama.embed(model=model, input=text)
    if isinstance(r, dict):
        if "embedding" in r and r["embedding"]: return r["embedding"]
        if "embeddings" in r and r["embeddings"]: return r["embeddings"][0]
    emb = getattr(r, "embedding", None)
    if emb: return emb
    embs = getattr(r, "embeddings", None)
    if embs: return embs[0]
    raise ValueError(f"Unexpected embed response format: {type(r)} -> {r}")

def chunk_text(text: str, chunk_words=240, overlap=40):
    words = re.findall(r"\S+", text or "")
    out, i, n = [], 0, len(words)
    while i < n:
        j = min(i + chunk_words, n)
        out.append(" ".join(words[i:j]))
        if j == n: break
        i = max(0, j - overlap)
    return out

def load_idoi_insurance_records(path, min_words=150, limit=None):
    keep_url = ["//www.in.gov/idoi/", "/idoi/", "insurance"]
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            url   = (rec.get("url") or "").lower()
            title = rec.get("title") or ""
            body  = rec.get("content") or ""
            if not any(k in url for k in keep_url):
                continue
            text = (title + "\n" + body).strip()
            if len(text.split()) < min_words:
                continue
            items.append({"url": rec.get("url",""), "title": title, "text": text})
            if limit and len(items) >= limit: break
    return items

DATA_PATH = r"C:\Users\19658\Downloads\genai\genai-hw1-document\out\indiana_consumer_corpus.jsonl"
records = load_idoi_insurance_records(DATA_PATH, min_words=150)
client = chromadb.PersistentClient(path="mydb_insurance/", settings=Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="idoi_insurance", metadata={"hnsw:space":"cosine"})

if collection.count() == 0:
    ids, docs, metas, embs = [], [], [], []
    idx = 0
    for rec in records:
        chunks = chunk_text(rec["text"], chunk_words=240, overlap=40)
        for ch in chunks:
            vec = get_embedding(ch, model="nomic-embed-text")
            ids.append(f"{idx}")
            docs.append(ch)
            metas.append({"title": rec["title"], "url": rec["url"]})
            embs.append(vec)
            idx += 1
            if len(ids) >= 64:
                collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                ids, docs, metas, embs = [], [], [], []
    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

INS_KWS = [
    "insurance","complaint","submit","file","claim","appeal","policy","insurer",
    "consumer services","consumer protection","coverage","denial","grievance",
    "department of insurance","IDOI","license","agent","producer","fraud","mediation"
]

def keyword_score(text: str, kws):
    t = text.lower()
    return sum(1 for k in kws if k in t)

def retrieve_insurance(prompt, k=5):
    q = get_embedding(prompt, model="nomic-embed-text")
    res = collection.query(query_embeddings=[q], n_results=max(k*4, 12),
                           include=["documents","metadatas","distances"])
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    packed, seen = [], set()
    for d,m in zip(docs, metas):
        url = (m or {}).get("url","")
        if url in seen:
            continue
        seen.add(url)
        packed.append({
            "text": d, "title": (m or {}).get("title",""),
            "url": url, "kw": keyword_score(d, INS_KWS)
        })
    packed.sort(key=lambda x: x["kw"], reverse=True)
    return packed[:k]


st.title("Indiana Insurance (IDOI) RAG Demo")
st.caption("Scope: Indiana Department of Insurance (IDOI) — complaints, claims, coverage, licensing, fraud.")
user_prompt = st.text_area("Enter an insurance-related question (Indiana):", height=180)
if st.button("Retrieve Context"):
    contexts = retrieve_insurance(user_prompt, k=5)
    if contexts:
        st.subheader("Relevant Context (IDOI / Insurance):")
        for c in contexts:
            st.markdown(f"**{c['title'] or '(no title)'}** — {c['url']}")
            st.write((c['text'][:1000] + " ...") if len(c['text'])>1000 else c['text'])
        st.subheader("Generated Response")
        ctx_block = "\n\n".join([f"[{i+1}] {c['title']}\n{c['url']}\n{c['text'][:1200]}" for i,c in enumerate(contexts)])
        resp = ollama.generate(model="hf.co/unsloth/Qwen3-8B-GGUF:UD-Q4_K_XL", prompt=f"Use the following Indiana consumer rights context to answer:\n{contexts}\n\nQuestion: {user_prompt}")
        st.write(resp.get("response", "No response generated"))
    else:
        st.error("No IDOI/insurance context found. Try a more insurance-specific query (e.g., 'file an insurance complaint to IDOI').")
