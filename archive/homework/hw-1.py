#!/usr/bin/env python3
# task2_1_easy.py
# Minimal RAG using your existing API key & URL (same style as lab2_first_api_query.py) python C:\Users\19658\Downloads\genai\hw-1.py --data "C:\Users\19658\Downloads\genai\genai-hw1-dacument\out\indiana_consumer_corpus.jsonl" --query "How do I submit an insurance complaint in Indiana?"
# Dependencies: pip install scikit-learn pandas requests

import os, json, argparse, requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ====== CONFIG ======
API_KEY = os.environ.get("LLM_API_KEY", "")
API_URL = "http://localhost:11434/api/generate"
MODEL   = "qwen3-8b-gguf"

def load_jsonl(path, min_words=80, limit=None):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = (rec.get("title","") + "\n" + (rec.get("content","") or "")).strip()
            if len(txt.split()) < min_words:
                continue
            items.append({"url": rec.get("url",""), "title": rec.get("title",""), "text": txt})
            if limit and len(items) >= limit:
                break
    return items

def top_k_context(corpus, query, k=3):
    if not corpus:
        return []
    texts = [c["text"] for c in corpus]
    vec = TfidfVectorizer(stop_words="english", max_df=0.9)
    X = vec.fit_transform(texts)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X)[0]
    idxs = sims.argsort()[::-1][:k]
    return [corpus[i] for i in idxs]

def call_model(api_url, api_key, model, user_prompt, contexts):
    if not contexts:
        ctx_text = "No Indiana.gov context retrieved."
    else:
        ctx_text = "\n\n".join([f"[{i+1}] {c['title']}\n{c['url']}\n{c['text'][:1200]}" for i, c in enumerate(contexts)])

    prompt = (
        "You are a helpful assistant that ONLY answers Indiana consumer-rights questions "
        "(insurance complaints, utilities, scams, banking/credit). Ground answers in the CONTEXT. "
        "If not found in context, say you cannot find an Indiana.gov source and suggest the right agency page.\n\n"
        f"QUESTION:\n{user_prompt}\n\nCONTEXT:\n{ctx_text}\n\n"
        "Answer concisely and include 2–3 citations with title + URL from context. "
        "Finish with: 'Disclaimer: This is general information, not legal advice.'"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter 推荐的附加头
        "HTTP-Referer": "http://localhost",
        "X-Title": "IU-HW1-Indiana-Consumer-Copilot"
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.2
    }
    r = requests.post(api_url, headers=headers, json=body, timeout=60)
    try:
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        # 打印原始响应，方便定位（很关键）
        return f"[API Error] status={r.status_code}, text={r.text[:500]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to indiana_consumer_corpus.jsonl or clean_corpus_min.jsonl")
    ap.add_argument("--query", required=True, help="Your question")
    ap.add_argument("--limit", type=int, default=6000, help="Max docs to load for TF-IDF")
    ap.add_argument("--k", type=int, default=3, help="Top-k contexts")
    args = ap.parse_args()

    corpus = load_jsonl(args.data, min_words=80, limit=args.limit)
    if not corpus:
        print("No documents loaded. Check your --data path and filters.")
        return

    contexts = top_k_context(corpus, args.query, k=args.k)
    print(f"Retrieved {len(contexts)} contexts:")
    for c in contexts:
        print(f"- {c['title']} — {c['url']}")

    # 优先使用环境变量的 Key（更安全）
    api_key = os.getenv("MY_API_KEY") or API_KEY
    api_url = os.getenv("MY_API_URL") or API_URL

    answer = call_model(api_url, api_key, MODEL, args.query, contexts)
    print("\n=== Answer ===\n")
    print(answer)

if __name__ == "__main__":
    main()
