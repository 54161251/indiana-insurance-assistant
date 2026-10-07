import json
with open("C:/Users/19658/Downloads/genai/genai-hw1-document/out/indiana_consumer_corpus.jsonl","r",encoding="utf-8") as f, \
        open("C:/Users/19658/Downloads/genai/genai-hw1-document/out/indiana_consumer_clean.jsonl","w",encoding="utf-8") as out:
    for line in f:
        rec = json.loads(line)
        url = rec.get("url","")
        if any(key in url for key in ["consumer","idoi","attorneygeneral"]):
            out.write(line)