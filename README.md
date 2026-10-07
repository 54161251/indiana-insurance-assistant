# Indiana Insurance & General Chat Assistant

A Streamlit chatbot using a local Ollama model and LangChain. The most complete supplied version, `project.zip/project/final.py`, routes questions to Indiana insurance document retrieval, weather lookup (Open-Meteo), US ZIP-code lookup (Zippopotam.us), or a local date-window calculator. Questions can also be answered without a tool. The router produces JSON tool calls; the app executes them and asks the model to compose a structured response.

## Run

Use Python 3.11 or 3.12 and install Ollama separately. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
python scripts/restore_database.py
ollama pull nomic-embed-text
ollama pull huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF:latest
.\.venv\Scripts\python.exe -m streamlit run final.py
```

Ollama must be running locally. The model identifier above is preserved from the author's code; if unavailable, select an available equivalent and update `DEFAULT_MODEL` in `final.py` and the archived midterm if needed. Insurance retrieval uses `nomic-embed-text`. Weather and ZIP lookup require internet access. The main app does not require an API key.

## Contents

| Path | Purpose |
| --- | --- |
| `final.py` | Final tool-routing app, preserved byte-for-byte |
| `archive/mid.py` | Earlier insurance/general chatbot; passes conversation history to the model |
| `archive/homework/` | Earlier retrieval prototypes and corpus filtering script |
| `examples/`, `notebooks/` | Related API, local-model, prompting, and memory exercises |
| `docs/MidTerm.pdf` | Author's midterm project report; describes the earlier app |
| `docs/PROJECT_NOTES.md` | Version comparison, limitations, and packaging changes |
| `data/` | Original database snapshot split into parts below 25 MiB, with checksum |
| `scripts/restore_database.py` | Packaging helper that verifies and restores the snapshot |

The snapshot contains 2,368 records in `idoi_insurance` and 361 in a separate `docs` collection. These are stored records/chunks, not a verified count of distinct source documents. Metadata includes titles and URLs. The raw collection/crawling pipeline was not supplied in the archives. The restore helper preserves the original database and vectors rather than rebuilding them.

## Behavior and limits

- Answer styles set temperature to 0.1 (strict), 0.4 (normal), or 0.8 (creative).
- The final app displays session history but does **not** send earlier turns to its router or answer model. The archived midterm does include history in its prompts.
- Insurance retrieval requests four records. The final citation helper expects `source`/`file_path`, but the supplied insurance records use `title`/`url`; citations can show `unknown_source`.
- Model-produced JSON is parsed but not schema-validated. Malformed routing responses, service outages, and missing models can affect execution.
- Deadline calculations add user-specified calendar days; they do not determine legally applicable filing periods.
- The report's prompt-engineering claims are the author's description, not independent performance results. No accuracy benchmark is supplied.

## Dependencies and security

`requirements.txt` covers the main app and midterm with the legacy LangChain API family used in the code. The original version lock was not supplied. Historical exercises additionally use `ollama`, `scikit-learn`, `langchain-openai`, `pydantic`, Jupyter/IPython, and `tqdm`; some notebook APIs may require older versions. Historical scripts retain their original local corpus paths and need path adjustment and the relevant data to run.

Hard-coded credentials in two historical scripts were replaced with environment-variable lookups. Notebook outputs were cleared. `.env`, restored databases, model caches, and local secrets are ignored. Any previously exposed credentials should be revoked by their owner. No license was added because the author has not selected one.

The main code and database were preserved. Packaging, documentation, and credential cleanup were prepared with Codex assistance. Syntax, snapshot integrity, and secret-pattern checks were performed; the full app has not been run against Ollama or live APIs in this environment.
