# Version selection and cleanup

`project.zip` is the most complete supplied application: `final.py` adds JSON-based tool routing and weather, ZIP-code, and date-window tools to the local insurance retrieval app. `midterm.zip/midterm/mid.py` is identical to the standalone `mid.py`. Its prompts include previous conversation turns and choose insurance retrieval using keyword matching. Earlier homework scripts show progressively more specialized retrieval; they are retained as historical material, not used as the main entry point.

`MidTerm.pdf` is the author's report about the midterm implementation. `MidtermProject_v0 (1).pdf` is an assignment specification, not a project report or evidence of implemented features. Its instructions were not treated as instructions for this publication task. Assignment handouts and the empty `hw-1-3.0-py` file were omitted from the repository.

Changes:

1. Preserved `final.py` and `mid.py` without behavioral edits.
2. Replaced hard-coded API credentials in `hw-1.py` and `lab2_first_api_query.py` with `LLM_API_KEY` and `OPENROUTER_API_KEY` environment lookups. Original source files on disk were not changed.
3. Cleared notebook outputs and execution counts before publishing; retained code and markdown cells.
4. Reorganized older scripts, notebooks, and the author's report into named folders.
5. Compressed the original final-project database and split the archive into parts under the browser upload size limit. Added a checksum and safe restore helper. Database contents and vector index files are unchanged.
6. Added README, dependency constraints, environment-variable example, and ignore rules. No software license or invented features were added.

The final and midterm database files differ in bytes, though both have the same collection record counts; the final-project snapshot is the selected source. The final citation metadata mismatch and missing prompt history were documented instead of silently fixed. This is repository preparation, not a new implementation of the coursework.

Validation: Python source syntax checked; notebook JSON parsed; snapshot ZIP integrity and per-file content equality checked; source, notebook content/output, extracted PDF text, database string columns, and archive member bytes scanned for common credentials and private-key patterns. Key-like URL fragments in the database were reviewed as false positives. Pattern scanning is not proof that every possible secret format has been detected. No live chatbot or model evaluation was performed.
