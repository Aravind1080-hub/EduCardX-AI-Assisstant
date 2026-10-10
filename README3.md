# EduCardX Assistant API

A FastAPI backend for the **EduCardX AI assistant**. Each student question is routed by a fast rule-based router to the best source of information, and Gemini writes the final answer.

| Intent | When it is used | What happens |
|---|---|---|
| `RAG` | Questions about EduCardX policies, KYC, repayment, financing, scholarships, etc. | Retrieves relevant chunks from the EduCardX documents (Chroma + MMR search) |
| `WEB_SEARCH` | Current info: latest scholarships, study abroad, deadlines, visas, IELTS/TOEFL, etc. | Searches Google through Serper and gives the results to Gemini |
| `GENERAL` | Greetings and general education questions | Answered directly by Gemini |

Answers are returned as Markdown. Text and microphone voice input are supported.

## Highlights

- **One Gemini call per request.** Intent detection and search-query building are done with plain Python, so responses are faster and cheaper.
- **Retrieval-Augmented Generation** over 12 EduCardX Word documents using ChromaDB and local embeddings.
- **Live web search** (India region) through Serper.dev for current information.
- **Flexible document loading:** finds `.docx` files by folder, exact path or filename prefix, and skips missing ones instead of crashing.
- **Voice input** with Indian English (`en-IN`) speech recognition.

## Tech stack

- **FastAPI** + Uvicorn
- **Google Gemini** (`google-genai`, model `gemini-3.8-flash`)
- **LangChain**, **ChromaDB**, **HuggingFace** `all-MiniLM-L6-v2` embeddings
- **Serper.dev** for Google search
- **SpeechRecognition** (Google Web Speech) for microphone input

## Project structure

```
.
├── main.py                   # the app (main3.py renamed)
├── requirements.txt
├── .env                      # your API keys (do not commit)
├── chroma_db/                # created automatically (vector store)
├── 01_EduCardX_Terms_Conditions.docx    # knowledge-base documents
├── 02_EduCardX_Privacy_Policy.docx
├── ...
└── 12_EduCardX_AI_Mentor_Operating_Policy_ML_Spec.docx
```

## Setup

### 1. Prerequisites

- Python 3.10+
- A [Gemini API key](https://aistudio.google.com/apikey)
- A [Serper API key](https://serper.dev)
- A working microphone (only needed for `/voice-chat`)

### 2. Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**PyAudio notes** (needed only for voice input):

- **Windows:** `pip install pyaudio` usually works. If it fails, use `pip install pipwin && pipwin install pyaudio`.
- **macOS:** `brew install portaudio && pip install pyaudio`
- **Linux (Debian/Ubuntu):** `sudo apt install portaudio19-dev python3-dev && pip install pyaudio`

### 3. Configure environment variables

Create a `.env` file next to the script:

```env
SERPER_API_KEY=your_serper_key
GEMINI_API_KEY=your_gemini_key
```

The app refuses to start if either key is missing.

### 4. Add the knowledge-base documents

Put the 12 EduCardX Word documents in the same folder as the script. Each entry in the `documents` list is resolved in this order:

1. a **folder** with that name: every `.docx` inside it is loaded
2. an **exact file path**
3. any `.docx` in the project folder whose name **starts with** the entry (so `01_EduCardX_Terms_Conditions` matches `01_EduCardX_Terms_Conditions.docx` or `01_EduCardX_Terms_Conditions[1].docx`)

Word temporary files (`~$...`) are ignored. If no file is found for an entry, a warning is printed and that entry is skipped. At startup the console shows each loaded file and a total count, so check it says the number you expect.

### 5. Run

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

If you keep the filename `main3.py`, use `uvicorn main3:app ...` instead. The first start downloads the embedding model and builds the vector store, so it can take a minute. Interactive docs are at **http://localhost:8000/docs**.

## API reference

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /chat`

Send a text message (also use this if your mobile app does speech-to-text on the device).

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the repayment policy?"}'
```

Response:

```json
{
  "intent": "RAG",
  "answer": "## Repayment Policy\n..."
}
```

### `POST /voice-chat`

Listens on the **server machine's** microphone (up to 6 s to start speaking, 10 s of speech), converts it to text, and answers it. No request body.

```bash
curl -X POST http://localhost:8000/voice-chat
```

Response:

```json
{
  "transcript": "What scholarships are available for engineering students",
  "intent": "WEB_SEARCH",
  "answer": "..."
}
```

> Because this uses the server's own microphone, it works for local testing only. For a deployed app, do speech-to-text on the client and call `/chat`.

### Error codes

| Code | Meaning |
|---|---|
| 400 | Empty message |
| 422 | Speech could not be captured or understood |
| 502 | Gemini / Serper / retrieval error (details in the response) |

---

# How it works

## The big picture

When a student asks a question, the system decides what kind of question it is, gathers the right information, and asks Gemini to write the answer.

```
(Voice → Text) → Rule-based intent router → (RAG | Web search | Nothing) → Gemini → Answer
```

"What is the repayment policy?" needs the company's documents. "Latest scholarships for 2026" needs the live internet. "Hi!" needs neither.

## The models and tools

| Component | What it does | Where in the code |
|---|---|---|
| **Gemini** (`gemini-3.8-flash`) | Writes the final, formatted answer from the question and the gathered context. It is the only Gemini call. | `answer_question()` |
| **all-MiniLM-L6-v2** (HuggingFace) | Turns text into vectors so documents can be searched by meaning. Runs locally, so it costs nothing per request. | `HuggingFaceEmbeddings` |
| **Google Web Speech** | Turns the student's voice into text (`en-IN`). | `voice_to_text()` |
| **Rule-based router** (plain Python) | Decides the intent with regex and keyword lists. No AI call. | `detect_intent()` |
| **Query builder** (plain Python) | Builds the Google search query. No AI call. | `create_search_query()` |

## Intent routing (no AI call)

`detect_intent()` lowercases the message and checks it in this order, stopping at the first match:

1. **Greeting or small talk** (hi, hello, thanks, bye, "who are you", "what can you do"): `GENERAL`
2. **Contains "educardx"**: `RAG`
3. **Contains a web keyword** such as latest, current, upcoming, deadline, visa, IELTS, PTE, TOEFL, GRE, GMAT, abroad, "study in", tuition fee, "government scholarship", 2025/2026/2027: `WEB_SEARCH` (matched as whole words)
4. **Contains a document keyword** such as terms, privacy, KYC, repay, EMI, loan, interest, eligib, scholarship, course, college, placement, career, apply, policy, fee, refund, card, credit, student, FAQ: `RAG`
5. **Anything else**: `GENERAL`

Doing this in code instead of asking an LLM makes routing instant and predictable, and saves one API call per request.

## The three paths

### Path 1: RAG (EduCardX documents)

RAG (Retrieval-Augmented Generation) means Gemini doesn't answer from memory. It first looks up relevant passages in your documents and answers from them.

**At startup (indexing):**

1. **Load:** the Word documents are found with `find_docx_files()` and read with `Docx2txtLoader`.
2. **Chunk:** `RecursiveCharacterTextSplitter` cuts them into pieces of about 700 characters with a 100-character overlap. Small chunks keep search precise, and the overlap stops meaning being lost at the cut.
3. **Embed:** MiniLM converts each chunk into a vector.
4. **Store:** the vectors go into **ChromaDB**, saved in `./chroma_db`.

**At question time:**

1. The question is embedded with the same MiniLM model.
2. Chroma finds the closest chunks using **MMR (Maximal Marginal Relevance)**. It fetches the 20 closest (`fetch_k=20`), then picks 4 (`k=4`) that are relevant but different from each other, so you don't get four near-duplicates.
3. Those chunks become the context for Gemini.

### Path 2: Web search (current information)

Documents can't tell you this year's deadlines, so this path uses the internet.

1. `create_search_query()` takes the student's message, adds " India" if it isn't there, and adds " 2026" if the message has no year.
2. `serper_search()` sends the query to **Serper.dev**, which returns Google results as JSON (India region, English, 8 results).
3. `format_web_results()` turns each result into a block with a title, URL and snippet.
4. Gemini answers from these results only, with rules not to invent URLs, deadlines or amounts, and to say the information comes from current web results.

### Path 3: General

Greetings and general education questions need no lookup. The context is just a placeholder, and Gemini answers directly.

## The final prompt

Much of the answer quality comes from the last prompt. It includes:

- **The question and detected intent**, so the model knows which rules apply.
- **The retrieved context**, the "open book" it answers from.
- **Intent-specific rules:** no invented policies for RAG, no invented facts or links for web search, no mention of RAG or search for general chat.
- **Honesty rules:** if the context lacks the answer, say so instead of guessing.
- **Style rules:** Markdown, short paragraphs, bullets and numbered steps, bold key terms, sparing emojis, direct answer first.

## Voice input

1. `adjust_for_ambient_noise` calibrates to background noise.
2. The app waits up to 6 seconds for speech to start and records up to 10 seconds.
3. The audio goes to Google's speech-to-text.
4. Each failure (no speech, unintelligible speech, network error, microphone error) is caught separately, and the endpoint returns a clear 422 error.

The transcript then follows exactly the same path as a typed message.

## The API layer

**FastAPI** exposes `GET /health`, `POST /chat` and `POST /voice-chat`. Pydantic models (`ChatRequest`, `ChatResponse`, `VoiceChatResponse`) validate data in and out. CORS is enabled so a web frontend can call the API during testing.

## Configuration and safety

API keys are loaded from `.env` with `python-dotenv`, never hard-coded, and the app refuses to start if either is missing. The prompts act as guardrails: prefer official sources, don't invent facts, and admit when information can't be verified.

---

# Known limitations

The rule-based router is fast but simple, so it can misroute some questions:

- **Web keywords win over document keywords.** "What is the *current* repayment policy?" matches "current" and goes to `WEB_SEARCH` instead of `RAG`.
- **Document keywords match substrings, not whole words.** "emi" also matches "acad**emi**c" and "ch**emi**stry", "fee" matches "coffee", and "interest" matches "interested", so these can go to `RAG` by mistake.
- **"student" is a document keyword**, so general student questions (for example study tips) may be sent to `RAG`.
- **Search queries are not rewritten.** The student's text is sent to Google as typed, plus " India 2026", so casual phrasing may give less targeted results.
- **The year `2026` is hard-coded** in `create_search_query()` and will need updating.

## Things to check before deploying

- **Model name:** confirm `gemini-3.8-flash` is available for your API key.
- **Duplicate embeddings:** `Chroma.from_documents(...)` re-adds all chunks on every start, so the persisted `chroma_db` grows with duplicates. Delete `chroma_db/` when documents change, or build the index once and load it afterwards with `Chroma(persist_directory=..., embedding_function=...)`.
- **Empty knowledge base:** if no documents are found, the vector store is built from nothing. Check the console for `Warning: no .docx found`.
- **Source paths:** document metadata stores full file paths. Strip them to file names if you ever show sources to users.
- **CORS:** `allow_origins=["*"]` is for testing. Restrict it to your real domains in production.
- **Secrets:** add `.env` and `chroma_db/` to `.gitignore`.

## License

Add your license here.
