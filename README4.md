# EduCardX Assistant API

A FastAPI backend for the **EduCardX AI assistant**. Each student question is routed by a fast rule-based router to the best source of information, and Gemini writes the final answer. The assistant remembers recent messages in each conversation, so follow-up questions work.

| Intent | When it is used | What happens |
|---|---|---|
| `RAG` | Questions about EduCardX policies, KYC, repayment, financing, scholarships, etc. | Retrieves relevant chunks from the EduCardX documents (Chroma + MMR search) |
| `WEB_SEARCH` | Current info: latest scholarships, study abroad, deadlines, visas, IELTS/TOEFL, etc. | Searches Google through Serper and gives the results to Gemini |
| `GENERAL` | Greetings and general education questions | Answered directly by Gemini |

Answers are returned as Markdown. Text and microphone voice input are supported.

## Highlights

- **Chat memory:** each conversation has a `session_id`, and the last 10 question/answer pairs are sent to Gemini so it understands follow-ups like "what about its fees?".
- **Redis or in-memory storage:** history is stored in Redis if it is running, and automatically falls back to an in-memory store if not. No setup is required to get started.
- **One Gemini call per request.** Intent detection and search-query building are done with plain Python, so responses are faster and cheaper.
- **Retrieval-Augmented Generation** over 12 EduCardX Word documents using ChromaDB and local embeddings.
- **Live web search** (India region) through Serper.dev for current information.
- **Flexible document loading:** finds `.docx` files by folder, exact path or filename prefix, and skips missing ones instead of crashing.
- **Voice input** with Indian English (`en-IN`) speech recognition.

## Tech stack

- **FastAPI** + Uvicorn
- **Google Gemini** (`google-genai`, model `gemini-2.5-flash`)
- **LangChain**, **ChromaDB**, **HuggingFace** `all-MiniLM-L6-v2` embeddings
- **Serper.dev** for Google search
- **Redis** (optional) for chat history
- **SpeechRecognition** (Google Web Speech) for microphone input

## Project structure

```
.
├── main.py                   # the app (rename main4.py to main.py)
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
- Redis (optional, see below)

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

### 5. (Optional) Start Redis

Redis keeps chat history across server restarts. Without it, the app uses an in-memory store and history is lost when the server stops.

```bash
# Docker (easiest)
docker run -d --name redis -p 6379:6379 redis

# macOS
brew install redis && brew services start redis

# Linux (Debian/Ubuntu)
sudo apt install redis-server && sudo systemctl start redis
```

On startup the console prints either `Connected to Redis.` or `Redis not available, using in-memory chat history.` The app looks for Redis at `localhost:6379`.

### 6. Run

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

If you keep the filename `main4.py`, use `uvicorn main4:app ...` instead. The first start downloads the embedding model and builds the vector store, so it can take a minute. Interactive docs are at **http://localhost:8000/docs**.

## API reference

### `GET /health`

```json
{ "status": "ok", "redis": true }
```

`redis` is `false` when the app is using the in-memory history store.

### `POST /chat`

Send a text message (also use this if your mobile app does speech-to-text on the device). Leave `session_id` out on the first message. The response returns one, and you send it back with every later message in that conversation.

```bash
# First message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the repayment policy?"}'

# Follow-up (use the session_id from the first response)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID", "message": "Can you explain that again?"}'
```

Response:

```json
{
  "session_id": "3f2b8c1e-....",
  "intent": "RAG",
  "answer": "## Repayment Policy\n..."
}
```

### `POST /voice-chat`

Listens on the **server machine's** microphone (up to 6 s to start speaking, 10 s of speech), converts it to text, and answers it. Pass `session_id` as a query parameter to continue a conversation.

```bash
curl -X POST "http://localhost:8000/voice-chat?session_id=YOUR_SESSION_ID"
```

Response:

```json
{
  "session_id": "3f2b8c1e-....",
  "transcript": "What scholarships are available for engineering students",
  "intent": "WEB_SEARCH",
  "answer": "..."
}
```

> Because this uses the server's own microphone, it works for local testing only. For a deployed app, do speech-to-text on the client and call `/chat`.

### `GET /history/{session_id}`

Returns the saved messages for a session.

```json
{
  "session_id": "3f2b8c1e-....",
  "history": [
    { "role": "user", "content": "What is the repayment policy?" },
    { "role": "assistant", "content": "## Repayment Policy\n..." }
  ]
}
```

### `DELETE /history/{session_id}`

Clears a session's history, which starts a fresh conversation.

```json
{ "session_id": "3f2b8c1e-....", "cleared": true }
```

### Error codes

| Code | Meaning |
|---|---|
| 400 | Empty message |
| 422 | Speech could not be captured or understood |
| 502 | Gemini / Serper / retrieval error (details in the response) |

---

# How it works

## The big picture

When a student asks a question, the system loads that conversation's recent history, decides what kind of question it is, gathers the right information, and asks Gemini to write the answer. The new question and answer are then saved to the history.

```
(Voice → Text) → Load history → Rule-based intent router → (RAG | Web search | Nothing) → Gemini → Answer → Save history
```

"What is the repayment policy?" needs the company's documents. "Latest scholarships for 2026" needs the live internet. "Hi!" needs neither.

## The models and tools

| Component | What it does | Where in the code |
|---|---|---|
| **Gemini** (`gemini-2.5-flash`) | Writes the final, formatted answer from the question, history and gathered context. It is the only Gemini call. | `answer_question()` |
| **all-MiniLM-L6-v2** (HuggingFace) | Turns text into vectors so documents can be searched by meaning. Runs locally, so it costs nothing per request. | `HuggingFaceEmbeddings` |
| **Google Web Speech** | Turns the student's voice into text (`en-IN`). | `voice_to_text()` |
| **Rule-based router** (plain Python) | Decides the intent with regex and keyword lists. No AI call. | `detect_intent()` |
| **Query builder** (plain Python) | Builds the Google search query. No AI call. | `create_search_query()` |
| **Redis / dict** | Stores each conversation's messages. | `get_history()`, `save_history()` |

## Chat history (short-term memory)

Gemini has no memory between calls, so the app supplies it.

1. **Session ID:** the first message creates a `session_id` (a UUID) that the client sends back with each later message. Each session has its own message list.
2. **Storage:** history is kept in **Redis** under the key `chat:<session_id>` and expires after 2 hours (`HISTORY_TTL_SECONDS`). If Redis can't be reached at startup (2-second timeout), the app uses an in-memory dictionary instead.
3. **What Gemini sees:** `format_history()` turns the last 20 messages (`MAX_HISTORY_MESSAGES`, so 10 question/answer pairs) into text like `Student: ...` / `Assistant: ...` and puts it at the top of the prompt. The prompt tells Gemini to use it for follow-up questions and not to repeat earlier answers.
4. **What is stored:** after each successful answer the new pair is appended and the list is trimmed before saving.
5. **Management:** `GET /history/{session_id}` reads a conversation and `DELETE /history/{session_id}` clears it.

If a request fails, nothing is saved for that turn.

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

- **Conversation history**, so follow-up questions make sense.
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

The transcript then follows exactly the same path as a typed message, including history.

## The API layer

**FastAPI** exposes `GET /health`, `POST /chat`, `POST /voice-chat`, `GET /history/{session_id}` and `DELETE /history/{session_id}`. Pydantic models (`ChatRequest`, `ChatResponse`, `VoiceChatResponse`) validate data in and out. CORS is enabled so a web frontend can call the API during testing.

## Configuration and safety

API keys are loaded from `.env` with `python-dotenv`, never hard-coded, and the app refuses to start if either is missing. The prompts act as guardrails: prefer official sources, don't invent facts, and admit when information can't be verified.

---

# Known limitations

**Routing** (the rule-based router is fast but simple):

- **Web keywords win over document keywords.** "What is the *current* repayment policy?" matches "current" and goes to `WEB_SEARCH` instead of `RAG`.
- **Document keywords match substrings, not whole words.** "emi" also matches "acad**emi**c" and "ch**emi**stry", "fee" matches "coffee", and "interest" matches "interested", so these can go to `RAG` by mistake.
- **"student" is a document keyword**, so general student questions (for example study tips) may be sent to `RAG`.
- **Search queries are not rewritten.** The student's text is sent to Google as typed, plus " India 2026".
- **The year `2026` is hard-coded** in `create_search_query()` and will need updating.

**Chat history:**

- **Routing and retrieval ignore history.** Only the current message is used to pick the intent, search documents and build the web query. A follow-up like "explain that again" has no keywords, so it goes to `GENERAL` with no document lookup, and Gemini has to rely on the history alone.
- **More tokens per request.** Up to 20 previous messages, including full earlier answers, are added to every prompt, so longer conversations use more tokens.
- **In-memory fallback is temporary.** History is lost on restart, isn't shared between multiple server workers, and is never cleaned up automatically.
- **Redis is only checked at startup.** If Redis stops while the app is running, requests that read history will fail.
- **Redis address is fixed** at `localhost:6379`.

## Things to check before deploying

- **Model name:** confirm `gemini-2.5-flash` is available for your API key.
- **Duplicate embeddings:** `Chroma.from_documents(...)` re-adds all chunks on every start, so the persisted `chroma_db` grows with duplicates. Delete `chroma_db/` when documents change, or build the index once and load it afterwards with `Chroma(persist_directory=..., embedding_function=...)`.
- **Empty knowledge base:** if no documents are found, the vector store is built from nothing. Check the console for `Warning: no .docx found`.
- **Source paths:** document metadata stores full file paths. Strip them to file names if you ever show sources to users.
- **Session access:** anyone who knows a `session_id` can read or clear its history through `/history/{session_id}`. Add authentication before exposing this publicly.
- **CORS:** `allow_origins=["*"]` is for testing. Restrict it to your real domains in production.
- **Secrets:** add `.env` and `chroma_db/` to `.gitignore`.

## License

Add your license here.
