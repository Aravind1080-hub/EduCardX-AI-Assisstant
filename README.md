# EduCardX Assistant API

A FastAPI backend for the **EduCardX AI assistant**. Gemini classifies every student question and routes it to the best source of information:

| Intent | When it is used | What happens |
|---|---|---|
| `RAG` | Questions about EduCardX policies, KYC, repayment, financing, scholarships, etc. | Retrieves relevant chunks from the EduCardX documents (Chroma + MMR search) |
| `WEB_SEARCH` | Current info: latest scholarships, study abroad, deadlines, visas, IELTS/TOEFL, etc. | Gemini rewrites the question into a search query, Google results are fetched via Serper |
| `GENERAL` | Greetings and general education questions | Answered directly by Gemini |

The final answer is written by Gemini and returned as Markdown. Text and microphone voice input are supported.

## Tech stack

- **FastAPI** + Uvicorn
- **Google Gemini** (`google-genai`, model `gemini-3.6-flash`)
- **LangChain**, **ChromaDB**, **HuggingFace** `all-MiniLM-L6-v2` embeddings
- **Serper.dev** for Google search
- **SpeechRecognition** (Google Web Speech) for microphone input

## Project structure

```
.
├── main.py
├── requirements.txt
├── .env                      # your API keys (do not commit)
├── chroma_db/                # created automatically (vector store)
├── 01_EduCardX_Terms_Conditions        # knowledge-base documents (.docx)
├── 02_EduCardX_Privacy_Policy
├── ...
└── 12_EduCardX_AI_Mentor_Operating_Policy_ML_Spec
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

Create a `.env` file next to `main.py`:

```env
SERPER_API_KEY=your_serper_key
GEMINI_API_KEY=your_gemini_key
```

The app refuses to start if either key is missing.

### 4. Add the knowledge-base documents

Place the 12 EduCardX Word documents listed in the `documents` list in `main.py` in the folder you run the app from. The loader opens each file by exactly the name in that list, so either:

- name the files exactly as listed (no extension), **or**
- add `.docx` to the names in the list (e.g. `"01_EduCardX_Terms_Conditions.docx"`).

### 5. Run

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The first start downloads the embedding model and builds the vector store, so it can take a minute. Interactive docs are at **http://localhost:8000/docs**.

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

When a student asks a question, the system first decides what kind of question it is, then gathers the right information, and finally asks Gemini to write the answer. "What is the repayment policy?" needs the company's documents. "Latest scholarships for 2026" needs the live internet. "Hi!" needs neither.

```
Question → Intent detection → (RAG | Web search | Nothing) → Final answer
```

## One model, three jobs

The app uses a single Gemini model (`gemini-3.6-flash`) but calls it three times, each with its own prompt and its own job:

| Step | Function | Job |
|---|---|---|
| 1 | `detect_intent()` | Classify the question as `RAG`, `WEB_SEARCH` or `GENERAL` |
| 2 | `create_search_query()` | Turn the student's message into one good Google search query (web search only) |
| 3 | `answer_question()` | Write the final, formatted answer from the gathered context |

**1. The router (intent classifier).** Reads one message and returns one word. The prompt describes each category with examples and says to return only the label, which makes the output quick to check in code.

**2. The search query writer.** Students write things like "any scholarships for me after 12th?", which makes a poor Google query. This step rewrites it into a focused search, for example adding "India" and "2026" and preferring official government or university sources. Better queries mean better Serper results.

**3. The answer writer.** This is what the student actually sees. It receives the question, the detected intent and the retrieved context, then writes a friendly Markdown answer.

Splitting the work into separate, focused prompts keeps each one short and reliable instead of one giant prompt doing everything. Because all three steps share one model, you can also change models later by editing a single value (see "Things to check").

## The three paths

### Path 1: RAG (EduCardX documents)

RAG (Retrieval-Augmented Generation) means the model does not answer from memory. It first looks up relevant passages in your documents and answers from them, so it can explain EduCardX's rules without inventing anything.

**At startup (indexing):**

1. **Load:** the 12 Word documents are read with `Docx2txtLoader`, and each is tagged with its source name.
2. **Chunk:** `RecursiveCharacterTextSplitter` cuts the text into pieces of about 700 characters with a 100-character overlap. Small chunks keep searches precise, and the overlap stops a sentence from losing meaning when it is cut at a boundary.
3. **Embed:** each chunk is converted into a vector by `all-MiniLM-L6-v2`, a HuggingFace sentence-transformer. Chunks with similar meaning get similar vectors, so "when do I pay back my loan" can match a chunk about repayment even without shared keywords.
4. **Store:** the vectors go into **ChromaDB**, a vector database saved in `./chroma_db`.

**At question time (retrieval):**

1. The question is embedded the same way.
2. Chroma finds the closest chunks using **MMR (Maximal Marginal Relevance)**. It fetches the 20 closest chunks (`fetch_k=20`), then picks the 4 (`k=4`) that are both relevant and different from each other. Plain similarity search often returns four near-identical chunks, and MMR avoids that.
3. Those 4 chunks are joined into the context for the final answer.

### Path 2: Web search (current information)

Documents can't tell you this year's scholarship deadlines, so this path goes to the internet.

1. `create_search_query()` writes the search query.
2. `serper_search()` sends it to **Serper.dev**, an API that returns Google results as JSON. It is set to India (`gl: "in"`) and English, and asks for 8 results.
3. `format_web_results()` turns each result into a block with a title, URL and snippet.
4. The final step receives these as context and is told to use only what is there, never invent URLs, deadlines or amounts, and to mention that the information comes from current web results.

### Path 3: General

For greetings and general education questions no lookup is needed. The context is just the placeholder "No document retrieval required", and Gemini answers directly.

## The final prompt

The last prompt is where much of the answer quality comes from. It contains:

- **The question and the detected intent**, so the model knows which rules apply.
- **The retrieved context**, the "open book" the model answers from.
- **Intent-specific rules:** RAG answers must not invent policies, web answers must not invent facts or links, and general answers should not mention RAG or search.
- **Honesty rules:** if the context doesn't contain the answer, say so instead of guessing.
- **Style rules:** Markdown, short paragraphs, bullets for lists, numbers for steps, bold key terms, sparing emojis, and the direct answer first.

These rules reduce hallucination, which matters for an app that talks about student finance.

## Voice input

`/voice-chat` uses the **SpeechRecognition** library and the server's microphone:

1. It calibrates for background noise (`adjust_for_ambient_noise`).
2. It listens, waiting up to 6 seconds for speech to start and recording up to 10 seconds of it.
3. It sends the audio to Google's speech-to-text using `en-IN` (Indian English).
4. The text goes through the same pipeline as a typed message.

Each failure (no speech, unintelligible speech, network error, microphone error) is caught separately, and the endpoint returns a clear error.

## The API layer

**FastAPI** exposes three endpoints: `GET /health`, `POST /chat` and `POST /voice-chat`. Pydantic models (`ChatRequest`, `ChatResponse`, `VoiceChatResponse`) validate data in and out. Empty messages return 400, failures in the AI pipeline return 502, and unrecognized speech returns 422. CORS is enabled so a web frontend can call the API during testing.

## Configuration and safety

API keys are loaded from `.env` with `python-dotenv`, never hard-coded, and the app refuses to start if either is missing. The prompts also act as guardrails: prefer official sources, don't invent facts, and admit when information can't be verified.

## Things to check before deploying

- **Model name:** `gemini-3.6-flash` is hard-coded in three places. Confirm it is available for your API key, and move it into one constant (e.g. `MODEL = "gemini-3.6-flash"`) so it is easy to change.
- **Duplicate embeddings:** `Chroma.from_documents(...)` re-adds all chunks on every start, so the persisted `chroma_db` grows with duplicates. Delete `chroma_db/` when documents change, or build the index once and load it afterwards with `Chroma(persist_directory=..., embedding_function=...)`.
- **CORS:** `allow_origins=["*"]` is for testing. Restrict it to your real domains in production.
- **Secrets:** add `.env` and `chroma_db/` to `.gitignore`.
- **Intent output:** the router must return exactly `RAG`, `GENERAL` or `WEB_SEARCH`. Any other output (extra words, different casing) silently falls into the `GENERAL` path, so consider `.strip().upper()` on the result.

## License

Add your license here.
