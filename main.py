import os
import re
import traceback
import requests
import speech_recognition as sr
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel


# ==========================================
# KEYS (set these as environment variables)
#   export SERPER_API_KEY="..."
#   export GEMINI_API_KEY="..."
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SERPER_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Set SERPER_API_KEY and GEMINI_API_KEY environment variables.")

# Gemini API key
client = genai.Client(
    api_key=GEMINI_API_KEY
)

# Documents
documents = [
    "01_EduCardX_Terms_Conditions",
    "02_EduCardX_Privacy_Policy",
    "03_EduCardX_KYC_Student_Verification",
    "04_EduCardX_Financing_Explanation_Guide",
    "05_EduCardX_Repayment_Policy",
    "06_EduCardX_FAQ",
    "07_EduCardX_Student_Guide",
    "08_EduCardX_Scholarship_Information_Guide",
    "09_EduCardX_College_Information_Guide",
    "10_EduCardX_Career_Placement_Guide",
    "11_EduCardX_Financial_Literacy_Guide",
    "12_EduCardX_AI_Mentor_Operating_Policy_ML_Spec",
]


# Load Documents
docs = []

def find_docx_files(name):
   
    path = os.path.join(BASE_DIR, name)

    # Entry is a folder -> every .docx inside it
    if os.path.isdir(path):
        return [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.lower().endswith(".docx") and not f.startswith("~$")
        ]

    # Entry is already a full file name
    if os.path.isfile(path):
        return [path]

    # Entry is a file name without extension / with a suffix like "[1]"
    return [
        os.path.join(BASE_DIR, f)
        for f in sorted(os.listdir(BASE_DIR))
        if f.lower().endswith(".docx")
        and not f.startswith("~$")
        and f.startswith(name)
    ]


for file in documents:

    paths = find_docx_files(file)

    if not paths:
        print(f"Warning: no .docx found for '{file}' in {BASE_DIR}, skipping.")
        continue

    for file_path in paths:
        print(f"Loading document: {file_path}")
        loader = Docx2txtLoader(file_path)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = file_path
        docs.extend(loaded_docs)

print(f"Loaded {len(docs)} documents")


# chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100)
chunks = splitter.split_documents(docs)


# embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Chroma Database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


# Retriving
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20
    }
)



# SPEECH RECOGNITION (live microphone)
# Listens on the microphone of the machine running this server,
# so it works for local use / testing on your PC.

def voice_to_text():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:

            print("\n Listening...")
            print("Speak now...")

            # Reduce background-noise problems
            recognizer.adjust_for_ambient_noise(
                source,
                duration=0.5
            )

            audio = recognizer.listen(
                source,
                timeout=6,
                phrase_time_limit=10
            )

        print("Converting speech to text...")

        text = recognizer.recognize_google(
            audio,
            language="en-IN"
        )

        print(text)

        return text

    except sr.WaitTimeoutError:
        print("❌ No speech detected.")
        return None

    except sr.UnknownValueError:
        print("❌ Sorry, I couldn't understand your speech.")
        return None

    except sr.RequestError as e:
        print(f"❌ Speech recognition error: {e}")
        return None

    except Exception as e:
        print(f"❌ Microphone error: {e}")
        return None


# ==========================================
# SERPER WEB SEARCH

def serper_search(query, num_results=8):

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "q": query,
        "gl": "in",
        "hl": "en",
        "num": num_results
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=10
    )

    response.raise_for_status()

    return response.json()


# ==========================================
# FORMAT WEB RESULTS

def format_web_results(data):

    results = data.get("organic", [])

    if not results:
        return "No web search results found."

    formatted_results = []

    for i, result in enumerate(results[:8], start=1):

        title = result.get("title", "")
        link = result.get("link", "")
        snippet = result.get("snippet", "")

        formatted_results.append(
            f"""
SOURCE {i}

Title:
{title}

URL:
{link}

Description:
{snippet}
"""
        )

    return "\n".join(formatted_results)

# search engine query

def create_search_query(user):
    """No Gemini call: build the Google query with plain Python."""

    query = user.strip()

    # Add India + the current year so results are current and relevant
    if "india" not in query.lower():
        query += " India"

    if not re.search(r"\b20\d{2}\b", query):
        query += " 2026"

    return query

# intent function (no Gemini call: simple keyword router)

GREETING_PATTERN = re.compile(
    r"^\s*(hi|hii+|hello|hey|heyy+|good\s+(morning|afternoon|evening|night)|"
    r"thanks|thank\s+you|ok|okay|bye|goodbye|who\s+are\s+you|what\s+can\s+you\s+do)\b[\s!.?]*$",
    re.IGNORECASE,
)

WEB_KEYWORDS = [
    "latest", "newest", "recent", "currently", "current", "today", "this year",
    "upcoming", "2025", "2026", "2027",
    "deadline", "last date", "visa", "ielts", "pte", "toefl", "gre", "gmat",
    "abroad", "overseas", "study in", "universities in", "university in",
    "government scholarship", "government scheme", "scholarship scheme",
    "new scholarship", "student scheme", "tuition fee", "tuition fees",
    "admission open", "admissions open",
]

RAG_KEYWORDS = [
    "educardx", "terms", "conditions", "privacy", "kyc", "verification",
    "repay", "emi", "installment", "financing", "finance", "loan", "interest",
    "eligib", "scholarship", "course", "college", "placement", "career",
    "application", "apply", "policy", "policies", "fee", "refund",
    "card", "credit", "limit", "document", "student", "mentor", "faq",
]


def detect_intent(user):

    text = user.lower().strip()

    # 1. Greetings / small talk
    if GREETING_PATTERN.match(text):
        return "GENERAL"

    # 2. Questions that clearly name EduCardX always use the documents
    if "educardx" in text:
        return "RAG"

    # 3. Needs current information from the internet
    for word in WEB_KEYWORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", text):
            return "WEB_SEARCH"

    # 4. EduCardX document topics
    for word in RAG_KEYWORDS:
        if word in text:
            return "RAG"

    # 5. Everything else: general question
    return "GENERAL"


# --------------------------------------------------
# BOT (one question in -> one answer out)
# --------------------------------------------------

def answer_question(user):

    # --------------------------------
    # STEP 1: DETECT INTENT
    # --------------------------------

    intent = detect_intent(user)

    context = ""

    # --------------------------------
    # STEP 4: RAG
    # --------------------------------

    if intent == "RAG":

        results = retriever.invoke(user)

        context = "\n\n".join(
            doc.page_content
            for doc in results
        )

    # --------------------------------
    # STEP 5: WEB SEARCH
    # --------------------------------

    elif intent == "WEB_SEARCH":

        # Convert student's question
        # into a better Google search query
        search_query = create_search_query(user)

        # Search Google through Serper
        search_data = serper_search(search_query)

        # Format results for Gemini
        context = format_web_results(search_data)

    # --------------------------------
    # STEP 6: GENERAL
    # --------------------------------

    else:

        context = "No document retrieval required."

    # --------------------------------
    # STEP 6: FINAL GEMINI RESPONSE
    # --------------------------------

    prompt = f"""
You are EduCardX Assistant.

User question:
{user}

Detected intent:
{intent}

Retrieved EduCardX information:
{context}

Instructions:

1. Answer the user's question directly.
2. If the user is greeting you, respond naturally.
3. For EduCardX policy questions, use the retrieved document information.
4. Do not invent EduCardX policies.
5. If the required EduCardX information is not present in
   the retrieved context, give any relavant answer to user.

6. If the intent is RAG:
   - Use the supplied EduCardX information.
   - Do not invent EduCardX policies.

7. If the intent is WEB_SEARCH:
   - Use the supplied live web search results.
   - Do not invent facts.
   - Do not invent URLs.
   - Do not invent deadlines.
   - Do not invent scholarship amounts.
   - Prefer official government, university, embassy
     and scholarship organization sources.
   - Include useful source URLs from the search results.
   - Clearly indicate that the information comes from
     current web search results.

8. If the intent is GENERAL:
   - Answer normally.
   - Do not mention RAG or web search.

9. If the supplied information is insufficient,
   clearly say that the information could not be verified.

Give a clear, student-friendly answer.

STYLE RULES:
- Give a clear, professional and friendly answer.
- Use Markdown formatting.
- Use short paragraphs.
- Use headings when useful.
- Use bullet points for multiple items.
- Use numbered lists for steps or procedures.
- Bold important terms.
- Use emojis sparingly to improve readability.
- Do not make the answer unnecessarily long.
- Directly answer the user's question first.
- If the information contains several policies or terms, organize them into sections.
- Do not invent information that is not present in the provided information.
- If the information does not contain the answer, clearly say that the available EduCardX information does not specify it.

IMPORTANT:
The answer should feel like a modern AI student/finance assistant,
not like a raw document summary.

Return only the final answer.
"""

    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    return intent, response.output_text


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(title="EduCardX Assistant API", version="1.0.0")

# Mobile apps don't need CORS, but it is useful for web testing.
# Restrict allow_origins to your real domains in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    intent: str
    answer: str


class VoiceChatResponse(BaseModel):
    transcript: str
    intent: str
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


# Text input (also use this when the app does on-device speech-to-text)
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    user = req.message.strip()

    if not user:
        raise HTTPException(status_code=400, detail="Message is empty.")

    try:
        intent, answer = answer_question(user)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Gemini Error: {e}")

    return ChatResponse(intent=intent, answer=answer)


# Voice input: speak into the microphone, it is converted to text and answered
@app.post("/voice-chat", response_model=VoiceChatResponse)
def voice_chat():

    user = voice_to_text()

    if not user:
        raise HTTPException(
            status_code=422,
            detail="Could not capture or understand speech. Please try again.",
        )

    try:
        intent, answer = answer_question(user)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Gemini Error: {e}")

    return VoiceChatResponse(transcript=user, intent=intent, answer=answer)


# Run:  python main.py   (or: uvicorn main:app --host 0.0.0.0 --port 8000)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
