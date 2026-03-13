# main.py

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from rag import load_clinic_data, search_clinic_data
import os

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/app")
def serve_frontend():
    return FileResponse("index.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv(
    "dental-receptionist"))

# Load clinic data when the server starts
# Later this will be dynamic — for now we load one clinic


@app.on_event("startup")
async def startup_event():
    if os.path.exists("clinic_data.txt"):
        load_clinic_data("clinic_abc", "clinic_data.txt")
        print("Clinic data loaded successfully!")
    else:
        print("Warning: clinic_data.txt not found!")


class ChatRequest(BaseModel):
    message: str
    clinic_id: str = "clinic_abc"


@app.get("/")
def root():
    return {"status": "Dental chatbot backend is running"}


@app.post("/chat")
def chat(request: ChatRequest):

    # Step 1: Search clinic data for relevant context
    relevant_chunks = search_clinic_data(request.clinic_id, request.message)

    # Step 2: Build context string from search results
    if relevant_chunks:
        context = "\n\n".join(relevant_chunks)
    else:
        context = "No specific information found for this query."

    # Step 3: Build the system prompt with real clinic data injected
    system_prompt = f"""You are a friendly receptionist chatbot for a dental clinic.
Answer the patient's question using ONLY the clinic information provided below.
If the answer is not in the information, say "I don't have that information, 
please call us directly."
Be warm, concise, and helpful.

CLINIC INFORMATION:
{context}"""

    # Step 4: Call Groq with the context-aware prompt
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ],
        max_tokens=500
    )

    reply = response.choices[0].message.content
    return {"reply": reply, "clinic_id": request.clinic_id}

# Endpoint to load new clinic data (we'll use this more in Phase 3)


@app.post("/load-data/{clinic_id}")
def load_data(clinic_id: str, file_path: str):
    count = load_clinic_data(clinic_id, file_path)
    return {"message": f"Loaded {count} chunks for {clinic_id}"}
