# main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from rag import load_clinic_data, search_clinic_data
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

# Fix: correctly read the API key and create the client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@app.get("/app")
def serve_frontend():
    return FileResponse("index.html")


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

    relevant_chunks = search_clinic_data(request.clinic_id, request.message)

    if relevant_chunks:
        context = "\n\n".join(relevant_chunks)
        system_prompt = f"""You are a friendly and helpful dental clinic receptionist chatbot.

You have two jobs:
1. Answer questions about this specific clinic using the information below
2. Answer general dental or health questions, and handle normal conversation naturally

CLINIC INFORMATION:
{context}

Rules:
- For clinic-specific questions (hours, doctors, services, prices, location), use ONLY the clinic info above
- For general questions (dental tips, what is a root canal, how to brush teeth, greetings, small talk), answer helpfully from your own knowledge
- Always be warm, friendly and professional
- Never make up clinic-specific details not in the clinic info"""

    else:
        system_prompt = """You are a friendly dental clinic receptionist chatbot.
Answer general dental questions, give helpful advice, and have natural conversations.
Be warm, friendly and professional."""

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


@app.post("/load-data/{clinic_id}")
def load_data(clinic_id: str, file_path: str):
    count = load_clinic_data(clinic_id, file_path)
    return {"message": f"Loaded {count} chunks for {clinic_id}"}
