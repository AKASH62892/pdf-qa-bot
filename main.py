from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from groq import Groq
from dotenv import load_dotenv
from rag import build_vector_store, get_relevant_chunks
import uuid
import os
import shutil

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    pdf_path = f"{UPLOAD_DIR}/{session_id}.pdf"
    
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    num_chunks = build_vector_store(pdf_path, session_id)
    
    return {
        "session_id": session_id,
        "message": f"PDF processed successfully with {num_chunks} chunks"
    }

@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    session_id: str = Form(...)
):
    relevant_chunks = get_relevant_chunks(question, session_id)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions based only on the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{relevant_chunks}\n\nQuestion: {question}"
            }
        ]
    )

    return {"answer": response.choices[0].message.content}

@app.get("/")
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())
