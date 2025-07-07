from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from models.response import EvaluationRequest, EvaluationResponse
from services.openai_client import transcribe_audio, evaluate_speaking_transcript
import os
import tempfile

app = FastAPI()

# ✅ Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your deployed frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🎤 Route: Transcribe uploaded audio
@app.post("/api/speech/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save the uploaded audio to a temporary file
    suffix = os.path.splitext(file.filename)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_file_path = tmp.name

    try:
        transcript = transcribe_audio(temp_file_path)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return {"transcript": transcript}


# 🤖 Route: Evaluate transcript with GPT feedback
@app.post("/api/speech/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    feedback = evaluate_speaking_transcript(request.transcript)
    return {"feedback": feedback}
