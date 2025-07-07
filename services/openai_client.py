import requests
from utils.config import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_ENDPOINT,
    AZURE_SPEECH_DEPLOYMENT_ID,
    AZURE_SPEECH_API_VERSION,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_ID,
    AZURE_OPENAI_API_VERSION
)


def transcribe_audio(file_path: str) -> str:
    url = f"{AZURE_SPEECH_ENDPOINT}/openai/deployments/{AZURE_SPEECH_DEPLOYMENT_ID}/audio/transcriptions?api-version={AZURE_SPEECH_API_VERSION}"

    headers = {
        "api-key": AZURE_SPEECH_KEY,
    }

    files = {
        "file": open(file_path, "rb"),
        "model": (None, AZURE_SPEECH_DEPLOYMENT_ID),
        "language": (None, "en"),
        "response_format": (None, "text"),
    }

    response = requests.post(url, headers=headers, files=files)
    response.raise_for_status()
    return response.text.strip()


def evaluate_speaking_transcript(transcript: str) -> str:
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_ID}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY,
    }

    prompt = f"""
You are an IELTS Speaking examiner. Evaluate the following candidate's **spoken** response based on IELTS Speaking Band Descriptors.

The response is transcribed below (please infer pronunciation and fluency from likely speech patterns):

\"\"\"{transcript}\"\"\"

Respond with structured feedback in this format:

1. **Band Score**: Overall score from 1 to 9

2. **Fluency and Coherence**
- Is the speech smooth and logical?
- Are there hesitations, fillers, or self-corrections?

3. **Lexical Resource**
- Vocabulary range and appropriateness
- Use of idiomatic or topic-specific expressions

4. **Grammatical Range and Accuracy**
- Use of various sentence structures
- Grammatical accuracy

5. **Pronunciation**
- Clarity, stress, intonation, rhythm (infer based on errors or awkward phrasing in transcript)

6. **Suggestions for Improvement**
- Give 2–3 targeted tips

Format in Markdown.
"""

    body = {
        "messages": [
            {"role": "system", "content": "You are a certified IELTS Speaking examiner."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
