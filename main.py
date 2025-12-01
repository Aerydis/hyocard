from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
import os
import base64
import asyncio
import json

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini Client
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Prompts
EXPLAIN_PROMPT = (
    "You are a clear, concise tutor in Korean. "
    "Extract the text from the image and write a short explanation (3–6 sentences)."
)

# We update this prompt to be very specific about the JSON schema we want
FLASHCARD_PROMPT = (
    "You are an assistant that creates Anki flashcards from study notes.\n"
    "Extract the key concepts from the image and create a list of Q&A pairs.\n"
    "Return the output in this specific JSON structure:\n"
    "[\n"
    "  {\"question\": \"Concept or Question in Korean\", \"answer\": \"Definition or Answer in Korean/English\"}\n"
    "]"
)

def call_gemini(prompt_template: str, image_bytes: bytes, mime_type: str, mode: str):
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    
    # Configure strict JSON output if we are in flashcard mode
    config = None
    if mode == "flashcards":
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "question": {"type": "STRING"},
                        "answer": {"type": "STRING"}
                    }
                }
            }
        )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt_template},
                        {"inline_data": {"mime_type": mime_type, "data": encoded_image}}
                    ]
                }
            ],
            config=config # Apply the JSON config here
        )

        if hasattr(response, "text"):
            return response.text
        return "No text returned"

    except Exception as e:
        print(f"Gemini Error: {e}")
        raise RuntimeError(f"Gemini request failed: {e}")


@app.post("/process")
async def process(image: UploadFile = File(...), mode: str = Form("explain")):
    try:
        image_bytes = await image.read()
        mime_type = image.content_type or "image/png"
        
        prompt = EXPLAIN_PROMPT if mode == "explain" else FLASHCARD_PROMPT

        loop = asyncio.get_running_loop()
        # Pass 'mode' to the helper so it knows when to enforce JSON
        result_text = await loop.run_in_executor(
            None, lambda: call_gemini(prompt, image_bytes, mime_type, mode)
        )

        return {"result": result_text}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health():
    return {"status": "ok"}