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
# Allow all origins, methods, and headers for the frontend hosted on GitHub pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini Client
API_KEY = os.getenv("GEMINI_API_KEY")
# Using gemini-2.5-flash which is the modern recommended fast model
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") 

# Check for API Key before client initialization
if not API_KEY:
    # This will cause the Render service to fail if the environment variable isn't set
    print("FATAL: GEMINI_API_KEY environment variable is not set.")
    # Exit or raise error if needed, but for now, we'll initialize the client
    # This allows it to run locally/in environments where the key is provided differently
    pass 
    
client = genai.Client(api_key=API_KEY)


# Prompts
EXPLAIN_PROMPT = (
    "You are a clear, friendly Korean tutor. "
    "Extract the text from the image and write a short explanation (3–6 sentences) in Korean."
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
        # Re-raise the error to be caught by the FastAPI handler
        raise RuntimeError(f"Gemini request failed: {e}")


# -----------------------------------------------
# FIX 2: Add Root Route for Render Health Check
# -----------------------------------------------
@app.get("/")
async def root():
    return {"message": "Hyocard FastAPI service is running. Use /process endpoint for image processing."}

@app.get("/health")
async def health():
    return {"status": "ok"}
# -----------------------------------------------

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

    except RuntimeError as e:
        # Catch explicit RuntimeError from Gemini client and return 500
        return JSONResponse(status_code=500, content={"error": f"API 처리 오류: {e}"})
    except Exception as e:
        # Catch general server errors
        return JSONResponse(status_code=500, content={"error": f"서버 처리 오류: {e}"})

