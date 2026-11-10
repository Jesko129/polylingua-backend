import base64
import os
from pathlib import Path
from typing import Final

import edge_tts
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

PORT = int(os.getenv("PORT", "8000"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()

SUPPORTED_LANGUAGES: Final[set[str]] = {
    "English",
    "Mandarin",
    "Filipino",
}

ALLOWED_IMAGE_TYPES: Final[set[str]] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

MAX_IMAGE_SIZE_BYTES: Final[int] = 10 * 1024 * 1024

# Free Microsoft Edge neural voices (no API key or billing needed)
EDGE_VOICES: Final[dict[str, dict[str, str]]] = {
    "English": {
        "female": "en-US-JennyNeural",
        "male": "en-US-GuyNeural",
    },
    "Mandarin": {
        "female": "zh-CN-XiaoxiaoNeural",
        "male": "zh-CN-YunxiNeural",
    },
    "Filipino": {
        "female": "fil-PH-BlessicaNeural",
        "male": "fil-PH-AngeloNeural",
    },
}


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Polylingua API",
    version="1.7.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request / response models
# --------------------------------------------------

class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    sourceLanguage: str
    targetLanguage: str


class TranslationResponse(BaseModel):
    translatedText: str


class OcrResponse(BaseModel):
    text: str


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    language: str
    voiceStyle: str | None = None  # "male" or "female" (optional)
    speechRate: float = Field(1.0, ge=0.5, le=1.5)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def require_gemini_key() -> None:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=f"GEMINI_API_KEY is missing. Add it to {ENV_FILE}",
        )


def normalize_language(value: str) -> str:
    return value.strip().capitalize()


def validate_language(value: str, role: str) -> str:
    language = normalize_language(value)
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported {role} language: {value}. "
                "Use English, Mandarin, or Filipino."
            ),
        )
    return language


def gemini_generate_url() -> str:
    return (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{GEMINI_MODEL}:generateContent"
    )


def extract_gemini_text(response_data: dict) -> str:
    candidates = response_data.get("candidates", [])
    if not candidates:
        raise HTTPException(status_code=502, detail="Gemini returned no candidates")

    first_candidate = candidates[0]
    content = first_candidate.get("content", {})
    parts = content.get("parts", [])

    result = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict)
    ).strip()

    if not result:
        finish_reason = first_candidate.get("finishReason", "unknown")
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned no text. Finish reason: {finish_reason}",
        )

    return result


async def call_gemini(payload: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                gemini_generate_url(),
                params={"key": GEMINI_API_KEY},
                json=payload,
            )
    except httpx.TimeoutException as error:
        raise HTTPException(
            status_code=504,
            detail="Gemini request timed out",
        ) from error
    except httpx.RequestError as error:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Gemini",
        ) from error

    try:
        response_data = response.json()
    except ValueError as error:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned invalid JSON",
        ) from error

    if response.status_code >= 400:
        error_message = response_data.get("error", {}).get(
            "message",
            "Gemini request failed",
        )
        raise HTTPException(status_code=502, detail=error_message)

    return response_data


def pick_edge_voice(language: str, voice_style: str | None) -> str:
    style = "male" if (voice_style or "").strip().lower() == "male" else "female"
    return EDGE_VOICES[language][style]


async def synthesize_edge(
    text: str,
    voice: str,
    speech_rate: float = 1.0,
) -> bytes:
    percent = round((speech_rate - 1.0) * 100)
    rate = f"{percent:+d}%"

    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        audio = bytearray()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Text-to-speech failed: {error}",
        ) from error

    if not audio:
        raise HTTPException(
            status_code=502,
            detail="Text-to-speech returned empty audio",
        )

    return bytes(audio)


# --------------------------------------------------
# Basic route
# --------------------------------------------------

@app.get("/")
async def root() -> dict[str, object]:
    return {
        "ok": True,
        "message": "Polylingua FastAPI backend is running",
    }


# --------------------------------------------------
# Translation endpoint
# --------------------------------------------------

@app.post("/api/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest) -> TranslationResponse:
    require_gemini_key()

    text = request.text.strip()
    source_language = validate_language(request.sourceLanguage, "source")
    target_language = validate_language(request.targetLanguage, "target")

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    if source_language == target_language:
        raise HTTPException(
            status_code=400,
            detail="Source and target languages must be different",
        )

    prompt = f"""
Translate the following text from {source_language} to {target_language}.

Only these languages are supported:
English, Mandarin, and Filipino.

Return only the translation.
Do not include explanations, quotation marks, labels, pronunciation guides, or extra formatting.

Text:
{text}
""".strip()

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }

    response_data = await call_gemini(payload)
    return TranslationResponse(
        translatedText=extract_gemini_text(response_data),
    )


# --------------------------------------------------
# OCR endpoint
# --------------------------------------------------

@app.post("/api/ocr", response_model=OcrResponse)
async def extract_text_from_image(
    image: UploadFile = File(...),
) -> OcrResponse:
    require_gemini_key()

    content_type = (image.content_type or "").lower().strip()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Use JPEG, PNG, or WebP.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="The image must be smaller than 10 MB",
        )

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
Extract all visible text from this image.

Preserve the original wording and line breaks as much as possible.

Return only the extracted text.
Do not translate the text.
Do not describe the image.
Do not add labels or explanations.

If no readable text exists, return an empty response.
""".strip()

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": content_type,
                            "data": encoded_image,
                        },
                    },
                ],
            }
        ],
        "generationConfig": {"temperature": 0},
    }

    response_data = await call_gemini(payload)

    try:
        extracted_text = extract_gemini_text(response_data)
    except HTTPException:
        extracted_text = ""

    return OcrResponse(text=extracted_text)


# --------------------------------------------------
# TTS endpoint (free, via edge-tts)
# --------------------------------------------------

@app.post("/api/tts")
async def tts(request: TtsRequest) -> Response:
    text = request.text.strip()
    language = validate_language(request.language, "tts")

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    voice = pick_edge_voice(language, request.voiceStyle)
    audio_bytes = await synthesize_edge(text, voice, request.speechRate)

    return Response(content=audio_bytes, media_type="audio/mpeg")


# --------------------------------------------------
# Local development entry point
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    )