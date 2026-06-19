import io
import os
import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import soundfile as sf

from kittentts import KittenTTS
from pydub import AudioSegment
from pydub.utils import which

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kittentts-api")

# Configure pydub to use ffmpeg
AudioSegment.converter = which("ffmpeg")

app = FastAPI(title="KittenTTS API")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# All models that should be preloaded and kept resident in memory at startup.
# Maps a short, OpenAI-style model id (what clients pass in `model`) to the
# actual KittenTTS / HuggingFace repo id.
AVAILABLE_MODELS = {
    "tts-1": "KittenML/kitten-tts-mini-0.8",       # default / OpenAI-compatible alias
    "tts-1-hd": "KittenML/kitten-tts-mini-0.8",    # alias to the higher quality mini model
    "kitten-tts-mini-0.8": "KittenML/kitten-tts-mini-0.8",
    "kitten-tts-micro-0.8": "KittenML/kitten-tts-micro-0.8",
}

DEFAULT_MODEL_ID = os.environ.get("KITTEN_TTS_DEFAULT_MODEL", "tts-1")

# Optionally restrict which repos actually get preloaded (comma-separated env var),
# otherwise preload every unique repo referenced in AVAILABLE_MODELS.
_preload_env = os.environ.get("KITTEN_TTS_PRELOAD_MODELS")
if _preload_env:
    REPOS_TO_PRELOAD = [r.strip() for r in _preload_env.split(",") if r.strip()]
else:
    REPOS_TO_PRELOAD = sorted(set(AVAILABLE_MODELS.values()))

# repo_id -> loaded KittenTTS instance, kept alive for the lifetime of the process
MODEL_INSTANCES: dict[str, KittenTTS] = {}


def _load_all_models() -> None:
    for repo_id in REPOS_TO_PRELOAD:
        if repo_id in MODEL_INSTANCES:
            continue
        logger.info("Loading KittenTTS model: %s", repo_id)
        MODEL_INSTANCES[repo_id] = KittenTTS(repo_id)
        logger.info("Loaded KittenTTS model: %s", repo_id)


@app.on_event("startup")
async def startup_event():
    _load_all_models()


def resolve_model(model_id: str) -> KittenTTS:
    """Resolve an incoming OpenAI-style `model` field to a loaded KittenTTS instance."""
    repo_id = AVAILABLE_MODELS.get(model_id)
    if repo_id is None:
        # Unknown id — fall back to default rather than erroring, for compatibility
        # with clients that always send "tts-1" etc. Change to raise if you'd
        # rather be strict.
        repo_id = AVAILABLE_MODELS[DEFAULT_MODEL_ID]

    instance = MODEL_INSTANCES.get(repo_id)
    if instance is None:
        # Lazy-load as a fallback in case it wasn't in the preload list
        logger.warning("Model %s not preloaded, loading on demand", repo_id)
        instance = KittenTTS(repo_id)
        MODEL_INSTANCES[repo_id] = instance
    return instance


# ---------------------------------------------------------------------------
# Voice mapping
# ---------------------------------------------------------------------------
VOICE_MAPPING = {
    # OpenAI-style -> KittenTTS
    "alloy": "Jasper",
    "echo": "Bruno",
    "fable": "Luna",
    "onyx": "Bruno",
    "shimmer": "Kiki",
    # Direct mapping for KittenTTS voices
    "Jasper": "Jasper",
    "Bella": "Bella",
    "Luna": "Luna",
    "Bruno": "Bruno",
    "Rosie": "Rosie",
    "Hugo": "Hugo",
    "Kiki": "Kiki",
    "Leo": "Leo",
}

DEFAULT_VOICE = "Jasper"
SAMPLE_RATE = 24000


class TTSRequest(BaseModel):
    model: str = DEFAULT_MODEL_ID
    input: str
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0
    stream_format: str = "audio"


def convert_audio(audio_data, output_format: str, sample_rate: int = SAMPLE_RATE):
    """Convert audio to the requested format."""

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
    wav_buffer.seek(0)

    if output_format == "wav":
        wav_buffer.seek(0)
        return wav_buffer, "audio/wav", "wav"

    audio_segment = AudioSegment.from_wav(wav_buffer)
    output_buffer = io.BytesIO()

    if output_format == "mp3":
        audio_segment.export(output_buffer, format="mp3", bitrate="128k")
        media_type = "audio/mpeg"
        extension = "mp3"
    elif output_format == "ogg":
        audio_segment.export(output_buffer, format="ogg", bitrate="128k")
        media_type = "audio/ogg"
        extension = "ogg"
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    output_buffer.seek(0)
    return output_buffer, media_type, extension


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """OpenAI-compatible TTS endpoint, backed by a preloaded model of choice."""

    if not request.input:
        raise HTTPException(status_code=400, detail="Input text is required")

    supported_formats = ["wav", "mp3", "ogg"]
    if request.response_format not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(supported_formats)}"
        )

    if request.stream_format != "audio":
        raise HTTPException(
            status_code=400,
            detail="Unsupported stream format. Supported: audio"
        )

    mapped_voice = VOICE_MAPPING.get(request.voice, DEFAULT_VOICE)
    model_instance = resolve_model(request.model)

    try:
        audio = model_instance.generate(
            request.input, voice=mapped_voice, speed=request.speed
        )

        audio_buffer, media_type, extension = convert_audio(
            audio, request.response_format, SAMPLE_RATE
        )

        return Response(
            content=audio_buffer.read(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{extension}"
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "kitten",
                "repo": repo_id,
                "loaded": repo_id in MODEL_INSTANCES,
            }
            for model_id, repo_id in AVAILABLE_MODELS.items()
        ]
    }


@app.get("/v1/voices")
async def list_voices():
    """List available voices."""
    voices = list(VOICE_MAPPING.keys())
    return {
        "object": "list",
        "data": [
            {"id": voice, "name": voice.title(), "object": "voice"}
            for voice in voices
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint, reports which models are resident in memory."""
    return {
        "status": "healthy",
        "loaded_models": list(MODEL_INSTANCES.keys()),
    }
