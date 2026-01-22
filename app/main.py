import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .pipeline import ClipSuggestion, suggest_clips
from .utils import command_exists, cut_clip, ensure_dir


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
WORK_DIR = DATA_DIR / "work"

ensure_dir(str(UPLOAD_DIR))
ensure_dir(str(WORK_DIR))

app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/media", StaticFiles(directory=str(DATA_DIR)), name="media")


def _safe_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _sanitize_filename(filename: str) -> str:
    """Remove extension and sanitize filename for use in folder name."""
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Replace spaces and special chars with underscores
    name = re.sub(r"[^\w\-]", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    # Trim and limit length
    name = name.strip("_")[:30]
    return name or "video"


def _generate_folder_name(filename: str) -> str:
    """Generate a folder name with timestamp and video name."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_name = _sanitize_filename(filename)
    short_id = uuid.uuid4().hex[:6]
    return f"{timestamp}_{video_name}_{short_id}"


async def _save_upload(file: UploadFile, dest_path: Path) -> None:
    with dest_path.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _serialize_clip(clip: ClipSuggestion) -> dict:
    return {
        "start": round(clip.start, 2),
        "end": round(clip.end, 2),
        "score": round(clip.score, 3),
        "text": clip.text[:400],
        "text_score": round(clip.text_score, 2),
        "coherence_score": round(clip.coherence_score, 3),
        "distinctiveness_score": round(clip.distinctiveness_score, 3),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/clip-suggestions")
async def clip_suggestions(
    video: UploadFile = File(...),
    min_seconds: int = Form(60),
    max_seconds: int = Form(180),
    max_results: int = Form(5),
    model_name: str = Form("base"),
    scoring_mode: str = Form("combined"),
    api_key: str = Form(""),
    llm_model: str = Form("gpt-4o-mini"),
    llm_provider: str = Form("openai"),
    ollama_url: str = Form("http://127.0.0.1:11434"),
    render: int = Form(0),
) -> JSONResponse:
    if not video.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    min_seconds = _safe_int(min_seconds, 10, 300)
    max_seconds = _safe_int(max_seconds, min_seconds, 300)
    max_results = _safe_int(max_results, 1, 20)

    if scoring_mode not in ("coherence", "topic", "combined", "llm"):
        scoring_mode = "combined"

    valid_providers = ("openai", "gemini", "huggingface", "ollama")
    if llm_provider not in valid_providers:
        llm_provider = "openai"

    # Ollama is local, no API key needed
    if scoring_mode == "llm" and llm_provider != "ollama" and not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"LLM mode with {llm_provider} requires an API key",
        )

    folder_name = _generate_folder_name(video.filename)
    file_ext = os.path.splitext(video.filename)[1] or ".mp4"
    video_path = UPLOAD_DIR / f"{folder_name}{file_ext}"
    work_dir = WORK_DIR / folder_name
    ensure_dir(str(work_dir))

    await _save_upload(video, video_path)

    clips, warning = suggest_clips(
        video_path=str(video_path),
        work_dir=str(work_dir),
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        max_results=max_results,
        model_name=model_name,
        scoring_mode=scoring_mode,
        api_key=api_key,
        llm_model=llm_model,
        llm_provider=llm_provider,
        ollama_url=ollama_url,
    )

    if not clips:
        raise HTTPException(
            status_code=400,
            detail=warning or "No clips could be generated",
        )

    response_clips = [_serialize_clip(clip) for clip in clips]

    if render:
        if not command_exists("ffmpeg"):
            raise HTTPException(status_code=400, detail="ffmpeg not found on PATH")
        clips_dir = work_dir / "clips"
        ensure_dir(str(clips_dir))
        for idx, clip in enumerate(clips):
            output_name = f"clip_{idx + 1}.mp4"
            output_path = clips_dir / output_name
            cut_clip(str(video_path), str(output_path), clip.start, clip.end)
            response_clips[idx]["file"] = f"/media/work/{folder_name}/clips/{output_name}"

    return JSONResponse(
        {
            "ok": True,
            "warning": warning,
            "clips": response_clips,
        }
    )
