import os
import uuid
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
    render: int = Form(0),
) -> JSONResponse:
    if not video.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    min_seconds = _safe_int(min_seconds, 10, 300)
    max_seconds = _safe_int(max_seconds, min_seconds, 300)
    max_results = _safe_int(max_results, 1, 20)

    if scoring_mode not in ("coherence", "topic", "combined"):
        scoring_mode = "combined"

    request_id = uuid.uuid4().hex
    file_ext = os.path.splitext(video.filename)[1] or ".mp4"
    video_path = UPLOAD_DIR / f"{request_id}{file_ext}"
    work_dir = WORK_DIR / request_id
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
            response_clips[idx]["file"] = f"/media/work/{request_id}/clips/{output_name}"

    return JSONResponse(
        {
            "ok": True,
            "warning": warning,
            "clips": response_clips,
        }
    )
