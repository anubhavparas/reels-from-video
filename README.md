# Reels From Video

Web app that suggests 60-120 second clip ranges from an uploaded video.

## What it does
- Upload a video
- Generate clip ranges based on transcript timing
- Fall back to time-based clips if transcription is unavailable
- Optionally render clip files with ffmpeg

## Install
```
cd /Users/aparas/Documents/Projects/reels_from_video
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Transcription support
For transcript-based clip scoring, install ffmpeg:
```
conda install -c conda-forge ffmpeg
```
or on macOS with Homebrew:
```
brew install ffmpeg
```

The app uses OpenAI Whisper (included in requirements.txt).

## Run
```
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000
