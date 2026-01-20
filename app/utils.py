import os
import shutil
import subprocess
from typing import Optional


def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cmd(args: list[str]) -> str:
    result = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_duration_seconds(video_path: str) -> Optional[float]:
    if not command_exists("ffprobe"):
        return None
    try:
        output = run_cmd(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]
        )
        return float(output)
    except Exception:
        return None


def extract_audio(video_path: str, audio_path: str) -> None:
    if not command_exists("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            audio_path,
        ]
    )


def cut_clip(video_path: str, output_path: str, start: float, end: float) -> None:
    if not command_exists("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.2f}",
            "-to",
            f"{end:.2f}",
            "-i",
            video_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            output_path,
        ]
    )
