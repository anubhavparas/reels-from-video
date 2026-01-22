import importlib.util
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .utils import command_exists, extract_audio, get_duration_seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _has_openai() -> bool:
    return importlib.util.find_spec("openai") is not None


def _has_google_genai() -> bool:
    return importlib.util.find_spec("google.generativeai") is not None


def _has_huggingface_hub() -> bool:
    return importlib.util.find_spec("huggingface_hub") is not None


def _has_requests() -> bool:
    return importlib.util.find_spec("requests") is not None


# LLM Provider constants
LLM_PROVIDER_OPENAI = "openai"
LLM_PROVIDER_GEMINI = "gemini"
LLM_PROVIDER_HUGGINGFACE = "huggingface"
LLM_PROVIDER_OLLAMA = "ollama"


FILLER_WORDS = {
    "um",
    "uh",
    "like",
    "you",
    "know",
    "so",
    "actually",
    "basically",
    "right",
}

# Embedding model singleton
_embedding_model = None


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class TopicBlock:
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float


@dataclass
class ClipSuggestion:
    start: float
    end: float
    score: float
    text: str
    text_score: float = 0.0
    coherence_score: float = 0.0
    distinctiveness_score: float = 0.0


def _has_whisper() -> bool:
    return importlib.util.find_spec("whisper") is not None


def _has_sentence_transformers() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def transcribe_video(
    video_path: str,
    work_dir: str,
    model_name: str,
) -> tuple[Optional[list[TranscriptSegment]], Optional[str]]:
    if not _has_whisper():
        return None, "openai-whisper not installed"
    if not command_exists("ffmpeg"):
        return None, "ffmpeg not found on PATH"

    audio_path = os.path.join(work_dir, "audio.wav")
    extract_audio(video_path, audio_path)

    import whisper

    logger.info("=" * 60)
    logger.info("WHISPER TRANSCRIPTION")
    logger.info("=" * 60)
    logger.info(f"Loading Whisper model: {model_name}")

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)

    logger.info(f"Whisper returned {len(result.get('segments', []))} segments")
    logger.info("-" * 40)
    logger.info("WHISPER OUTPUT (first 10 segments):")
    for i, seg in enumerate(result.get("segments", [])[:10]):
        logger.info(f"  [{seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s]: {seg.get('text', '')[:80]}...")
    if len(result.get("segments", [])) > 10:
        logger.info(f"  ... and {len(result.get('segments', [])) - 10} more segments")
    logger.info("-" * 40)

    results: list[TranscriptSegment] = []
    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if not text:
            continue
        results.append(
            TranscriptSegment(
                start=float(segment.get("start", 0)),
                end=float(segment.get("end", 0)),
                text=text,
            )
        )
    return results, None


def compute_embeddings(segments: list[TranscriptSegment]) -> None:
    """Compute embeddings for all segments using sentence-transformers."""
    if not _has_sentence_transformers():
        return
    if not segments:
        return

    model = _get_embedding_model()
    texts = [seg.text for seg in segments]
    embeddings = model.encode(texts, convert_to_numpy=True)

    for seg, emb in zip(segments, embeddings):
        seg.embedding = emb


def detect_topic_shifts(
    segments: list[TranscriptSegment],
    similarity_threshold: float = 0.3,
) -> list[int]:
    """
    Detect topic shift indices based on embedding similarity.
    Returns list of segment indices where a topic shift occurs.
    """
    if len(segments) < 2:
        return []

    shift_indices = []
    for i in range(1, len(segments)):
        prev_emb = segments[i - 1].embedding
        curr_emb = segments[i].embedding

        if prev_emb is None or curr_emb is None:
            continue

        similarity = _cosine_similarity(prev_emb, curr_emb)
        if similarity < similarity_threshold:
            shift_indices.append(i)

    return shift_indices


def build_topic_blocks(
    segments: list[TranscriptSegment],
    shift_indices: list[int],
) -> list[TopicBlock]:
    """Build contiguous topic blocks from segments and shift indices."""
    if not segments:
        return []

    blocks = []
    start_idx = 0

    for shift_idx in shift_indices:
        if shift_idx > start_idx:
            blocks.append(
                TopicBlock(
                    start_idx=start_idx,
                    end_idx=shift_idx - 1,
                    start_time=segments[start_idx].start,
                    end_time=segments[shift_idx - 1].end,
                )
            )
        start_idx = shift_idx

    # Add final block
    if start_idx < len(segments):
        blocks.append(
            TopicBlock(
                start_idx=start_idx,
                end_idx=len(segments) - 1,
                start_time=segments[start_idx].start,
                end_time=segments[-1].end,
            )
        )

    return blocks


def score_text(text: str) -> float:
    """Original text-based scoring."""
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if not words:
        return 0.0

    word_count = len(words)
    unique_count = len(set(words))
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    excitement = text.count("!") * 2.0 + text.count("?") * 1.5

    return word_count + unique_count * 0.5 + excitement - filler_count * 1.5


def score_coherence(segments: list[TranscriptSegment]) -> float:
    """
    Score internal coherence of a clip.
    Higher score = sentences within the clip are more related to each other.
    """
    if len(segments) < 2:
        return 1.0

    embeddings = [seg.embedding for seg in segments if seg.embedding is not None]
    if len(embeddings) < 2:
        return 0.5

    # Compute pairwise similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.5


def score_distinctiveness(
    clip_segments: list[TranscriptSegment],
    all_segments: list[TranscriptSegment],
) -> float:
    """
    Score how distinctive/unique a clip is compared to the overall video.
    Higher score = clip covers unique content not repeated elsewhere.
    """
    if not clip_segments:
        return 0.0

    clip_embeddings = [seg.embedding for seg in clip_segments if seg.embedding is not None]
    if not clip_embeddings:
        return 0.5

    # Compute clip centroid
    clip_centroid = np.mean(clip_embeddings, axis=0)

    # Get embeddings outside this clip
    clip_set = set(id(seg) for seg in clip_segments)
    other_embeddings = [
        seg.embedding for seg in all_segments
        if seg.embedding is not None and id(seg) not in clip_set
    ]

    if not other_embeddings:
        return 1.0

    # Compute similarity to other content
    other_centroid = np.mean(other_embeddings, axis=0)
    similarity_to_rest = _cosine_similarity(clip_centroid, other_centroid)

    # Lower similarity to rest = more distinctive
    return 1.0 - similarity_to_rest


def _remove_overlapping_clips(
    candidates: list[ClipSuggestion],
    max_results: int,
    overlap_threshold: float = 0.5,
) -> list[ClipSuggestion]:
    """Remove overlapping clips, preferring higher scored ones."""
    selected: list[ClipSuggestion] = []
    for candidate in candidates:
        overlaps = False
        for existing in selected:
            overlap_start = max(candidate.start, existing.start)
            overlap_end = min(candidate.end, existing.end)
            overlap_duration = max(0, overlap_end - overlap_start)
            candidate_duration = candidate.end - candidate.start
            if overlap_duration > candidate_duration * overlap_threshold:
                overlaps = True
                break
        if not overlaps:
            selected.append(candidate)
        if len(selected) >= max_results:
            break
    return selected


def build_clips_coherence_mode(
    segments: list[TranscriptSegment],
    min_seconds: int,
    max_seconds: int,
    max_results: int,
) -> list[ClipSuggestion]:
    """
    Build clips scored primarily by internal coherence.
    Higher coherence = sentences within the clip are more related.
    """
    if not segments:
        return []

    candidates: list[ClipSuggestion] = []
    n = len(segments)

    for i in range(n):
        start = segments[i].start
        end = start
        collected_segments: list[TranscriptSegment] = []
        text_parts: list[str] = []
        max_gap = 3.0  # Allow slightly larger gaps for longer clips
        last_end = segments[i].start

        for j in range(i, n):
            gap = segments[j].start - last_end
            if gap > max_gap and j > i:
                break

            end = segments[j].end
            text_parts.append(segments[j].text)
            collected_segments.append(segments[j])
            last_end = segments[j].end
            duration = end - start

            if duration >= min_seconds:
                if duration <= max_seconds:
                    text = " ".join(text_parts).strip()
                    text_score = score_text(text)
                    coherence = score_coherence(collected_segments)
                    distinctiveness = score_distinctiveness(collected_segments, segments)

                    # Coherence-focused score
                    text_norm = min(text_score / 100.0, 2.0)
                    combined = (
                        text_norm * 0.2
                        + coherence * 0.6  # Emphasize coherence
                        + distinctiveness * 0.2
                    )

                    candidates.append(
                        ClipSuggestion(
                            start=start,
                            end=end,
                            score=round(combined, 3),
                            text=text,
                            text_score=round(text_score, 2),
                            coherence_score=round(coherence, 3),
                            distinctiveness_score=round(distinctiveness, 3),
                        )
                    )
                break
            if duration > max_seconds:
                break

    candidates.sort(key=lambda c: c.score, reverse=True)
    return _remove_overlapping_clips(candidates, max_results)


def build_clips_topic_mode(
    segments: list[TranscriptSegment],
    topic_blocks: list[TopicBlock],
    min_seconds: int,
    max_seconds: int,
    max_results: int,
) -> list[ClipSuggestion]:
    """
    Build clips that stay within topic boundaries.
    Clips are constrained to not cross topic shifts.
    """
    if not segments:
        return []

    candidates: list[ClipSuggestion] = []

    # Build clips within each topic block
    for block in topic_blocks:
        block_duration = block.end_time - block.start_time

        # Skip blocks that are too short
        if block_duration < min_seconds:
            continue

        # Build clips within this block
        block_segments = segments[block.start_idx : block.end_idx + 1]
        n = len(block_segments)

        for i in range(n):
            start = block_segments[i].start
            end = start
            collected_segments: list[TranscriptSegment] = []
            text_parts: list[str] = []
            max_gap = 3.0
            last_end = block_segments[i].start

            for j in range(i, n):
                gap = block_segments[j].start - last_end
                if gap > max_gap and j > i:
                    break

                end = block_segments[j].end
                text_parts.append(block_segments[j].text)
                collected_segments.append(block_segments[j])
                last_end = block_segments[j].end
                duration = end - start

                if duration >= min_seconds:
                    if duration <= max_seconds:
                        text = " ".join(text_parts).strip()
                        text_score = score_text(text)
                        coherence = score_coherence(collected_segments)
                        distinctiveness = score_distinctiveness(collected_segments, segments)

                        # Topic-focused score (bonus for staying in topic)
                        text_norm = min(text_score / 100.0, 2.0)
                        topic_bonus = 1.2  # Bonus for staying within topic
                        combined = (
                            text_norm * 0.3
                            + coherence * 0.3
                            + distinctiveness * 0.4  # Emphasize distinctiveness
                        ) * topic_bonus

                        candidates.append(
                            ClipSuggestion(
                                start=start,
                                end=end,
                                score=round(combined, 3),
                                text=text,
                                text_score=round(text_score, 2),
                                coherence_score=round(coherence, 3),
                                distinctiveness_score=round(distinctiveness, 3),
                            )
                        )
                    break
                if duration > max_seconds:
                    break

    candidates.sort(key=lambda c: c.score, reverse=True)
    return _remove_overlapping_clips(candidates, max_results)


def build_clips_combined_mode(
    segments: list[TranscriptSegment],
    topic_blocks: list[TopicBlock],
    min_seconds: int,
    max_seconds: int,
    max_results: int,
) -> list[ClipSuggestion]:
    """
    Build clips using both coherence and topic segmentation.
    Clips prefer to stay within topic boundaries but can cross with penalty.
    """
    if not segments:
        return []

    candidates: list[ClipSuggestion] = []
    n = len(segments)

    # Create a set of topic shift indices for quick lookup
    shift_set = set()
    for block in topic_blocks:
        if block.start_idx > 0:
            shift_set.add(block.start_idx)

    for i in range(n):
        start = segments[i].start
        end = start
        collected_segments: list[TranscriptSegment] = []
        text_parts: list[str] = []
        max_gap = 3.0
        last_end = segments[i].start
        crossed_topic = False

        for j in range(i, n):
            # Check for topic shift
            if j in shift_set and j > i:
                crossed_topic = True

            gap = segments[j].start - last_end
            if gap > max_gap and j > i:
                break

            end = segments[j].end
            text_parts.append(segments[j].text)
            collected_segments.append(segments[j])
            last_end = segments[j].end
            duration = end - start

            if duration >= min_seconds:
                if duration <= max_seconds:
                    text = " ".join(text_parts).strip()
                    text_score = score_text(text)
                    coherence = score_coherence(collected_segments)
                    distinctiveness = score_distinctiveness(collected_segments, segments)

                    # Penalty for crossing topic boundaries
                    topic_penalty = 0.8 if crossed_topic else 1.0

                    # Combined score (balanced)
                    text_norm = min(text_score / 100.0, 2.0)
                    combined = (
                        text_norm * 0.3
                        + coherence * 0.4
                        + distinctiveness * 0.3
                    ) * topic_penalty

                    candidates.append(
                        ClipSuggestion(
                            start=start,
                            end=end,
                            score=round(combined, 3),
                            text=text,
                            text_score=round(text_score, 2),
                            coherence_score=round(coherence, 3),
                            distinctiveness_score=round(distinctiveness, 3),
                        )
                    )
                break
            if duration > max_seconds:
                break

    candidates.sort(key=lambda c: c.score, reverse=True)
    return _remove_overlapping_clips(candidates, max_results)


def build_clip_suggestions(
    segments: list[TranscriptSegment],
    min_seconds: int,
    max_seconds: int,
    max_results: int,
) -> list[ClipSuggestion]:
    """Original clip building without embeddings (fallback)."""
    if not segments:
        return []

    candidates: list[ClipSuggestion] = []
    n = len(segments)
    for i in range(n):
        start = segments[i].start
        end = start
        text_parts: list[str] = []
        max_gap = 2.5
        last_end = segments[i].start
        for j in range(i, n):
            gap = segments[j].start - last_end
            if gap > max_gap and j > i:
                break
            end = segments[j].end
            text_parts.append(segments[j].text)
            last_end = segments[j].end
            duration = end - start
            if duration >= min_seconds:
                if duration <= max_seconds:
                    text = " ".join(text_parts).strip()
                    score = score_text(text)
                    candidates.append(
                        ClipSuggestion(
                            start=start,
                            end=end,
                            score=score,
                            text=text,
                            text_score=score,
                        )
                    )
                break
            if duration > max_seconds:
                break

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_results]


def build_time_based_clips(
    duration: Optional[float],
    min_seconds: int,
    max_seconds: int,
    max_results: int,
) -> list[ClipSuggestion]:
    if not duration:
        return []

    target = 90
    clip_len = min(max_seconds, max(min_seconds, target))
    clips: list[ClipSuggestion] = []
    start = 0.0
    while start + min_seconds <= duration and len(clips) < max_results:
        end = min(start + clip_len, duration)
        clips.append(
            ClipSuggestion(
                start=start,
                end=end,
                score=0.0,
                text="",
            )
        )
        start += clip_len
    return clips


# =============================================================================
# LLM-BASED SEGMENTATION
# =============================================================================

def _format_transcript_for_llm(
    segments: list[TranscriptSegment],
    max_chars: int = 12000,
) -> str:
    """Format transcript with timestamps for LLM input."""
    lines = []
    total_chars = 0

    for seg in segments:
        line = f"[{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text}"
        if total_chars + len(line) > max_chars:
            lines.append("... (transcript truncated)")
            break
        lines.append(line)
        total_chars += len(line) + 1

    return "\n".join(lines)


def _build_llm_prompt(
    transcript: str,
    min_seconds: int,
    max_seconds: int,
    max_results: int,
) -> str:
    """Build the prompt for LLM-based segmentation."""
    target_duration = (min_seconds + max_seconds) // 2
    return f"""You are analyzing a video transcript to identify the best clip segments for social media content.

TRANSCRIPT WITH TIMESTAMPS:
{transcript}

CRITICAL DURATION REQUIREMENTS:
- MINIMUM duration: {min_seconds} seconds (MUST be at least this long)
- MAXIMUM duration: {max_seconds} seconds
- TARGET duration: around {target_duration} seconds per clip
- DO NOT return any segment shorter than {min_seconds} seconds

TASK:
1. Identify distinct topics or coherent segments in the transcript, basically check where a new topic starts or ends
2. Select the top {max_results} most engaging segments
3. Each segment MUST not be longer than {max_seconds} seconds
4. Prefer LONGER segments that cover complete topics over short snippets
5. Combine related content to reach the minimum duration if needed
6. Prioritize segments that are: informative, entertaining, surprising, or emotionally engaging
7. Make sure that the output segments do not overlap with each other by more than 20% of the segment duration

RESPOND WITH ONLY A JSON ARRAY (no other text):
[
  {{
    "start": <start_time_in_seconds>,
    "end": <end_time_in_seconds>,
    "rank": <1_to_{max_results}_where_1_is_best>,
    "topic": "<brief_topic_description>",
    "reason": "<why_this_segment_is_engaging>"
  }},
  ...
]

REQUIREMENTS:
- Each segment MUST have (end - start) >= {min_seconds} seconds
- Use the exact timestamps from the transcript
- Ensure start and end times align with sentence boundaries
- Return exactly {max_results} segments (or fewer if not enough content)
- Segments should not overlap significantly
- JSON must be valid and parseable
- If the start of a segment's topic is not matching the timestamps of the transcript, you can adjust the start time to match the transcript. For example, if the segment starts at 10.0 seconds but the topic starts at 15.0 seconds, you can adjust the start time to 15.0 seconds based on the transcript maximum duration.

EXAMPLE: If min_seconds={min_seconds}, a valid segment would be:
{{"start": 10.0, "end": {10.0 + min_seconds + 30}, "rank": 1, "topic": "...", "reason": "..."}}"""


@dataclass
class LLMSegment:
    start: float
    end: float
    rank: int
    topic: str
    reason: str


def _parse_llm_response(response_text: str) -> list[LLMSegment]:
    """Parse the LLM response into segments."""
    # Try to extract JSON from the response
    response_text = response_text.strip()

    # Handle markdown code blocks
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        response_text = response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        response_text = response_text[start:end].strip()

    # Find JSON array
    if "[" in response_text:
        start = response_text.find("[")
        end = response_text.rfind("]") + 1
        response_text = response_text[start:end]

    try:
        data = json.loads(response_text)
        segments = []
        for item in data:
            segments.append(
                LLMSegment(
                    start=float(item.get("start", 0)),
                    end=float(item.get("end", 0)),
                    rank=int(item.get("rank", 1)),
                    topic=str(item.get("topic", "")),
                    reason=str(item.get("reason", "")),
                )
            )
        return segments
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return []


def _call_openai(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str,
) -> tuple[str, Optional[str]]:
    """Call OpenAI API."""
    if not _has_openai():
        return "", "openai package not installed"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content or "", None
    except Exception as e:
        return "", f"OpenAI API error: {str(e)}"


def _call_gemini(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str,
) -> tuple[str, Optional[str]]:
    """Call Google Gemini API."""
    if not _has_google_genai():
        return "", "google-generativeai package not installed"

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(model)

        full_prompt = f"{system_prompt}\n\n{prompt}"
        response = gen_model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2000,
            ),
        )
        return response.text or "", None
    except Exception as e:
        return "", f"Gemini API error: {str(e)}"


def _call_huggingface(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str,
) -> tuple[str, Optional[str]]:
    """Call HuggingFace Inference API."""
    if not _has_huggingface_hub():
        return "", "huggingface_hub package not installed"

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=api_key)

        full_prompt = f"{system_prompt}\n\n{prompt}"
        response = client.text_generation(
            full_prompt,
            model=model,
            max_new_tokens=2000,
            temperature=0.3,
        )
        return response or "", None
    except Exception as e:
        return "", f"HuggingFace API error: {str(e)}"


def _call_ollama(
    prompt: str,
    system_prompt: str,
    model: str,
    ollama_url: str = "http://127.0.0.1:11434",
) -> tuple[str, Optional[str]]:
    """Call local Ollama API."""
    if not _has_requests():
        return "", "requests package not installed"

    try:
        import requests

        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": f"{system_prompt}\n\n{prompt}",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 2000,
                },
            },
            timeout=120,
        )

        if response.status_code != 200:
            return "", f"Ollama error: HTTP {response.status_code}"

        data = response.json()
        return data.get("response", ""), None
    except requests.exceptions.ConnectionError:
        return "", "Ollama not running. Start with: ollama serve"
    except Exception as e:
        return "", f"Ollama API error: {str(e)}"


def call_llm_for_segmentation(
    segments: list[TranscriptSegment],
    min_seconds: int,
    max_seconds: int,
    max_results: int,
    api_key: str,
    llm_model: str = "gpt-4o-mini",
    llm_provider: str = LLM_PROVIDER_OPENAI,
    ollama_url: str = "http://127.0.0.1:11434",
) -> tuple[list[LLMSegment], Optional[str]]:
    """Call LLM API to get segment suggestions. Supports multiple providers."""
    transcript = _format_transcript_for_llm(segments)
    prompt = _build_llm_prompt(transcript, min_seconds, max_seconds, max_results)
    system_prompt = "You are an expert video editor who identifies the most engaging clips from video transcripts. Always respond with valid JSON only."

    # Log LLM input
    logger.info("=" * 60)
    logger.info("LLM INPUT")
    logger.info("=" * 60)
    logger.info(f"Provider: {llm_provider}")
    logger.info(f"Model: {llm_model}")
    logger.info(f"Min seconds: {min_seconds}, Max seconds: {max_seconds}, Max results: {max_results}")
    logger.info("-" * 40)
    logger.info("PROMPT (first 2000 chars):")
    logger.info(prompt[:2000])
    if len(prompt) > 2000:
        logger.info(f"... (truncated, total {len(prompt)} chars)")
    logger.info("-" * 40)

    # Route to appropriate provider
    if llm_provider == LLM_PROVIDER_OPENAI:
        if not api_key:
            return [], "OpenAI API key not provided"
        response_text, error = _call_openai(prompt, system_prompt, api_key, llm_model)

    elif llm_provider == LLM_PROVIDER_GEMINI:
        if not api_key:
            return [], "Gemini API key not provided"
        response_text, error = _call_gemini(prompt, system_prompt, api_key, llm_model)

    elif llm_provider == LLM_PROVIDER_HUGGINGFACE:
        if not api_key:
            return [], "HuggingFace API token not provided"
        response_text, error = _call_huggingface(prompt, system_prompt, api_key, llm_model)

    elif llm_provider == LLM_PROVIDER_OLLAMA:
        # Ollama is local, no API key needed
        response_text, error = _call_ollama(prompt, system_prompt, llm_model, ollama_url)

    else:
        return [], f"Unknown LLM provider: {llm_provider}"

    # Log LLM output
    logger.info("=" * 60)
    logger.info("LLM OUTPUT")
    logger.info("=" * 60)
    if error:
        logger.error(f"LLM Error: {error}")
        return [], error

    logger.info("RAW RESPONSE:")
    logger.info(response_text)
    logger.info("-" * 40)

    llm_segments = _parse_llm_response(response_text)

    if not llm_segments:
        logger.error("Failed to parse LLM response into segments")
        return [], "Failed to parse LLM response"

    logger.info(f"PARSED {len(llm_segments)} SEGMENTS:")
    for seg in llm_segments:
        duration = seg.end - seg.start
        logger.info(f"  Rank {seg.rank}: [{seg.start:.1f}s - {seg.end:.1f}s] ({duration:.1f}s) - {seg.topic}")
    logger.info("=" * 60)

    return llm_segments, None


def build_clips_llm_mode(
    segments: list[TranscriptSegment],
    llm_segments: list[LLMSegment],
    all_segments: list[TranscriptSegment],
    min_seconds: int = 60,
    max_seconds: int = 300,
) -> list[ClipSuggestion]:
    """
    Build clips based on LLM suggestions.
    Uses LLM-provided boundaries and adds computed scores.
    Extends short clips to meet minimum duration.
    """
    if not llm_segments:
        return []

    candidates: list[ClipSuggestion] = []

    for llm_seg in llm_segments:
        # Find segments within the LLM-suggested time range
        clip_segments = [
            seg for seg in segments
            if seg.start >= llm_seg.start - 1.0 and seg.end <= llm_seg.end + 1.0
        ]

        if not clip_segments:
            # Use the LLM boundaries directly
            duration = llm_seg.end - llm_seg.start
            # Extend if too short
            if duration < min_seconds:
                # Try to extend by adding nearby content
                center = (llm_seg.start + llm_seg.end) / 2
                half_target = min_seconds / 2 + 5  # Add buffer
                llm_seg.start = max(0, center - half_target)
                llm_seg.end = center + half_target

            text = llm_seg.topic + ": " + llm_seg.reason
            candidates.append(
                ClipSuggestion(
                    start=llm_seg.start,
                    end=llm_seg.end,
                    score=round(1.0 - (llm_seg.rank - 1) * 0.1, 3),
                    text=text,
                    text_score=0.0,
                    coherence_score=0.0,
                    distinctiveness_score=0.0,
                )
            )
            continue

        # Compute actual boundaries from segments
        start = min(seg.start for seg in clip_segments)
        end = max(seg.end for seg in clip_segments)

        # Extend clip if too short by including more adjacent segments
        duration = end - start
        if duration < min_seconds and segments:
            # Find the index range of current clip segments
            all_indices = [i for i, seg in enumerate(segments) if seg in clip_segments]
            if all_indices:
                first_idx = min(all_indices)
                last_idx = max(all_indices)

                # Extend backwards and forwards until we reach min_seconds
                while duration < min_seconds:
                    extended = False
                    # Try extending backwards
                    if first_idx > 0:
                        first_idx -= 1
                        clip_segments.insert(0, segments[first_idx])
                        start = segments[first_idx].start
                        extended = True
                    # Try extending forwards
                    if last_idx < len(segments) - 1:
                        last_idx += 1
                        clip_segments.append(segments[last_idx])
                        end = segments[last_idx].end
                        extended = True

                    duration = end - start
                    if not extended:
                        break

                    # Safety cap
                    if duration > max_seconds:
                        break

        text = " ".join(seg.text for seg in clip_segments).strip()

        # Compute scores if embeddings available
        text_score = score_text(text)
        coherence = score_coherence(clip_segments)
        distinctiveness = score_distinctiveness(clip_segments, all_segments)

        # LLM rank contributes to score (rank 1 = highest)
        llm_score = 1.0 - (llm_seg.rank - 1) * 0.1
        text_norm = min(text_score / 100.0, 2.0)

        combined = (
            llm_score * 0.4  # LLM ranking is important
            + text_norm * 0.2
            + coherence * 0.2
            + distinctiveness * 0.2
        )

        # Add topic/reason to text for display
        display_text = f"[{llm_seg.topic}] {text[:300]}"
        if llm_seg.reason:
            display_text += f" | Reason: {llm_seg.reason}"

        candidates.append(
            ClipSuggestion(
                start=start,
                end=end,
                score=round(combined, 3),
                text=display_text[:400],
                text_score=round(text_score, 2),
                coherence_score=round(coherence, 3),
                distinctiveness_score=round(distinctiveness, 3),
            )
        )

    # Sort by score (which incorporates LLM rank)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def suggest_clips(
    video_path: str,
    work_dir: str,
    min_seconds: int,
    max_seconds: int,
    max_results: int,
    model_name: str,
    scoring_mode: str = "combined",
    api_key: str = "",
    llm_model: str = "gpt-4o-mini",
    llm_provider: str = LLM_PROVIDER_OPENAI,
    ollama_url: str = "http://127.0.0.1:11434",
) -> tuple[list[ClipSuggestion], Optional[str]]:
    """
    Generate clip suggestions from a video.

    scoring_mode options:
    - "coherence": Prioritize internal coherence (sentences relate to each other)
    - "topic": Prioritize staying within topic boundaries
    - "combined": Balance both coherence and topic segmentation
    - "llm": Use LLM to identify and rank segments

    llm_provider options:
    - "openai": OpenAI GPT models (requires API key)
    - "gemini": Google Gemini models (requires API key)
    - "huggingface": HuggingFace Inference API (requires API token)
    - "ollama": Local Ollama instance (no API key needed)
    """
    duration = get_duration_seconds(video_path)
    segments, warning = transcribe_video(video_path, work_dir, model_name)

    if segments:
        # LLM mode - use LLM to segment and rank
        if scoring_mode == "llm":
            # Ollama doesn't need an API key
            if llm_provider != LLM_PROVIDER_OLLAMA and not api_key:
                return [], f"LLM mode with {llm_provider} requires an API key"

            # Compute embeddings for scoring (optional but helpful)
            if _has_sentence_transformers():
                compute_embeddings(segments)

            llm_segments, llm_error = call_llm_for_segmentation(
                segments,
                min_seconds,
                max_seconds,
                max_results,
                api_key,
                llm_model,
                llm_provider,
                ollama_url,
            )

            if llm_error:
                # Fallback to combined mode on LLM error
                warning = warning or ""
                warning += f" LLM error: {llm_error}. Falling back to combined mode."
                # Fall through to combined mode below
            else:
                clips = build_clips_llm_mode(
                    segments, llm_segments, segments, min_seconds, max_seconds
                )
                return clips, warning

        # Try embedding-based scoring if sentence-transformers is available
        if _has_sentence_transformers():
            compute_embeddings(segments)
            shift_indices = detect_topic_shifts(segments, similarity_threshold=0.3)
            topic_blocks = build_topic_blocks(segments, shift_indices)

            if scoring_mode == "coherence":
                clips = build_clips_coherence_mode(
                    segments, min_seconds, max_seconds, max_results
                )
                return clips, warning
            elif scoring_mode == "topic":
                clips = build_clips_topic_mode(
                    segments, topic_blocks, min_seconds, max_seconds, max_results
                )
                return clips, warning
            else:  # combined (or fallback from LLM error)
                clips = build_clips_combined_mode(
                    segments, topic_blocks, min_seconds, max_seconds, max_results
                )
                return clips, warning
        else:
            # Fallback to original text-based scoring
            return (
                build_clip_suggestions(segments, min_seconds, max_seconds, max_results),
                warning or "sentence-transformers not installed, using text-based scoring",
            )

    return (
        build_time_based_clips(duration, min_seconds, max_seconds, max_results),
        warning or "No transcript segments available, using time-based clips",
    )
