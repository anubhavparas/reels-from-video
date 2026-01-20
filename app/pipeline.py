import importlib.util
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .utils import command_exists, extract_audio, get_duration_seconds


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

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)

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


def suggest_clips(
    video_path: str,
    work_dir: str,
    min_seconds: int,
    max_seconds: int,
    max_results: int,
    model_name: str,
    scoring_mode: str = "combined",
) -> tuple[list[ClipSuggestion], Optional[str]]:
    """
    Generate clip suggestions from a video.

    scoring_mode options:
    - "coherence": Prioritize internal coherence (sentences relate to each other)
    - "topic": Prioritize staying within topic boundaries
    - "combined": Balance both coherence and topic segmentation
    """
    duration = get_duration_seconds(video_path)
    segments, warning = transcribe_video(video_path, work_dir, model_name)

    if segments:
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
            else:  # combined
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
