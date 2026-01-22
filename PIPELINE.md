# Reels From Video - Pipeline Documentation

## High-Level Architecture

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|   Web Browser    |---->|   FastAPI App    |---->|   Processing     |
|   (Frontend)     |<----|   (Backend)      |<----|   Pipeline       |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
                                  |
                                  v
                    +---------------------------+
                    |       External Tools      |
                    |  - FFmpeg (audio/video)   |
                    |  - Whisper (transcribe)   |
                    |  - Sentence-Transformers  |
                    |  - OpenAI API (LLM mode)  |
                    +---------------------------+
```

## Complete Pipeline Flow

```
USER UPLOAD
    |
    v
+-----------------------------------------------------------------------+
|                         1. API ENDPOINT                                |
|   POST /api/clip-suggestions                                          |
|   Input: FormData { video, min_seconds, max_seconds, max_results,     |
|                     model_name, scoring_mode, api_key, llm_model,     |
|                     llm_provider, ollama_url, render }                |
+-----------------------------------------------------------------------+
    |
    v
+-----------------------------------------------------------------------+
|                         2. FILE HANDLING                               |
|   - Save uploaded video to: data/uploads/{request_id}.mp4             |
|   - Create work directory: data/work/{request_id}/                    |
+-----------------------------------------------------------------------+
    |
    v
+-----------------------------------------------------------------------+
|                         3. TRANSCRIPTION                               |
|   transcribe_video()                                                  |
|   +-------------------------------------------------------------------+
|   |  3a. AUDIO EXTRACTION (FFmpeg)                                    |
|   |      Input:  video file (mp4/mov/etc)                             |
|   |      Output: data/work/{request_id}/audio.wav                     |
|   |      Command: ffmpeg -i video -ac 1 -ar 16000 -vn audio.wav      |
|   +-------------------------------------------------------------------+
|   |  3b. SPEECH-TO-TEXT (OpenAI Whisper)                              |
|   |      Model: whisper.load_model(model_name)                        |
|   |      Input:  audio.wav                                            |
|   |      Output: List[TranscriptSegment]                              |
|   |              - start: float (seconds)                             |
|   |              - end: float (seconds)                               |
|   |              - text: str                                          |
|   +-------------------------------------------------------------------+
+-----------------------------------------------------------------------+
    |
    v
+-----------------------------------------------------------------------+
|                         4. EMBEDDING COMPUTATION                       |
|   compute_embeddings() (if sentence-transformers installed)           |
|   +-------------------------------------------------------------------+
|   |  Model: SentenceTransformer("all-MiniLM-L6-v2")                   |
|   |  Input:  List[str] (segment texts)                                |
|   |  Output: List[np.ndarray] (384-dim vectors)                       |
|   |          Stored in: TranscriptSegment.embedding                   |
|   +-------------------------------------------------------------------+
+-----------------------------------------------------------------------+
    |
    v
+-----------------------------------------------------------------------+
|                         5. TOPIC DETECTION                             |
|   detect_topic_shifts()                                               |
|   +-------------------------------------------------------------------+
|   |  Method: Cosine similarity between consecutive embeddings         |
|   |  Threshold: 0.3 (similarity below = topic shift)                  |
|   |  Input:  List[TranscriptSegment] with embeddings                  |
|   |  Output: List[int] (indices where topic shifts occur)             |
|   +-------------------------------------------------------------------+
|   build_topic_blocks()                                                |
|   +-------------------------------------------------------------------+
|   |  Input:  segments, shift_indices                                  |
|   |  Output: List[TopicBlock]                                         |
|   |          - start_idx, end_idx: int                                |
|   |          - start_time, end_time: float                            |
|   +-------------------------------------------------------------------+
+-----------------------------------------------------------------------+
    |
    v
+-----------------------------------------------------------------------+
|                         6. CLIP BUILDING                               |
|   (based on scoring_mode)                                             |
|   +-------------------------------------------------------------------+
|   |  MODE: "coherence"  --> build_clips_coherence_mode()              |
|   |        - Iterates through all segments                            |
|   |        - Extends clips until min_seconds reached                  |
|   |        - Stops at max_seconds or 3s gap                           |
|   |        - Score: 20% text + 60% coherence + 20% distinctiveness    |
|   +-------------------------------------------------------------------+
|   |  MODE: "topic"      --> build_clips_topic_mode()                  |
|   |        - Only builds clips WITHIN topic blocks                    |
|   |        - Will not cross topic boundaries                          |
|   |        - Score: 30% text + 30% coherence + 40% distinctiveness    |
|   |        - 1.2x bonus for staying in topic                          |
|   +-------------------------------------------------------------------+
|   |  MODE: "combined"   --> build_clips_combined_mode()               |
|   |        - Can cross topic boundaries (with penalty)                |
|   |        - Score: 30% text + 40% coherence + 30% distinctiveness    |
|   |        - 0.8x penalty if crossing topic                           |
|   +-------------------------------------------------------------------+
|   |  MODE: "llm"        --> build_clips_llm_mode()                    |
|   |        - Sends transcript to OpenAI GPT for segmentation          |
|   |        - LLM identifies topics and ranks best segments            |
|   |        - Score: 40% LLM rank + 20% text + 20% coherence           |
|   |                + 20% distinctiveness                              |
|   |        - Requires: api_key (OpenAI API key)                       |
|   +-------------------------------------------------------------------+
+-----------------------------------------------------------------------+
    |
    v
+-----------------------------------------------------------------------+
|                         7. SCORING FUNCTIONS                           |
|   +-------------------------------------------------------------------+
|   |  score_text(text) -> float                                        |
|   |      Formula: word_count + unique_words*0.5 + excitement          |
|   |               - filler_words*1.5                                  |
|   |      excitement: "!" = +2, "?" = +1.5                             |
|   |      filler_words: um, uh, like, you know, so, actually, etc.     |
|   +-------------------------------------------------------------------+
|   |  score_coherence(segments) -> float                               |
|   |      Method: Mean cosine similarity of consecutive embeddings     |
|   |      Range: 0.0 to 1.0 (higher = more coherent)                   |
|   +-------------------------------------------------------------------+
|   |  score_distinctiveness(clip_segments, all_segments) -> float      |
|   |      Method: 1 - cosine_sim(clip_centroid, rest_centroid)         |
|   |      Range: 0.0 to 1.0 (higher = more unique content)             |
|   +-------------------------------------------------------------------+
+-----------------------------------------------------------------------+
    |
    v
+-----------------------------------------------------------------------+
|                         8. OVERLAP REMOVAL                             |
|   _remove_overlapping_clips()                                         |
|   +-------------------------------------------------------------------+
|   |  - Sort candidates by score (descending)                          |
|   |  - For each candidate, check overlap with selected clips          |
|   |  - If overlap > 50% of clip duration, skip                        |
|   |  - Return top N non-overlapping clips                             |
|   +-------------------------------------------------------------------+
+-----------------------------------------------------------------------+
    |
    v
+-----------------------------------------------------------------------+
|                         9. CLIP RENDERING (optional)                   |
|   If render=1                                                         |
|   +-------------------------------------------------------------------+
|   |  cut_clip() using FFmpeg                                          |
|   |  Input:  video_path, start, end                                   |
|   |  Output: data/work/{request_id}/clips/clip_{n}.mp4                |
|   |  Command: ffmpeg -ss {start} -to {end} -i video                   |
|   |           -c:v libx264 -c:a aac -movflags +faststart output.mp4   |
|   +-------------------------------------------------------------------+
+-----------------------------------------------------------------------+
    |
    v
+-----------------------------------------------------------------------+
|                         10. API RESPONSE                               |
|   Output: JSON                                                        |
|   {                                                                   |
|     "ok": true,                                                       |
|     "warning": "string or null",                                      |
|     "clips": [                                                        |
|       {                                                               |
|         "start": 0.0,           // seconds                            |
|         "end": 120.0,           // seconds                            |
|         "score": 0.856,         // combined score                     |
|         "text": "transcript...",// first 400 chars                    |
|         "text_score": 45.5,     // raw text score                     |
|         "coherence_score": 0.72,// 0-1                                |
|         "distinctiveness_score": 0.65, // 0-1                         |
|         "file": "/media/work/{id}/clips/clip_1.mp4" // if rendered    |
|       },                                                              |
|       ...                                                             |
|     ]                                                                 |
|   }                                                                   |
+-----------------------------------------------------------------------+
```

## Data Structures

### TranscriptSegment
```python
@dataclass
class TranscriptSegment:
    start: float          # Start time in seconds
    end: float            # End time in seconds
    text: str             # Transcribed text
    embedding: Optional[np.ndarray] = None  # 384-dim vector
```

### TopicBlock
```python
@dataclass
class TopicBlock:
    start_idx: int        # Index of first segment in block
    end_idx: int          # Index of last segment in block
    start_time: float     # Start time in seconds
    end_time: float       # End time in seconds
```

### ClipSuggestion
```python
@dataclass
class ClipSuggestion:
    start: float                    # Clip start time
    end: float                      # Clip end time
    score: float                    # Combined score (0-2+)
    text: str                       # Transcript text
    text_score: float = 0.0         # Raw text score
    coherence_score: float = 0.0    # Internal coherence (0-1)
    distinctiveness_score: float = 0.0  # Uniqueness (0-1)
```

## Model Dependencies

| Component | Model/Tool | Purpose | Input | Output |
|-----------|------------|---------|-------|--------|
| Audio extraction | FFmpeg | Extract audio from video | video file | audio.wav (16kHz mono) |
| Transcription | OpenAI Whisper | Speech-to-text | audio.wav | segments with timestamps |
| Embeddings | all-MiniLM-L6-v2 | Semantic vectors | text strings | 384-dim vectors |
| LLM Segmentation | OpenAI/Gemini/HF/Ollama | Intelligent topic segmentation | transcript + timestamps | ranked segments |
| Clip cutting | FFmpeg | Render video clips | video + timestamps | clip mp4 files |

## Whisper Model Options

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| tiny | 39M | Fastest | Lower | ~1GB |
| base | 74M | Fast | Good | ~1GB |
| small | 244M | Medium | Better | ~2GB |
| medium | 769M | Slow | High | ~5GB |
| large | 1550M | Slowest | Highest | ~10GB |

## Scoring Mode Comparison

```
+------------------+------------------+------------------+------------------+
|    COHERENCE     |      TOPIC       |    COMBINED      |       LLM        |
+------------------+------------------+------------------+------------------+
| Can cross topics | Stays in topics  | Cross with cost  | AI-driven cuts   |
| 60% coherence    | 40% distinctive  | 40% coherence    | 40% LLM rank     |
| Best for:        | Best for:        | Best for:        | Best for:        |
| - Interviews     | - Lectures       | - General use    | - Any content    |
| - Conversations  | - Tutorials      | - Mixed content  | - High quality   |
| - Podcasts       | - Presentations  |                  | - Smart ranking  |
| FREE             | FREE             | FREE             | REQUIRES API KEY |
+------------------+------------------+------------------+------------------+
```

## LLM Mode Details

### How It Works

```
TRANSCRIPT WITH TIMESTAMPS
         |
         v
+----------------------------------+
|   Format for LLM Input           |
|   [0.0s - 5.2s]: Hello world...  |
|   [5.2s - 10.1s]: Today we...    |
|   ...                            |
+----------------------------------+
         |
         v
+----------------------------------+
|   OpenAI API Call                |
|   Model: gpt-4o-mini (default)   |
|   Prompt: Identify top K clips   |
+----------------------------------+
         |
         v
+----------------------------------+
|   LLM Response (JSON)            |
|   [                              |
|     {                            |
|       "start": 30.0,             |
|       "end": 150.0,              |
|       "rank": 1,                 |
|       "topic": "Main insight",   |
|       "reason": "Key takeaway"   |
|     },                           |
|     ...                          |
|   ]                              |
+----------------------------------+
         |
         v
+----------------------------------+
|   Build Clips                    |
|   - Use LLM timestamps           |
|   - Add computed scores          |
|   - Display topic + reason       |
+----------------------------------+
```

### LLM Prompt Template

The prompt asks the LLM to:
1. Identify distinct topics in the transcript
2. Select top K most engaging segments
3. Ensure clips are within duration constraints
4. Prioritize informative, entertaining, surprising content
5. Return JSON with start, end, rank, topic, reason

### LLM Provider Options

| Provider | API Key Required | Cost | Notes |
|----------|------------------|------|-------|
| OpenAI | Yes (sk-...) | Pay per token | Best quality, reliable |
| Gemini | Yes (AIza...) | Free tier available | Good quality, fast |
| HuggingFace | Yes (hf_...) | Free tier available | Open-source models |
| Ollama | No | Free (local) | Requires local install |

### LLM Model Options by Provider

**OpenAI:**
| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| gpt-3.5-turbo | Fastest | Good | Cheapest |
| gpt-4o-mini | Fast | Better | Low |
| gpt-4o | Medium | Best | Medium |
| gpt-4-turbo | Slow | Excellent | Higher |

**Google Gemini:**
| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| gemini-2.5-flash | Fast | Best value | Recommended |
| gemini-2.0-flash | Fast | Good | Stable |
| gemini-2.0-flash-lite | Fastest | Good | Low cost |
| gemini-3-flash-preview | Fast | Better | Preview |
| gemini-3-pro-preview | Medium | Best | Preview |

**HuggingFace:**
| Model | Size | Notes |
|-------|------|-------|
| Mixtral-8x7B-Instruct | 47B | Fast, good quality |
| Llama-2-70b-chat | 70B | Best quality |
| CodeLlama-34b-Instruct | 34B | Good for technical |

**Ollama (Local):**
| Model | Size | Notes |
|-------|------|-------|
| llama3 | 8B | Fast, good quality |
| llama3:70b | 70B | Best quality (needs GPU) |
| mistral | 7B | Fast |
| mixtral | 47B | Better quality |
| phi3 | 3.8B | Smallest, fastest |
| gemma | 7B | Google's open model |

### Ollama Setup (Optional)

To use Ollama for local LLM inference:

1. Install Ollama: https://ollama.ai
2. Start the server: `ollama serve`
3. Pull a model: `ollama pull llama3`
4. Select "Ollama (Local)" in the UI
5. No API key needed - runs entirely on your machine

## File Structure

```
reels_from_video/
|-- app/
|   |-- main.py           # FastAPI endpoints
|   |-- pipeline.py       # Core processing logic
|   |-- utils.py          # FFmpeg helpers
|   |-- static/
|   |   |-- app.js        # Frontend JavaScript
|   |   |-- styles.css    # Styling
|   |-- templates/
|       |-- index.html    # Main UI
|-- data/
|   |-- uploads/          # Uploaded videos
|   |   |-- {request_id}.mp4
|   |-- work/             # Processing workspace
|       |-- {request_id}/
|           |-- audio.wav
|           |-- clips/
|               |-- clip_1.mp4
|               |-- clip_2.mp4
|-- requirements.txt
|-- README.md
|-- PIPELINE.md           # This file
```

## Error Handling

| Condition | Warning/Error |
|-----------|---------------|
| Whisper not installed | Falls back to time-based clips |
| FFmpeg not found | Cannot extract audio or render clips |
| sentence-transformers not installed | Falls back to text-only scoring |
| No transcript segments | Uses time-based clips |
| Clip too short | Skipped, extends to next segment |
| Topic block too short | Skipped in topic mode |
| LLM mode without API key | Returns error: "LLM mode with {provider} requires an API key" |
| OpenAI API error | Falls back to combined mode with warning |
| Gemini API error | Falls back to combined mode with warning |
| HuggingFace API error | Falls back to combined mode with warning |
| Ollama not running | Returns error: "Ollama not running. Start with: ollama serve" |
| LLM response parse failure | Falls back to combined mode with warning |

## Performance Considerations

1. **First request is slow**: Whisper and SentenceTransformer models load on first use
2. **Large videos**: Transcription time scales with video length
3. **Rendering clips**: Each clip requires an FFmpeg encode pass
4. **Memory**: Large Whisper models (medium/large) need significant RAM/VRAM
