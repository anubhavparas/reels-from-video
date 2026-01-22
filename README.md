# Reels From Video

AI-powered tool that extracts the best 1-5 minute clips from long-form video content. Perfect for content creators who want to turn podcasts, interviews, lectures, or vlogs into viral-ready reels and shorts.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **AI Transcription** - Uses OpenAI Whisper for accurate speech-to-text with timestamps
- **Smart Clip Detection** - Multiple scoring modes to find the most engaging segments
- **Topic Segmentation** - Detects topic boundaries to avoid awkward cuts
- **LLM-Powered Ranking** - Optional GPT/Gemini integration for intelligent clip selection
- **Video Export** - Renders clips as downloadable MP4 files
- **Beautiful UI** - Modern, techy interface with animated backgrounds

## How It Works

See [PIPELINE.md](PIPELINE.md) for a detailed technical breakdown of the processing pipeline.

**Quick Overview:**
1. Upload video -> Extract audio (FFmpeg)
2. Transcribe with timestamps (Whisper)
3. Generate semantic embeddings (Sentence Transformers)
4. Detect topic shifts and score segments
5. Rank and select top clips
6. Render video clips (FFmpeg)

---

## Requirements

### System Dependencies

| Dependency | Purpose | Required |
|------------|---------|----------|
| **Python 3.10+** | Runtime | Yes |
| **FFmpeg** | Audio extraction & video rendering | Yes |
| **Git** | Clone repository | Yes |

### Python Libraries

| Library | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `openai-whisper` | Speech-to-text transcription |
| `sentence-transformers` | Semantic embeddings for coherence scoring |
| `numpy` | Numerical operations |
| `openai` | OpenAI GPT API (optional, for LLM mode) |
| `google-generativeai` | Google Gemini API (optional, for LLM mode) |
| `huggingface_hub` | HuggingFace Inference API (optional, for LLM mode) |

### AI Models (Auto-downloaded on first use)

| Model | Size | Purpose |
|-------|------|---------|
| Whisper `tiny` | ~39 MB | Fast transcription (lower accuracy) |
| Whisper `base` | ~74 MB | Balanced speed/accuracy (default) |
| Whisper `small` | ~244 MB | Better accuracy |
| Whisper `medium` | ~769 MB | High accuracy |
| Whisper `large` | ~1.5 GB | Best accuracy (needs GPU) |
| `all-MiniLM-L6-v2` | ~80 MB | Sentence embeddings |

---

## Installation

### macOS

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install system dependencies
brew install python@3.10 ffmpeg git

# 3. Clone the repository
git clone https://github.com/YOUR_USERNAME/reels-from-video.git
cd reels-from-video

# 4. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 5. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)

```bash
# 1. Update package list
sudo apt update

# 2. Install system dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip ffmpeg git

# 3. Clone the repository
git clone https://github.com/YOUR_USERNAME/reels-from-video.git
cd reels-from-video

# 4. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 5. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux (Fedora/RHEL)

```bash
# 1. Install system dependencies
sudo dnf install -y python3.10 python3-pip ffmpeg git

# 2. Clone the repository
git clone https://github.com/YOUR_USERNAME/reels-from-video.git
cd reels-from-video

# 3. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Using Conda (Alternative)

```bash
# 1. Create conda environment
conda create -n reels python=3.10 -y
conda activate reels

# 2. Install FFmpeg via conda
conda install -c conda-forge ffmpeg -y

# 3. Clone and install
git clone https://github.com/YOUR_USERNAME/reels-from-video.git
cd reels-from-video
pip install -r requirements.txt
```

---

## Running the App

### Start the Server

```bash
# Activate virtual environment (if not already)
source .venv/bin/activate  # or: conda activate reels

# Run the server
uvicorn app.main:app --reload
```

### Access the App

Open your browser and go to: **http://127.0.0.1:8000**

---

## Usage

1. **Upload a video** - Supports MP4, MOV, AVI, MKV, WebM
2. **Configure settings:**
   - **Min/Max seconds** - Clip duration range (10-300 seconds)
   - **Max results** - Number of clips to generate
   - **Scoring mode** - Coherence, Topic, Combined, or LLM
   - **Whisper model** - tiny (fast) to large (accurate)
3. **Check "Render clip files"** to export as MP4
4. **Click "Generate Clips"**
5. **Download** your clips!

---

## Scoring Modes

| Mode | Best For | How It Works |
|------|----------|--------------|
| **Coherence** | Interviews, podcasts | Prioritizes internal flow and consistency |
| **Topic** | Lectures, tutorials | Stays within topic boundaries |
| **Combined** | General content | Balanced approach (default) |
| **LLM** | Any content | AI analyzes transcript for best moments |

### LLM Mode (Optional)

LLM mode requires an API key from one of these providers:

| Provider | Get API Key | Cost |
|----------|-------------|------|
| OpenAI | [platform.openai.com](https://platform.openai.com) | Pay per token |
| Google Gemini | [ai.google.dev](https://ai.google.dev) | Free tier available |
| HuggingFace | [huggingface.co](https://huggingface.co) | Free tier available |
| Ollama | Local install | Free (local only) |

**Note:** Ollama only works when running locally. It's not available on hosted deployments.

---

## Using Ollama (Local LLM)

For completely free, private LLM processing:

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama server
ollama serve

# 3. Pull a model (in another terminal)
ollama pull llama3.2       # 2GB, fast
ollama pull gemma3         # 3.3GB, good quality
ollama pull mistral        # 4.1GB, excellent

# 4. Select "Ollama (Local)" in the app
```

---

## Docker Deployment

### Build and Run Locally

```bash
# Build the image
docker build -t reels-from-video .

# Run the container
docker run -p 7860:7860 reels-from-video

# Access at http://localhost:7860
```

### Deploy to Hugging Face Spaces

1. Push your code to GitHub
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
3. Create new Space with **Docker** SDK
4. Connect your GitHub repo
5. It auto-deploys!

### Deploy to Render

1. Push your code to GitHub
2. Go to [render.com](https://render.com)
3. Create new Web Service
4. Connect GitHub repo
5. Select "Docker" environment
6. Deploy!

---

## Project Structure

```
reels-from-video/
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
|   |-- uploads/          # Uploaded videos (gitignored)
|   |-- work/             # Processing workspace (gitignored)
|-- Dockerfile            # Docker configuration
|-- requirements.txt      # Python dependencies
|-- PIPELINE.md           # Technical pipeline documentation
|-- README.md             # This file
```

---

## Troubleshooting

### FFmpeg not found
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Verify installation
ffmpeg -version
```

### Whisper model download fails
Models are downloaded to `~/.cache/whisper/`. Ensure you have internet access and disk space.

### Out of memory with large Whisper models
Use smaller models (`tiny` or `base`) or process shorter videos.

### LLM API errors
- Check your API key is correct
- Verify you have API credits/quota
- Try a different provider

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

MIT License - feel free to use this for personal or commercial projects.

---

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Sentence Transformers](https://www.sbert.net/) - Semantic embeddings
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [FFmpeg](https://ffmpeg.org/) - Video processing
