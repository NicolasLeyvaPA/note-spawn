# NoteSpawn

[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Flask](https://img.shields.io/badge/flask-3.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)

**Stop taking notes. Start paying attention.**

## The Problem

Students spend lectures frantically scribbling notes instead of actually *listening*. You can't fully absorb what a professor is explaining while simultaneously trying to write it all down. Something always gets lost.

## The Solution

NoteSpawn records your lecture, transcribes it in real-time using OpenAI Whisper (fully offline, on your machine), and optionally uses AI to structure the raw transcript into clean, organized notes. You sit back, engage with the material, ask questions, and walk out with perfect notes you never had to write.

## How It Works

```
Lecture audio  -->  Whisper (local, offline)  -->  Raw transcript
                                                        |
                                                   AI enhancement
                                                   (Claude / Ollama)
                                                        |
                                                  Structured notes
                                                  saved as Markdown
```

1. **Click "Start Recording"** in the browser
2. Whisper transcribes audio in chunks (configurable: 30-120 seconds)
3. Each chunk is optionally enhanced by AI into structured notes (key concepts, details, questions)
4. At the end, a full summary is generated
5. Everything is saved as clean Markdown files

## Features

- **One-click recording** - Start/stop from the browser
- **Real-time transcription** - Notes appear live as the lecture progresses
- **AI-powered enhancement** - Turns raw speech into structured, organized notes
- **Dual AI backend** - Use Anthropic Claude (API) or Ollama (free, local)
- **Works offline** - Whisper runs entirely on your machine, no internet needed for transcription
- **Session history** - Browse, search, and revisit past lectures
- **Class organization** - Group sessions by subject
- **Export** - Download notes as Markdown or plain text

## Quick Start

### Windows (one click)

1. Double-click **`run.bat`**
2. Open **http://localhost:5000**
3. Hit **Start Recording**

### Manual Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install FFmpeg (required by Whisper)
winget install ffmpeg

# Optional: enable AI note enhancement
# Option A - Anthropic Claude (requires API key)
set ANTHROPIC_API_KEY=your-key-here

# Option B - Ollama (free, local)
# Install from https://ollama.com then: ollama pull llama3.2

# Start the app
python app.py
```

Open **http://localhost:5000** in your browser.

## Configuration

| Setting | Description |
|---------|-------------|
| **Microphone** | Select your input device from the dropdown |
| **Chunk Duration** | How often to process audio (30-120s). 60-90s works well for most lectures |
| **Whisper Model** | Speed vs accuracy tradeoff (see below) |

### Whisper Models

| Model | Speed | Accuracy | RAM |
|-------|-------|----------|-----|
| tiny | Fast | Low | ~1GB |
| base | Moderate | Good | ~1GB |
| small | Slow | Great | ~2GB |
| medium | Very slow | Excellent | ~5GB |

**Recommendation:** `base` or `small` for most lectures. Use `small` if the speaker has a strong accent.

## Tech Stack

- **Flask + Socket.IO** - Real-time web server
- **OpenAI Whisper** - Local, offline speech-to-text
- **Anthropic Claude / Ollama** - AI note enhancement (optional)
- **Vanilla JS** - Zero build step frontend

## Tips

- Sit near the speaker or use an external mic for best results
- Without an API key, NoteSpawn still transcribes everything — it just skips AI enhancement
- Notes are saved to `./notes/` as Markdown files

## License

MIT
