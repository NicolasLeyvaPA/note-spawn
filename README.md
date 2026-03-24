# NoteSpawn

[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Flask](https://img.shields.io/badge/flask-3.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)

**Stop taking notes. Start paying attention.**

## Why I Built This

Halfway through my 3rd year of Computer Science, I noticed something — the students who looked the most "focused" in lecture were really just focused on writing things down. Heads down, scribbling non-stop, trying to capture every word. They weren't actually *thinking* about what the professor was saying. Neither was I.

You can't deeply engage with a concept and transcribe it at the same time. Your brain has to pick one. So I built NoteSpawn to handle the transcription part, so I could finally just sit there, listen, and actually understand.

## What It Does

You open NoteSpawn in your browser, hit record, and put your laptop to the side. That's it. While you're actually engaging with the lecture — asking questions, following the reasoning, building intuition — Whisper is running locally on your machine turning speech into text in real-time. No cloud, no API calls, no internet required for the transcription itself. It all happens on your hardware.

But raw transcription isn't enough. Professors go on tangents, repeat themselves, mumble through transitions, and your mic won't catch every word. So after each chunk of audio is transcribed, an AI layer (either Ollama running locally for free, or Claude via API) takes that messy raw text and restructures it — filling in gaps where the mic dropped words, organizing scattered points into key concepts and details, and flagging things that need clarification. The result reads like notes a top student would write, except nobody had to write them.

At the end of the session, everything gets compiled into clean Markdown files: structured notes and the raw transcript side by side, with a full summary of the lecture. You walk out of class having actually *learned* something, and the notes are already waiting for you.

## How It Works

```
Browser (UI)  <-- Socket.IO -->  Flask Server  <-- Whisper -->  Transcription
                                      |
                                      +-- Ollama / Claude -->  Note Enhancement
                                      |
                                      +-- File I/O ----------> ./notes/*.md
```

1. Hit **Start Recording** — audio streams to the Flask backend via your mic
2. Every 30-120 seconds (your choice), Whisper transcribes the latest chunk
3. AI enhances the raw transcript — structures it, fills gaps, organizes key ideas
4. Notes appear in real-time in the browser as each chunk is processed
5. When you stop, a full summary is generated and everything is saved as Markdown

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

## Contributing

Contributions are welcome. To get started:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Commit your changes
4. Push and open a pull request

If you find a bug or have a feature idea, open an issue.

## License

MIT
