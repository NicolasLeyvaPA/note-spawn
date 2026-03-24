#!/usr/bin/env python3
"""
Lecture Notes AI - Enhanced Version with Note Management
Inspired by NotebookLM
"""

import os
import sys
import wave
import queue
import logging
import tempfile
import threading
import json
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

script_dir = Path(__file__).parent.resolve()
os.chdir(script_dir)

logger.info("Starting Lecture Notes AI...")

def install(package):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

for module, package in [("flask", "flask"), ("flask_socketio", "flask-socketio"), 
                         ("numpy", "numpy"), ("sounddevice", "sounddevice")]:
    try:
        __import__(module)
    except ImportError:
        logger.info(f"Installing {package}...")
        install(package)

import numpy as np
import sounddevice as sd
from flask import Flask, jsonify, Response, request
from flask_socketio import SocketIO, emit

# Optional imports
whisper = None
ollama_available = False

try:
    import whisper as w
    whisper = w
    logger.info("Whisper available")
except ImportError:
    logger.warning("Whisper not installed (will install on first use)")

try:
    import urllib.request
    req = urllib.request.Request(f"{OLLAMA_API_URL}/api/tags")
    urllib.request.urlopen(req, timeout=2)
    ollama_available = True
    logger.info("Ollama connected (free local AI)")
except:
    logger.warning("Ollama not running - install from ollama.com for free AI enhancement")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lecture-notes'
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:5000').split(',')
socketio = SocketIO(app, cors_allowed_origins=CORS_ORIGINS, async_mode='threading')

# Audio constants
SAMPLE_RATE = 16000       # Hz — required by Whisper
AUDIO_CHANNELS = 1        # Mono recording
AUDIO_BLOCK_DURATION = 0.5  # Seconds per audio callback block

# Whisper configuration
DEFAULT_WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
WHISPER_LANGUAGE = 'en'
WHISPER_FP16 = False  # Set True if you have a CUDA GPU

# Ollama configuration
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')

# Transcription thresholds
MIN_TRANSCRIPT_LENGTH = 10  # Minimum chars to consider a valid transcript

# State
recorder = None
session_active = False
whisper_model = None
notes_dir = Path("notes")
notes_dir.mkdir(exist_ok=True)
metadata_file = notes_dir / "metadata.json"

# Pre-warm state for latency optimization
whisper_ready = threading.Event()
whisper_loading = False
warm_recorder = None
warm_device_id = None
warm_recorder_lock = threading.Lock()
default_model_size = DEFAULT_WHISPER_MODEL

import time  # For performance timing

def prewarm_whisper(model_size='base'):
    """Pre-load Whisper model in background to eliminate cold-start latency"""
    global whisper_model, whisper, whisper_loading
    whisper_loading = True
    try:
        if whisper is None:
            logger.info("Installing Whisper...")
            os.system(f'"{sys.executable}" -m pip install openai-whisper -q')
            import whisper as w
            whisper = w

        logger.info(f"Pre-loading Whisper model ({model_size})...")
        whisper_model = whisper.load_model(model_size)
        logger.info(f"Whisper model ready ({model_size})")
        whisper_ready.set()
    except Exception as e:
        logger.error(f"Whisper pre-load failed: {e}")
    finally:
        whisper_loading = False

# Start Whisper pre-warm in background thread
prewarm_thread = threading.Thread(target=prewarm_whisper, args=(default_model_size,), daemon=True)
prewarm_thread.start()

# Your classes
CLASSES = [
    {"id": "chatbots", "name": "AI: Chatbots & Recommendation Engines", "professor": "Miguel González-Fierro", "color": "#f87171"},
    {"id": "computer-vision", "name": "AI: Computer Vision", "professor": "Rubén Sánchez García", "color": "#fb923c"},
    {"id": "nlp", "name": "AI: NLP & Semantic Analysis", "professor": "Multiple Instructors", "color": "#a78bfa"},
    {"id": "reinforcement-learning", "name": "AI: Reinforcement Learning", "professor": "José Manuel Rey González", "color": "#4ade80"},
    {"id": "statistical-learning", "name": "AI: Statistical Learning & Prediction", "professor": "Luciano Dyballa", "color": "#60a5fa"},
    {"id": "entrepreneurship", "name": "Entrepreneurial Mindset & Practice", "professor": "Claudia Caso Fernandez", "color": "#f472b6"},
]

for c in CLASSES:
    (notes_dir / c["id"]).mkdir(exist_ok=True)


# ============== METADATA MANAGER ==============
class MetadataManager:
    def __init__(self):
        self.data = self._load()
        self._migrate_to_lectures()  # Migrate existing sessions to lecture hierarchy
        self._sync_with_files()  # Scan for any orphaned files

    def _load(self):
        if metadata_file.exists():
            try:
                data = json.loads(metadata_file.read_text(encoding='utf-8'))
                # Ensure lectures dict exists
                if 'lectures' not in data:
                    data['lectures'] = {}
                logger.info(f"Loaded metadata: {len(data.get('sessions', {}))} sessions, {len(data.get('lectures', {}))} lectures")
                return data
            except Exception as e:
                logger.warning(f"Metadata corrupted, creating backup: {e}")
                # Backup corrupted file
                backup = metadata_file.with_suffix('.json.bak')
                try:
                    metadata_file.rename(backup)
                except:
                    pass
        return {"sessions": {}, "lectures": {}}

    def _migrate_to_lectures(self):
        """Migrate existing standalone sessions to lecture containers (idempotent)"""
        migrated = 0
        for session_id, session in list(self.data.get("sessions", {}).items()):
            if session.get("lecture_id"):
                continue  # Already associated with a lecture

            # Create a new lecture container for this session
            lecture_id = f"lecture_{session_id}"
            if lecture_id not in self.data["lectures"]:
                self.data["lectures"][lecture_id] = {
                    "id": lecture_id,
                    "class_id": session.get("class_id", ""),
                    "title": session.get("title", "Migrated Lecture"),
                    "created": session.get("created", datetime.now().isoformat()),
                    "modified": datetime.now().isoformat(),
                    "sessions": [session_id]
                }
                session["lecture_id"] = lecture_id
                migrated += 1

        if migrated > 0:
            self._save()
            logger.info(f"Migrated {migrated} sessions to lecture containers")
    
    def _save(self):
        try:
            # Write to temp file first, then rename (atomic operation)
            temp_file = metadata_file.with_suffix('.json.tmp')
            temp_file.write_text(json.dumps(self.data, indent=2), encoding='utf-8')
            temp_file.replace(metadata_file)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _sync_with_files(self):
        """Scan for lecture files that aren't in metadata (recovery)"""
        found_new = False
        for c in CLASSES:
            class_dir = notes_dir / c["id"]
            if not class_dir.exists():
                continue
            for f in class_dir.glob("lecture_*.md"):
                session_id = f.stem.replace("lecture_", "")
                if session_id not in self.data["sessions"]:
                    logger.info(f"Found orphaned session: {session_id} in {c['id']}")
                    # Extract title from file
                    try:
                        content = f.read_text(encoding='utf-8')
                        lines = content.split('\n')
                        title = "Recovered Lecture"
                        for line in lines[:5]:
                            if line.startswith('## ') and not line.startswith('## 📚'):
                                title = line[3:].strip()
                                break
                        self.data["sessions"][session_id] = {
                            "class_id": c["id"],
                            "title": title,
                            "tags": ["recovered"],
                            "created": datetime.now().isoformat(),
                            "modified": datetime.now().isoformat()
                        }
                        found_new = True
                    except:
                        pass
        if found_new:
            self._save()
            logger.info("Recovered orphaned sessions")
    
    def get_session(self, session_id):
        return self.data["sessions"].get(session_id, {})

    def set_session(self, session_id, class_id, title=None, tags=None, lecture_id=None):
        if session_id not in self.data["sessions"]:
            self.data["sessions"][session_id] = {}

        self.data["sessions"][session_id].update({
            "class_id": class_id,
            "title": title or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "tags": tags or [],
            "created": self.data["sessions"].get(session_id, {}).get("created", datetime.now().isoformat()),
            "modified": datetime.now().isoformat(),
            "completed": False
        })

        # Associate with lecture if provided
        if lecture_id:
            self.data["sessions"][session_id]["lecture_id"] = lecture_id

        self._save()
        return self.data["sessions"][session_id]

    # ============== LECTURE METHODS ==============
    def create_lecture(self, class_id, title):
        """Create a new lecture container"""
        lecture_id = f"lecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.data["lectures"][lecture_id] = {
            "id": lecture_id,
            "class_id": class_id,
            "title": title or f"Lecture {datetime.now().strftime('%Y-%m-%d')}",
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "sessions": []
        }
        self._save()
        return lecture_id

    def get_lecture(self, lecture_id):
        """Get lecture metadata"""
        return self.data["lectures"].get(lecture_id, {})

    def get_all_lectures(self):
        """Get all lectures"""
        return self.data.get("lectures", {})

    def add_session_to_lecture(self, lecture_id, session_id):
        """Add a session to an existing lecture"""
        if lecture_id in self.data["lectures"]:
            if session_id not in self.data["lectures"][lecture_id]["sessions"]:
                self.data["lectures"][lecture_id]["sessions"].append(session_id)
                self.data["lectures"][lecture_id]["modified"] = datetime.now().isoformat()
            if session_id in self.data["sessions"]:
                self.data["sessions"][session_id]["lecture_id"] = lecture_id
            self._save()
            return True
        return False

    def update_lecture(self, lecture_id, **kwargs):
        """Update lecture metadata"""
        if lecture_id in self.data["lectures"]:
            self.data["lectures"][lecture_id].update(kwargs)
            self.data["lectures"][lecture_id]["modified"] = datetime.now().isoformat()
            self._save()
            return True
        return False

    def get_lecture_with_sessions(self, lecture_id):
        """Get lecture with full session details"""
        lecture = self.get_lecture(lecture_id)
        if not lecture:
            return None
        result = dict(lecture)
        result["session_details"] = [
            self.get_session(sid) for sid in lecture.get("sessions", [])
        ]
        # Calculate aggregates
        result["total_words"] = sum(s.get("words", 0) for s in result["session_details"])
        result["total_duration_mins"] = sum(s.get("duration_mins", 0) for s in result["session_details"])
        return result
    
    def update_session(self, session_id, **kwargs):
        if session_id in self.data["sessions"]:
            self.data["sessions"][session_id].update(kwargs)
            self.data["sessions"][session_id]["modified"] = datetime.now().isoformat()
            self._save()
            return True
        return False
    
    def delete_session(self, session_id):
        if session_id in self.data["sessions"]:
            del self.data["sessions"][session_id]
            self._save()
            return True
        return False
    
    def move_session(self, session_id, old_class, new_class):
        old_notes = notes_dir / old_class / f"lecture_{session_id}.md"
        new_notes = notes_dir / new_class / f"lecture_{session_id}.md"
        old_transcript = notes_dir / old_class / f"transcript_{session_id}.md"
        new_transcript = notes_dir / new_class / f"transcript_{session_id}.md"
        
        new_class_name = next((c["name"] for c in CLASSES if c["id"] == new_class), new_class)
        
        # Move notes file
        if old_notes.exists():
            content = old_notes.read_text(encoding='utf-8')
            lines = content.split('\n')
            if lines and lines[0].startswith('# '):
                lines[0] = f"# {new_class_name}"
            content = '\n'.join(lines)
            new_notes.write_text(content, encoding='utf-8')
            old_notes.unlink()
        
        # Move transcript file
        if old_transcript.exists():
            content = old_transcript.read_text(encoding='utf-8')
            lines = content.split('\n')
            # Update class reference in transcript
            for i, line in enumerate(lines):
                if line.startswith('**Class:**'):
                    lines[i] = f"**Class:** {new_class_name}"
                    break
            content = '\n'.join(lines)
            new_transcript.write_text(content, encoding='utf-8')
            old_transcript.unlink()
        
        self.update_session(session_id, class_id=new_class)
        return True
    
    def get_all_sessions(self):
        return self.data["sessions"]
    
    def search(self, query):
        results = []
        query_lower = query.lower()
        
        for session_id, meta in self.data["sessions"].items():
            # Search in title and tags
            if query_lower in meta.get("title", "").lower() or any(query_lower in t.lower() for t in meta.get("tags", [])):
                results.append({"id": session_id, **meta, "match": "metadata"})
                continue
            
            # Search in content
            class_id = meta.get("class_id", "")
            filepath = notes_dir / class_id / f"lecture_{session_id}.md"
            if filepath.exists():
                content = filepath.read_text(encoding='utf-8').lower()
                if query_lower in content:
                    results.append({"id": session_id, **meta, "match": "content"})
        
        return results


metadata_mgr = MetadataManager()


# ============== HTML PAGE ==============
HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lecture Notes AI</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/11.1.1/marked.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg: #0d1117; --bg2: #161b22; --bg3: #21262d; --bg4: #2d333b;
            --border: #30363d; --text: #e6edf3; --text2: #8b949e; --text3: #6e7681;
            --accent: #58a6ff; --success: #3fb950; --danger: #f85149; --warning: #d29922;
        }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; }
        
        /* Sidebar */
        .sidebar { width: 280px; background: var(--bg2); border-right: 1px solid var(--border); display: flex; flex-direction: column; height: 100vh; position: fixed; }
        .sidebar-header { padding: 16px; border-bottom: 1px solid var(--border); }
        .logo { font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
        .search-box { display: flex; align-items: center; background: var(--bg3); border-radius: 8px; padding: 8px 12px; gap: 8px; }
        .search-box input { flex: 1; background: none; border: none; color: var(--text); font-size: 14px; outline: none; }
        .search-box input::placeholder { color: var(--text3); }
        
        .class-list { flex: 1; overflow-y: auto; padding: 8px; }
        .class-section { margin-bottom: 4px; }
        .class-header { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; }
        .class-header:hover { background: var(--bg3); }
        .class-header.active { background: var(--bg3); }
        .class-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
        .class-name { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .class-count { font-size: 11px; color: var(--text3); background: var(--bg4); padding: 2px 6px; border-radius: 10px; }
        .class-sessions { margin-left: 16px; display: none; }
        .class-section.expanded .class-sessions { display: block; }
        .class-section.expanded .class-arrow { transform: rotate(90deg); }
        .class-arrow { font-size: 10px; color: var(--text3); transition: transform 0.2s; }
        
        .session-item { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; color: var(--text2); }
        .session-item:hover { background: var(--bg3); color: var(--text); }
        .session-item.active { background: var(--bg4); color: var(--text); }
        .session-item.incomplete { border-left: 2px solid var(--warning); }
        .session-status { font-size: 10px; opacity: 0.6; }
        .session-stats { font-size: 11px; color: var(--text3); margin-left: auto; }
        .session-title { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .session-date { font-size: 11px; color: var(--text3); }
        
        .sidebar-footer { padding: 12px; border-top: 1px solid var(--border); }
        .new-btn { width: 100%; padding: 10px; background: var(--accent); color: #000; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px; }
        .new-btn:hover { background: #79b8ff; }
        
        /* Main Content */
        .main { flex: 1; margin-left: 280px; display: flex; flex-direction: column; height: 100vh; }
        .main-header { padding: 16px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 16px; background: var(--bg2); }
        .status { display: flex; align-items: center; gap: 8px; padding: 6px 12px; background: var(--bg3); border-radius: 16px; font-size: 13px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text3); }
        .dot.recording { background: var(--danger); animation: pulse 1.5s infinite; }
        .dot.connected { background: var(--success); }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        
        .main-content { flex: 1; overflow-y: auto; }
        
        /* Recording Panel */
        .record-panel { padding: 24px; max-width: 800px; margin: 0 auto; }
        .record-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 24px; }
        .record-title { font-size: 18px; font-weight: 600; margin-bottom: 20px; }
        .record-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
        .control { display: flex; flex-direction: column; gap: 6px; }
        .control label { font-size: 12px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; }
        select, input[type="text"] { padding: 10px 12px; background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: 14px; }
        select:focus, input:focus { outline: none; border-color: var(--accent); }
        .record-btn { width: 100%; padding: 14px; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .record-btn.start { background: var(--accent); color: #000; }
        .record-btn.stop { background: var(--danger); color: #fff; }
        .record-btn:hover { opacity: 0.9; }
        .hidden { display: none !important; }
        
        /* Notes View */
        .notes-view { padding: 24px; max-width: 900px; margin: 0 auto; }
        .notes-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 24px; gap: 16px; }
        .notes-title-section { flex: 1; }
        .notes-title { font-size: 28px; font-weight: 600; margin-bottom: 8px; outline: none; }
        .notes-title:focus { border-bottom: 2px solid var(--accent); }
        .notes-meta { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
        .notes-class { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text2); padding: 4px 10px; background: var(--bg3); border-radius: 6px; cursor: pointer; }
        .notes-class:hover { background: var(--bg4); }
        .notes-date { font-size: 13px; color: var(--text3); }
        .notes-actions { display: flex; gap: 8px; }
        .action-btn { padding: 8px 12px; background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; color: var(--text2); font-size: 13px; cursor: pointer; display: flex; align-items: center; gap: 6px; }
        .action-btn:hover { background: var(--bg4); color: var(--text); }
        .action-btn.danger:hover { background: rgba(248, 81, 73, 0.2); color: var(--danger); border-color: var(--danger); }
        
        .tags-section { margin-bottom: 20px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
        .tag { padding: 4px 10px; background: var(--bg3); border-radius: 12px; font-size: 12px; color: var(--text2); display: flex; align-items: center; gap: 4px; }
        .tag .remove { cursor: pointer; opacity: 0.6; }
        .tag .remove:hover { opacity: 1; }
        .add-tag { padding: 4px 10px; border: 1px dashed var(--border); border-radius: 12px; font-size: 12px; color: var(--text3); cursor: pointer; }
        .add-tag:hover { border-color: var(--text2); color: var(--text2); }
        
        .notes-content { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 24px; }
        .chunk { margin-bottom: 24px; padding-bottom: 24px; border-bottom: 1px solid var(--border); }
        .chunk:last-child { margin-bottom: 0; padding-bottom: 0; border-bottom: none; }
        .chunk-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .chunk-num { font-size: 12px; font-weight: 600; color: var(--accent); text-transform: uppercase; }
        .chunk-time { font-size: 12px; color: var(--text3); }
        .transcript { color: var(--text2); font-style: italic; padding: 12px; background: var(--bg3); border-radius: 8px; margin-bottom: 12px; font-size: 14px; }
        .notes-text { line-height: 1.7; }
        .notes-text h1, .notes-text h2, .notes-text h3 { margin: 16px 0 8px 0; }
        .notes-text h1:first-child, .notes-text h2:first-child { margin-top: 0; }
        .notes-text ul, .notes-text ol { margin-left: 20px; margin-bottom: 12px; }
        .notes-text li { margin-bottom: 4px; }
        .notes-text strong { color: var(--accent); }
        .summary-box { background: linear-gradient(135deg, #1a2332, #162029); border: 1px solid var(--accent); border-radius: 12px; padding: 20px; margin-bottom: 24px; }
        .summary-title { font-size: 16px; font-weight: 600; color: var(--accent); margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
        
        /* Live Recording View */
        .live-chunks { padding: 24px; max-width: 900px; margin: 0 auto; }
        .live-chunk { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 16px; }
        
        /* Modal */
        .modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 1000; }
        .modal { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 24px; width: 400px; max-width: 90%; }
        .modal-title { font-size: 18px; font-weight: 600; margin-bottom: 16px; }
        .modal-body { margin-bottom: 20px; }
        .modal-actions { display: flex; gap: 12px; justify-content: flex-end; }
        .modal-btn { padding: 10px 20px; border-radius: 8px; font-size: 14px; font-weight: 500; cursor: pointer; border: none; }
        .modal-btn.primary { background: var(--accent); color: #000; }
        .modal-btn.secondary { background: var(--bg3); color: var(--text); }
        .modal-btn.danger { background: var(--danger); color: #fff; }
        
        /* Search Results */
        .search-results { padding: 24px; }
        .search-title { font-size: 18px; font-weight: 600; margin-bottom: 16px; }
        .search-result { padding: 16px; background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 8px; cursor: pointer; }
        .search-result:hover { border-color: var(--accent); }
        .search-result-title { font-weight: 500; margin-bottom: 4px; }
        .search-result-meta { font-size: 12px; color: var(--text3); }
        
        /* Empty State */
        .empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; color: var(--text3); text-align: center; }
        .empty-icon { font-size: 48px; margin-bottom: 16px; opacity: 0.5; }
        
        /* View Tabs */
        .view-tabs { display: flex; gap: 8px; margin-bottom: 16px; }
        .view-tab { padding: 10px 16px; background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; color: var(--text2); font-size: 14px; cursor: pointer; display: flex; align-items: center; gap: 6px; }
        .view-tab:hover:not(:disabled) { background: var(--bg4); color: var(--text); }
        .view-tab.active { background: var(--accent); color: #000; border-color: var(--accent); }
        .view-tab:disabled { opacity: 0.4; cursor: not-allowed; }
        
        /* Transcript View */
        .transcript-full { line-height: 1.8; font-size: 15px; }
        .transcript-full h3 { color: var(--accent); font-size: 13px; margin-top: 24px; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
        .transcript-full h3:first-child { margin-top: 0; }
        .transcript-full p { margin-bottom: 16px; color: var(--text); }
        .transcript-full hr { border: none; border-top: 1px solid var(--border); margin: 20px 0; }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text3); }
    </style>
</head>
<body>
    <aside class="sidebar">
        <div class="sidebar-header">
            <div class="logo">🎓 Lecture Notes</div>
            <div class="search-box">
                <span>🔍</span>
                <input type="text" id="searchInput" placeholder="Search notes...">
            </div>
        </div>
        <div class="class-list" id="classList"></div>
        <div class="sidebar-footer">
            <button class="new-btn" id="newRecordingBtn">⏺ New Recording</button>
        </div>
    </aside>
    
    <main class="main">
        <header class="main-header">
            <div class="status">
                <div class="dot" id="dot"></div>
                <span id="statusText">Connecting...</span>
            </div>
        </header>
        
        <div class="main-content" id="mainContent">
            <!-- Dynamic content loaded here -->
        </div>
    </main>
    
    <div class="modal-overlay hidden" id="modalOverlay">
        <div class="modal" id="modal"></div>
    </div>

    <script>
        const socket = io();
        const $ = id => document.getElementById(id);
        const $statusDot = () => document.getElementById('dot');
        let classes = [];
        let sessions = {};
        let lectures = {};
        let currentView = 'record';
        let currentSession = null;
        let currentLecture = null;
        let isRecording = false;
        let whisperReady = false;

        // Alias for statusDot
        Object.defineProperty(window, 'statusDot', { get: () => $('dot') });

        // Initialize
        async function init() {
            const classRes = await fetch('/api/classes');
            classes = await classRes.json();
            await loadSessions();
            await loadLectures();
            renderSidebar();
            showRecordPanel();
        }

        async function loadSessions() {
            const res = await fetch('/api/sessions/all');
            sessions = await res.json();
        }

        async function loadLectures() {
            const res = await fetch('/api/lectures');
            lectures = await res.json();
        }

        function updateStartButtonState() {
            const btn = $('startBtn');
            if (btn) {
                btn.disabled = !whisperReady;
                if (!whisperReady) {
                    btn.textContent = '⏳ Loading AI...';
                    btn.title = 'Please wait for AI model to load';
                } else {
                    btn.textContent = '⏺ Start Recording';
                    btn.title = '';
                }
            }
        }

        function renderSidebar() {
            // Count lectures per class
            const lectureCounts = {};
            classes.forEach(c => lectureCounts[c.id] = 0);
            Object.values(lectures).forEach(l => {
                if (lectureCounts[l.class_id] !== undefined) lectureCounts[l.class_id]++;
            });

            $('classList').innerHTML = classes.map(c => {
                // Get lectures for this class, sorted by date (newest first)
                const classLectures = Object.entries(lectures)
                    .filter(([id, l]) => l.class_id === c.id)
                    .sort((a, b) => (b[1].created || b[0]).localeCompare(a[1].created || a[0]));

                return `
                    <div class="class-section" data-class="${c.id}">
                        <div class="class-header" onclick="toggleClass('${c.id}')">
                            <span class="class-arrow">▶</span>
                            <span class="class-dot" style="background: ${c.color}"></span>
                            <span class="class-name">${c.name}</span>
                            <span class="class-count">${lectureCounts[c.id]}</span>
                        </div>
                        <div class="class-sessions">
                            ${classLectures.map(([id, l]) => {
                                const sessionCount = l.sessions?.length || 0;
                                return `
                                    <div class="session-item" data-lecture-id="${id}" onclick="loadLectureView('${id}')">
                                        <span class="session-title">${l.title || 'Untitled Lecture'}</span>
                                        <span class="session-stats">${sessionCount} session${sessionCount !== 1 ? 's' : ''}</span>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                `;
            }).join('');
        }

        async function loadLectureView(lectureId) {
            currentView = 'lecture';
            currentLecture = lectureId;
            currentSession = null;

            document.querySelectorAll('.session-item').forEach(el => {
                el.classList.toggle('active', el.dataset.lectureId === lectureId);
            });

            const res = await fetch('/api/lectures/' + lectureId);
            const lecture = await res.json();
            if (!lecture || lecture.error) return;

            const classInfo = classes.find(c => c.id === lecture.class_id) || {};

            $('mainContent').innerHTML = `
                <div class="notes-view">
                    <div class="notes-header">
                        <div class="notes-title-section">
                            <div class="notes-title">${lecture.title || 'Untitled Lecture'}</div>
                            <div class="notes-meta">
                                <div class="notes-class" style="border-left: 3px solid ${classInfo.color || '#888'}">
                                    ${classInfo.name || 'Unknown Class'}
                                </div>
                                <div class="notes-date">${lecture.session_details?.length || 0} sessions</div>
                                ${lecture.total_duration_mins ? `<div class="notes-date">⏱ ${lecture.total_duration_mins} min total</div>` : ''}
                                ${lecture.total_words ? `<div class="notes-date">💬 ${lecture.total_words.toLocaleString()} words</div>` : ''}
                            </div>
                        </div>
                        <div class="notes-actions">
                            <button class="action-btn" onclick="addSessionToLecture('${lectureId}')">+ Add Session</button>
                        </div>
                    </div>

                    <div class="notes-content">
                        <h3 style="margin-bottom: 16px;">Sessions</h3>
                        ${lecture.session_details?.length > 0 ? `
                            <div class="session-timeline">
                                ${lecture.session_details.map((s, i) => {
                                    const sessionId = lecture.sessions[i];
                                    return `
                                        <div class="session-card" onclick="loadSession('${sessionId}')" style="padding: 16px; background: var(--bg3); border-radius: 8px; margin-bottom: 12px; cursor: pointer;">
                                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                                <div>
                                                    <div style="font-weight: 500; margin-bottom: 4px;">Session ${i + 1}: ${s.title || 'Untitled'}</div>
                                                    <div style="font-size: 12px; color: var(--text3);">
                                                        ${formatDate(sessionId)} •
                                                        ${s.duration_mins || 0} min •
                                                        ${s.words || 0} words •
                                                        ${s.chunks || 0} chunks
                                                    </div>
                                                </div>
                                                <span style="color: var(--text3);">→</span>
                                            </div>
                                        </div>
                                    `;
                                }).join('')}
                            </div>
                        ` : `
                            <div class="empty-state">
                                <div class="empty-icon">📝</div>
                                <div>No sessions yet</div>
                                <button class="action-btn" style="margin-top: 16px;" onclick="addSessionToLecture('${lectureId}')">Start First Session</button>
                            </div>
                        `}
                    </div>
                </div>
            `;
        }

        function addSessionToLecture(lectureId) {
            // Pre-select this lecture and go to record panel
            showRecordPanel(lectureId);
        }

        function toggleClass(classId) {
            const section = document.querySelector(`.class-section[data-class="${classId}"]`);
            section.classList.toggle('expanded');
        }

        function showRecordPanel(preselectedLectureId = null) {
            currentView = 'record';
            currentSession = null;
            document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));

            // Build lecture options
            const lectureOptions = Object.entries(lectures)
                .sort((a, b) => (b[1].created || '').localeCompare(a[1].created || ''))
                .map(([id, l]) => {
                    const classInfo = classes.find(c => c.id === l.class_id);
                    return `<option value="${id}" ${id === preselectedLectureId ? 'selected' : ''}>${l.title} (${classInfo?.name || l.class_id})</option>`;
                }).join('');

            $('mainContent').innerHTML = `
                <div class="record-panel">
                    <div class="record-card">
                        <div class="record-title">New Recording</div>
                        <div class="record-grid">
                            <div class="control">
                                <label>Recording Type</label>
                                <select id="lectureSelect" onchange="onLectureTypeChange()">
                                    <option value="new" ${!preselectedLectureId ? 'selected' : ''}>📝 Create New Lecture</option>
                                    <optgroup label="Add to Existing Lecture">
                                        ${lectureOptions}
                                    </optgroup>
                                </select>
                            </div>
                            <div class="control" id="classSelectControl" ${preselectedLectureId ? 'style="display:none"' : ''}>
                                <label>Class</label>
                                <select id="classSelect">
                                    ${classes.map(c => `<option value="${c.id}">${c.name}</option>`).join('')}
                                </select>
                            </div>
                            <div class="control">
                                <label>Session Title</label>
                                <input type="text" id="sessionTitle" placeholder="e.g., Monte Carlo Methods">
                            </div>
                            <div class="control">
                                <label>Microphone</label>
                                <select id="deviceSelect" onchange="warmDevice()"></select>
                            </div>
                            <div class="control">
                                <label>Chunk Duration</label>
                                <select id="durationSelect">
                                    <option value="30">30 seconds</option>
                                    <option value="60" selected>60 seconds</option>
                                    <option value="90">90 seconds</option>
                                </select>
                            </div>
                        </div>
                        <button class="record-btn start" id="startBtn" onclick="startRecording()" ${!whisperReady ? 'disabled' : ''}>
                            ${whisperReady ? '⏺ Start Recording' : '⏳ Loading AI...'}
                        </button>
                        <button class="record-btn stop hidden" id="stopBtn" onclick="stopRecording()">⬛ Stop Recording</button>
                    </div>
                </div>
                <div class="live-chunks" id="liveChunks"></div>
            `;
            loadDevices();
        }

        function onLectureTypeChange() {
            const lectureSelect = $('lectureSelect');
            const classControl = $('classSelectControl');
            if (lectureSelect.value === 'new') {
                classControl.style.display = '';
            } else {
                classControl.style.display = 'none';
                // Auto-fill class from selected lecture
                const lecture = lectures[lectureSelect.value];
                if (lecture && $('classSelect')) {
                    $('classSelect').value = lecture.class_id;
                }
            }
        }

        function warmDevice() {
            const deviceId = parseInt($('deviceSelect').value);
            socket.emit('warm_device', { device_id: deviceId });
        }

        async function loadDevices() {
            const res = await fetch('/api/devices');
            const devices = await res.json();
            if ($('deviceSelect')) {
                $('deviceSelect').innerHTML = '<option value="-1">System Default</option>' +
                    devices.map(d => `<option value="${d.id}">${d.name}</option>`).join('');
                // Pre-warm the default device
                warmDevice();
            }
        }

        function startRecording() {
            // OPTIMISTIC UI: Show recording state immediately
            performance.mark('record_click');
            isRecording = true;
            $('dot').classList.add('recording');
            $('statusText').textContent = 'Starting...';
            if ($('startBtn')) $('startBtn').classList.add('hidden');
            if ($('stopBtn')) $('stopBtn').classList.remove('hidden');

            const title = $('sessionTitle').value || `Lecture ${new Date().toLocaleDateString()}`;
            const lectureId = $('lectureSelect')?.value || null;

            socket.emit('start_recording', {
                class_id: $('classSelect').value,
                title: title,
                chunk_duration: parseInt($('durationSelect').value),
                device_id: parseInt($('deviceSelect').value),
                model_size: 'base',
                lecture_id: lectureId === 'new' ? null : lectureId
            });
        }

        function stopRecording() {
            socket.emit('stop_recording');
        }

        // Revert optimistic UI on error
        function revertOptimisticUI() {
            if (!isRecording) return;
            isRecording = false;
            $('dot').classList.remove('recording');
            $('statusText').textContent = whisperReady ? 'Connected' : 'Loading AI model...';
            if ($('startBtn')) $('startBtn').classList.remove('hidden');
            if ($('stopBtn')) $('stopBtn').classList.add('hidden');
        }

        async function loadSession(sessionId) {
            currentView = 'session';
            currentSession = sessionId;
            
            document.querySelectorAll('.session-item').forEach(el => {
                el.classList.toggle('active', el.dataset.id === sessionId);
            });
            
            const meta = sessions[sessionId];
            if (!meta) return;
            
            const res = await fetch(`/api/sessions/${meta.class_id}/${sessionId}`);
            const data = await res.json();
            if (!data.content) return;
            
            const classInfo = classes.find(c => c.id === meta.class_id) || {};
            const tags = meta.tags || [];
            const hasTranscript = !!data.transcript;
            
            $('mainContent').innerHTML = `
                <div class="notes-view">
                    <div class="notes-header">
                        <div class="notes-title-section">
                            <div class="notes-title" contenteditable="true" id="noteTitle" onblur="updateTitle('${sessionId}')">${meta.title || 'Untitled'}</div>
                            <div class="notes-meta">
                                <div class="notes-class" onclick="showMoveModal('${sessionId}')" style="border-left: 3px solid ${classInfo.color || '#888'}">
                                    ${classInfo.name || 'Unknown Class'} ▾
                                </div>
                                <div class="notes-date">${formatDate(sessionId)}</div>
                                ${meta.duration_mins ? `<div class="notes-date">⏱ ${meta.duration_mins} min</div>` : ''}
                                ${meta.chunks ? `<div class="notes-date">📝 ${meta.chunks} chunks</div>` : ''}
                                ${meta.words ? `<div class="notes-date">💬 ${meta.words.toLocaleString()} words</div>` : ''}
                                ${meta.completed === false ? '<div class="notes-date" style="color: var(--warning)">⚠ Incomplete</div>' : ''}
                            </div>
                        </div>
                        <div class="notes-actions">
                            <button class="action-btn" onclick="exportSession('${sessionId}')">📥 Export</button>
                            <button class="action-btn danger" onclick="showDeleteModal('${sessionId}')">🗑 Delete</button>
                        </div>
                    </div>
                    <div class="tags-section" id="tagsSection">
                        ${tags.map(t => `<span class="tag">${t} <span class="remove" onclick="removeTag('${sessionId}', '${t}')">×</span></span>`).join('')}
                        <span class="add-tag" onclick="showAddTagModal('${sessionId}')">+ Add tag</span>
                    </div>
                    
                    <div class="view-tabs">
                        <button class="view-tab active" data-view="notes" onclick="switchView('notes')">📝 Enhanced Notes</button>
                        <button class="view-tab" data-view="transcript" onclick="switchView('transcript')" ${!hasTranscript ? 'disabled title="No transcript available"' : ''}>📜 Full Transcript</button>
                    </div>
                    
                    <div class="notes-content" id="notesView">
                        <div class="notes-text">${marked.parse(data.content)}</div>
                    </div>
                    
                    <div class="notes-content hidden" id="transcriptView">
                        ${hasTranscript ? `<div class="transcript-full">${marked.parse(data.transcript)}</div>` : '<div class="empty-state"><div class="empty-icon">📜</div><div>No transcript available</div></div>'}
                    </div>
                </div>
            `;
        }
        
        function switchView(view) {
            document.querySelectorAll('.view-tab').forEach(t => t.classList.toggle('active', t.dataset.view === view));
            $('notesView').classList.toggle('hidden', view !== 'notes');
            $('transcriptView').classList.toggle('hidden', view !== 'transcript');
        }

        function formatDate(sessionId) {
            try {
                if (!sessionId || sessionId.length < 13) return sessionId;
                const y = sessionId.slice(0,4), m = sessionId.slice(4,6), d = sessionId.slice(6,8);
                const h = sessionId.slice(9,11), min = sessionId.slice(11,13);
                if (isNaN(+y) || isNaN(+m) || isNaN(+d)) return sessionId;
                return `${y}-${m}-${d} ${h}:${min}`;
            } catch { return sessionId; }
        }

        async function updateTitle(sessionId) {
            const title = $('noteTitle').textContent.trim();
            await fetch(`/api/sessions/${sessionId}/update`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({title})
            });
            sessions[sessionId].title = title;
            renderSidebar();
        }

        function showMoveModal(sessionId) {
            const meta = sessions[sessionId];
            $('modal').innerHTML = `
                <div class="modal-title">Move to Class</div>
                <div class="modal-body">
                    <div class="control">
                        <label>Select Class</label>
                        <select id="moveClassSelect">
                            ${classes.map(c => `<option value="${c.id}" ${c.id === meta.class_id ? 'selected' : ''}>${c.name}</option>`).join('')}
                        </select>
                    </div>
                </div>
                <div class="modal-actions">
                    <button class="modal-btn secondary" onclick="hideModal()">Cancel</button>
                    <button class="modal-btn primary" onclick="moveSession('${sessionId}')">Move</button>
                </div>
            `;
            $('modalOverlay').classList.remove('hidden');
        }

        async function moveSession(sessionId) {
            const newClass = $('moveClassSelect').value;
            const oldClass = sessions[sessionId].class_id;
            if (newClass === oldClass) { hideModal(); return; }
            
            await fetch(`/api/sessions/${sessionId}/move`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({old_class: oldClass, new_class: newClass})
            });
            
            sessions[sessionId].class_id = newClass;
            hideModal();
            renderSidebar();
            loadSession(sessionId);
        }

        function showDeleteModal(sessionId) {
            $('modal').innerHTML = `
                <div class="modal-title">Delete Session?</div>
                <div class="modal-body">
                    <p style="color: var(--text2)">This will permanently delete "${sessions[sessionId]?.title || 'this session'}" and cannot be undone.</p>
                </div>
                <div class="modal-actions">
                    <button class="modal-btn secondary" onclick="hideModal()">Cancel</button>
                    <button class="modal-btn danger" onclick="deleteSession('${sessionId}')">Delete</button>
                </div>
            `;
            $('modalOverlay').classList.remove('hidden');
        }

        async function deleteSession(sessionId) {
            const meta = sessions[sessionId];
            await fetch(`/api/sessions/${meta.class_id}/${sessionId}`, {method: 'DELETE'});
            delete sessions[sessionId];
            hideModal();
            renderSidebar();
            showRecordPanel();
        }

        function showAddTagModal(sessionId) {
            $('modal').innerHTML = `
                <div class="modal-title">Add Tag</div>
                <div class="modal-body">
                    <div class="control">
                        <label>Tag Name</label>
                        <input type="text" id="tagInput" placeholder="e.g., important, exam-topic">
                    </div>
                </div>
                <div class="modal-actions">
                    <button class="modal-btn secondary" onclick="hideModal()">Cancel</button>
                    <button class="modal-btn primary" onclick="addTag('${sessionId}')">Add</button>
                </div>
            `;
            $('modalOverlay').classList.remove('hidden');
            setTimeout(() => $('tagInput').focus(), 100);
        }

        async function addTag(sessionId) {
            const tag = $('tagInput').value.trim();
            if (!tag) return;
            
            const tags = sessions[sessionId].tags || [];
            if (!tags.includes(tag)) {
                tags.push(tag);
                await fetch(`/api/sessions/${sessionId}/update`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({tags})
                });
                sessions[sessionId].tags = tags;
            }
            hideModal();
            loadSession(sessionId);
        }

        async function removeTag(sessionId, tag) {
            const tags = (sessions[sessionId].tags || []).filter(t => t !== tag);
            await fetch(`/api/sessions/${sessionId}/update`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({tags})
            });
            sessions[sessionId].tags = tags;
            loadSession(sessionId);
        }

        function exportSession(sessionId) {
            const meta = sessions[sessionId];
            showExportModal(sessionId, meta.class_id);
        }
        
        function showExportModal(sessionId, classId) {
            $('modal').innerHTML = `
                <div class="modal-title">Export Session</div>
                <div class="modal-body">
                    <p style="color: var(--text2); margin-bottom: 16px;">Choose what to export:</p>
                    <div style="display: flex; flex-direction: column; gap: 8px;">
                        <button class="action-btn" onclick="downloadFile('${classId}', '${sessionId}', 'notes')" style="justify-content: flex-start;">📝 Enhanced Notes (.md)</button>
                        <button class="action-btn" onclick="downloadFile('${classId}', '${sessionId}', 'transcript')" style="justify-content: flex-start;">📜 Full Transcript (.md)</button>
                        <button class="action-btn" onclick="downloadFile('${classId}', '${sessionId}', 'both')" style="justify-content: flex-start;">📦 Both Files (.zip)</button>
                    </div>
                </div>
                <div class="modal-actions">
                    <button class="modal-btn secondary" onclick="hideModal()">Cancel</button>
                </div>
            `;
            $('modalOverlay').classList.remove('hidden');
        }
        
        function downloadFile(classId, sessionId, type) {
            window.open(`/api/sessions/${classId}/${sessionId}/download?type=${type}`, '_blank');
            hideModal();
        }

        function hideModal() {
            $('modalOverlay').classList.add('hidden');
        }

        // Search
        let searchTimeout;
        $('searchInput').addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => search(e.target.value), 300);
        });

        async function search(query) {
            if (!query.trim()) {
                renderSidebar();
                if (currentSession) loadSession(currentSession);
                else showRecordPanel();
                return;
            }
            
            const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
            const results = await res.json();
            
            $('mainContent').innerHTML = `
                <div class="search-results">
                    <div class="search-title">Search Results for "${query}"</div>
                    ${results.length ? results.map(r => {
                        const classInfo = classes.find(c => c.id === r.class_id) || {};
                        return `
                            <div class="search-result" onclick="loadSession('${r.id}')">
                                <div class="search-result-title">${r.title || 'Untitled'}</div>
                                <div class="search-result-meta">
                                    <span style="color: ${classInfo.color}">${classInfo.name || 'Unknown'}</span>
                                    · ${formatDate(r.id)}
                                    · Match in ${r.match}
                                </div>
                            </div>
                        `;
                    }).join('') : '<div class="empty-state"><div class="empty-icon">🔍</div><div>No results found</div></div>'}
                </div>
            `;
        }

        // Socket events
        socket.on('connect', () => {
            $('dot').classList.add('connected');
            $('statusText').textContent = whisperReady ? 'Connected' : 'Loading AI model...';
        });

        socket.on('status', data => {
            $('statusText').textContent = data.message;
            if (data.recording !== undefined) {
                isRecording = data.recording;
                $('dot').classList.toggle('recording', isRecording);
                if ($('startBtn')) $('startBtn').classList.toggle('hidden', isRecording);
                if ($('stopBtn')) $('stopBtn').classList.toggle('hidden', !isRecording);
            }
            if (data.whisper_ready !== undefined) {
                whisperReady = data.whisper_ready;
                updateStartButtonState();
            }
        });

        socket.on('whisper_status', data => {
            whisperReady = data.ready;
            if (data.ready) {
                $('statusText').textContent = isRecording ? 'Recording...' : 'Connected';
            }
            updateStartButtonState();
        });

        socket.on('device_warmed', data => {
            console.log('Device warmed:', data);
        });

        socket.on('perf_timing', data => {
            console.log('Performance timing:', data);
            performance.mark('server_' + data.phase);
        });

        socket.on('error', data => {
            alert(data.message);
            revertOptimisticUI();
        });

        socket.on('session_started', async (data) => {
            performance.mark('session_started');
            if (performance.getEntriesByName('record_click').length) {
                performance.measure('record_to_started', 'record_click', 'session_started');
                const measure = performance.getEntriesByName('record_to_started')[0];
                console.log(`Record start latency: ${measure.duration.toFixed(0)}ms`);
            }
            if ($('liveChunks')) $('liveChunks').innerHTML = '';
            await loadSessions();
            await loadLectures();
            renderSidebar();
        });

        socket.on('transcript', data => {
            const chunk = document.createElement('div');
            chunk.className = 'live-chunk';
            chunk.id = `chunk-${data.chunk}`;
            chunk.innerHTML = `
                <div class="chunk-header">
                    <span class="chunk-num">Chunk ${data.chunk}</span>
                    <span class="chunk-time">${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="transcript">"${data.text}"</div>
                <div class="notes-text">Processing...</div>
            `;
            if ($('liveChunks')) $('liveChunks').prepend(chunk);
        });

        socket.on('notes', data => {
            const chunk = $(`chunk-${data.chunk}`);
            if (chunk) chunk.querySelector('.notes-text').innerHTML = marked.parse(data.content);
        });

        socket.on('summary', data => {
            const div = document.createElement('div');
            div.className = 'summary-box';
            div.innerHTML = `<div class="summary-title">📚 Summary</div><div class="notes-text">${marked.parse(data.content)}</div>`;
            if ($('liveChunks')) $('liveChunks').prepend(div);
        });

        socket.on('session_ended', async (data) => {
            await loadSessions();
            await loadLectures();
            renderSidebar();
            if (data.session_id) loadSession(data.session_id);
        });

        // Event listeners
        $('newRecordingBtn').onclick = showRecordPanel;
        $('modalOverlay').onclick = (e) => { if (e.target === $('modalOverlay')) hideModal(); };

        // Init
        init();
    </script>
</body>
</html>
'''


# ============== AUDIO RECORDER ==============
class AudioRecorder:
    """Records audio from a microphone and provides chunked access for transcription.

    Uses sounddevice for low-latency audio capture with a callback-based
    architecture. Audio is collected into a thread-safe queue and can be
    retrieved in fixed-duration chunks for processing by Whisper.

    Args:
        sample_rate: Recording sample rate in Hz (default: 16000, required by Whisper).
        channels: Number of audio channels (default: 1 for mono).
        device: Audio input device ID, or None for system default.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, channels=AUDIO_CHANNELS, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None

    def callback(self, indata, frames, time_info, status):
        if status: logger.warning(f"Audio: {status}")
        self.audio_queue.put(indata.copy())

    def start(self):
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, channels=self.channels, dtype=np.float32,
            callback=self.callback, blocksize=int(self.sample_rate * AUDIO_BLOCK_DURATION),
            device=self.device if self.device != -1 else None
        )
        self.stream.start()

    def stop(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        while not self.audio_queue.empty():
            try: self.audio_queue.get_nowait()
            except: break

    def get_chunk(self, duration):
        samples = int(self.sample_rate * duration)
        collected = []
        count = 0
        while count < samples and self.is_recording:
            try:
                data = self.audio_queue.get(timeout=1.0)
                collected.append(data)
                count += len(data)
            except queue.Empty: continue
        if not collected: return None
        return np.concatenate(collected)[:samples].flatten()

    def save(self, audio, path):
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(path, 'wb') as f:
            f.setnchannels(self.channels)
            f.setsampwidth(2)
            f.setframerate(self.sample_rate)
            f.writeframes(audio_int16.tobytes())


# ============== SESSION MANAGER ==============
class SessionManager:
    """Manages the lifecycle of a recording session.

    Handles creating session files, appending transcription chunks,
    and finalizing sessions with summaries and metadata. Each session
    produces two Markdown files: enhanced notes and raw transcript.

    Session lifecycle: start() -> add_chunk() (repeated) -> finalize()
    """

    def __init__(self):
        self.session_id = None
        self.class_id = None
        self.title = None
        self.lecture_id = None
        self.notes = []
        self.transcripts = []
        self.chunk_count = 0
        self.start_time = None

    def start(self, class_id, title, lecture_id=None):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.class_id = class_id
        self.title = title
        self.lecture_id = lecture_id
        self.notes = []
        self.transcripts = []
        self.chunk_count = 0
        self.start_time = datetime.now()

        class_name = next((c["name"] for c in CLASSES if c["id"] == class_id), "Unknown")
        class_dir = notes_dir / class_id
        class_dir.mkdir(exist_ok=True)

        try:
            # Create notes file
            filepath = class_dir / f"lecture_{self.session_id}.md"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {class_name}\n")
                f.write(f"## {title}\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n")

            # Create transcript file
            transcript_path = class_dir / f"transcript_{self.session_id}.md"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(f"# Full Transcript: {title}\n")
                f.write(f"**Class:** {class_name}\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n")

            # Set session metadata with lecture association
            metadata_mgr.set_session(self.session_id, class_id, title, lecture_id=lecture_id)

            # If lecture_id provided, add session to that lecture
            if lecture_id:
                metadata_mgr.add_session_to_lecture(lecture_id, self.session_id)

            logger.info(f"Session started: {self.session_id} in {class_id}" +
                  (f" (lecture: {lecture_id})" if lecture_id else ""))
            return self.session_id
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            return None

    def add_chunk(self, transcript, enhanced):
        """Add a chunk with automatic retry and backup"""
        self.chunk_count += 1
        self.notes.append(enhanced)
        self.transcripts.append(transcript)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        try:
            # Save enhanced notes
            filepath = notes_dir / self.class_id / f"lecture_{self.session_id}.md"
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"### Chunk {self.chunk_count}\n\n")
                f.write(f"{enhanced}\n\n")
                f.write(f"<details>\n<summary>📝 View Transcript [{timestamp}]</summary>\n\n")
                f.write(f"> {transcript}\n\n")
                f.write(f"</details>\n\n---\n\n")
            
            # Save raw transcript
            transcript_path = notes_dir / self.class_id / f"transcript_{self.session_id}.md"
            with open(transcript_path, 'a', encoding='utf-8') as f:
                f.write(f"### [{self.chunk_count}] {timestamp}\n\n")
                f.write(f"{transcript}\n\n---\n\n")
            
            # Update metadata with progress
            metadata_mgr.update_session(self.session_id, 
                chunks=self.chunk_count,
                last_update=datetime.now().isoformat()
            )
            
            logger.info(f"Chunk {self.chunk_count} saved")
            return True
            
        except Exception as e:
            logger.error(f"Error saving chunk {self.chunk_count}: {e}")
            # Emergency backup to temp file
            try:
                backup_path = notes_dir / f"_backup_{self.session_id}_{self.chunk_count}.txt"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(f"BACKUP - Chunk {self.chunk_count}\n")
                    f.write(f"Transcript: {transcript}\n\n")
                    f.write(f"Enhanced: {enhanced}\n")
                logger.warning(f"Backup saved to {backup_path}")
            except:
                pass
            return False

    def finalize(self, summary):
        """Finalize session with summary and statistics"""
        try:
            duration_mins = 0
            if self.start_time:
                duration_mins = int((datetime.now() - self.start_time).total_seconds() / 60)
            
            filepath = notes_dir / self.class_id / f"lecture_{self.session_id}.md"
            
            # Add summary if available
            if summary:
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write(f"\n## 📚 Session Summary\n\n{summary}\n")
            
            # Add session stats
            full_transcript = " ".join(self.transcripts)
            word_count = len(full_transcript.split())
            
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"\n---\n\n### 📊 Session Statistics\n")
                f.write(f"- **Duration:** {duration_mins} minutes\n")
                f.write(f"- **Chunks:** {self.chunk_count}\n")
                f.write(f"- **Words transcribed:** {word_count}\n")
            
            # Finalize transcript file
            transcript_path = notes_dir / self.class_id / f"transcript_{self.session_id}.md"
            if transcript_path.exists():
                with open(transcript_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n---\n\n## 📊 Statistics\n\n")
                    f.write(f"- **Total words:** {word_count}\n")
                    f.write(f"- **Total chunks:** {self.chunk_count}\n")
                    f.write(f"- **Duration:** {duration_mins} minutes\n")
            
            # Update metadata with final stats
            metadata_mgr.update_session(self.session_id,
                chunks=self.chunk_count,
                words=word_count,
                duration_mins=duration_mins,
                completed=True,
                completed_at=datetime.now().isoformat()
            )
            
            logger.info(f"Session finalized: {self.chunk_count} chunks, {word_count} words, {duration_mins} min")
            return True
            
        except Exception as e:
            logger.error(f"Error finalizing session: {e}")
            return False


session_mgr = SessionManager()


# ============== AI FUNCTIONS ==============
def ollama_generate(prompt, model=OLLAMA_MODEL):
    import urllib.request
    import json as json_lib
    try:
        data = json_lib.dumps({"model": model, "prompt": prompt, "stream": False}).encode('utf-8')
        req = urllib.request.Request(f"{OLLAMA_API_URL}/api/generate", data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json_lib.loads(resp.read().decode('utf-8')).get("response", "")
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return None

def load_whisper(model_size):
    global whisper_model, whisper
    if whisper is None:
        socketio.emit('status', {'message': 'Installing Whisper...'})
        os.system(f'"{sys.executable}" -m pip install openai-whisper -q')
        import whisper as w
        whisper = w
    if whisper_model is None:
        socketio.emit('status', {'message': f'Loading Whisper ({model_size})...'})
        whisper_model = whisper.load_model(model_size)
    return whisper_model

def transcribe(path):
    result = whisper_model.transcribe(path, language=WHISPER_LANGUAGE, fp16=WHISPER_FP16, verbose=False)
    return result["text"].strip()

def enhance(text):
    if not ollama_available: return f"**Transcript:**\n{text}"
    prompt = f"""Transform this lecture transcript into structured notes. Be concise.

TRANSCRIPT: {text}

Create notes with:
- **Key Concepts** - Main ideas (2-3 bullets)
- **Details** - Important points (2-3 bullets)
- **Questions** - Things to clarify (1-2 bullets)

Output only the notes."""
    return ollama_generate(prompt) or f"**Transcript:**\n{text}"

def summarize(notes):
    if not ollama_available or not notes: return ""
    prompt = f"""Summarize these lecture notes:

{chr(10).join(notes)}

Create:
1. **Summary** (2-3 sentences)
2. **Key Takeaways** (4-5 bullets)
3. **Review Topics** (2-3 items)"""
    return ollama_generate(prompt) or ""


# ============== RECORDING LOOP ==============
def recording_loop(class_id, title, chunk_duration, device_id, model_size, lecture_id=None):
    global recorder, session_active, warm_recorder, warm_device_id

    timings = {'t0_loop_start': time.perf_counter()}

    try:
        # Use pre-warmed recorder if available and matching device
        with warm_recorder_lock:
            if warm_recorder and warm_device_id == device_id:
                recorder = warm_recorder
                warm_recorder = None
                warm_device_id = None
                timings['t1_recorder_reused'] = time.perf_counter()
            else:
                recorder = AudioRecorder(device=device_id)
                timings['t1_recorder_created'] = time.perf_counter()

        recorder.start()
        timings['t2_recorder_started'] = time.perf_counter()

        session_id = session_mgr.start(class_id, title, lecture_id)
        timings['t3_session_created'] = time.perf_counter()

        if not session_id:
            socketio.emit('error', {'message': 'Failed to start session'})
            session_active = False
            return

        # Emit session_started BEFORE loading Whisper for faster UI feedback
        socketio.emit('session_started', {'session_id': session_id, 'class_id': class_id})
        socketio.emit('status', {'message': 'Recording started...', 'recording': True})
        timings['t4_session_started_emitted'] = time.perf_counter()

        # Emit timing data for profiling
        socketio.emit('perf_timing', {
            'phase': 'record_start',
            'timings': {k: round((v - timings['t0_loop_start']) * 1000, 2) for k, v in timings.items()}
        })

        # Load Whisper (should be instant if pre-warmed)
        if not whisper_ready.is_set():
            socketio.emit('status', {'message': 'Waiting for AI model...', 'recording': True})
            whisper_ready.wait()  # Wait for pre-warm to complete
        load_whisper(model_size)
        timings['t5_whisper_loaded'] = time.perf_counter()

        # Emit full timing after Whisper load
        socketio.emit('perf_timing', {
            'phase': 'whisper_loaded',
            'total_ms': round((timings['t5_whisper_loaded'] - timings['t0_loop_start']) * 1000, 2)
        })

        while session_active:
            try:
                socketio.emit('status', {'message': 'Recording...', 'recording': True})
                audio = recorder.get_chunk(chunk_duration)
                
                if audio is None or len(audio) == 0:
                    continue

                # Skip silent or near-silent audio chunks
                if np.max(np.abs(audio)) < 0.001:
                    logger.info("Skipping silent audio chunk")
                    continue

                # Save audio to temp file
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        temp_path = f.name
                    recorder.save(audio, temp_path)

                    socketio.emit('status', {'message': 'Transcribing...', 'recording': True})
                    text = transcribe(temp_path)
                    
                finally:
                    # Always clean up temp file
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass

                if not text or len(text.strip()) < MIN_TRANSCRIPT_LENGTH:
                    socketio.emit('status', {'message': 'No speech detected...', 'recording': True})
                    continue

                socketio.emit('transcript', {'chunk': session_mgr.chunk_count + 1, 'text': text})
                socketio.emit('status', {'message': 'Enhancing...', 'recording': True})
                
                try:
                    enhanced = enhance(text)
                except Exception as e:
                    logger.error(f"Enhancement failed: {e}")
                    enhanced = f"**Raw Transcript:**\n\n{text}"
                
                session_mgr.add_chunk(text, enhanced)
                socketio.emit('notes', {'chunk': session_mgr.chunk_count, 'content': enhanced})
                
            except Exception as e:
                logger.error(f"Error in recording loop iteration: {e}")
                socketio.emit('status', {'message': f'Error: {str(e)[:50]}...', 'recording': True})
                continue

        # Finalize session
        recorder.stop()
        socketio.emit('status', {'message': 'Generating summary...'})
        
        try:
            summary = summarize(session_mgr.notes)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            summary = ""
        
        session_mgr.finalize(summary)
        
        if summary:
            socketio.emit('summary', {'content': summary})
        
        socketio.emit('session_ended', {'session_id': session_mgr.session_id})
        socketio.emit('status', {'message': 'Done', 'recording': False})
        
    except Exception as e:
        logger.critical(f"Recording failed: {e}")
        socketio.emit('error', {'message': f'Recording failed: {str(e)}'})
        socketio.emit('status', {'message': 'Error - recording stopped', 'recording': False})
        session_active = False
        
        # Try to save what we have
        if session_mgr.chunk_count > 0:
            try:
                session_mgr.finalize("")
                logger.info("Emergency save completed")
            except:
                pass


# ============== ROUTES ==============
@app.route('/')
def index():
    return Response(HTML_PAGE, mimetype='text/html')

@app.route('/api/devices')
def get_devices():
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0:
            devices.append({'id': i, 'name': d['name'], 'default': i == sd.default.device[0]})
    return jsonify(devices)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'whisper_ready': whisper_ready.is_set(),
        'ollama_available': ollama_available,
        'recording': session_active,
    })

@app.route('/api/classes')
def get_classes():
    return jsonify(CLASSES)

# ============== LECTURE API ==============
@app.route('/api/lectures')
def get_lectures():
    """Get all lectures"""
    return jsonify(metadata_mgr.get_all_lectures())

@app.route('/api/lectures', methods=['POST'])
def create_lecture():
    """Create a new lecture"""
    data = request.json or {}
    lecture_id = metadata_mgr.create_lecture(
        data.get('class_id', CLASSES[0]['id']),
        data.get('title', f"Lecture {datetime.now().strftime('%Y-%m-%d')}")
    )
    return jsonify({'lecture_id': lecture_id, 'success': True})

@app.route('/api/lectures/<lecture_id>')
def get_lecture(lecture_id):
    """Get lecture with session details"""
    lecture = metadata_mgr.get_lecture_with_sessions(lecture_id)
    if not lecture:
        return jsonify({'error': 'Lecture not found'}), 404
    return jsonify(lecture)

@app.route('/api/lectures/<lecture_id>', methods=['PUT'])
def update_lecture(lecture_id):
    """Update lecture metadata"""
    data = request.json or {}
    success = metadata_mgr.update_lecture(lecture_id, **data)
    if success:
        return jsonify({'success': True})
    return jsonify({'error': 'Lecture not found'}), 404

@app.route('/api/sessions/all')
def get_all_sessions():
    return jsonify(metadata_mgr.get_all_sessions())

@app.route('/api/sessions/<class_id>/<sid>')
def get_session(class_id, sid):
    filepath = notes_dir / class_id / f"lecture_{sid}.md"
    transcript_path = notes_dir / class_id / f"transcript_{sid}.md"
    # Also check for old .txt format
    transcript_path_txt = notes_dir / class_id / f"transcript_{sid}.txt"
    
    result = {}
    if filepath.exists():
        result['content'] = filepath.read_text(encoding='utf-8')
    
    if transcript_path.exists():
        result['transcript'] = transcript_path.read_text(encoding='utf-8')
    elif transcript_path_txt.exists():
        # Convert old format on the fly
        result['transcript'] = f"# Full Transcript\n\n{transcript_path_txt.read_text(encoding='utf-8')}"
    
    if result:
        return jsonify(result)
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/sessions/<class_id>/<sid>/download')
def download_session(class_id, sid):
    export_type = request.args.get('type', 'notes')
    
    notes_path = notes_dir / class_id / f"lecture_{sid}.md"
    transcript_path = notes_dir / class_id / f"transcript_{sid}.md"
    
    if export_type == 'notes':
        if notes_path.exists():
            content = notes_path.read_text(encoding='utf-8')
            return Response(content, mimetype='text/markdown', 
                          headers={'Content-Disposition': f'attachment; filename="notes_{sid}.md"'})
    
    elif export_type == 'transcript':
        if transcript_path.exists():
            content = transcript_path.read_text(encoding='utf-8')
            return Response(content, mimetype='text/markdown',
                          headers={'Content-Disposition': f'attachment; filename="transcript_{sid}.md"'})
    
    elif export_type == 'both':
        import io
        import zipfile
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            if notes_path.exists():
                zf.writestr(f"notes_{sid}.md", notes_path.read_text(encoding='utf-8'))
            if transcript_path.exists():
                zf.writestr(f"transcript_{sid}.md", transcript_path.read_text(encoding='utf-8'))
        
        zip_buffer.seek(0)
        return Response(zip_buffer.getvalue(), mimetype='application/zip',
                      headers={'Content-Disposition': f'attachment; filename="lecture_{sid}.zip"'})
    
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/sessions/<class_id>/<sid>', methods=['DELETE'])
def delete_session(class_id, sid):
    notes_path = notes_dir / class_id / f"lecture_{sid}.md"
    transcript_path = notes_dir / class_id / f"transcript_{sid}.md"
    
    deleted = False
    if notes_path.exists():
        notes_path.unlink()
        deleted = True
    if transcript_path.exists():
        transcript_path.unlink()
        deleted = True
    
    if deleted:
        metadata_mgr.delete_session(sid)
        return jsonify({'success': True})
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/sessions/<sid>/update', methods=['POST'])
def update_session(sid):
    data = request.json
    metadata_mgr.update_session(sid, **data)
    return jsonify({'success': True})

@app.route('/api/sessions/<sid>/move', methods=['POST'])
def move_session(sid):
    data = request.json
    metadata_mgr.move_session(sid, data['old_class'], data['new_class'])
    return jsonify({'success': True})

@app.route('/api/search')
def search_sessions():
    query = request.args.get('q', '')
    return jsonify(metadata_mgr.search(query))


# ============== SOCKET EVENTS ==============
@socketio.on('connect')
def on_connect():
    is_whisper_ready = whisper_ready.is_set()
    emit('status', {
        'message': 'Connected' if is_whisper_ready else 'Loading AI model...',
        'recording': session_active,
        'has_api_key': ollama_available,
        'whisper_ready': is_whisper_ready
    })
    # If Whisper is still loading, notify when ready
    if not is_whisper_ready:
        def notify_when_ready():
            whisper_ready.wait()
            socketio.emit('whisper_status', {'ready': True, 'message': 'AI model ready'})
        threading.Thread(target=notify_when_ready, daemon=True).start()

@socketio.on('start_recording')
def on_start(data):
    global session_active
    if session_active:
        emit('error', {'message': 'Already recording'})
        return

    # Handle lecture context - create new lecture or use existing
    lecture_id = data.get('lecture_id')
    class_id = data.get('class_id', CLASSES[0]['id'])
    title = data.get('title', 'Untitled Session')

    if not lecture_id:
        # Create a new lecture for this recording
        lecture_id = metadata_mgr.create_lecture(class_id, title)

    session_active = True
    t = threading.Thread(target=recording_loop, args=(
        class_id,
        title,
        data.get('chunk_duration', 60),
        data.get('device_id', -1),
        data.get('model_size', 'base'),
        lecture_id  # Pass lecture context
    ))
    t.daemon = True
    t.start()

@socketio.on('stop_recording')
def on_stop():
    global session_active
    session_active = False

@socketio.on('warm_device')
def on_warm_device(data):
    """Pre-warm audio device to eliminate initialization latency on record start"""
    global warm_recorder, warm_device_id
    device_id = data.get('device_id', -1)

    with warm_recorder_lock:
        # Clean up previous warmed device
        if warm_recorder:
            try:
                warm_recorder.stop()
            except:
                pass
            warm_recorder = None

        # Pre-warm new device
        try:
            warm_recorder = AudioRecorder(device=device_id)
            warm_device_id = device_id
            emit('device_warmed', {'device_id': device_id, 'success': True})
        except Exception as e:
            warm_recorder = None
            warm_device_id = None
            emit('device_warmed', {'device_id': device_id, 'success': False, 'error': str(e)})


# ============== MAIN ==============
if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("LECTURE NOTES AI")
    logger.info("=" * 50)
    logger.info("Open http://localhost:5000")
    if not ollama_available:
        logger.info("For FREE AI enhancement:")
        logger.info("  1. Install from https://ollama.com")
        logger.info("  2. Run: ollama pull llama3.2")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
