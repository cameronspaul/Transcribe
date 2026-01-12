# 🦜 Video Transcriber - Parakeet TDT 0.6B V3

A powerful, high-performance video-to-SRT transcription tool powered by NVIDIA's **Parakeet TDT 0.6B V3** multilingual speech recognition model. Designed for creators, this tool provides accurate timestamps, smart segment splitting, and specialized features for short-form content.

## ✨ Key Features

- **🚀 State-of-the-Art Accuracy**: Uses NVIDIA's latest Parakeet TDT 0.6B V3 model.
- **🌍 Multilingual**: Automatically transcribes 25 European languages.
- **⏱️ Precise Timestamps**: Accurate word-level and segment-level timing.
- **📱 Short-Form Optimized**: Default settings (5 words per segment) perfect for TikTok, Reels, and Shorts.
- **💎 Premium Formatting**: Support for all-caps text and punctuation removal.
- **🔌 Silence Detection**: Automatically identifying and marking silent periods.
- **💾 Memory Efficient**: Processes long videos in chunks to prevent GPU out-of-memory errors.

## 🚀 Installation

### 1. Prerequisites
- **Python 3.8+**
- **FFmpeg**: Required for audio extraction.
  - Windows: `winget install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- **NVIDIA GPU** (Recommended): For lightning-fast processing.

### 2. Quick Setup
Run the automated setup script:
```powershell
.\setup.bat
```
*Or manually:*
```bash
python -m venv venv
.\venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
pip install -r requirements.txt
```

## 💻 Usage

### Basic Transcription
```bash
python transcribe.py video.mp4
```
This generates `video.srt` in the same directory.

### Advanced Formatting
```bash
# perfect for viral-style captions (Shorts/Reels)
python transcribe.py video.mp4 --max-words 3 --uppercase --remove-punctuation
```

### Options Overview

| Flag | Description | Default |
|------|-------------|---------|
| `-o`, `--output` | Specify output SRT path | `[video_name].srt` |
| `--max-words` | Max words per caption segment | `5` |
| `--uppercase` | Convert all text to UPPERCASE | `False` |
| `--remove-punctuation` | Strip commas, periods, etc. | `False` |
| `--offset` | Shift all timestamps (e.g., `-0.1` to advance) | `0.0` |
| `--word-gap` | Max silence between words in a segment | `0.5s` |
| `--include-silence` | Add `[SILENCE]` markers to SRT | `False` |
| `--chunk-duration` | Audio chunk size for GPU memory | `60.0s` |

## 🛠️ Optimization Tips

### For Social Media (TikTok/Reels/Shorts)
Use these settings for those snappy, centered captions:
```bash
python transcribe.py video.mp4 --max-words 1 --uppercase --remove-punctuation
```

### Fixing Sync Issues
If your audio is slightly out of sync with the video, use the `--offset` flag:
```bash
# If captions appear 0.2s late, advance them:
python transcribe.py video.mp4 --offset -0.2
```

### Handling Long Videos
The tool automatically uses local attention and chunking for long files. If you have 8GB+ VRAM, you can increase `--chunk-duration` to `300` for slightly better coherence in very fast speech.

## 📊 Model Performance

| Metric | Performance |
|--------|-------------|
| **WER (clean)** | 1.93% |
| **WER (noisy)** | 3.59% |
| **Languages** | 25+ |
| **Architecture** | Transformer-based TDT |

## 📄 License

This project utilizes NVIDIA's Parakeet model under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

## 🔗 Credits
- Model: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- Toolkit: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
