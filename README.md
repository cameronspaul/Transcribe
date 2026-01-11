# 🦜 Video Transcriber - Parakeet TDT 0.6B V3

A powerful video-to-SRT transcription tool using NVIDIA's **Parakeet TDT 0.6B V3** multilingual speech recognition model. This tool extracts audio from video files, transcribes speech with accurate timestamps, and optionally detects silence periods.

## ✨ Features

- **Multilingual Support**: Transcribes 25 European languages automatically
- **Accurate Timestamps**: Word-level and segment-level timing
- **Silence Detection**: Optionally marks silent periods in the transcript
- **Long Audio Support**: Handles videos up to 3 hours with local attention
- **Punctuation & Capitalization**: Automatic formatting included
- **SRT Output**: Industry-standard subtitle format

## 🌍 Supported Languages

Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish, Russian, Ukrainian

## 📋 Prerequisites

- **Python 3.8+**
- **FFmpeg** - Required for video processing
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `winget install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
- **NVIDIA GPU** (Recommended) - CUDA-enabled GPU for faster processing
- **~4GB+ RAM** - For model loading

## 🚀 Installation

### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd "c:\Users\camer\Videos\OBS\Trasnscribe"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (CMD)
.\venv\Scripts\activate.bat

# Linux/macOS
source venv/bin/activate
```

### 2. Install PyTorch (with CUDA support)

For GPU acceleration, install PyTorch with CUDA first:

```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (slower)
pip install torch torchaudio
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Transcription

```bash
python transcribe.py video.mp4
```

This creates `video.srt` in the same directory.

### Specify Output File

```bash
python transcribe.py video.mp4 -o subtitles.srt
```

### Include Silence Markers

```bash
python transcribe.py video.mp4 --include-silence
```

This adds `[SILENCE]` markers in the SRT file where pauses are detected.

### Adjust Silence Detection

```bash
python transcribe.py video.mp4 --include-silence --silence-threshold -35 --min-silence 1.0
```

- `--silence-threshold`: Volume threshold in dB (default: -40, lower = more sensitive)
- `--min-silence`: Minimum silence duration in seconds (default: 0.5)

### Keep Extracted Audio

```bash
python transcribe.py video.mp4 --keep-audio
```

This keeps the extracted WAV file alongside the video.

### Full Options

```bash
python transcribe.py --help
```

```
usage: transcribe.py [-h] [-o OUTPUT] [--include-silence]
                     [--silence-threshold SILENCE_THRESHOLD]
                     [--min-silence MIN_SILENCE] [--keep-audio]
                     video

Transcribe video to SRT using NVIDIA Parakeet TDT 0.6B V3

positional arguments:
  video                 Path to the input video file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path for the output SRT file
  --include-silence     Include [SILENCE] markers in the SRT output
  --silence-threshold   Volume threshold (dB) for silence detection (default: -40)
  --min-silence         Minimum silence duration in seconds (default: 0.5)
  --keep-audio          Keep the extracted audio file
```

## 📁 Output Format

The output is a standard SRT file:

```srt
1
00:00:00,000 --> 00:00:03,500
Hello, welcome to this video tutorial.

2
00:00:03,500 --> 00:00:07,200
Today we'll be learning about transcription.

3
00:00:07,200 --> 00:00:08,500
[SILENCE]

4
00:00:08,500 --> 00:00:12,000
Let's get started with the first example.
```

## 🔧 Troubleshooting

### CUDA Not Available

If you see "CUDA is not available" warnings:
1. Ensure you have an NVIDIA GPU
2. Install the correct CUDA toolkit version
3. Reinstall PyTorch with CUDA support

### FFmpeg Not Found

Ensure FFmpeg is in your PATH:
```bash
ffmpeg -version
```

If not, install it and restart your terminal.

### Out of Memory

For very long videos, the model uses local attention automatically. If you still run out of memory:
- Close other applications
- Try processing shorter video segments
- Use a GPU with more VRAM

## 📊 Model Performance

| Dataset | WER |
|---------|-----|
| LibriSpeech (clean) | 1.93% |
| LibriSpeech (other) | 3.59% |
| FLEURS English | 4.85% |
| Average across 25 languages | ~11% |

## 📄 License

This project uses the NVIDIA Parakeet model which is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## 🔗 References

- [Parakeet TDT 0.6B V3 on Hugging Face](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)
- [Technical Report](https://arxiv.org/abs/2509.14128)
