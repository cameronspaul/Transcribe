"""
Video to SRT Transcription using NVIDIA Parakeet TDT 0.6B V3

This script transcribes video files to SRT format with accurate timestamps,
including detection of speech segments and silence periods.
"""

import os
import sys
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import soundfile as sf
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available in the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio(video_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """
    Extract audio from video file and convert to mono WAV at specified sample rate.
    
    Args:
        video_path: Path to the input video file
        output_path: Path for the output audio file
        sample_rate: Target sample rate (default: 16000 Hz for Parakeet)
    
    Returns:
        Path to the extracted audio file
    """
    console.print(f"[cyan]Extracting audio from video...[/cyan]")
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", str(sample_rate),  # Sample rate
        "-ac", "1",  # Mono
        "-y",  # Overwrite output
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        console.print(f"[red]FFmpeg error: {result.stderr}[/red]")
        raise RuntimeError(f"Failed to extract audio: {result.stderr}")
    
    console.print(f"[green]✓ Audio extracted to {output_path}[/green]")
    return output_path


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def detect_silence_segments(
    audio_path: str,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Detect silence segments in audio file.
    
    Args:
        audio_path: Path to the audio file
        threshold_db: Volume threshold for silence detection (in dB)
        min_silence_duration: Minimum duration to consider as silence (in seconds)
    
    Returns:
        List of (start_time, end_time) tuples for silence segments
    """
    console.print("[cyan]Detecting silence segments...[/cyan]")
    
    # Load audio
    audio, sample_rate = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Calculate RMS energy in frames
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)    # 10ms hop
    
    # Pad audio
    audio = np.pad(audio, (frame_length // 2, frame_length // 2), mode='reflect')
    
    # Calculate frame energies
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    energies = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        energies[i] = 20 * np.log10(rms + 1e-10)
    
    # Find silence frames
    is_silence = energies < threshold_db
    
    # Group consecutive silence frames
    silence_segments = []
    in_silence = False
    silence_start = 0
    
    for i, silent in enumerate(is_silence):
        time = i * hop_length / sample_rate
        
        if silent and not in_silence:
            in_silence = True
            silence_start = time
        elif not silent and in_silence:
            in_silence = False
            duration = time - silence_start
            if duration >= min_silence_duration:
                silence_segments.append((silence_start, time))
    
    # Handle case where audio ends in silence
    if in_silence:
        end_time = num_frames * hop_length / sample_rate
        duration = end_time - silence_start
        if duration >= min_silence_duration:
            silence_segments.append((silence_start, end_time))
    
    console.print(f"[green]✓ Found {len(silence_segments)} silence segments[/green]")
    return silence_segments


def load_asr_model():
    """
    Load the NVIDIA Parakeet TDT 0.6B V3 model.
    
    Returns:
        Loaded ASR model
    """
    console.print("[cyan]Loading Parakeet TDT 0.6B V3 model...[/cyan]")
    console.print("[dim](This may take a moment on first run as the model downloads)[/dim]")
    
    import nemo.collections.asr as nemo_asr
    
    # Load the model
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3"
    )
    
    # For long audio, use local attention
    asr_model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[256, 256]
    )
    
    console.print("[green]✓ Model loaded successfully[/green]")
    return asr_model


def transcribe_audio_chunk(asr_model, audio_path: str, chunk_offset: float = 0.0, max_words: int = 0, word_gap: float = 0.5, time_offset: float = 0.0) -> List[dict]:
    """
    Transcribe a single audio chunk using the ASR model with timestamps.
    
    Args:
        asr_model: Loaded ASR model
        audio_path: Path to the audio chunk file
        chunk_offset: Time offset to add to timestamps
        max_words: Maximum words per segment (0 for default sentence-based)
        word_gap: Maximum silence (seconds) between words in a segment
        time_offset: Global time offset to apply to all timestamps
    
    Returns:
        List of segment dictionaries with text and timestamps
    """
    import torch
    
    # Clear CUDA cache before transcription
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Transcribe with timestamps enabled
    output = asr_model.transcribe([audio_path], timestamps=True)
    
    segments = []
    
    if output and len(output) > 0:
        result = output[0]
        
        # Get timestamps if available
        has_ts = hasattr(result, 'timestamp') and result.timestamp
        word_timestamps = result.timestamp.get('word', []) if has_ts else []
        segment_timestamps = result.timestamp.get('segment', []) if has_ts else []
        
    # Priority 1: Group by word count if requested and word timestamps are available
    if max_words > 0 and word_timestamps:
        current_words = []
        segment_start = None
        prev_word_end = None
        
        for word_info in word_timestamps:
            word = word_info.get('word', word_info.get('char', '')).strip()
            if not word:
                continue
            
            w_start = word_info['start'] + chunk_offset + time_offset
            w_end = word_info['end'] + chunk_offset + time_offset
            
            # Check if we should start a new segment due to a large gap
            should_break_gap = prev_word_end is not None and (w_start - prev_word_end) > word_gap
            
            if segment_start is None:
                segment_start = w_start
            
            # If we should break, and we have words, emit current segment
            # Breaks on: large gap, max words reached, or previous word had punctuation
            if (should_break_gap or len(current_words) >= max_words or (current_words and current_words[-1].endswith(('.', '!', '?', ':', ',')))) and current_words:
                segments.append({
                    'start': segment_start,
                    'end': prev_word_end,
                    'text': ' '.join(current_words)
                })
                current_words = []
                segment_start = w_start
            
            current_words.append(word)
            prev_word_end = w_end
        
        # Add remaining words
        if current_words:
            segments.append({
                'start': segment_start,
                'end': prev_word_end,
                'text': ' '.join(current_words)
            })
        
    # Priority 2: Use native segments if max_words is 0 or word timestamps missing
    elif segment_timestamps:
        for stamp in segment_timestamps:
            text = stamp['segment'].strip()
            start = stamp['start'] + chunk_offset + time_offset
            end = stamp['end'] + chunk_offset + time_offset
            
            words = text.split()
            if not words:
                continue

            # Split segment if it has punctuation or exceeds max_words
            duration = end - start
            current_chunk = []
            chunk_start_idx = 0
            
            for i, word in enumerate(words):
                # Check if we should break BEFORE adding the word
                should_break_punct = current_chunk and current_chunk[-1].endswith(('.', '!', '?', ':', ','))
                should_break_count = max_words > 0 and len(current_chunk) >= max_words
                
                if (should_break_punct or should_break_count) and current_chunk:
                    chunk_start = start + (chunk_start_idx / len(words)) * duration
                    chunk_end = start + (i / len(words)) * duration
                    segments.append({
                        'start': chunk_start,
                        'end': chunk_end,
                        'text': ' '.join(current_chunk)
                    })
                    current_chunk = []
                    chunk_start_idx = i
                
                current_chunk.append(word)
            
            # Add remaining words for this segment
            if current_chunk:
                chunk_start = start + (chunk_start_idx / len(words)) * duration
                segments.append({
                    'start': chunk_start,
                    'end': end,
                    'text': ' '.join(current_chunk)
                })
        
    # Priority 3: Use word timestamps with sentence logic (fallback)
    elif word_timestamps:
        current_segment = {
            'start': word_timestamps[0]['start'] + chunk_offset + time_offset,
            'text': '',
            'words': []
        }
        
        for word_info in word_timestamps:
            word = word_info.get('word', word_info.get('char', ''))
            current_segment['words'].append(word)
            current_segment['end'] = word_info['end'] + chunk_offset + time_offset
            
            # Check for punctuation (including commas)
            if word.rstrip().endswith(('.', '!', '?', ':', ',')):
                current_segment['text'] = ' '.join(current_segment['words']).strip()
                del current_segment['words']
                segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    'start': word_info['end'] + chunk_offset + time_offset,
                    'text': '',
                    'words': []
                }
        
        # Add remaining segment
        if current_segment.get('words'):
            current_segment['text'] = ' '.join(current_segment['words']).strip()
            del current_segment['words']
            if current_segment['text']:
                segments.append(current_segment)
    
    # Priority 4: Last fallback: single segment with full text
    else:
        text = ""
        if hasattr(result, 'text'):
            text = result.text
        elif isinstance(result, str):
            text = result
        else:
            text = str(result)
            
        if text:
            segments.append({
                'start': chunk_offset + time_offset,
                'end': chunk_offset + time_offset,  # Error case, but keeps it running
                'text': text
            })
    
    # Clear CUDA cache after transcription
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return segments


def transcribe_audio(asr_model, audio_path: str, chunk_duration: float = 60.0, max_words: int = 0, word_gap: float = 0.5, offset: float = 0.0) -> List[dict]:
    """
    Transcribe audio file using the ASR model with timestamps.
    Processes audio in chunks to avoid CUDA out of memory errors.
    
    Args:
        asr_model: Loaded ASR model
        audio_path: Path to the audio file
        chunk_duration: Duration of each chunk in seconds (default: 60s)
        max_words: Maximum words per segment (0 for default sentence-based)
        word_gap: Maximum silence (seconds) between words in a segment
        offset: Global time offset to apply to all timestamps
    
    Returns:
        List of segment dictionaries with text and timestamps
    """
    import torch
    
    console.print("[cyan]Transcribing audio...[/cyan]")
    
    # Get audio info
    audio, sample_rate = sf.read(audio_path)
    total_duration = len(audio) / sample_rate
    
    # If audio is short enough, transcribe in one go
    if total_duration <= chunk_duration:
        console.print(f"[dim]Processing audio in single pass ({total_duration:.1f}s)[/dim]")
        return transcribe_audio_chunk(asr_model, audio_path, 0.0, max_words=max_words, word_gap=word_gap, time_offset=offset)
    
    # Process in chunks for longer audio
    console.print(f"[dim]Processing {total_duration:.1f}s audio in {chunk_duration:.0f}s chunks to save memory[/dim]")
    
    all_segments = []
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(2.0 * sample_rate)  # 2 second overlap for smooth transitions
    
    num_chunks = int(np.ceil(len(audio) / (chunk_samples - overlap_samples)))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(num_chunks):
            start_sample = i * (chunk_samples - overlap_samples)
            end_sample = min(start_sample + chunk_samples, len(audio))
            chunk_offset = start_sample / sample_rate
            
            console.print(f"[dim]  Chunk {i+1}/{num_chunks}: {chunk_offset:.1f}s - {end_sample/sample_rate:.1f}s[/dim]")
            
            # Extract chunk
            chunk_audio = audio[start_sample:end_sample]
            chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
            sf.write(chunk_path, chunk_audio, sample_rate)
            
            # Transcribe chunk
            chunk_segments = transcribe_audio_chunk(
                asr_model, chunk_path, chunk_offset, 
                max_words=max_words, word_gap=word_gap, time_offset=offset
            )
            
            # For overlapping chunks, merge carefully to avoid duplicates
            if i > 0 and all_segments:
                # The overlap region starts at chunk_offset
                # The previous chunk already covered up to this point (and slightly beyond)
                # To avoid duplicates, we only take segments from the new chunk that start
                # after the midpoint of the overlap or after the last segment ends.
                last_end_time = all_segments[-1]['end']
                # Midpoint of overlap is chunk_offset + overlap_duration/2
                overlap_midpoint = chunk_offset + (overlap_samples / sample_rate) / 2
                
                # Filter segments that were likely already covered by the previous chunk
                # We use the midpoint of the overlap as a cleaner cut-off than a fixed 0.5s
                chunk_segments = [s for s in chunk_segments if s['start'] >= max(last_end_time, overlap_midpoint)]
            
            all_segments.extend(chunk_segments)
            
            # Clean up chunk file
            os.remove(chunk_path)
            
            # Clear CUDA cache between chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    console.print(f"[green]✓ Transcribed {len(all_segments)} segments[/green]")
    return all_segments



def merge_with_silence(
    speech_segments: List[dict],
    silence_segments: List[Tuple[float, float]],
    include_silence: bool = True,
    audio_duration: float = None
) -> List[dict]:
    """
    Merge speech segments with silence indicators.
    
    Args:
        speech_segments: List of transcribed speech segments
        silence_segments: List of detected silence periods
        include_silence: Whether to include silence markers in output
        audio_duration: Total audio duration for final segment
    
    Returns:
        Merged list of segments including silence markers
    """
    if not include_silence or not silence_segments:
        return speech_segments
    
    all_segments = []
    
    # Convert silence to segment format
    silence_entries = [
        {'start': s, 'end': e, 'text': '[SILENCE]', 'is_silence': True}
        for s, e in silence_segments
    ]
    
    # Combine and sort by start time
    combined = speech_segments + silence_entries
    combined.sort(key=lambda x: x['start'])
    
    return combined


def write_srt(
    segments: List[dict],
    output_path: str,
    include_silence_markers: bool = False,
    remove_punctuation: bool = False,
    uppercase: bool = False
) -> str:
    """
    Write segments to SRT file format.
    
    Args:
        segments: List of segment dictionaries
        output_path: Path for the output SRT file
        include_silence_markers: Whether to include [SILENCE] markers
        remove_punctuation: Whether to remove common punctuation marks
        uppercase: Whether to convert text to UPPERCASE
        Path to the written SRT file
    """
    console.print(f"[cyan]Writing SRT file...[/cyan]")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        subtitle_index = 1
        
        for segment in segments:
            # Skip empty segments
            if not segment.get('text', '').strip():
                continue
            
            # Skip silence markers if not wanted
            if not include_silence_markers and segment.get('is_silence', False):
                continue
            
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            text = segment['text'].strip()
            
            if remove_punctuation:
                import re
                # Remove common punctuation: . , ! ? : ;
                text = re.sub(r'[,.!?:;]', '', text)
            
            if uppercase:
                text = text.upper()
            
            f.write(f"{subtitle_index}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n")
            f.write("\n")
            
            subtitle_index += 1
    
    console.print(f"[green]✓ SRT file written: {output_path}[/green]")
    console.print(f"[green]  Total subtitles: {subtitle_index - 1}[/green]")
    
    return output_path


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    audio, sample_rate = sf.read(audio_path)
    return len(audio) / sample_rate


def transcribe_video(
    video_path: str,
    output_path: Optional[str] = None,
    include_silence: bool = False,
    silence_threshold_db: float = -40.0,
    min_silence_duration: float = 0.5,
    keep_audio: bool = False,
    chunk_duration: float = 60.0,
    max_words: int = 0,
    word_gap: float = 0.5,
    offset: float = 0.0,
    remove_punctuation: bool = False,
    uppercase: bool = False
) -> str:
    """
    Main function to transcribe a video file to SRT format.
    
    Args:
        video_path: Path to the input video file
        output_path: Path for the output SRT file (default: video_name.srt)
        include_silence: Whether to include [SILENCE] markers
        silence_threshold_db: Volume threshold for silence detection
        min_silence_duration: Minimum duration to consider as silence
        keep_audio: Whether to keep the extracted audio file
        chunk_duration: Duration of each audio chunk for memory-efficient processing
        max_words: Maximum words per segment (0 for default sentence-based)
        word_gap: Maximum silence (seconds) between words in a segment
        offset: Global time offset to apply to all timestamps
        remove_punctuation: Whether to remove commas and full stops
        uppercase: Whether to convert text to UPPERCASE
    
    Returns:
        Path to the generated SRT file
    """
    video_path = Path(video_path).resolve()
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Check ffmpeg
    if not check_ffmpeg():
        raise RuntimeError(
            "FFmpeg is not installed or not in PATH. "
            "Please install FFmpeg: https://ffmpeg.org/download.html"
        )
    
    # Set default output path
    if output_path is None:
        output_path = video_path.with_suffix('.srt')
    else:
        output_path = Path(output_path).resolve()
    
    console.print(f"\n[bold blue]═══════════════════════════════════════════════════════════[/bold blue]")
    console.print(f"[bold blue]  Video to SRT Transcription - Parakeet TDT 0.6B V3[/bold blue]")
    console.print(f"[bold blue]═══════════════════════════════════════════════════════════[/bold blue]\n")
    console.print(f"[yellow]Input:[/yellow]  {video_path}")
    console.print(f"[yellow]Output:[/yellow] {output_path}\n")
    
    # Create temp directory for audio
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "audio.wav"
        
        # If keeping audio, save alongside video
        if keep_audio:
            audio_path = video_path.with_suffix('.wav')
        
        # Step 1: Extract audio
        extract_audio(str(video_path), str(audio_path))
        
        # Get audio duration
        audio_duration = get_audio_duration(str(audio_path))
        console.print(f"[dim]Audio duration: {audio_duration:.2f} seconds[/dim]\n")
        
        # Step 2: Detect silence (optional)
        silence_segments = []
        if include_silence:
            silence_segments = detect_silence_segments(
                str(audio_path),
                threshold_db=silence_threshold_db,
                min_silence_duration=min_silence_duration
            )
        
        # Step 3: Load model and transcribe
        asr_model = load_asr_model()
        speech_segments = transcribe_audio(
            asr_model, str(audio_path), 
            chunk_duration=chunk_duration, 
            max_words=max_words,
            word_gap=word_gap,
            offset=offset
        )
        
        # Step 4: Merge with silence markers
        all_segments = merge_with_silence(
            speech_segments,
            silence_segments,
            include_silence=include_silence,
            audio_duration=audio_duration
        )
        
        # Step 5: Write SRT file
        write_srt(
            all_segments,
            str(output_path),
            include_silence_markers=include_silence,
            remove_punctuation=remove_punctuation,
            uppercase=uppercase
        )
    
    console.print(f"\n[bold green]✓ Transcription complete![/bold green]\n")
    
    return str(output_path)


def main():
    """Command-line interface for video transcription."""
    parser = argparse.ArgumentParser(
        description="Transcribe video to SRT using NVIDIA Parakeet TDT 0.6B V3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py video.mp4
  python transcribe.py video.mp4 -o subtitles.srt
  python transcribe.py video.mp4 --include-silence
  python transcribe.py video.mp4 --silence-threshold -35 --min-silence 1.0
        """
    )
    
    parser.add_argument(
        "video",
        help="Path to the input video file"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path for the output SRT file (default: same as video with .srt extension)"
    )
    
    parser.add_argument(
        "--include-silence",
        action="store_true",
        help="Include [SILENCE] markers in the SRT output"
    )
    
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=-40.0,
        help="Volume threshold (dB) for silence detection (default: -40)"
    )
    
    parser.add_argument(
        "--min-silence",
        type=float,
        default=0.5,
        help="Minimum silence duration in seconds (default: 0.5)"
    )
    
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the extracted audio file"
    )
    
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=60.0,
        help="Duration of each audio chunk in seconds for memory-efficient processing (default: 60). Lower values use less GPU memory but may affect accuracy at chunk boundaries."
    )
    
    parser.add_argument(
        "--max-words",
        type=int,
        default=5,
        help="Maximum words per SRT segment (default: 5, use 0 for original sentence-based segments)"
    )
    
    parser.add_argument(
        "--word-gap",
        type=float,
        default=0.5,
        help="Maximum silence in seconds between words before starting a new segment (default: 0.5)"
    )
    
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Global time offset in seconds to apply to all captions (e.g., -0.2 to advance captions by 200ms)"
    )
    
    parser.add_argument(
        "--remove-punctuation",
        action="store_true",
        help="Remove commas and full stops from the SRT output"
    )
    
    parser.add_argument(
        "--uppercase",
        action="store_true",
        help="Convert all text to UPPERCASE"
    )
    
    args = parser.parse_args()
    
    try:
        transcribe_video(
            video_path=args.video,
            output_path=args.output,
            include_silence=args.include_silence,
            silence_threshold_db=args.silence_threshold,
            min_silence_duration=args.min_silence,
            keep_audio=args.keep_audio,
            chunk_duration=args.chunk_duration,
            max_words=args.max_words,
            word_gap=args.word_gap,
            offset=args.offset,
            remove_punctuation=args.remove_punctuation,
            uppercase=args.uppercase
        )
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
