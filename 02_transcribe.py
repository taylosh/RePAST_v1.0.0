"""
02_transcribe.py - Consolidated Transcription Hub
@author: taylosh
Created on Aug 21 2024  
Last edited on Mar 16 2026

Main transcription engine for the overhauled ASR pipeline.
- Utilizes OpenAI Whisper for high-fidelity speech-to-text.
- Integrated active hook for libs/diarize.py subprocess.
- Hardware-aware via consolidated GPU backend.
- Strictly processes "_preprocessed" audio files from Phase 1.
- Outputs original-named TextGrids and transcripts (suffix-stripped).
"""

import os
import sys
import json
import logging
import numpy as np
import subprocess
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

project_root = Path(__file__).parent
libs_path = project_root / "libs"
if str(libs_path) not in sys.path:
    sys.path.insert(0, str(libs_path))

try:
    from wrap_accel import get_c_acceleration_wrapper
    wrapper = get_c_acceleration_wrapper(enable_c_accel=True, enable_gpu_accel=True)
    ACCEL_READY = wrapper.available
except ImportError:
    ACCEL_READY = False

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.prompt import Prompt, IntPrompt
    from rich.live import Live
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    class FallbackConsole:
        def print(self, *args, **kwargs): print(*args)
    console = FallbackConsole()

# ============================================================================
# LANGUAGE MAPPING
# ============================================================================

LANGUAGE_MAPPING = {
    1: "en", 2: "es", 3: "fr", 4: "de", 5: "it", 6: "pt", 7: "nl", 8: "ru",
    9: "ja", 10: "zh", 11: "ar", 12: "hi", 13: "ko", 14: "auto"
}

LANGUAGE_NAMES = {
    1: "English", 2: "Spanish", 3: "French", 4: "German", 5: "Italian",
    6: "Portuguese", 7: "Dutch", 8: "Russian", 9: "Japanese", 10: "Chinese",
    11: "Arabic", 12: "Hindi", 13: "Korean", 14: "Auto-detect"
}

MODEL_MAPPING = {
    1: "tiny", 2: "base", 3: "small", 4: "medium", 5: "large"
}

PRECISION_MAPPING = {
    "1": "low", "2": "medium", "3": "high"
}

# ============================================================================
# INTERACTIVE CONFIGURATION FUNCTIONS
# ============================================================================

def get_speaker_profiles_from_user():
    embeddings_dir = Path("./models/embeddings")
    if not embeddings_dir.exists():
        if RICH_AVAILABLE:
            console.print("[yellow]No speaker profiles directory found.[/yellow]")
        else:
            print("No speaker profiles directory found.")
        return []
    
    profile_files = list(embeddings_dir.glob("*.json"))
    if not profile_files:
        if RICH_AVAILABLE:
            console.print("[yellow]No speaker profiles found in ./models/embeddings/[/yellow]")
        else:
            print("No speaker profiles found in ./models/embeddings/")
        return []
    
    profiles = [f.stem for f in profile_files]
    
    if RICH_AVAILABLE:
        table = Table(title="Available Speaker Profiles")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Speaker Name", style="green")
        
        for i, name in enumerate(profiles, 1):
            table.add_row(str(i), name)
        
        console.print(table)
        choice = Prompt.ask("Select profiles to use (comma-separated numbers, or 'all', or 'none')", default="none")
    else:
        print("\nAvailable Speaker Profiles:")
        for i, name in enumerate(profiles, 1):
            print(f"  {i}. {name}")
        choice = input("\nSelect profiles to use (comma-separated numbers, or 'all', or 'none') [default: none]: ").strip() or "none"
    
    if choice.lower() == 'none':
        return []
    elif choice.lower() == 'all':
        return profiles
    else:
        selected = []
        try:
            for num in choice.split(','):
                idx = int(num.strip()) - 1
                if 0 <= idx < len(profiles):
                    selected.append(profiles[idx])
        except ValueError:
            if RICH_AVAILABLE:
                console.print("[red]Invalid selection, using no profiles[/red]")
            else:
                print("Invalid selection, using no profiles")
            return []
        
        if selected and RICH_AVAILABLE:
            console.print(f"[green]Selected: {', '.join(selected)}[/green]")
        elif selected:
            print(f"Selected: {', '.join(selected)}")
        return selected

def get_segmentation_parameters():
    """Get segmentation parameters from user, with option for adaptive or manual"""
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]=== Silence Segmentation Parameters ===[/bold cyan]")
        console.print("These parameters control how audio is split into segments based on silence.")
    else:
        print("\n=== Silence Segmentation Parameters ===")
        print("These parameters control how audio is split into segments based on silence.")
    
    # Ask about adaptive mode first
    if RICH_AVAILABLE:
        adaptive_choice = Prompt.ask(
            "Use Automatic Adaptive Thresholds?", 
            choices=["y", "n", "yes", "no"], 
            default="y"
        )
    else:
        adaptive_choice = input("Use Automatic Adaptive Thresholds? (Y/n): ").strip().lower() or "y"
    
    use_adaptive = adaptive_choice in ['y', 'yes']
    
    if use_adaptive:
        # Adaptive mode - still need min silence length
        if RICH_AVAILABLE:
            console.print("\n[bold]Adaptive Mode[/bold] - Threshold will be calculated per file")
            console.print("  Algorithm analyzes each file's noise floor and speech levels")
            console.print("  You only need to set the minimum silence length")
        else:
            print("\nAdaptive Mode - Threshold will be calculated per file")
            print("  Algorithm analyzes each file's noise floor and speech levels")
            print("  You only need to set the minimum silence length")
        
        # Get min silence length with adaptive-specific suggestions
        if RICH_AVAILABLE:
            console.print("\n[bold]Minimum Silence Length (ms)[/bold]")
            console.print("  Shorter values = more segments (detects brief pauses)")
            console.print("  Longer values = fewer segments (requires longer pauses)")
            console.print("  Suggestions:")
            console.print("    - Conversational speech: 200-300 ms")
            console.print("    - Monologue/presentation: 400-600 ms")
            console.print("    - Very segmented speech: 100-150 ms")
        else:
            print("\nMinimum Silence Length (ms)")
            print("  Shorter values = more segments (detects brief pauses)")
            print("  Longer values = fewer segments (requires longer pauses)")
            print("  Suggestions:")
            print("    - Conversational speech: 200-300 ms")
            print("    - Monologue/presentation: 400-600 ms")
            print("    - Very segmented speech: 100-150 ms")
        
        while True:
            try:
                if RICH_AVAILABLE:
                    len_input = Prompt.ask("Enter minimum silence length in ms", default="250")
                else:
                    len_input = input("Enter minimum silence length in ms [default: 250]: ").strip() or "250"
                
                min_silence_len = int(len_input)
                if min_silence_len < 10:
                    if RICH_AVAILABLE:
                        console.print("[yellow]Minimum silence length should be at least 10 ms[/yellow]")
                    else:
                        print("Minimum silence length should be at least 10 ms")
                    continue
                break
            except ValueError:
                if RICH_AVAILABLE:
                    console.print("[red]Please enter a valid number[/red]")
                else:
                    print("Please enter a valid number")
        
        return {
            'use_adaptive': True,
            'min_silence_len': min_silence_len,
            'silence_threshold': None  # Will be calculated per file
        }
    
    else:
        # Manual mode - get both threshold and min silence length
        if RICH_AVAILABLE:
            console.print("\n[bold]Manual Mode[/bold] - You set fixed threshold for all files")
        else:
            print("\nManual Mode - You set fixed threshold for all files")
        
        # Get silence threshold
        if RICH_AVAILABLE:
            console.print("\n[bold]Silence Threshold (dB)[/bold]")
            console.print("  Lower values = more sensitive (detects quieter pauses)")
            console.print("  Higher values = less sensitive (ignores background noise)")
            console.print("  Suggestions:")
            console.print("    - Noisy background: -20 to -25 dB")
            console.print("    - Normal conversation: -28 to -32 dB")
            console.print("    - Very clean audio: -35 to -40 dB")
        else:
            print("\nSilence Threshold (dB)")
            print("  Lower values = more sensitive (detects quieter pauses)")
            print("  Higher values = less sensitive (ignores background noise)")
            print("  Suggestions:")
            print("    - Noisy background: -20 to -25 dB")
            print("    - Normal conversation: -28 to -32 dB")
            print("    - Very clean audio: -35 to -40 dB")
        
        while True:
            try:
                if RICH_AVAILABLE:
                    thresh_input = Prompt.ask("Enter silence threshold in dB", default="-30")
                else:
                    thresh_input = input("Enter silence threshold in dB [default: -30]: ").strip() or "-30"
                
                silence_thresh = int(thresh_input)
                if silence_thresh > 0:
                    if RICH_AVAILABLE:
                        console.print("[yellow]Threshold should be negative (e.g., -35)[/yellow]")
                    else:
                        print("Threshold should be negative (e.g., -35)")
                    continue
                break
            except ValueError:
                if RICH_AVAILABLE:
                    console.print("[red]Please enter a valid number[/red]")
                else:
                    print("Please enter a valid number")
        
        # Get min silence length
        if RICH_AVAILABLE:
            console.print("\n[bold]Minimum Silence Length (ms)[/bold]")
            console.print("  Shorter values = more segments (detects brief pauses)")
            console.print("  Longer values = fewer segments (requires longer pauses)")
            console.print("  Suggestions:")
            console.print("    - Conversational speech: 200-300 ms")
            console.print("    - Monologue/presentation: 400-600 ms")
            console.print("    - Very segmented speech: 100-150 ms")
        else:
            print("\nMinimum Silence Length (ms)")
            print("  Shorter values = more segments (detects brief pauses)")
            print("  Longer values = fewer segments (requires longer pauses)")
            print("  Suggestions:")
            print("    - Conversational speech: 200-300 ms")
            print("    - Monologue/presentation: 400-600 ms")
            print("    - Very segmented speech: 100-150 ms")
        
        while True:
            try:
                if RICH_AVAILABLE:
                    len_input = Prompt.ask("Enter minimum silence length in ms", default="200")
                else:
                    len_input = input("Enter minimum silence length in ms [default: 200]: ").strip() or "200"
                
                min_silence_len = int(len_input)
                if min_silence_len < 10:
                    if RICH_AVAILABLE:
                        console.print("[yellow]Minimum silence length should be at least 10 ms[/yellow]")
                    else:
                        print("Minimum silence length should be at least 10 ms")
                    continue
                break
            except ValueError:
                if RICH_AVAILABLE:
                    console.print("[red]Please enter a valid number[/red]")
                else:
                    print("Please enter a valid number")
        
        return {
            'use_adaptive': False,
            'min_silence_len': min_silence_len,
            'silence_threshold': silence_thresh
        }

def get_language_from_user():
    languages = {
        1: {"name": "English", "code": "en"},
        2: {"name": "Spanish", "code": "es"},
        3: {"name": "French", "code": "fr"},
        4: {"name": "German", "code": "de"},
        5: {"name": "Italian", "code": "it"},
        6: {"name": "Portuguese", "code": "pt"},
        7: {"name": "Dutch", "code": "nl"},
        8: {"name": "Russian", "code": "ru"},
        9: {"name": "Japanese", "code": "ja"},
        10: {"name": "Chinese", "code": "zh"},
        11: {"name": "Arabic", "code": "ar"},
        12: {"name": "Hindi", "code": "hi"},
        13: {"name": "Korean", "code": "ko"},
        14: {"name": "Auto-detect", "code": "auto"}
    }
    
    if RICH_AVAILABLE:
        table = Table(title="Available Languages")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Language", style="green")
        table.add_column("Code", style="dim")
        
        for num, lang_info in languages.items():
            table.add_row(str(num), lang_info["name"], lang_info["code"])
        
        console.print(table)
        choice = IntPrompt.ask("Select language", default=14)
    else:
        print("\nAvailable Languages:")
        for num, lang_info in languages.items():
            print(f"  {num}. {lang_info['name']} ({lang_info['code']})")
        try:
            choice = int(input("\nSelect language (default: 14 - Auto-detect): ").strip() or "14")
        except ValueError:
            choice = 14
    
    if choice not in languages:
        if RICH_AVAILABLE:
            console.print(f"[yellow]Invalid choice {choice}, using Auto-detect[/yellow]")
        else:
            print(f"Invalid choice {choice}, using Auto-detect")
        return "auto"
    
    selected = languages[choice]
    if RICH_AVAILABLE:
        console.print(f"[green]Selected: {selected['name']}[/green]")
    else:
        print(f"Selected: {selected['name']}")
    return selected["code"]

def get_model_from_user():
    models = {
        1: {"name": "Tiny", "size": "tiny"},
        2: {"name": "Base", "size": "base"},
        3: {"name": "Small", "size": "small"},
        4: {"name": "Medium", "size": "medium"},
        5: {"name": "Large", "size": "large"}
    }
    
    if RICH_AVAILABLE:
        table = Table(title="Available Whisper Models")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Model", style="green")
        table.add_column("Size", style="dim")
        
        for num, model_info in models.items():
            table.add_row(str(num), model_info["name"], model_info["size"])
        
        console.print(table)
        choice = IntPrompt.ask("Select model", default=2)
    else:
        print("\nAvailable Models:")
        for num, model_info in models.items():
            print(f"  {num}. {model_info['name']} ({model_info['size']})")
        try:
            choice = int(input("\nSelect model (default: 2 - Base): ").strip() or "2")
        except ValueError:
            choice = 2
    
    if choice not in models:
        if RICH_AVAILABLE:
            console.print(f"[yellow]Invalid choice {choice}, using Base[/yellow]")
        else:
            print(f"Invalid choice {choice}, using Base")
        return "base"
    
    selected = models[choice]
    if RICH_AVAILABLE:
        console.print(f"[green]Selected: {selected['name']}[/green]")
    else:
        print(f"Selected: {selected['name']}")
    return selected["size"]

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command line arguments for transcription"""
    parser = argparse.ArgumentParser(description='ASR Transcription Phase')
    
    # Input/output directories
    parser.add_argument('--input-dir', '-i', type=str, default='./preprocessed_audio',
                        help='Input directory containing preprocessed WAV files')
    parser.add_argument('--output-dir', '-o', type=str, default='./initial_transcription',
                        help='Output directory for transcriptions')
    
    # Core transcription options
    parser.add_argument('--language', '-l', type=str, choices=['en', 'es', 'fr', 'de', 'it', 'pt', 
                       'nl', 'ru', 'ja', 'zh', 'ar', 'hi', 'ko', 'auto'], default='auto',
                       help='Language code for transcription')
    parser.add_argument('--model', '-m', type=str, choices=['tiny', 'base', 'small', 'medium', 'large'],
                       default='base', help='Whisper model size')
    
    # Diarization options
    parser.add_argument('--use-diarization', action='store_true',
                       help='Enable speaker diarization')
    parser.add_argument('--no-diarization', action='store_true',
                       help='Disable speaker diarization')
    parser.add_argument('--diarization-precision', type=str, choices=['low', 'medium', 'high'],
                       default='medium', help='Diarization precision level')
    parser.add_argument('--diarization-threshold', type=float, default=0.5,
                       help='Diarization matching threshold (0.0-1.0)')
    parser.add_argument('--speaker-profiles', type=str, nargs='+',
                       help='Speaker profile names to use')
    
    # Segmentation options (when diarization is disabled)
    parser.add_argument('--silence-threshold', type=int, default=-28,
                       help='Silence threshold in dB for segmentation')
    parser.add_argument('--min-silence-len', type=int, default=100,
                       help='Minimum silence length in ms for segmentation')
    
    # Batch mode (suppress interactive prompts)
    parser.add_argument('--batch-mode', action='store_true',
                       help='Run in batch mode (no interactive prompts)')
    
    # Verbose logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()

# ============================================================================
# SPEAKER PROFILE SELECTION (Batch mode version)
# ============================================================================

def get_speaker_profiles(profile_names: Optional[List[str]] = None) -> List[str]:
    """Get speaker profiles either from args or by scanning directory"""
    if profile_names:
        return profile_names
    
    embeddings_dir = Path("./models/embeddings")
    if not embeddings_dir.exists():
        return []
    
    profile_files = list(embeddings_dir.glob("*.json"))
    return [f.stem for f in profile_files]

# ============================================================================
# REST OF THE ORIGINAL CODE (with modifications for batch mode)
# ============================================================================

def run_diarization_stage(input_path: Path, output_base: Path, relative_path: Path, 
                         config: dict, batch_mode: bool = False) -> Optional[Path]:
    """Run diarization as a subprocess"""
    try:
        file_stem = input_path.stem.replace("_preprocessed", "")
        diarization_dir = output_base / "diarization_data" / relative_path / file_stem
        json_path = diarization_dir / "enhanced_diarization.json"
        
        if json_path.exists():
            logger.info(f"Using existing diarization data for: {relative_path / file_stem}")
            return json_path

        diarize_script = libs_path / "diarize.py"
        if not diarize_script.exists():
            logger.error(f"diarize.py not found at {diarize_script}")
            return None
        
        # In batch mode, we don't show the interactive messages
        if not batch_mode and RICH_AVAILABLE:
            print("\n" + "="*60)
            print("SPEAKER DIARIZATION PHASE")
            print("="*60)
            print("Please complete the speaker identification process below.")
            print("Transcription will begin automatically after diarization completes.\n")
        
        cmd = [
            sys.executable, str(diarize_script),
            "--audio", str(input_path),
            "--output", str(output_base / "diarization_data" / relative_path),
            "--precision", config.get('precision', 'medium'),
            "--threshold", str(config.get('threshold', 0.5))
        ]
        
        if config.get('selected_profiles'):
            cmd.extend(["--profiles", ",".join(config['selected_profiles'])])
        
        # In batch mode, we might want to suppress input prompts
        stdin = subprocess.DEVNULL if batch_mode else sys.stdin
        stdout = subprocess.DEVNULL if batch_mode and not config.get('verbose') else sys.stdout
        stderr = subprocess.DEVNULL if batch_mode and not config.get('verbose') else sys.stderr
        
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            check=False
        )
        
        if not batch_mode and RICH_AVAILABLE:
            print("\n" + "="*60)
            print("DIARIZATION COMPLETE - PREPARING TRANSCRIPTION")
            print("="*60 + "\n")
        
        if result.returncode != 0:
            logger.error(f"diarize.py exited with error code: {result.returncode}")
            return None
            
        if json_path.exists():
            return json_path
        else:
            possible_paths = list(diarization_dir.glob("*.json"))
            if possible_paths:
                logger.info(f"Found diarization output: {possible_paths[0]}")
                return possible_paths[0]
            else:
                logger.error(f"Diarization completed but no output found for {relative_path / file_stem}")
                return None
            
    except Exception as e:
        logger.error(f"Diarization stage failed for {input_path.name}: {e}")
        return None

class WhisperEngine:
    """Whisper transcription engine"""
    def __init__(self, model_size: str, language: str):
        self.model_size = model_size
        self.language = language
        self.model = None
        self.device = "cuda" if (ACCEL_READY and wrapper.gpu_backend.gpu_is_available()) else "cpu"

    def initialize(self):
        import whisper
        logger.info(f"Loading Whisper {self.model_size} on {self.device}...")
        
        if RICH_AVAILABLE and not batch_mode:
            console.print(f"[dim]Loading model {self.model_size}...[/dim]")
        else:
            print(f"Loading model {self.model_size}...")
        
        self.model = whisper.load_model(self.model_size, device=self.device)

    def transcribe(self, audio_source: Any) -> str:
        if self.model is None:
            self.initialize()
        
        if isinstance(audio_source, np.ndarray):
            if audio_source.dtype != np.float32:
                audio_source = audio_source.astype(np.float32)
        
        result = self.model.transcribe(
            audio_source, 
            language=self.language if self.language != "auto" else None,
            fp16=(self.device == "cuda")
        )
        return result.get("text", "").strip()

def detect_speech_segments_adaptive(audio_path: Path, 
                                   min_silence_len: int = 250,
                                   silence_percentile: int = 20,
                                   adapt_duration: bool = True,
                                   use_adaptive: bool = True,
                                   manual_threshold: Optional[int] = None) -> List[Tuple[float, float]]:
    """Detect speech segments, either adaptively or with manual threshold"""
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
        import numpy as np
        
        audio = AudioSegment.from_wav(str(audio_path))
        samples = np.array(audio.get_array_of_samples())
        
        # If not using adaptive mode and manual threshold is provided, use it directly
        if not use_adaptive and manual_threshold is not None:
            silence_thresh = manual_threshold
            logger.info(f"Using manual silence threshold: {silence_thresh} dB for {audio_path.name}")
            logger.info(f"  Using min silence length: {min_silence_len} ms")
            
            # Skip adaptive analysis and just use the manual threshold
            nonsilent_segments = detect_nonsilent(
                audio, 
                silence_thresh=silence_thresh,
                min_silence_len=min_silence_len
            )
            
            if not nonsilent_segments:
                logger.warning(f"No speech detected in {audio_path.name} with manual threshold {silence_thresh} dB")
                return [(0.0, audio.duration_seconds)]
            
            segments_in_seconds = [(start/1000, end/1000) for start, end in nonsilent_segments]
            logger.info(f"Manual detection found {len(segments_in_seconds)} segments")
            return segments_in_seconds
        
        # Otherwise use adaptive algorithm (original code)
        window_size = int(audio.frame_rate * 0.025)
        stride = int(audio.frame_rate * 0.010)
        
        rms_values = []
        timestamps = []
        for i in range(0, len(samples) - window_size, stride):
            window = samples[i:i+window_size]
            rms = np.sqrt(np.mean(window**2))
            if rms > 0:
                rms_db = 20 * np.log10(rms / 32768)
            else:
                rms_db = -100
            rms_values.append(rms_db)
            timestamps.append(i / audio.frame_rate)
        
        rms_values = np.array(rms_values)
        noise_floor = np.percentile(rms_values, silence_percentile)
        
        speech_samples = rms_values[rms_values > noise_floor + 5]
        if len(speech_samples) > 0:
            typical_speech = np.median(speech_samples)
        else:
            typical_speech = noise_floor + 15
        
        silence_thresh = noise_floor + (typical_speech - noise_floor) * 0.33
        
        if adapt_duration:
            temp_segments = detect_nonsilent(
                audio, 
                silence_thresh=silence_thresh,
                min_silence_len=100
            )
            
            if len(temp_segments) > 1:
                gaps = []
                for i in range(len(temp_segments) - 1):
                    gap = (temp_segments[i+1][0] - temp_segments[i][1]) / 1000
                    if gap > 0.05:
                        gaps.append(gap)
                
                if gaps:
                    gaps = np.array(gaps)
                    typical_pause = np.median(gaps) * 1000
                    short_pause = np.percentile(gaps, 10) * 1000
                    adapted_min_len = min(min_silence_len, int(typical_pause * 0.8))
                    adapted_min_len = max(150, min(adapted_min_len, 800))
                    min_silence_len = adapted_min_len
        
        logger.info(f"Adaptive threshold analysis for {audio_path.name}:")
        logger.info(f"  Noise floor: {noise_floor:.1f} dBFS")
        logger.info(f"  Typical speech: {typical_speech:.1f} dBFS")
        logger.info(f"  Using silence threshold: {silence_thresh:.1f} dBFS")
        logger.info(f"  Using min silence length: {min_silence_len} ms")
        
        nonsilent_segments = detect_nonsilent(
            audio, 
            silence_thresh=silence_thresh,
            min_silence_len=min_silence_len
        )
        
        if not nonsilent_segments:
            logger.warning(f"No speech detected in {audio_path.name}, using full duration")
            return [(0.0, audio.duration_seconds)]
        
        segments_in_seconds = [(start/1000, end/1000) for start, end in nonsilent_segments]
        return segments_in_seconds
        
    except Exception as e:
        logger.error(f"Error in adaptive detection for {audio_path.name}: {str(e)}")
        return [(0.0, get_audio_duration(audio_path))]

def get_audio_duration(audio_path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(audio_path))
    return info.duration

def save_outputs(audio_stem: str, transcriptions: Dict[str, List[Tuple]], 
                output_dir: Path, duration: float, relative_path: Path):
    """Save transcription outputs to TextGrid and text files"""
    from praatio import textgrid
    
    tg = textgrid.Textgrid()
    tg.minTimestamp = 0.0
    tg.maxTimestamp = duration
    
    combined_text = []
    
    for speaker_id, intervals in transcriptions.items():
        tier_name = f"{speaker_id}_phrases" if speaker_id != "default" else "phrases"
        
        if RICH_AVAILABLE and not batch_mode:
            console.print(f"Creating tier '{tier_name}' with {len(intervals)} intervals")
        
        for start, end, text in intervals:
            logger.debug(f"  Interval: {start:.3f}-{end:.3f}: '{text[:50]}...'")
        
        tier = textgrid.IntervalTier(tier_name, intervals, 0.0, duration)
        tg.addTier(tier)
        
        for start, end, text in intervals:
            combined_text.append(f"[{speaker_id}] {start:.2f}-{end:.2f}s: {text}")
    
    textgrid_dir = output_dir / "textgrids" / relative_path
    transcript_dir = output_dir / "transcripts" / relative_path
    
    tg_out = textgrid_dir / f"{audio_stem}.TextGrid"
    tg_out.parent.mkdir(parents=True, exist_ok=True)
    
    tg.save(str(tg_out), format="long_textgrid", includeBlankSpaces=True)
    
    txt_out = transcript_dir / f"{audio_stem}.txt"
    txt_out.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_text))
    
    if RICH_AVAILABLE and not batch_mode:
        console.print(f"Saved TextGrid and transcript for {audio_stem}")

def transcribe_only(input_file: Path, output_dir: Path, relative_path: Path,
                   use_diarization: bool, diarization_config: dict,
                   lang_code: str, model_size: str,
                   silence_thresh: int = -28, min_silence_len: int = 100,
                   use_adaptive: bool = True,
                   batch_mode: bool = False):
    """Transcribe a single audio file"""
    try:
        import soundfile as sf
        
        original_stem = input_file.stem.replace("_preprocessed", "")
        
        json_path = None
        if use_diarization:
            file_stem = input_file.stem.replace("_preprocessed", "")
            json_path = output_dir / "diarization_data" / relative_path / file_stem / "enhanced_diarization.json"
            
            if not json_path.exists():
                logger.warning(f"No diarization data found for {relative_path / original_stem}, falling back to silence-based segmentation")
                use_diarization = False
        
        engine = WhisperEngine(model_size, lang_code)
        
        audio_info = sf.info(input_file)
        transcriptions_by_speaker = {}

        if use_diarization and json_path and json_path.exists():
            with open(json_path, 'r') as jf:
                data = json.load(jf)
            
            audio_data, sr = sf.read(input_file)
            
            total_segments = len(data["segments"])
            for idx, seg in enumerate(data["segments"], 1):
                logger.info(f"Processing segment {idx}/{total_segments} for {relative_path / original_stem}")
                
                speaker_id = seg.get("assigned_name", seg.get("orig_label", f"SPEAKER_{idx}"))
                
                start_s, end_s = seg["start"], seg["end"]
                # Extract the exact segment
                start_sample = int(start_s * sr)
                end_sample = int(end_s * sr)
                buffer = audio_data[start_sample:end_sample]
                
                if buffer.dtype != np.float32:
                    buffer = buffer.astype(np.float32)
                
                # Transcribe just this segment
                text = engine.transcribe(buffer)
                
                if speaker_id not in transcriptions_by_speaker:
                    transcriptions_by_speaker[speaker_id] = []
                # Use the original timing, not whatever Whisper returns
                transcriptions_by_speaker[speaker_id].append((start_s, end_s, text))
        
        else:  # NON-DIARIZATION PATH - ADAPTIVE OR MANUAL SEGMENTATION
            logger.info(f"Using {'adaptive' if use_adaptive else 'manual'} silence-based segmentation for {relative_path / original_stem}")
            
            # Get speech segments from the adaptive detector
            speech_segments = detect_speech_segments_adaptive(
                input_file, 
                min_silence_len=min_silence_len,
                silence_percentile=20,
                adapt_duration=True,
                use_adaptive=use_adaptive,
                manual_threshold=silence_thresh if not use_adaptive else None
            )
            
            logger.info(f"Detection returned {len(speech_segments)} speech segments for {relative_path / original_stem}")
            
            # Load the full audio data once
            audio_data, sr = sf.read(input_file)
            
            clean_segments = []
            
            # Process each speech segment individually
            for i, (start, end) in enumerate(speech_segments):
                # Calculate sample indices for this segment
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                
                # Extract just this segment's audio
                audio_chunk = audio_data[start_sample:end_sample]
                
                if len(audio_chunk) == 0:
                    logger.warning(f"Empty audio chunk for segment {i+1}, skipping")
                    continue
                
                if audio_chunk.dtype != np.float32:
                    audio_chunk = audio_chunk.astype(np.float32)
                
                logger.info(f"Transcribing segment {i+1}/{len(speech_segments)}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
                
                # Transcribe ONLY this segment
                text = engine.transcribe(audio_chunk)
                
                # Store with the ORIGINAL timing from the segment detector
                clean_segments.append({
                    "start": start,
                    "end": end,
                    "text": text,
                    "id": i
                })
                
                # Small progress indicator
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(speech_segments)} segments")
            
            # Add all segments to the transcription dictionary
            if clean_segments:
                # Initialize default speaker if needed
                if "default" not in transcriptions_by_speaker:
                    transcriptions_by_speaker["default"] = []
                
                # Add each segment with its original timing
                for seg in clean_segments:
                    transcriptions_by_speaker["default"].append((
                        seg["start"], 
                        seg["end"], 
                        seg["text"]
                    ))
                
                logger.info(f"Created {len(clean_segments)} transcribed segments for {relative_path / original_stem}")
                
                # Log first few segments for debugging
                for i, seg in enumerate(clean_segments[:3]):
                    logger.debug(f"  Segment {i+1}: {seg['start']:.2f}-{seg['end']:.2f}s: '{seg['text'][:50]}...'")
            else:
                logger.warning(f"No segments returned for {relative_path / original_stem}, using full file")
                full_text = engine.transcribe(str(input_file))
                transcriptions_by_speaker["default"] = [(0.0, audio_info.duration, full_text)]

        # Save the outputs with the correct timing
        save_outputs(original_stem, transcriptions_by_speaker, output_dir, audio_info.duration, relative_path)
        return True
    
    except Exception as e:
        logger.error(f"Failed to process {input_file.name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Global batch mode flag for modules that need it
batch_mode = False

def main():
    global batch_mode
    
    # FIRST: Check if we're running interactively (no arguments)
    if len(sys.argv) == 1:
        # No arguments provided - run in interactive mode
        if RICH_AVAILABLE:
            console.print("=== [bold cyan]02: TRANSCRIPTION & DIARIZATION HUB - INTERACTIVE MODE[/bold cyan] ===")
        else:
            print("=== 02: TRANSCRIPTION & DIARIZATION HUB - INTERACTIVE MODE ===")
        
        # Interactive configuration
        if RICH_AVAILABLE:
            diarize_choice = Prompt.ask("Use Speaker Diarization?", choices=["y", "n", "yes", "no"], default="n")
        else:
            diarize_choice = input("Use Speaker Diarization? (y/N): ").strip().lower() or "n"
        
        use_diarization = diarize_choice in ['y', 'yes']
        
        diarization_config = {}
        if use_diarization:
            if RICH_AVAILABLE:
                speaker_id_choice = Prompt.ask(
                    "Use Speaker Identification? (assign names to speakers)", 
                    choices=["y", "n", "yes", "no"], 
                    default="n"
                )
            else:
                speaker_id_choice = input("Use Speaker Identification? (assign names to speakers) (y/N): ").strip().lower() or "n"
            
            use_speaker_id = speaker_id_choice in ['y', 'yes']
            diarization_config['use_speaker_id'] = use_speaker_id
            
            if use_speaker_id:
                if RICH_AVAILABLE:
                    console.print("[yellow]Note: Speaker identification will prompt you to name new speakers[/yellow]")
                else:
                    print("Note: Speaker identification will prompt you to name new speakers")
                
                selected_profiles = get_speaker_profiles_from_user()
                diarization_config['selected_profiles'] = selected_profiles
            else:
                if RICH_AVAILABLE:
                    console.print("[dim]Using diarization only (generic speaker labels)[/dim]")
                else:
                    print("Using diarization only (generic speaker labels)")
                diarization_config['selected_profiles'] = []
            
            if RICH_AVAILABLE:
                console.print("\n[bold cyan]=== Diarization Configuration ===[/bold cyan]")
            
            precision_map = {"1": "low", "2": "medium", "3": "high"}
            if RICH_AVAILABLE:
                if RICH_AVAILABLE:
                    console.print("\n[bold]Precision Level[/bold]")
                    console.print("  1. Low - Faster, lower precision")
                    console.print("  2. Medium - Balanced speed and precision (Default)")
                    console.print("  3. High - Slowest, highest timing accuracy")
                precision_choice = Prompt.ask("Select precision level", choices=["1", "2", "3"], default="2")
            else:
                print("\nPrecision Level:")
                print("  1. Low - Faster, lower precision")
                print("  2. Medium - Balanced speed and precision (Default)")
                print("  3. High - Slowest, highest timing accuracy")
                precision_choice = input("Select precision level (1-3) [default: 2]: ").strip() or "2"
            
            diarization_config['precision'] = precision_map.get(precision_choice, "medium")
            
            if use_speaker_id:
                if RICH_AVAILABLE:
                    threshold_input = Prompt.ask(
                        "Verification threshold (0.0-1.0, higher = stricter)", 
                        default="0.5"
                    )
                else:
                    threshold_input = input("Verification threshold (0.0-1.0) [default: 0.5]: ").strip() or "0.5"
                try:
                    threshold = float(threshold_input)
                    threshold = max(0.0, min(1.0, threshold))
                except ValueError:
                    threshold = 0.5
                diarization_config['threshold'] = threshold
                
                if RICH_AVAILABLE:
                    use_ecapa_choice = Prompt.ask(
                        "Use ECAPA-TDNN vocal fingerprinting?", 
                        choices=["y", "n", "yes", "no"], 
                        default="y"
                    )
                else:
                    use_ecapa_choice = input("Use ECAPA-TDNN vocal fingerprinting? (Y/n): ").strip().lower() or "y"
                diarization_config['use_ecapa'] = use_ecapa_choice in ['y', 'yes']
            else:
                diarization_config['threshold'] = 0.5
                diarization_config['use_ecapa'] = False
        
        # Segmentation parameters (if not using diarization)
        segmentation_config = {}
        if not use_diarization:
            segmentation_config = get_segmentation_parameters()
            use_adaptive = segmentation_config['use_adaptive']
            min_silence_len = segmentation_config['min_silence_len']
            silence_thresh = segmentation_config['silence_threshold']
        else:
            # Defaults when diarization is on (these won't be used anyway)
            use_adaptive = True
            min_silence_len = 100
            silence_thresh = None
        
        # Language selection
        lang_code = get_language_from_user()
        
        # Model selection
        model_size = get_model_from_user()
        
        # Set default directories
        in_dir = Path("./preprocessed_audio")
        out_dir = Path("./initial_transcription")
        
        # Set batch mode to False for interactive mode
        batch_mode = False
        
    else:
        # Arguments provided - use command-line parsing
        args = parse_arguments()
        batch_mode = args.batch_mode
        
        # Display header (only in non-batch mode)
        if not batch_mode and RICH_AVAILABLE:
            console.print("=== [bold cyan]02: TRANSCRIPTION & DIARIZATION HUB[/bold cyan] ===")
        elif not batch_mode:
            print("=== 02: TRANSCRIPTION & DIARIZATION HUB ===")
        
        # Determine if using diarization
        use_diarization = args.use_diarization
        if args.no_diarization:
            use_diarization = False
        
        # Set up diarization config if needed
        diarization_config = {}
        if use_diarization:
            selected_profiles = get_speaker_profiles(args.speaker_profiles)
            
            diarization_config = {
                'precision': args.diarization_precision,
                'threshold': args.diarization_threshold,
                'selected_profiles': selected_profiles,
                'verbose': args.verbose
            }
            
            if not batch_mode and RICH_AVAILABLE:
                console.print(f"\n[green]Diarization enabled with {len(selected_profiles)} profiles[/green]")
            
            # Set defaults for segmentation (won't be used)
            use_adaptive = True
            min_silence_len = 100
            silence_thresh = None
        else:
            # When diarization is off, use args for segmentation
            use_adaptive = True  # Default to adaptive in batch mode
            min_silence_len = args.min_silence_len
            silence_thresh = args.silence_threshold
        
        # Set input/output directories
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir)
        
        # Get language and model from args
        lang_code = args.language
        model_size = args.model
    
    # Find files to process
    wav_files = list(in_dir.rglob("*_preprocessed.wav"))
    if not wav_files:
        if RICH_AVAILABLE:
            console.print(f"[red]No _preprocessed.wav files found in {in_dir}.[/red]")
        else:
            print(f"No _preprocessed.wav files found in {in_dir}.")
        return

    if not batch_mode and RICH_AVAILABLE:
        console.print(f"\n[green]Found {len(wav_files)} files to process[/green]")
    elif not batch_mode:
        print(f"\nFound {len(wav_files)} files to process")

    stats = {'success': 0, 'error': 0, 'diarization_success': 0}
    
    # PASS 1: Run diarization on ALL files first (if enabled)
    if use_diarization:
        if not batch_mode and RICH_AVAILABLE:
            console.print("\n[bold cyan]=== PASS 1: SPEAKER DIARIZATION FOR ALL FILES ===[/bold cyan]")
        
        files_to_diarize = []
        files_with_existing = []
        
        for f in wav_files:
            relative_path = f.relative_to(in_dir).parent
            file_stem = f.stem.replace("_preprocessed", "")
            json_path = out_dir / "diarization_data" / relative_path / file_stem / "enhanced_diarization.json"
            
            if json_path.exists():
                files_with_existing.append((f, relative_path))
                logger.info(f"Using existing diarization data for: {relative_path / file_stem}")
            else:
                files_to_diarize.append((f, relative_path))
        
        if files_to_diarize:
            if not batch_mode and RICH_AVAILABLE:
                console.print(f"[yellow]Need to run diarization for {len(files_to_diarize)} files[/yellow]")
            
            # Use progress bar in non-batch mode with rich
            if not batch_mode and RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    refresh_per_second=10
                ) as progress:
                    task = progress.add_task("Running diarization...", total=len(files_to_diarize))
                    
                    for f, relative_path in files_to_diarize:
                        progress.update(task, description=f"Diarizing {relative_path / f.name}...")
                        
                        json_path = run_diarization_stage(f, out_dir, relative_path, diarization_config, batch_mode=batch_mode)
                        
                        if json_path and json_path.exists():
                            stats['diarization_success'] += 1
                            logger.info(f"Diarization complete for {relative_path / f.stem.replace('_preprocessed', '')}")
                        else:
                            stats['error'] += 1
                            logger.error(f"Diarization failed for {relative_path / f.name}")
                        
                        progress.advance(task)
            else:
                for f, relative_path in files_to_diarize:
                    print(f"Diarizing {relative_path / f.name}...")
                    json_path = run_diarization_stage(f, out_dir, relative_path, diarization_config, batch_mode=batch_mode)
                    
                    if json_path and json_path.exists():
                        stats['diarization_success'] += 1
                    else:
                        stats['error'] += 1
            
            if not batch_mode and RICH_AVAILABLE:
                console.print("[green]All diarization complete - models unloaded[/green]")
        else:
            stats['diarization_success'] = len(files_with_existing)
            if not batch_mode and RICH_AVAILABLE:
                console.print("[green]All files already have diarization data - skipping diarization phase[/green]")
        
        # Clear separation between PASS 1 and PASS 2
        if not batch_mode and RICH_AVAILABLE:
            console.print("\n" + "="*60)
            console.print("[bold green]PASS 1 COMPLETE: Diarization finished[/bold green]")
            console.print("[dim]Models unloaded - memory cleared[/dim]")
            console.print("[dim]Preparing for transcription phase...[/dim]")
            console.print("="*60 + "\n")
        
        # Small pause to ensure all subprocesses are cleaned up
        time.sleep(2)
    
    # PASS 2: Run transcription on ALL files
    if not batch_mode and RICH_AVAILABLE:
        console.print("\n[bold cyan]=== PASS 2: TRANSCRIPTION FOR ALL FILES ===[/bold cyan]")
    
    logger.info("Loading Whisper model for transcription phase...")
    
    if not batch_mode and RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            refresh_per_second=10
        ) as progress:
            task = progress.add_task("Transcribing files...", total=len(wav_files))
            
            for f in wav_files:
                relative_path = f.relative_to(in_dir).parent
                progress.update(task, description=f"Transcribing {relative_path / f.name}...")
                
                if transcribe_only(f, out_dir, relative_path, use_diarization, diarization_config, 
                                  lang_code, model_size, 
                                  silence_thresh if not use_diarization else -28, 
                                  min_silence_len if not use_diarization else 100,
                                  use_adaptive=use_adaptive if not use_diarization else True,
                                  batch_mode=batch_mode):
                    stats['success'] += 1
                else:
                    stats['error'] += 1
                
                progress.advance(task)
    else:
        for f in wav_files:
            relative_path = f.relative_to(in_dir).parent
            print(f"Transcribing {relative_path / f.name}...")
            
            if transcribe_only(f, out_dir, relative_path, use_diarization, diarization_config,
                              lang_code, model_size,
                              silence_thresh if not use_diarization else -28,
                              min_silence_len if not use_diarization else 100,
                              use_adaptive=use_adaptive if not use_diarization else True,
                              batch_mode=batch_mode):
                stats['success'] += 1
            else:
                stats['error'] += 1

    # Display summary
    if RICH_AVAILABLE:
        summary = Table(title="Transcription Phase Summary", box=None)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="bold")
        summary.add_row("Files Found", str(len(wav_files)))
        summary.add_row("Diarization Successes", str(stats['diarization_success']))
        summary.add_row("Transcription Successes", str(stats['success']))
        summary.add_row("Errors", str(stats['error']))
        summary.add_row("Diarization Active", "Yes" if use_diarization else "No")
        if use_diarization:
            summary.add_row("Profiles Loaded", str(len(diarization_config.get('selected_profiles', []))))
            summary.add_row("Matching Threshold", str(diarization_config.get('threshold', 0.5)))
            summary.add_row("Precision", diarization_config.get('precision', 'medium'))
        if not use_diarization:
            summary.add_row("Segmentation Mode", "Adaptive" if use_adaptive else "Manual")
            if not use_adaptive:
                summary.add_row("Silence Threshold", f"{silence_thresh} dB")
            summary.add_row("Min Silence Length", f"{min_silence_len} ms")
        summary.add_row("Language", lang_code)
        summary.add_row("Model", model_size)
        summary.add_row("GPU Used", "Yes" if (ACCEL_READY and wrapper.gpu_backend.gpu_is_available()) else "No")
        console.print(summary)
    else:
        print("\n=== Transcription Phase Summary ===")
        print(f"Files Found: {len(wav_files)}")
        print(f"Diarization Successes: {stats['diarization_success']}")
        print(f"Transcription Successes: {stats['success']}")
        print(f"Errors: {stats['error']}")
        print(f"Diarization Active: {'Yes' if use_diarization else 'No'}")
        if use_diarization:
            print(f"Profiles Loaded: {len(diarization_config.get('selected_profiles', []))}")
            print(f"Matching Threshold: {diarization_config.get('threshold', 0.5)}")
            print(f"Precision: {diarization_config.get('precision', 'medium')}")
        if not use_diarization:
            print(f"Segmentation Mode: {'Adaptive' if use_adaptive else 'Manual'}")
            if not use_adaptive:
                print(f"Silence Threshold: {silence_thresh} dB")
            print(f"Min Silence Length: {min_silence_len} ms")
        print(f"Language: {lang_code}")
        print(f"Model: {model_size}")
        print(f"GPU Used: {'Yes' if (ACCEL_READY and wrapper.gpu_backend.gpu_is_available()) else 'No'}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()
