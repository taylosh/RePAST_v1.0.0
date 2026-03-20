"""
03_annotate.py - Alignment & Annotation Hub
@author: taylosh
Created on Nov 15 2025
Last edited on Mar 16 2026

Main alignment engine for the overhauled ASR pipeline.
- Automatically generates MFA corpus from Phase 2 TextGrids.
- Uses C-accelerated audio_segment_engine for high-speed chunking (with Python fallback).
- Performs Segment-to-Phoneme alignment via Montreal Forced Aligner.
- Integrated subscripts: libs/syllabify.py and libs/tag.py.
- Preservation of original filenames with "_aligned" suffix.
"""

import os
import sys
import json
import logging
import subprocess
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Path resolution for consolidated libs/ and bin/
project_root = Path(__file__).parent
libs_path = project_root / "libs"
if str(libs_path) not in sys.path:
    sys.path.insert(0, str(libs_path))

# Import the Consolidated Accelerator Wrapper with fallback
try:
    from wrap_accel import get_c_acceleration_wrapper
    wrapper = get_c_acceleration_wrapper(enable_c_accel=True, enable_gpu_accel=True)
    ACCEL_READY = wrapper.available
    if ACCEL_READY:
        print("C acceleration wrapper loaded successfully")
    else:
        print("Warning: C acceleration wrapper loaded but not available - using Python fallbacks")
        wrapper = None
except ImportError as e:
    ACCEL_READY = False
    wrapper = None
    print(f"C acceleration wrapper not available: {e} - using Python implementations")

# Rich progress and console integration 
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    # Fallback console if Rich not available
    class FallbackConsole:
        def print(self, *args, **kwargs): print(*args)
    console = FallbackConsole()

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command line arguments for alignment and annotation."""
    parser = argparse.ArgumentParser(description='Alignment and Annotation Phase')
    
    # Core paths - now optional (for standalone mode)
    parser.add_argument('--textgrid-dir', '-t', type=str,
                        help='Directory containing transcribed TextGrid files from Phase 2')
    parser.add_argument('--audio-dir', '-a', type=str,
                        help='Directory containing preprocessed audio files')
    parser.add_argument('--output-dir', '-o', type=str, default='./aligned_textgrids',
                        help='Output directory for aligned TextGrids')
    parser.add_argument('--language', '-l', type=str, default='english',
                        choices=['english', 'spanish', 'french', 'german', 'italian', 
                                'portuguese', 'dutch', 'russian', 'mandarin', 'japanese', 
                                'korean', 'arabic'],
                        help='Language for MFA alignment')
    
    # Subscript configuration
    parser.add_argument('--enable-syllabification', action='store_true',
                        help='Enable syllabification after alignment')
    parser.add_argument('--enable-pos-tagging', action='store_true',
                        help='Enable POS tagging after alignment')
    
    # MFA model configuration
    parser.add_argument('--mfa-auto-download', action='store_true', default=True,
                        help='Auto-download missing MFA models')
    parser.add_argument('--mfa-no-auto-download', dest='mfa_auto_download', 
                        action='store_false', help='Disable auto-download of MFA models')
    
    # Syllabification arguments
    parser.add_argument('--syllabification-dict-dir', type=str, default='./models/syllable_dicts',
                        help='Directory for syllable dictionaries')
    parser.add_argument('--syllabification-use-dictionary', action='store_true', default=True,
                        help='Use dictionary lookup for syllabification')
    parser.add_argument('--syllabification-no-dictionary', dest='syllabification_use_dictionary', 
                        action='store_false', help='Disable dictionary lookup for syllabification')
    parser.add_argument('--syllabification-use-acoustic', action='store_true', default=True,
                        help='Use acoustic analysis for syllabification')
    parser.add_argument('--syllabification-no-acoustic', dest='syllabification_use_acoustic', 
                        action='store_false', help='Disable acoustic analysis for syllabification')
    parser.add_argument('--syllabification-threshold', type=float, default=0.3,
                        help='Peak detection threshold for acoustic syllabification (0.0-1.0)')
    parser.add_argument('--syllabification-frame-size', type=int, default=256,
                        help='Frame size for acoustic analysis (samples)')
    parser.add_argument('--syllabification-hop-size', type=int, default=128,
                        help='Hop size for acoustic analysis (samples)')
    parser.add_argument('--syllabification-fallback', type=str, 
                        choices=['dictionary_first', 'acoustic_first'], default='dictionary_first',
                        help='Fallback order for syllabification methods')
    
    # POS tagging arguments
    parser.add_argument('--pos-model-size', type=str, choices=['sm', 'md', 'lg'], default='sm',
                        help='Size of spaCy model for POS tagging')
    parser.add_argument('--pos-tag-type', type=str, choices=['universal', 'fine'], default='universal',
                        help='Type of POS tags to use')
    parser.add_argument('--pos-auto-download', action='store_true', default=True,
                        help='Auto-download missing spaCy models')
    parser.add_argument('--pos-no-auto-download', dest='pos_auto_download', 
                        action='store_false', help='Disable auto-download of spaCy models')
    
    # General
    parser.add_argument('--non-interactive', action='store_true',
                        help='Run in non-interactive mode (use provided arguments)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

# ============================================================================
# MFA MODEL MANAGEMENT
# ============================================================================

def check_mfa_model(language: str, auto_download: bool = True) -> bool:
    """Check if MFA model exists, download if missing."""
    import subprocess
    import sys
    
    model_name = f"{language}_mfa"
    
    # Try to check if model exists by attempting to get its info
    try:
        # Try to run a command that would fail if model doesn't exist
        result = subprocess.run(
            ["mfa", "model", "list", "acoustic"],
            capture_output=True,
            text=True
        )
        
        # Check if model appears in output
        if model_name in result.stdout:
            logger.info(f"MFA model '{model_name}' found")
            return True
        else:
            logger.info(f"MFA model '{model_name}' not found")
            if auto_download:
                console.print(f"[yellow]MFA model '{model_name}' not found. Downloading...[/yellow]" if RICH_AVAILABLE else f"MFA model '{model_name}' not found. Downloading...")
                try:
                    # Download both dictionary and acoustic models
                    logger.info(f"Downloading MFA dictionary: {model_name}")
                    subprocess.run(
                        ["mfa", "model", "download", "dictionary", model_name],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    logger.info(f"Downloading MFA acoustic model: {model_name}")
                    subprocess.run(
                        ["mfa", "model", "download", "acoustic", model_name],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    console.print(f"[green]MFA model '{model_name}' downloaded successfully[/green]" if RICH_AVAILABLE else f"MFA model '{model_name}' downloaded successfully")
                    return True
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to download MFA model: {e.stderr}")
                    console.print(f"[red]Failed to download MFA model '{model_name}'[/red]" if RICH_AVAILABLE else f"Failed to download MFA model '{model_name}'")
                    return False
            else:
                return False
                
    except Exception as e:
        logger.error(f"Error checking MFA model: {e}")
        return False

# Language selection mapping
LANGUAGE_OPTIONS = {
    1: {"name": "English", "code": "english"},
    2: {"name": "Spanish", "code": "spanish"},
    3: {"name": "French", "code": "french"},
    4: {"name": "German", "code": "german"},
    5: {"name": "Italian", "code": "italian"},
    6: {"name": "Portuguese", "code": "portuguese"},
    7: {"name": "Dutch", "code": "dutch"},
    8: {"name": "Russian", "code": "russian"},
    9: {"name": "Mandarin", "code": "mandarin"},
    10: {"name": "Japanese", "code": "japanese"},
    11: {"name": "Korean", "code": "korean"},
    12: {"name": "Arabic", "code": "arabic"}
}

def select_language() -> str:
    """Display numbered language menu and get user selection."""
    
    if RICH_AVAILABLE:
        # Rich menu display
        table = Table(title="Available Languages for MFA Alignment")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Language", style="green")
        table.add_column("Code", style="dim")
        
        for num, lang_info in LANGUAGE_OPTIONS.items():
            table.add_row(str(num), lang_info["name"], lang_info["code"])
        
        console.print(table)
        choice = IntPrompt.ask("Select language", default=1)
    else:
        # Fallback console menu
        print("\nAvailable Languages for MFA Alignment:")
        print("-" * 40)
        for num, lang_info in LANGUAGE_OPTIONS.items():
            print(f"  {num}. {lang_info['name']} ({lang_info['code']})")
        print("-" * 40)
        
        try:
            choice = int(input("\nSelect language (default: 1 - English): ").strip() or "1")
        except ValueError:
            choice = 1
    
    # Validate choice
    if choice not in LANGUAGE_OPTIONS:
        console.print(f"[yellow]Invalid choice {choice}, using English[/yellow]" if RICH_AVAILABLE else f"Invalid choice {choice}, using English")
        return "english"
    
    selected = LANGUAGE_OPTIONS[choice]
    console.print(f"[green]Selected: {selected['name']}[/green]" if RICH_AVAILABLE else f"Selected: {selected['name']}")
    return selected["code"]

# ============================================================================
# ENHANCED CONFIGURATION PROMPTS
# ============================================================================

def configure_syllabification() -> Dict[str, Any]:
    """Present all syllabification configuration options to user."""
    config = {
        'enabled': False,
        'use_dictionary': True,
        'use_acoustic': True,
        'dictionary_path': "./models/syllable_dicts",
        'acoustic_threshold': 0.3,
        'frame_size': 256,
        'hop_size': 128,
        'fallback_order': 'dictionary_first'  # or 'acoustic_first'
    }
    
    if not RICH_AVAILABLE:
        # Fallback text prompts
        print("\n--- SYLLABIFICATION CONFIGURATION ---")
        enable = input("Enable syllabification? (y/N): ").lower() == 'y'
        if not enable:
            return {'enabled': False}
        
        config['enabled'] = True
        config['use_dictionary'] = input("Use dictionary lookup? (Y/n): ").lower() != 'n'
        config['use_acoustic'] = input("Use acoustic analysis? (Y/n): ").lower() != 'n'
        
        if config['use_dictionary']:
            dict_path = input(f"Dictionary path [{config['dictionary_path']}]: ").strip()
            if dict_path:
                config['dictionary_path'] = dict_path
        
        if config['use_acoustic']:
            try:
                threshold = input(f"Peak detection threshold (0.0-1.0) [{config['acoustic_threshold']}]: ").strip()
                if threshold:
                    config['acoustic_threshold'] = float(threshold)
                
                frame_size = input(f"Frame size [{config['frame_size']}]: ").strip()
                if frame_size:
                    config['frame_size'] = int(frame_size)
                
                hop_size = input(f"Hop size [{config['hop_size']}]: ").strip()
                if hop_size:
                    config['hop_size'] = int(hop_size)
            except ValueError:
                print("Invalid input, using defaults")
        
        if config['use_dictionary'] and config['use_acoustic']:
            print("\nFallback order:")
            print("  1. dictionary_first (Try dictionary first, fall back to acoustic)")
            print("  2. acoustic_first (Try acoustic first, fall back to dictionary)")
            choice = input("Select fallback order (1/2) [1]: ").strip() or "1"
            config['fallback_order'] = 'dictionary_first' if choice == "1" else 'acoustic_first'
        
        return config
    
    # Rich interactive configuration
    console.print(Panel("[bold cyan]Syllabification Configuration[/bold cyan]"))
    
    config['enabled'] = Confirm.ask("Enable syllabification?", default=False)
    if not config['enabled']:
        return {'enabled': False}
    
    config['use_dictionary'] = Confirm.ask("Use dictionary lookup?", default=True)
    config['use_acoustic'] = Confirm.ask("Use acoustic analysis?", default=True)
    
    if config['use_dictionary']:
        config['dictionary_path'] = Prompt.ask(
            "Dictionary path", 
            default=config['dictionary_path']
        )
    
    if config['use_acoustic']:
        console.print("\n[dim]Acoustic analysis parameters:[/dim]")
        config['acoustic_threshold'] = FloatPrompt.ask(
            "Peak detection threshold (0.0-1.0)", 
            default=config['acoustic_threshold']
        )
        config['frame_size'] = IntPrompt.ask(
            "Frame size (samples)", 
            default=config['frame_size']
        )
        config['hop_size'] = IntPrompt.ask(
            "Hop size (samples)", 
            default=config['hop_size']
        )
    
    if config['use_dictionary'] and config['use_acoustic']:
        console.print("\n[dim]Fallback behavior:[/dim]")
        console.print("  1. dictionary_first (Try dictionary first, fall back to acoustic)")
        console.print("  2. acoustic_first (Try acoustic first, fall back to dictionary)")
        
        fallback_choice = Prompt.ask(
            "Select fallback order",
            choices=["1", "2"],
            default="1"
        )
        
        config['fallback_order'] = 'dictionary_first' if fallback_choice == "1" else 'acoustic_first'
    
    # Display summary
    console.print("\n[green]Syllabification Configuration Summary:[/green]")
    console.print(f"  Dictionary: {config['use_dictionary']}")
    if config['use_dictionary']:
        console.print(f"    Path: {config['dictionary_path']}")
    console.print(f"  Acoustic: {config['use_acoustic']}")
    if config['use_acoustic']:
        console.print(f"    Threshold: {config['acoustic_threshold']}")
        console.print(f"    Frame size: {config['frame_size']}")
        console.print(f"    Hop size: {config['hop_size']}")
    if config['use_dictionary'] and config['use_acoustic']:
        console.print(f"  Fallback order: {config['fallback_order']}")
    
    return config

def configure_pos_tagging() -> Dict[str, Any]:
    """Present all POS tagging configuration options to user."""
    config = {
        'enabled': False,
        'model_size': 'sm',  # sm, md, lg
        'tag_type': 'universal',  # universal or fine
        'auto_download': True,
        'model_map': {
            'english': 'en_core_web',
            'spanish': 'es_core_news',
            'french': 'fr_core_news',
            'german': 'de_core_news',
            'italian': 'it_core_news',
            'portuguese': 'pt_core_news',
            'dutch': 'nl_core_news',
            'russian': 'ru_core_news',
            'mandarin': 'zh_core_web',
            'japanese': 'ja_core_news',
            'korean': 'ko_core_news',
            'arabic': 'ar_core_news'
        }
    }
    
    if not RICH_AVAILABLE:
        # Fallback text prompts
        print("\n--- POS TAGGING CONFIGURATION ---")
        enable = input("Enable POS tagging? (y/N): ").lower() == 'y'
        if not enable:
            return {'enabled': False}
        
        config['enabled'] = True
        
        print("\nModel size options:")
        print("  sm - Small (fastest, least accurate)")
        print("  md - Medium (balanced)")
        print("  lg - Large (slowest, most accurate)")
        size = input("Select size [sm]: ").strip() or "sm"
        if size in ['sm', 'md', 'lg']:
            config['model_size'] = size
        
        print("\nTag type options:")
        print("  universal - Universal POS tags (e.g., NOUN, VERB)")
        print("  fine - Fine-grained tags (language-specific)")
        tag = input("Select type [universal]: ").strip() or "universal"
        if tag in ['universal', 'fine']:
            config['tag_type'] = tag
        
        auto = input("Auto-download missing models? (Y/n): ").strip().lower()
        config['auto_download'] = auto != 'n'
        
        return config
    
    # Rich interactive configuration
    console.print(Panel("[bold cyan]POS Tagging Configuration[/bold cyan]"))
    
    config['enabled'] = Confirm.ask("Enable POS tagging?", default=False)
    if not config['enabled']:
        return {'enabled': False}
    
    # Model size selection
    console.print("\n[dim]Model size options:[/dim]")
    size_table = Table(box=None)
    size_table.add_column("Size", style="cyan")
    size_table.add_column("Description", style="white")
    size_table.add_row("sm", "Small (fastest, least accurate)")
    size_table.add_row("md", "Medium (balanced)")
    size_table.add_row("lg", "Large (slowest, most accurate)")
    console.print(size_table)
    
    config['model_size'] = Prompt.ask(
        "Select model size",
        choices=["sm", "md", "lg"],
        default="sm"
    )
    
    # Tag type selection
    console.print("\n[dim]Tag type options:[/dim]")
    tag_table = Table(box=None)
    tag_table.add_column("Type", style="cyan")
    tag_table.add_column("Description", style="white")
    tag_table.add_row("universal", "Universal POS tags (e.g., NOUN, VERB, ADJ)")
    tag_table.add_row("fine", "Fine-grained tags (language-specific, e.g., NNP, VBD)")
    console.print(tag_table)
    
    config['tag_type'] = Prompt.ask(
        "Select tag type",
        choices=["universal", "fine"],
        default="universal"
    )
    
    config['auto_download'] = Confirm.ask(
        "Auto-download missing models?",
        default=True
    )
    
    # Display full model names that will be used
    console.print("\n[green]POS Tagging Configuration Summary:[/green]")
    console.print(f"  Model size: {config['model_size']}")
    console.print(f"  Tag type: {config['tag_type']}")
    console.print(f"  Auto-download: {config['auto_download']}")
    console.print("\n[dim]Models that will be used:[/dim]")
    for lang, base in config['model_map'].items():
        console.print(f"  {lang}: {base}_{config['model_size']}")
    
    return config

def select_audio_directory() -> Path:
    """Prompt user for audio directory path."""
    default_path = Path("./preprocessed_audio")
    
    if RICH_AVAILABLE:
        console.print(Panel("[bold cyan]Audio Directory Selection[/bold cyan]"))
        console.print(f"Default: [dim]{default_path}[/dim]")
        
        path_str = Prompt.ask(
            "Enter path to preprocessed audio files",
            default=str(default_path)
        )
    else:
        print(f"\nAudio Directory Selection (default: {default_path})")
        path_str = input("Enter path: ").strip() or str(default_path)
    
    audio_path = Path(path_str)
    
    # Validate directory exists
    if not audio_path.exists():
        console.print(f"[yellow]Warning: Directory {audio_path} does not exist[/yellow]" 
                     if RICH_AVAILABLE else f"Warning: Directory {audio_path} does not exist")
        
        if RICH_AVAILABLE:
            create = Confirm.ask("Create it?", default=True)
        else:
            create = input("Create it? (y/N): ").lower() == 'y'
        
        if create:
            audio_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]Created directory: {audio_path}[/green]" 
                         if RICH_AVAILABLE else f"Created directory: {audio_path}")
        else:
            console.print("[red]Audio directory not available. Exiting.[/red]" 
                         if RICH_AVAILABLE else "Audio directory not available. Exiting.")
            sys.exit(1)
    
    return audio_path


def select_textgrid_directory() -> Path:
    """Prompt user for TextGrid directory path."""
    default_path = Path("./initial_transcription/textgrids")
    
    if RICH_AVAILABLE:
        console.print(Panel("[bold cyan]TextGrid Directory Selection[/bold cyan]"))
        console.print(f"Default: [dim]{default_path}[/dim]")
        
        path_str = Prompt.ask(
            "Enter path to transcribed TextGrid files",
            default=str(default_path)
        )
    else:
        print(f"\nTextGrid Directory Selection (default: {default_path})")
        path_str = input("Enter path: ").strip() or str(default_path)
    
    tg_path = Path(path_str)
    
    if not tg_path.exists():
        console.print(f"[red]Error: TextGrid directory not found: {tg_path}[/red]" 
                     if RICH_AVAILABLE else f"Error: TextGrid directory not found: {tg_path}")
        sys.exit(1)
    
    return tg_path

# ============================================================================
# FAMILY-BASED PROJECT MATCHING
# ============================================================================

class AudioProjectFamily:
    """Represents a complete audio project family"""
    def __init__(self, audio_path: Path, project_key: str):
        self.audio_path = audio_path
        self.project_key = project_key
        self.speaker_segments: Dict[str, List[Dict]] = {}
        self.textgrid_path: Optional[Path] = None

def find_project_families(audio_dir: Path, textgrid_dir: Path) -> Dict[str, AudioProjectFamily]:
    """Find all audio-textgrid project families."""
    families = {}
    
    # Find all TextGrids
    tg_files = list(textgrid_dir.rglob("*.TextGrid"))
    
    for tg_file in tg_files:
        # Get the relative path from the textgrid base directory
        relative_path = tg_file.relative_to(textgrid_dir)
        
        # The project key is the path without the file extension
        # This preserves the folder structure
        project_key = str(relative_path.parent / relative_path.stem)
        
        # Construct the expected audio path maintaining the same relative structure
        # Get the stem without _transcribed if present
        base_stem = relative_path.stem
        if base_stem.endswith('_transcribed'):
            base_stem = base_stem[:-12]  # Remove '_transcribed'
        
        # Try with _preprocessed suffix first, preserving directory structure
        audio_relative_path = relative_path.parent / f"{base_stem}_preprocessed.wav"
        audio_file = audio_dir / audio_relative_path
        
        if audio_file.exists():
            if project_key not in families:
                families[project_key] = AudioProjectFamily(audio_file, project_key)
            families[project_key].textgrid_path = tg_file
        else:
            # Try without _preprocessed suffix
            audio_relative_path = relative_path.parent / f"{base_stem}.wav"
            audio_file = audio_dir / audio_relative_path
            if audio_file.exists():
                if project_key not in families:
                    families[project_key] = AudioProjectFamily(audio_file, project_key)
                families[project_key].textgrid_path = tg_file
            else:
                logger.warning(f"No matching audio found for TextGrid: {tg_file.name} (tried {base_stem}_preprocessed.wav and {base_stem}.wav in {audio_dir / relative_path.parent})")
    
    return families

# ============================================================================
# FALLBACK FUNCTIONS (used when C acceleration is not available)
# ============================================================================

def fallback_batch_extract(audio_data: np.ndarray, intervals: List[Tuple[int, int]]) -> List[np.ndarray]:
    """Python fallback for batch_extract when C acceleration is not available"""
    chunks = []
    for start_sample, end_sample in intervals:
        chunk = audio_data[start_sample:end_sample]
        chunks.append(chunk)
    return chunks

# ============================================================================
# SECTION 1: CORPUS GENERATION WITH FAMILY SUPPORT
# ============================================================================

def create_mfa_corpus(family: AudioProjectFamily, corpus_base: Path) -> List[Dict]:
    """Extract audio chunks and create .lab files for a project family."""
    try:
        from praatio import textgrid
        import soundfile as sf
    except ImportError as e:
        logger.error(f"Missing required module: {e}")
        return []
    
    if not family.textgrid_path:
        logger.error(f"No TextGrid found for family {family.project_key}")
        return []
    
    if not family.audio_path.exists():
        logger.error(f"Audio file not found: {family.audio_path}")
        return []
    
    try:
        tg = textgrid.openTextgrid(str(family.textgrid_path), includeEmptyIntervals=False)
        audio_data, sr = sf.read(family.audio_path)
    except Exception as e:
        logger.error(f"Failed to load files for {family.project_key}: {e}")
        return []
    
    chunk_metadata = []
    family.speaker_segments = {}
    
    for tier_name in tg.tierNames:
        if not tier_name.endswith("phrases"): 
            continue
        
        speaker_id = tier_name.replace("_phrases", "")
        tier = tg.getTier(tier_name)
        
        intervals = []
        texts = []
        for entry in tier.entries:
            if not entry.label.strip(): 
                continue
            # Convert to sample indices
            start_sample = int(entry.start * sr)
            end_sample = int(entry.end * sr)
            intervals.append((start_sample, end_sample))
            texts.append(entry.label.strip())
            
        if not intervals: 
            continue

        # Store segments for this speaker
        speaker_segments = []
        for i, (start_sample, end_sample) in enumerate(intervals):
            speaker_segments.append({
                'start_sample': start_sample,
                'end_sample': end_sample,
                'text': texts[i] if i < len(texts) else "",
                'start_time': start_sample / sr,
                'end_time': end_sample / sr
            })
        
        family.speaker_segments[speaker_id] = speaker_segments

        # Accelerated Chunking via Signal Core with Python fallback
        try:
            audio_float32 = audio_data.astype(np.float32)
            
            # Use C acceleration if available, otherwise use Python fallback
            if ACCEL_READY and wrapper and hasattr(wrapper, 'audio_segment') and hasattr(wrapper.audio_segment, 'batch_extract'):
                try:
                    logger.info(f"Using C-accelerated chunking for {family.project_key} - {speaker_id}")
                    chunks = wrapper.audio_segment.batch_extract(audio_float32, intervals)
                except Exception as e:
                    logger.warning(f"C-accelerated chunking failed: {e} - falling back to Python")
                    chunks = fallback_batch_extract(audio_float32, intervals)
            else:
                logger.info(f"Using Python fallback chunking for {family.project_key} - {speaker_id}")
                chunks = fallback_batch_extract(audio_float32, intervals)
            
            # Check if chunks is None or empty
            if chunks is None:
                logger.error(f"batch_extract returned None for {family.project_key} - {speaker_id}")
                continue
                
            if len(chunks) == 0:
                logger.warning(f"batch_extract returned empty list for {family.project_key} - {speaker_id}")
                continue
                
        except Exception as e:
            logger.error(f"Error in chunk extraction for {family.project_key} - {speaker_id}: {e}")
            continue

        # Process chunks
        for i, (chunk, text) in enumerate(zip(chunks, texts)):
            if chunk is None:
                logger.warning(f"Chunk {i} is None for {family.project_key} - {speaker_id}, skipping")
                continue
                
            chunk_id = f"{family.audio_path.stem}_{speaker_id}_{i:03d}"
            wav_path = corpus_base / f"{chunk_id}.wav"
            lab_path = corpus_base / f"{chunk_id}.lab"
            
            try:
                sf.write(str(wav_path), chunk, sr, subtype='PCM_16')
                lab_path.write_text(text, encoding='utf-8')
                
                chunk_metadata.append({
                    "chunk_id": chunk_id,
                    "speaker": speaker_id,
                    "offset": intervals[i][0] / sr,
                    "duration": len(chunk) / sr,
                    "text": text,
                    "wav_path": wav_path,
                    "lab_path": lab_path
                })
            except Exception as e:
                logger.error(f"Failed to write chunk {chunk_id}: {e}")
                continue
            
    return chunk_metadata

# ============================================================================
# SECTION 2: MFA EXECUTION & RECONSTRUCTION WITH FAMILY SUPPORT
# ============================================================================

def run_mfa_alignment(corpus_dir: Path, language: str, mfa_out: Path, auto_download: bool = True) -> bool:
    """Execute MFA align command."""
    # Check if model exists first
    if not check_mfa_model(language, auto_download):
        logger.error(f"MFA model for {language} not available")
        console.print(f"[red]MFA model for {language} not available. Run 'mfa model download {language}_mfa' to install.[/red]" if RICH_AVAILABLE else f"MFA model for {language} not available")
        return False
    
    # Check if corpus directory has files
    wav_files = list(corpus_dir.glob("*.wav"))
    if not wav_files:
        logger.error(f"No WAV files found in {corpus_dir}")
        return False
    
    console.print(f"Found {len(wav_files)} chunks to align" if RICH_AVAILABLE else f"Found {len(wav_files)} chunks to align")
    
    try:
        model_name = f"{language}_mfa"
        cmd = [
            "mfa", "align", 
            str(corpus_dir), 
            model_name, 
            model_name, 
            str(mfa_out), 
            "--clean", 
            "--overwrite",
            "--single_speaker"
        ]
        
        # Remove --beam and --retry-beam arguments as they can cause issues
        # If you want to keep them, use more reasonable values:
        # cmd.extend(["--beam", "10", "--retry-beam", "40"])
        
        console.print(f"Running MFA command: {' '.join(cmd)}" if RICH_AVAILABLE else f"Running MFA command: {' '.join(cmd)}")
        
        # Use subprocess.run with stdout/stderr to see what's happening
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log MFA output for debugging
        if result.stdout:
            logger.info(f"MFA stdout: {result.stdout[:500]}...")
        if result.stderr:
            logger.warning(f"MFA stderr: {result.stderr[:500]}")
        
        # Check if output was created
        if result.returncode == 0:
            # Check if any TextGrids were created
            output_files = list(mfa_out.glob("*.TextGrid"))
            if output_files:
                logger.info(f"MFA created {len(output_files)} TextGrid files")
                return True
            else:
                logger.error(f"MFA succeeded but no TextGrid files found in {mfa_out}")
                return False
        else:
            logger.error(f"MFA failed with return code: {result.returncode}")
            if result.stderr:
                logger.error(f"MFA error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"MFA execution failed: {e}")
        return False

def reconstruct_aligned_textgrid(family: AudioProjectFamily, mfa_raw_dir: Path, chunk_metadata: List[Dict], output_dir: Path, audio_base_dir: Path):
    """Reconstruct aligned TextGrid from chunk-level MFA results."""
    try:
        from praatio import textgrid
    except ImportError:
        logger.error("praatio module not found")
        return False
    
    if not family.textgrid_path:
        logger.error(f"No TextGrid found for family {family.project_key}")
        return False
    
    try:
        # Load the original TextGrid
        parent_tg = textgrid.openTextgrid(str(family.textgrid_path), includeEmptyIntervals=True)
        
        # Create a new TextGrid with the original's timestamps
        new_tg = textgrid.Textgrid()
        new_tg.minTimestamp = parent_tg.minTimestamp
        new_tg.maxTimestamp = parent_tg.maxTimestamp
        
        # Copy ALL tiers from the original
        for tier_name in parent_tg.tierNames:
            original_tier = parent_tg.getTier(tier_name)
            if hasattr(original_tier, 'entries'):  # IntervalTier
                # Create a new interval tier with the same entries
                new_tier = textgrid.IntervalTier(
                    tier_name,
                    original_tier.entries,
                    original_tier.minTimestamp,
                    original_tier.maxTimestamp
                )
                new_tg.addTier(new_tier)
            elif hasattr(original_tier, 'points'):  # PointTier
                # Create a new point tier with the same points
                new_tier = textgrid.PointTier(
                    tier_name,
                    original_tier.points,
                    original_tier.minTimestamp,
                    original_tier.maxTimestamp
                )
                new_tg.addTier(new_tier)
    except Exception as e:
        logger.error(f"Failed to open parent TextGrid {family.textgrid_path}: {e}")
        return False

    # Group chunks by speaker
    chunks_by_speaker: Dict[str, List[Dict]] = {}
    for chunk in chunk_metadata:
        speaker = chunk['speaker']
        if speaker not in chunks_by_speaker:
            chunks_by_speaker[speaker] = []
        chunks_by_speaker[speaker].append(chunk)
    
    # For each speaker, add the aligned word and phone tiers
    for speaker, speaker_chunks in chunks_by_speaker.items():
        words_intervals, phones_intervals = [], []
        
        for chunk in speaker_chunks:
            mfa_chunk_tg = mfa_raw_dir / f"{chunk['chunk_id']}.TextGrid"
            if not mfa_chunk_tg.exists():
                logger.warning(f"Missing chunk TextGrid: {mfa_chunk_tg}")
                continue
            
            try:
                chunk_tg = textgrid.openTextgrid(str(mfa_chunk_tg), includeEmptyIntervals=False)
                offset = chunk['offset']
                
                for tier_name in chunk_tg.tierNames:
                    tier = chunk_tg.getTier(tier_name)
                    entries = tier.entries if hasattr(tier, 'entries') else tier._entries if hasattr(tier, '_entries') else []
                    
                    for entry in entries:
                        adjusted_start = entry.start + offset
                        adjusted_end = entry.end + offset
                        
                        # Skip invalid intervals
                        if adjusted_start >= adjusted_end:
                            continue
                            
                        label = entry.label.strip()
                        if not label:
                            continue
                        
                        if "word" in tier_name.lower(): 
                            words_intervals.append((adjusted_start, adjusted_end, label))
                        elif "phone" in tier_name.lower(): 
                            phones_intervals.append((adjusted_start, adjusted_end, label))
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk['chunk_id']}: {e}")
                continue

        # Sort intervals by start time
        words_intervals.sort(key=lambda x: x[0])
        phones_intervals.sort(key=lambda x: x[0])
        
        # Add aligned tiers
        try:
            # Add word tier
            words_tier_name = f"{speaker}_words"
            if words_tier_name in new_tg.tierNames:
                new_tg.removeTier(words_tier_name)
            new_tg.addTier(textgrid.IntervalTier(words_tier_name, words_intervals, 0.0, new_tg.maxTimestamp))
            
            # Add phone tier
            phones_tier_name = f"{speaker}_phones"
            if phones_tier_name in new_tg.tierNames:
                new_tg.removeTier(phones_tier_name)
            new_tg.addTier(textgrid.IntervalTier(phones_tier_name, phones_intervals, 0.0, new_tg.maxTimestamp))
        except Exception as e:
            logger.error(f"Failed to add aligned tiers for {speaker}: {e}")

    try:
        # Preserve directory structure in output - mirror the audio input structure
        # Get the relative path from the audio directory to the audio file's parent
        if family.audio_path.parent != audio_base_dir:
            # Audio is in a subdirectory - preserve that structure
            relative_dir = family.audio_path.parent.relative_to(audio_base_dir)
            output_subdir = output_dir / relative_dir
        else:
            # Audio is in the root audio directory
            output_subdir = output_dir
        
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Clean filename: remove '_preprocessed' if present, then add '_aligned'
        clean_stem = family.audio_path.stem
        if clean_stem.endswith('_preprocessed'):
            clean_stem = clean_stem[:-13]  # Remove '_preprocessed' (13 characters)
        
        output_path = output_subdir / f"{clean_stem}_aligned.TextGrid"
        new_tg.save(str(output_path), format="long_textgrid", includeBlankSpaces=True)
        
        # DEBUG: Log what tiers were saved
        logger.info(f"Saved aligned TextGrid with {len(new_tg.tierNames)} tiers: {output_path}")
        for tier_name in new_tg.tierNames:
            tier = new_tg.getTier(tier_name)
            if hasattr(tier, 'entries'):
                logger.info(f"  - {tier_name}: {len(tier.entries)} intervals")
            elif hasattr(tier, 'points'):
                logger.info(f"  - {tier_name}: {len(tier.points)} points")
                
        console.print(f"Saved aligned TextGrid: {output_path}" if RICH_AVAILABLE else f"Saved aligned TextGrid: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save aligned TextGrid: {e}")
        return False

# ============================================================================
# SECTION 3: SUBSCRIPT ORCHESTRATION (UPDATED WITH CONFIG)
# ============================================================================

def run_syllabification_sub(aligned_dir: Path, audio_dir: Path, language: str, config: Dict):
    """Hook for libs/syllabify.py with full configuration."""
    if not config.get('enabled', False):
        console.print("Syllabification skipped (disabled in configuration)" if RICH_AVAILABLE else "Syllabification skipped")
        return
    
    try:
        from libs.syllabify import run_syllabification
        
        # Build kwargs dynamically based on config
        kwargs = {
            'aligned_dir': str(aligned_dir),
            'audio_dir': str(audio_dir),
            'language': language
        }
        
        # Add all configured parameters
        if 'dictionary_path' in config:
            kwargs['dict_dir'] = config['dictionary_path']
        if 'acoustic_threshold' in config:
            kwargs['threshold'] = config['acoustic_threshold']
        if 'frame_size' in config:
            kwargs['frame_size'] = config['frame_size']
        if 'hop_size' in config:
            kwargs['hop_size'] = config['hop_size']
        if 'use_dictionary' in config:
            kwargs['use_dictionary'] = config['use_dictionary']
        if 'use_acoustic' in config:
            kwargs['use_acoustic'] = config['use_acoustic']
        if 'fallback_order' in config:
            kwargs['fallback_order'] = config['fallback_order']
        
        console.print("\nRunning syllabification with custom configuration..." if RICH_AVAILABLE else "\nRunning syllabification...")
        run_syllabification(**kwargs)
        
    except ImportError:
        logger.error("libs/syllabify.py subscript not found.")
    except Exception as e:
        logger.error(f"Error in syllabification: {e}")

def run_tagging_sub(aligned_dir: Path, language: str, config: Dict):
    """Hook for libs/tag.py with full configuration."""
    if not config.get('enabled', False):
        console.print("POS tagging skipped (disabled in configuration)" if RICH_AVAILABLE else "POS tagging skipped")
        return
    
    try:
        from libs.tag import add_pos_tiers_to_textgrids
        
        # Build full model name based on language and size
        base_model = config['model_map'].get(language, 'en_core_web')
        model_name = f"{base_model}_{config['model_size']}"
        
        kwargs = {
            'aligned_dir': str(aligned_dir),
            'language': language,
            'model_name': model_name,
            'tag_type': config['tag_type'],
            'auto_download': config.get('auto_download', True)
        }
        
        console.print("\nRunning POS tagging with custom configuration..." if RICH_AVAILABLE else "\nRunning POS tagging...")
        add_pos_tiers_to_textgrids(**kwargs)
        
    except ImportError:
        logger.error("libs/tag.py subscript not found.")
    except Exception as e:
        logger.error(f"Error in POS tagging: {e}")

# ============================================================================
# MAIN WORKFLOW WITH FAMILY SUPPORT
# ============================================================================

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print("\n=== [bold cyan]03: ALIGNMENT & ANNOTATION HUB[/bold cyan] ===\n" if RICH_AVAILABLE else "\n=== 03: ALIGNMENT & ANNOTATION HUB ===\n")
    
    # Display acceleration status
    if ACCEL_READY:
        console.print("[green]C acceleration: Available (using for chunking)[/green]" if RICH_AVAILABLE else "C acceleration: Available (using for chunking)")
    else:
        console.print("[yellow]C acceleration: Not available (using Python fallbacks)[/yellow]" if RICH_AVAILABLE else "C acceleration: Not available (using Python fallbacks)")
    
    # Determine if we're in interactive mode
    # Interactive mode if: not --non-interactive AND (textgrid-dir and audio-dir not provided)
    interactive_mode = not args.non_interactive and (args.textgrid_dir is None or args.audio_dir is None)
    
    if interactive_mode:
        # 1. Basic inputs - interactive
        lang = select_language()
        
        # 2. Get directory paths from user
        tg_in = select_textgrid_directory()
        audio_in = select_audio_directory()
        
        # 3. Get full configurations for subscripts
        syllabify_config = configure_syllabification()
        pos_config = configure_pos_tagging()
        
        # Define paths
        out_dir = Path("./aligned_textgrids")
        mfa_auto_download = True  # Default to True in interactive mode
    else:
        # Non-interactive mode - use command line arguments
        # If --non-interactive is set but arguments missing, show error
        if args.non_interactive and (args.textgrid_dir is None or args.audio_dir is None):
            console.print("[red]Error: In non-interactive mode, --textgrid-dir and --audio-dir are required[/red]" if RICH_AVAILABLE else "Error: In non-interactive mode, --textgrid-dir and --audio-dir are required")
            sys.exit(1)
        
        console.print("[dim]Running in non-interactive mode[/dim]" if RICH_AVAILABLE else "Running in non-interactive mode")
        
        lang = args.language
        tg_in = Path(args.textgrid_dir)
        audio_in = Path(args.audio_dir)
        out_dir = Path(args.output_dir)
        mfa_auto_download = args.mfa_auto_download
        
        # Build syllabification config from arguments
        syllabify_config = {
            'enabled': args.enable_syllabification,
            'use_dictionary': args.syllabification_use_dictionary,
            'use_acoustic': args.syllabification_use_acoustic,
            'dictionary_path': args.syllabification_dict_dir,
            'acoustic_threshold': args.syllabification_threshold,
            'frame_size': args.syllabification_frame_size,
            'hop_size': args.syllabification_hop_size,
            'fallback_order': args.syllabification_fallback
        }
        
        # Build POS tagging config from arguments
        pos_config = {
            'enabled': args.enable_pos_tagging,
            'model_size': args.pos_model_size,
            'tag_type': args.pos_tag_type,
            'auto_download': args.pos_auto_download,
            'model_map': {
                'english': 'en_core_web',
                'spanish': 'es_core_news',
                'french': 'fr_core_news',
                'german': 'de_core_news',
                'italian': 'it_core_news',
                'portuguese': 'pt_core_news',
                'dutch': 'nl_core_news',
                'russian': 'ru_core_news',
                'mandarin': 'zh_core_web',
                'japanese': 'ja_core_news',
                'korean': 'ko_core_news',
                'arabic': 'ar_core_news'
            }
        }
        
        # Validate directories
        if not tg_in.exists():
            console.print(f"[red]Error: TextGrid directory not found: {tg_in}[/red]" if RICH_AVAILABLE else f"Error: TextGrid directory not found: {tg_in}")
            return
        
        if not audio_in.exists():
            console.print(f"[red]Error: Audio directory not found: {audio_in}[/red]" if RICH_AVAILABLE else f"Error: Audio directory not found: {audio_in}")
            return

    # Define paths
    temp_corpus = Path("./temp_mfa_corpus")
    mfa_temp_dir = Path("./mfa_temp")
    
    # Check if directories exist (tg_in already validated in select_textgrid_directory or args)
    if not audio_in.exists():  # Double-check in case directory was deleted between selection and now
        console.print(f"[red]Error: Audio directory not found: {audio_in}[/red]" if RICH_AVAILABLE else f"Error: Audio directory not found: {audio_in}")
        return
    
    # Find all project families
    families = find_project_families(audio_in, tg_in)
    
    if not families:
        console.print("[red]No project families found. Run Phase 2 first.[/red]" if RICH_AVAILABLE else "No project families found. Run Phase 2 first.")
        return
    
    console.print(f"\n[green]Found {len(families)} project families to process[/green]" if RICH_AVAILABLE else f"\nFound {len(families)} project families to process")

    # 2. Main Alignment Loop
    stats = {'success': 0, 'skip': 0}
    out_dir.mkdir(exist_ok=True)
    temp_corpus.mkdir(exist_ok=True)  # Create once at the start
    mfa_temp_dir.mkdir(exist_ok=True)

    if RICH_AVAILABLE:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn()) as progress:
            task = progress.add_task("Aligning families...", total=len(families))
            for project_key, family in families.items():
                console.print(f"\n[bold]Processing family:[/bold] {project_key}" if RICH_AVAILABLE else f"\nProcessing family: {project_key}")
                
                # Clear temp corpus for this family but keep directory
                for item in temp_corpus.glob("*"):
                    if item.is_file():
                        item.unlink()
                
                # Clear MFA temp directory
                for item in mfa_temp_dir.glob("*"):
                    if item.is_file():
                        item.unlink()
                
                # Create corpus for this family
                chunk_metadata = create_mfa_corpus(family, temp_corpus)
                
                if not chunk_metadata:
                    console.print(f"[yellow]No chunks created for {project_key}[/yellow]" if RICH_AVAILABLE else f"No chunks created for {project_key}")
                    stats['skip'] += 1
                    progress.advance(task)
                    continue
                
                console.print(f"Created {len(chunk_metadata)} chunks for {project_key}" if RICH_AVAILABLE else f"Created {len(chunk_metadata)} chunks for {project_key}")
                
                # Run MFA alignment for this family
                if run_mfa_alignment(temp_corpus, lang, mfa_temp_dir, mfa_auto_download):
                    # Reconstruct aligned TextGrid
                    if reconstruct_aligned_textgrid(family, mfa_temp_dir, chunk_metadata, out_dir, audio_in):
                        stats['success'] += 1
                        console.print(f"[green]Successfully aligned {project_key}[/green]" if RICH_AVAILABLE else f"Successfully aligned {project_key}")
                    else:
                        stats['skip'] += 1
                        console.print(f"[red]Failed to reconstruct TextGrid for {project_key}[/red]" if RICH_AVAILABLE else f"Failed to reconstruct TextGrid for {project_key}")
                else:
                    stats['skip'] += 1
                    console.print(f"[red]MFA alignment failed for {project_key}[/red]" if RICH_AVAILABLE else f"MFA alignment failed for {project_key}")
                
                progress.advance(task)
    else:
        for project_key, family in families.items():
            print(f"\nProcessing family: {project_key}")
            
            # Clear temp corpus for this family but keep directory
            for item in temp_corpus.glob("*"):
                if item.is_file():
                    item.unlink()
            
            # Clear MFA temp directory
            for item in mfa_temp_dir.glob("*"):
                if item.is_file():
                    item.unlink()
            
            # Create corpus for this family
            chunk_metadata = create_mfa_corpus(family, temp_corpus)
            
            if not chunk_metadata:
                print(f"No chunks created for {project_key}")
                stats['skip'] += 1
                continue
            
            print(f"Created {len(chunk_metadata)} chunks for {project_key}")
            
            # Run MFA alignment for this family
            if run_mfa_alignment(temp_corpus, lang, mfa_temp_dir, mfa_auto_download):
                # Reconstruct aligned TextGrid
                if reconstruct_aligned_textgrid(family, mfa_temp_dir, chunk_metadata, out_dir, audio_in):
                    stats['success'] += 1
                    print(f"Successfully aligned {project_key}")
                else:
                    stats['skip'] += 1
                    print(f"Failed to reconstruct TextGrid for {project_key}")
            else:
                stats['skip'] += 1
                print(f"MFA alignment failed for {project_key}")

    # Clean up temp directories
    if temp_corpus.exists():
        shutil.rmtree(temp_corpus)
        console.print(f"\nCleaned up temporary corpus" if RICH_AVAILABLE else f"\nCleaned up temporary corpus")
    
    if mfa_temp_dir.exists():
        shutil.rmtree(mfa_temp_dir)
        console.print(f"Cleaned up MFA temporary directory" if RICH_AVAILABLE else f"Cleaned up MFA temporary directory")

    # 3. Subscript Execution with full configurations
    run_syllabification_sub(out_dir, audio_in, lang, syllabify_config)
    run_tagging_sub(out_dir, lang, pos_config)

    # 4. Enhanced Summary
    if RICH_AVAILABLE:
        summary = Table(title="Annotation Phase Summary", box=None)
        summary.add_column("Component", style="cyan")
        summary.add_column("Configuration", style="bold")
        
        summary.add_row("Language", lang.capitalize())
        summary.add_row("TextGrid Directory", str(tg_in))
        summary.add_row("Audio Directory", str(audio_in))
        summary.add_row("Families Found", str(len(families)))
        summary.add_row("Successfully Aligned", str(stats['success']))
        summary.add_row("Errors/Skipped", str(stats['skip']))
        
        # Syllabification details
        if syllabify_config['enabled']:
            summary.add_row("", "")
            summary.add_row("[green]Syllabification[/green]", "ENABLED")
            summary.add_row("  Dictionary", "Yes" if syllabify_config.get('use_dictionary') else "No")
            if syllabify_config.get('use_dictionary'):
                summary.add_row("  Dict Path", syllabify_config['dictionary_path'])
            summary.add_row("  Acoustic", "Yes" if syllabify_config.get('use_acoustic') else "No")
            if syllabify_config.get('use_acoustic'):
                summary.add_row("  Threshold", str(syllabify_config['acoustic_threshold']))
                summary.add_row("  Frame/Hop", f"{syllabify_config['frame_size']}/{syllabify_config['hop_size']}")
            if syllabify_config.get('use_dictionary') and syllabify_config.get('use_acoustic'):
                summary.add_row("  Fallback", syllabify_config['fallback_order'])
        else:
            summary.add_row("", "")
            summary.add_row("[yellow]Syllabification[/yellow]", "DISABLED")
        
        # POS Tagging details
        if pos_config['enabled']:
            summary.add_row("", "")
            summary.add_row("[green]POS Tagging[/green]", "ENABLED")
            summary.add_row("  Model Size", pos_config['model_size'])
            summary.add_row("  Tag Type", pos_config['tag_type'])
            summary.add_row("  Auto-download", "Yes" if pos_config.get('auto_download') else "No")
            base_model = pos_config['model_map'].get(lang, 'en_core_web')
            summary.add_row("  Model", f"{base_model}_{pos_config['model_size']}")
        else:
            summary.add_row("", "")
            summary.add_row("[yellow]POS Tagging[/yellow]", "DISABLED")
        
        console.print(summary)
    else:
        print(f"\n{'='*50}")
        print(f"Selected Language: {lang.capitalize()}")
        print(f"TextGrid Directory: {tg_in}")
        print(f"Audio Directory: {audio_in}")
        print(f"{'='*50}")
        print(f"Annotation Phase Summary:")
        print(f"  Families Found: {len(families)}")
        print(f"  Successfully Aligned: {stats['success']}")
        print(f"  Errors/Skipped: {stats['skip']}")
        print(f"\nSyllabification: {'ENABLED' if syllabify_config['enabled'] else 'DISABLED'}")
        if syllabify_config['enabled']:
            print(f"  Dictionary: {syllabify_config.get('use_dictionary')}")
            print(f"  Acoustic: {syllabify_config.get('use_acoustic')}")
        print(f"\nPOS Tagging: {'ENABLED' if pos_config['enabled'] else 'DISABLED'}")
        if pos_config['enabled']:
            print(f"  Model Size: {pos_config['model_size']}")
            print(f"  Tag Type: {pos_config['tag_type']}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    main()