"""
99_main.py - Master Pipeline Controller
@author: taylosh
Created on Tue Oct 28 2025
Last edited on Mar 16 2026

Master Transcriber - controller for the complete ASR pipeline
Orchestrates the complete audio processing workflow using our modular components.

Currently implements:
- Phase 1: Audio Preprocessing
- Phase 2: Transcription with Optional Diarization
- Phase 3: Alignment & Annotation

This script serves as the central hub, calling individual phase scripts
with appropriate arguments based on user configuration.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time

# Rich progress and console integration
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    # Fallback console
    class FallbackConsole:
        def print(self, *args, **kwargs): print(*args)
        def input(self, prompt): return input(prompt)
    console = FallbackConsole()


# ============================================================================
# CONFIGURATION
# ============================================================================

# Phase scripts mapping
PHASE_SCRIPTS = {
    1: "01_preprocess.py",      # Audio preprocessing
    2: "02_transcribe.py",      # Transcription & diarization
    3: "03_annotate.py",        # Alignment & annotation
}

# Enhancement level mapping (mirrors 01_preprocess.py)
ENHANCEMENT_OPTIONS = {
    0: {"name": "None", "desc": "Standardize Only (No Enhancement)"},
    1: {"name": "Gentle", "desc": "HPF + LUFS Normalization"},
    2: {"name": "Aggressive", "desc": "Gentle + Noise Reduction + EQ"},
    3: {"name": "Experimental", "desc": "SpeechBrain AI Models (SepFormer, MetricGAN+)"}
}

# Language mapping (mirrors 02_transcribe.py)
LANGUAGE_OPTIONS = {
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

# Model mapping (mirrors 02_transcribe.py)
MODEL_OPTIONS = {
    1: {"name": "Tiny", "size": "tiny"},
    2: {"name": "Base", "size": "base"},
    3: {"name": "Small", "size": "small"},
    4: {"name": "Medium", "size": "medium"},
    5: {"name": "Large", "size": "large"}
}

# Precision mapping for diarization
PRECISION_OPTIONS = {
    1: {"name": "Low", "desc": "Faster, lower precision"},
    2: {"name": "Medium", "desc": "Balanced speed and precision"},
    3: {"name": "High", "desc": "Slowest, highest timing accuracy"}
}

def setup_logging():
    """Setup logging for the pipeline"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "pipeline.log"
    
    # Clear old log file if it exists
    if log_file.exists():
        log_file.unlink()
    
    level = logging.INFO
    handlers = [logging.FileHandler(log_file)]
    if RICH_AVAILABLE:
        handlers.append(RichHandler(markup=True, show_path=False))
    else:
        handlers.append(logging.StreamHandler())
        
    logging.basicConfig(
        level=level, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        handlers=handlers
    )
    return logging.getLogger(__name__)

logger = setup_logging()


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class PipelineConfig:
    """Stores and manages pipeline configuration"""
    
    def __init__(self):
        # Phase 1: Preprocessing config
        self.input_dir = "./original_audio"
        self.output_dir = "./preprocessed_audio"
        self.enhancement_level = 0
        self.debug_mode = False
        self.max_workers = None
        self.verbose = False
        
        # Phase 2: Transcription config
        self.transcription_output_dir = "./initial_transcription"
        self.language = "auto"
        self.model = "base"
        self.use_diarization = False
        self.diarization_precision = "medium"
        self.diarization_threshold = 0.5
        self.speaker_profiles = []
        self.silence_threshold = -28
        self.min_silence_len = 100
        
        # Phase 3: Alignment & Annotation config
        self.alignment_language = "english"
        self.enable_syllabification = False
        self.enable_pos_tagging = False
        self.syllabification_dict_dir = "./models/syllable_dicts"
        self.syllabification_use_dictionary = True
        self.syllabification_use_acoustic = True
        self.syllabification_threshold = 0.3
        self.syllabification_frame_size = 256
        self.syllabification_hop_size = 128
        self.syllabification_fallback = "dictionary_first"
        self.pos_model_size = "sm"
        self.pos_tag_type = "universal"
        self.pos_auto_download = True
        
        # Phase selection
        self.run_phase1 = True
        self.run_phase2 = True
        self.run_phase3 = False
        self.run_phase4 = False
        
        # System
        self.config_file = "./pipeline_config.json"
        self.non_interactive = False
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for saving"""
        return {
            # Phase 1
            'input_dir': self.input_dir,
            'output_dir': self.output_dir,
            'enhancement_level': self.enhancement_level,
            'debug_mode': self.debug_mode,
            'max_workers': self.max_workers,
            'verbose': self.verbose,
            
            # Phase 2
            'transcription_output_dir': self.transcription_output_dir,
            'language': self.language,
            'model': self.model,
            'use_diarization': self.use_diarization,
            'diarization_precision': self.diarization_precision,
            'diarization_threshold': self.diarization_threshold,
            'speaker_profiles': self.speaker_profiles,
            'silence_threshold': self.silence_threshold,
            'min_silence_len': self.min_silence_len,
            
            # Phase 3
            'alignment_language': self.alignment_language,
            'enable_syllabification': self.enable_syllabification,
            'enable_pos_tagging': self.enable_pos_tagging,
            'syllabification_dict_dir': self.syllabification_dict_dir,
            'syllabification_use_dictionary': self.syllabification_use_dictionary,
            'syllabification_use_acoustic': self.syllabification_use_acoustic,
            'syllabification_threshold': self.syllabification_threshold,
            'syllabification_frame_size': self.syllabification_frame_size,
            'syllabification_hop_size': self.syllabification_hop_size,
            'syllabification_fallback': self.syllabification_fallback,
            'pos_model_size': self.pos_model_size,
            'pos_tag_type': self.pos_tag_type,
            'pos_auto_download': self.pos_auto_download,
            
            # Phase selection
            'run_phase1': self.run_phase1,
            'run_phase2': self.run_phase2,
            'run_phase3': self.run_phase3,
            'run_phase4': self.run_phase4,
        }
    
    def from_dict(self, data: Dict):
        """Load config from dictionary"""
        # Phase 1
        self.input_dir = data.get('input_dir', self.input_dir)
        self.output_dir = data.get('output_dir', self.output_dir)
        self.enhancement_level = data.get('enhancement_level', self.enhancement_level)
        self.debug_mode = data.get('debug_mode', self.debug_mode)
        self.max_workers = data.get('max_workers', self.max_workers)
        self.verbose = data.get('verbose', self.verbose)
        
        # Phase 2
        self.transcription_output_dir = data.get('transcription_output_dir', self.transcription_output_dir)
        self.language = data.get('language', self.language)
        self.model = data.get('model', self.model)
        self.use_diarization = data.get('use_diarization', self.use_diarization)
        self.diarization_precision = data.get('diarization_precision', self.diarization_precision)
        self.diarization_threshold = data.get('diarization_threshold', self.diarization_threshold)
        self.speaker_profiles = data.get('speaker_profiles', self.speaker_profiles)
        self.silence_threshold = data.get('silence_threshold', self.silence_threshold)
        self.min_silence_len = data.get('min_silence_len', self.min_silence_len)
        
        # Phase 3
        self.alignment_language = data.get('alignment_language', self.alignment_language)
        self.enable_syllabification = data.get('enable_syllabification', self.enable_syllabification)
        self.enable_pos_tagging = data.get('enable_pos_tagging', self.enable_pos_tagging)
        self.syllabification_dict_dir = data.get('syllabification_dict_dir', self.syllabification_dict_dir)
        self.syllabification_use_dictionary = data.get('syllabification_use_dictionary', self.syllabification_use_dictionary)
        self.syllabification_use_acoustic = data.get('syllabification_use_acoustic', self.syllabification_use_acoustic)
        self.syllabification_threshold = data.get('syllabification_threshold', self.syllabification_threshold)
        self.syllabification_frame_size = data.get('syllabification_frame_size', self.syllabification_frame_size)
        self.syllabification_hop_size = data.get('syllabification_hop_size', self.syllabification_hop_size)
        self.syllabification_fallback = data.get('syllabification_fallback', self.syllabification_fallback)
        self.pos_model_size = data.get('pos_model_size', self.pos_model_size)
        self.pos_tag_type = data.get('pos_tag_type', self.pos_tag_type)
        self.pos_auto_download = data.get('pos_auto_download', self.pos_auto_download)
        
        # Phase selection
        self.run_phase1 = data.get('run_phase1', self.run_phase1)
        self.run_phase2 = data.get('run_phase2', self.run_phase2)
        self.run_phase3 = data.get('run_phase3', self.run_phase3)
        self.run_phase4 = data.get('run_phase4', self.run_phase4)
    
    def save(self, filepath: Optional[str] = None):
        """Save configuration to file"""
        if filepath:
            self.config_file = filepath
        with open(self.config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {self.config_file}")
    
    def load(self, filepath: Optional[str] = None):
        """Load configuration from file"""
        if filepath:
            self.config_file = filepath
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.from_dict(json.load(f))
            logger.info(f"Configuration loaded from {self.config_file}")
            return True
        return False


# ============================================================================
# INTERACTIVE CONFIGURATION
# ============================================================================

def get_transcription_config(config: PipelineConfig) -> PipelineConfig:
    """Interactively gather transcription configuration from user"""
    
    console.print("\n[bold cyan]╔════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║     TRANSCRIPTION CONFIGURATION        ║[/bold cyan]")
    console.print("[bold cyan]╚════════════════════════════════════════╝[/bold cyan]\n")
    
    # Transcription output directory
    default_trans_out = config.transcription_output_dir
    if RICH_AVAILABLE:
        config.transcription_output_dir = Prompt.ask("Transcription output directory", default=default_trans_out)
    else:
        config.transcription_output_dir = input(f"Transcription output directory [default: {default_trans_out}]: ").strip() or default_trans_out
    
    # Language selection
    if RICH_AVAILABLE:
        # Display language options
        table = Table(title="Available Languages")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Language", style="green")
        table.add_column("Code", style="dim")
        
        for num, lang_info in LANGUAGE_OPTIONS.items():
            table.add_row(str(num), lang_info["name"], lang_info["code"])
        
        console.print(table)
        lang_choice = IntPrompt.ask("Select language", default=14)
        
        if lang_choice in LANGUAGE_OPTIONS:
            config.language = LANGUAGE_OPTIONS[lang_choice]["code"]
            console.print(f"[green]Selected: {LANGUAGE_OPTIONS[lang_choice]['name']}[/green]")
        else:
            console.print("[yellow]Invalid choice, using auto-detect[/yellow]")
            config.language = "auto"
    else:
        print("\nAvailable Languages:")
        for num, lang_info in LANGUAGE_OPTIONS.items():
            print(f"  {num}. {lang_info['name']} ({lang_info['code']})")
        try:
            lang_choice = int(input("\nSelect language (default: 14 - Auto-detect): ").strip() or "14")
            if lang_choice in LANGUAGE_OPTIONS:
                config.language = LANGUAGE_OPTIONS[lang_choice]["code"]
            else:
                config.language = "auto"
        except ValueError:
            config.language = "auto"
    
    # Model selection
    if RICH_AVAILABLE:
        table = Table(title="Available Whisper Models")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Model", style="green")
        table.add_column("Size", style="dim")
        
        for num, model_info in MODEL_OPTIONS.items():
            table.add_row(str(num), model_info["name"], model_info["size"])
        
        console.print(table)
        model_choice = IntPrompt.ask("Select model", default=2)
        
        if model_choice in MODEL_OPTIONS:
            config.model = MODEL_OPTIONS[model_choice]["size"]
            console.print(f"[green]Selected: {MODEL_OPTIONS[model_choice]['name']}[/green]")
        else:
            console.print("[yellow]Invalid choice, using base[/yellow]")
            config.model = "base"
    else:
        print("\nAvailable Models:")
        for num, model_info in MODEL_OPTIONS.items():
            print(f"  {num}. {model_info['name']} ({model_info['size']})")
        try:
            model_choice = int(input("\nSelect model (default: 2 - Base): ").strip() or "2")
            if model_choice in MODEL_OPTIONS:
                config.model = MODEL_OPTIONS[model_choice]["size"]
            else:
                config.model = "base"
        except ValueError:
            config.model = "base"
    
    # Diarization selection
    if RICH_AVAILABLE:
        config.use_diarization = Confirm.ask("Use speaker diarization?", default=False)
    else:
        diar_choice = input("Use speaker diarization? [y/N]: ").strip().lower()
        config.use_diarization = diar_choice in ['y', 'yes']
    
    if config.use_diarization:
        # Diarization precision
        if RICH_AVAILABLE:
            table = Table(title="Diarization Precision Levels")
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Level", style="green")
            table.add_column("Description", style="white")
            
            for num, prec_info in PRECISION_OPTIONS.items():
                table.add_row(str(num), prec_info["name"], prec_info["desc"])
            
            console.print(table)
            prec_choice = IntPrompt.ask("Select precision level", default=2)
            
            if prec_choice in PRECISION_OPTIONS:
                config.diarization_precision = prec_choice
            else:
                config.diarization_precision = 2
        else:
            print("\nPrecision Levels:")
            print("  1. Low - Faster, lower precision")
            print("  2. Medium - Balanced speed and precision")
            print("  3. High - Slowest, highest timing accuracy")
            try:
                prec_choice = int(input("\nSelect precision level (default: 2): ").strip() or "2")
                config.diarization_precision = prec_choice if 1 <= prec_choice <= 3 else 2
            except ValueError:
                config.diarization_precision = 2
        
        # Convert numeric precision to string for the script
        precision_map = {1: "low", 2: "medium", 3: "high"}
        config.diarization_precision = precision_map[config.diarization_precision]
        
        # Diarization threshold
        if RICH_AVAILABLE:
            threshold_input = Prompt.ask(
                "Matching threshold (0.0-1.0, higher = stricter)", 
                default=str(config.diarization_threshold)
            )
        else:
            threshold_input = input(f"Matching threshold (0.0-1.0) [default: {config.diarization_threshold}]: ").strip() or str(config.diarization_threshold)
        
        try:
            config.diarization_threshold = float(threshold_input)
            config.diarization_threshold = max(0.0, min(1.0, config.diarization_threshold))
        except ValueError:
            pass  # Keep default
        
        # Speaker profiles (optional)
        profiles_dir = Path("./models/embeddings")
        if profiles_dir.exists():
            profile_files = list(profiles_dir.glob("*.json"))
            if profile_files:
                available_profiles = [f.stem for f in profile_files]
                
                if RICH_AVAILABLE:
                    console.print("\n[bold]Available speaker profiles:[/bold]")
                    for i, name in enumerate(available_profiles, 1):
                        console.print(f"  {i}. {name}")
                    
                    profile_choice = Prompt.ask(
                        "Select profiles to use (comma-separated numbers, or 'all', or 'none')",
                        default="none"
                    )
                else:
                    print("\nAvailable speaker profiles:")
                    for i, name in enumerate(available_profiles, 1):
                        print(f"  {i}. {name}")
                    profile_choice = input("Select profiles to use (comma-separated numbers, or 'all', or 'none') [default: none]: ").strip() or "none"
                
                if profile_choice.lower() == 'all':
                    config.speaker_profiles = available_profiles
                elif profile_choice.lower() != 'none':
                    selected = []
                    try:
                        for num in profile_choice.split(','):
                            idx = int(num.strip()) - 1
                            if 0 <= idx < len(available_profiles):
                                selected.append(available_profiles[idx])
                        config.speaker_profiles = selected
                    except ValueError:
                        config.speaker_profiles = []
                else:
                    config.speaker_profiles = []
    
    else:  # No diarization - segmentation parameters
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]=== Silence Segmentation Parameters ===[/bold cyan]")
            console.print("These parameters control how audio is split into segments based on silence.")
            
            console.print("\n[bold]Silence Threshold (dB)[/bold]")
            console.print("  Lower values = more sensitive (detects quieter pauses)")
            console.print("  Higher values = less sensitive (ignores background noise)")
            console.print("  Suggestions:")
            console.print("    - Noisy background: -20 to -25 dB")
            console.print("    - Normal conversation: -28 to -32 dB")
            console.print("    - Very clean audio: -35 to -40 dB")
        else:
            print("\n=== Silence Segmentation Parameters ===")
            print("These parameters control how audio is split into segments based on silence.")
        
        while True:
            try:
                if RICH_AVAILABLE:
                    thresh_input = Prompt.ask("Enter silence threshold in dB", default=str(config.silence_threshold))
                else:
                    thresh_input = input(f"Enter silence threshold in dB [default: {config.silence_threshold}]: ").strip() or str(config.silence_threshold)
                
                silence_thresh = int(thresh_input)
                if silence_thresh > 0:
                    if RICH_AVAILABLE:
                        console.print("[yellow]Threshold should be negative (e.g., -35)[/yellow]")
                    else:
                        print("Threshold should be negative (e.g., -35)")
                    continue
                config.silence_threshold = silence_thresh
                break
            except ValueError:
                if RICH_AVAILABLE:
                    console.print("[red]Please enter a valid number[/red]")
                else:
                    print("Please enter a valid number")
        
        if RICH_AVAILABLE:
            console.print("\n[bold]Minimum Silence Length (ms)[/bold]")
            console.print("  Shorter values = more segments (detects brief pauses)")
            console.print("  Longer values = fewer segments (requires longer pauses)")
        else:
            print("\nMinimum Silence Length (ms)")
        
        while True:
            try:
                if RICH_AVAILABLE:
                    len_input = Prompt.ask("Enter minimum silence length in ms", default=str(config.min_silence_len))
                else:
                    len_input = input(f"Enter minimum silence length in ms [default: {config.min_silence_len}]: ").strip() or str(config.min_silence_len)
                
                min_len = int(len_input)
                if min_len < 10:
                    if RICH_AVAILABLE:
                        console.print("[yellow]Minimum silence length should be at least 10 ms[/yellow]")
                    else:
                        print("Minimum silence length should be at least 10 ms")
                    continue
                config.min_silence_len = min_len
                break
            except ValueError:
                if RICH_AVAILABLE:
                    console.print("[red]Please enter a valid number[/red]")
                else:
                    print("Please enter a valid number")
    
    return config

def get_alignment_config(config: PipelineConfig) -> PipelineConfig:
    """Interactively gather alignment and annotation configuration from user"""
    
    console.print("\n[bold cyan]╔════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║     ALIGNMENT & ANNOTATION CONFIG      ║[/bold cyan]")
    console.print("[bold cyan]╚════════════════════════════════════════╝[/bold cyan]\n")
    
    # Language for MFA alignment
    mfa_language_map = {
        1: "english", 2: "spanish", 3: "french", 4: "german", 5: "italian",
        6: "portuguese", 7: "dutch", 8: "russian", 9: "mandarin", 10: "japanese",
        11: "korean", 12: "arabic"
    }
    
    mfa_language_names = {
        1: "English", 2: "Spanish", 3: "French", 4: "German", 5: "Italian",
        6: "Portuguese", 7: "Dutch", 8: "Russian", 9: "Mandarin", 10: "Japanese",
        11: "Korean", 12: "Arabic"
    }
    
    if RICH_AVAILABLE:
        table = Table(title="Available Languages for MFA Alignment")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Language", style="green")
        
        for num, name in mfa_language_names.items():
            table.add_row(str(num), name)
        
        console.print(table)
        lang_choice = IntPrompt.ask("Select alignment language", default=1)
        
        if lang_choice in mfa_language_map:
            config.alignment_language = mfa_language_map[lang_choice]
            console.print(f"[green]Selected: {mfa_language_names[lang_choice]}[/green]")
        else:
            console.print("[yellow]Invalid choice, using English[/yellow]")
            config.alignment_language = "english"
    else:
        print("\nAvailable Languages for MFA Alignment:")
        for num, name in mfa_language_names.items():
            print(f"  {num}. {name}")
        try:
            lang_choice = int(input("\nSelect alignment language (default: 1 - English): ").strip() or "1")
            if lang_choice in mfa_language_map:
                config.alignment_language = mfa_language_map[lang_choice]
            else:
                config.alignment_language = "english"
        except ValueError:
            config.alignment_language = "english"
    
    # Syllabification configuration
    if RICH_AVAILABLE:
        config.enable_syllabification = Confirm.ask("\nEnable syllabification?", default=False)
    else:
        syll_input = input("\nEnable syllabification? [y/N]: ").strip().lower()
        config.enable_syllabification = syll_input in ['y', 'yes']
    
    if config.enable_syllabification:
        if RICH_AVAILABLE:
            config.syllabification_use_dictionary = Confirm.ask("  Use dictionary lookup?", default=True)
            config.syllabification_use_acoustic = Confirm.ask("  Use acoustic analysis?", default=True)
            
            if config.syllabification_use_dictionary:
                config.syllabification_dict_dir = Prompt.ask(
                    "  Dictionary path", 
                    default=config.syllabification_dict_dir
                )
            
            if config.syllabification_use_acoustic:
                console.print("\n  [dim]Acoustic analysis parameters:[/dim]")
                config.syllabification_threshold = FloatPrompt.ask(
                    "    Peak detection threshold (0.0-1.0)", 
                    default=config.syllabification_threshold
                )
                config.syllabification_frame_size = IntPrompt.ask(
                    "    Frame size (samples)", 
                    default=config.syllabification_frame_size
                )
                config.syllabification_hop_size = IntPrompt.ask(
                    "    Hop size (samples)", 
                    default=config.syllabification_hop_size
                )
            
            if config.syllabification_use_dictionary and config.syllabification_use_acoustic:
                console.print("\n  [dim]Fallback behavior:[/dim]")
                console.print("    1. dictionary_first (Try dictionary first, fall back to acoustic)")
                console.print("    2. acoustic_first (Try acoustic first, fall back to dictionary)")
                
                fallback_choice = Prompt.ask(
                    "    Select fallback order",
                    choices=["1", "2"],
                    default="1"
                )
                config.syllabification_fallback = 'dictionary_first' if fallback_choice == "1" else 'acoustic_first'
        else:
            # Non-Rich fallback
            dict_choice = input("  Use dictionary lookup? [Y/n]: ").strip().lower()
            config.syllabification_use_dictionary = dict_choice != 'n'
            
            acoustic_choice = input("  Use acoustic analysis? [Y/n]: ").strip().lower()
            config.syllabification_use_acoustic = acoustic_choice != 'n'
            
            if config.syllabification_use_dictionary:
                dict_path = input(f"  Dictionary path [{config.syllabification_dict_dir}]: ").strip()
                if dict_path:
                    config.syllabification_dict_dir = dict_path
            
            if config.syllabification_use_acoustic:
                try:
                    thresh = input(f"    Peak detection threshold (0.0-1.0) [{config.syllabification_threshold}]: ").strip()
                    if thresh:
                        config.syllabification_threshold = float(thresh)
                    
                    frame = input(f"    Frame size (samples) [{config.syllabification_frame_size}]: ").strip()
                    if frame:
                        config.syllabification_frame_size = int(frame)
                    
                    hop = input(f"    Hop size (samples) [{config.syllabification_hop_size}]: ").strip()
                    if hop:
                        config.syllabification_hop_size = int(hop)
                except ValueError:
                    print("    Invalid input, using defaults")
            
            if config.syllabification_use_dictionary and config.syllabification_use_acoustic:
                print("\n  Fallback behavior:")
                print("    1. dictionary_first (Try dictionary first, fall back to acoustic)")
                print("    2. acoustic_first (Try acoustic first, fall back to dictionary)")
                choice = input("    Select fallback order (1/2) [1]: ").strip() or "1"
                config.syllabification_fallback = 'dictionary_first' if choice == "1" else 'acoustic_first'
    
    # POS Tagging configuration
    if RICH_AVAILABLE:
        config.enable_pos_tagging = Confirm.ask("\nEnable POS tagging?", default=False)
    else:
        pos_input = input("\nEnable POS tagging? [y/N]: ").strip().lower()
        config.enable_pos_tagging = pos_input in ['y', 'yes']
    
    if config.enable_pos_tagging:
        if RICH_AVAILABLE:
            console.print("\n  [dim]Model size options:[/dim]")
            console.print("    sm - Small (fastest, least accurate)")
            console.print("    md - Medium (balanced)")
            console.print("    lg - Large (slowest, most accurate)")
            
            config.pos_model_size = Prompt.ask(
                "  Select model size",
                choices=["sm", "md", "lg"],
                default=config.pos_model_size
            )
            
            console.print("\n  [dim]Tag type options:[/dim]")
            console.print("    universal - Universal POS tags (e.g., NOUN, VERB, ADJ)")
            console.print("    fine - Fine-grained tags (language-specific, e.g., NNP, VBD)")
            
            config.pos_tag_type = Prompt.ask(
                "  Select tag type",
                choices=["universal", "fine"],
                default=config.pos_tag_type
            )
            
            config.pos_auto_download = Confirm.ask(
                "  Auto-download missing models?",
                default=config.pos_auto_download
            )
        else:
            # Non-Rich fallback
            print("\n  Model size options:")
            print("    sm - Small (fastest, least accurate)")
            print("    md - Medium (balanced)")
            print("    lg - Large (slowest, most accurate)")
            size = input(f"  Select model size [sm/md/lg, default: {config.pos_model_size}]: ").strip()
            if size in ['sm', 'md', 'lg']:
                config.pos_model_size = size
            
            print("\n  Tag type options:")
            print("    universal - Universal POS tags (e.g., NOUN, VERB, ADJ)")
            print("    fine - Fine-grained tags (language-specific, e.g., NNP, VBD)")
            tag = input(f"  Select tag type [universal/fine, default: {config.pos_tag_type}]: ").strip()
            if tag in ['universal', 'fine']:
                config.pos_tag_type = tag
            
            auto = input("  Auto-download missing models? [Y/n, default: y]: ").strip().lower()
            config.pos_auto_download = auto != 'n'
    
    return config

def get_user_config(config: PipelineConfig) -> PipelineConfig:
    """Interactively gather complete configuration from user"""
    
    console.print("\n[bold cyan]╔════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║     ASR PIPELINE CONFIGURATION        ║[/bold cyan]")
    console.print("[bold cyan]╚════════════════════════════════════════╝[/bold cyan]\n")
    
    # Phase selection
    if RICH_AVAILABLE:
        console.print("[bold]Select phases to run:[/bold]")
        config.run_phase1 = Confirm.ask("Run Phase 1 (Preprocessing)?", default=config.run_phase1)
        config.run_phase2 = Confirm.ask("Run Phase 2 (Transcription)?", default=config.run_phase2)
        config.run_phase3 = Confirm.ask("Run Phase 3 (Alignment & Annotation)?", default=config.run_phase3)
    else:
        print("\nSelect phases to run:")
        phase1_input = input(f"Run Phase 1 (Preprocessing)? [Y/n, default: {'Y' if config.run_phase1 else 'N'}]: ").strip().lower()
        config.run_phase1 = phase1_input != 'n' if phase1_input else config.run_phase1
        phase2_input = input(f"Run Phase 2 (Transcription)? [Y/n, default: {'Y' if config.run_phase2 else 'N'}]: ").strip().lower()
        config.run_phase2 = phase2_input != 'n' if phase2_input else config.run_phase2
        phase3_input = input(f"Run Phase 3 (Alignment & Annotation)? [y/N, default: {'Y' if config.run_phase3 else 'N'}]: ").strip().lower()
        config.run_phase3 = phase3_input == 'y' if phase3_input else config.run_phase3
    
    if config.run_phase1:
        # Input directory
        default_input = config.input_dir
        if RICH_AVAILABLE:
            config.input_dir = Prompt.ask("Input directory (containing original WAV files)", default=default_input)
        else:
            config.input_dir = input(f"Input directory [default: {default_input}]: ").strip() or default_input
        
        # Output directory for preprocessing
        default_output = config.output_dir
        if RICH_AVAILABLE:
            config.output_dir = Prompt.ask("Preprocessing output directory", default=default_output)
        else:
            config.output_dir = input(f"Preprocessing output directory [default: {default_output}]: ").strip() or default_output
        
        # Enhancement level
        if RICH_AVAILABLE:
            # Display enhancement options
            table = Table(title="Enhancement Levels")
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Level", style="green")
            table.add_column("Description", style="white")
            
            for num, level_info in ENHANCEMENT_OPTIONS.items():
                table.add_row(str(num), level_info["name"], level_info["desc"])
            
            console.print(table)
            config.enhancement_level = IntPrompt.ask("Select enhancement level", default=config.enhancement_level)
            
            # Show additional info for experimental level
            if config.enhancement_level == 3:
                console.print("[yellow]Note: Experimental level will prompt for SpeechBrain model selection[/yellow]")
        else:
            print("\nEnhancement Levels:")
            for num, level_info in ENHANCEMENT_OPTIONS.items():
                print(f" {num}. {level_info['name']} - {level_info['desc']}")
            try:
                choice = input(f"\nSelect enhancement [0-3, default: {config.enhancement_level}]: ").strip()
                config.enhancement_level = int(choice) if choice else config.enhancement_level
            except ValueError:
                pass
        
        # Debug mode
        if RICH_AVAILABLE:
            config.debug_mode = Confirm.ask("Enable debug mode? (saves intermediate files)", default=config.debug_mode)
        else:
            debug_input = input(f"Enable debug mode? [y/n, default: {'y' if config.debug_mode else 'n'}]: ").strip().lower()
            config.debug_mode = debug_input == 'y'
        
        # Max workers (multiprocessing)
        if RICH_AVAILABLE:
            use_mp = Confirm.ask("Use multiprocessing for faster processing?", default=config.max_workers is not None)
            if use_mp:
                config.max_workers = IntPrompt.ask("Number of parallel workers", default=4)
            else:
                config.max_workers = None
        else:
            use_mp = input("Use multiprocessing? [y/n, default: n]: ").strip().lower() == 'y'
            if use_mp:
                try:
                    workers = input("Number of workers [default: 4]: ").strip()
                    config.max_workers = int(workers) if workers else 4
                except ValueError:
                    config.max_workers = 4
        
        # Verbose logging
        if RICH_AVAILABLE:
            config.verbose = Confirm.ask("Enable verbose logging?", default=config.verbose)
        else:
            verbose_input = input(f"Enable verbose logging? [y/n, default: {'y' if config.verbose else 'n'}]: ").strip().lower()
            config.verbose = verbose_input == 'y'
    
    if config.run_phase2:
        config = get_transcription_config(config)
    
    if config.run_phase3:
        config = get_alignment_config(config)
    
    return config


def display_config_summary(config: PipelineConfig):
    """Display current configuration"""
    if RICH_AVAILABLE:
        summary = Table(title="Pipeline Configuration", box=None)
        summary.add_column("Setting", style="cyan")
        summary.add_column("Value", style="bold")
        
        # Phase 1 settings
        if config.run_phase1:
            summary.add_row("Phase 1", "[green]Enabled[/green]")
            summary.add_row("├─ Input Directory", config.input_dir)
            summary.add_row("├─ Output Directory", config.output_dir)
            summary.add_row("├─ Enhancement Level", f"{config.enhancement_level} - {ENHANCEMENT_OPTIONS[config.enhancement_level]['name']}")
            summary.add_row("├─ Debug Mode", "Yes" if config.debug_mode else "No")
            summary.add_row("├─ Multiprocessing", f"Yes ({config.max_workers} workers)" if config.max_workers else "No")
            summary.add_row("└─ Verbose Logging", "Yes" if config.verbose else "No")
        else:
            summary.add_row("Phase 1", "[yellow]Disabled[/yellow]")
        
        # Phase 2 settings
        if config.run_phase2:
            summary.add_row("Phase 2", "[green]Enabled[/green]")
            summary.add_row("├─ Transcription Output", config.transcription_output_dir)
            summary.add_row("├─ Language", f"{config.language}")
            summary.add_row("├─ Model", config.model)
            summary.add_row("├─ Diarization", "Yes" if config.use_diarization else "No")
            if config.use_diarization:
                summary.add_row("│  ├─ Precision", config.diarization_precision)
                summary.add_row("│  ├─ Threshold", str(config.diarization_threshold))
                summary.add_row("│  └─ Profiles", str(len(config.speaker_profiles)) if config.speaker_profiles else "None")
            else:
                summary.add_row("│  ├─ Silence Threshold", f"{config.silence_threshold} dB")
                summary.add_row("│  └─ Min Silence Length", f"{config.min_silence_len} ms")
        else:
            summary.add_row("Phase 2", "[yellow]Disabled[/yellow]")
        
        # Phase 3 settings
        if config.run_phase3:
            summary.add_row("Phase 3", "[green]Enabled[/green]")
            summary.add_row("├─ Alignment Language", config.alignment_language.capitalize())
            summary.add_row("├─ Syllabification", "Yes" if config.enable_syllabification else "No")
            if config.enable_syllabification:
                summary.add_row("│  ├─ Dictionary", "Yes" if config.syllabification_use_dictionary else "No")
                if config.syllabification_use_dictionary:
                    summary.add_row("│  │  └─ Path", config.syllabification_dict_dir)
                summary.add_row("│  ├─ Acoustic", "Yes" if config.syllabification_use_acoustic else "No")
                if config.syllabification_use_acoustic:
                    summary.add_row("│  │  ├─ Threshold", str(config.syllabification_threshold))
                    summary.add_row("│  │  ├─ Frame", str(config.syllabification_frame_size))
                    summary.add_row("│  │  └─ Hop", str(config.syllabification_hop_size))
                if config.syllabification_use_dictionary and config.syllabification_use_acoustic:
                    summary.add_row("│  └─ Fallback", config.syllabification_fallback)
            summary.add_row("└─ POS Tagging", "Yes" if config.enable_pos_tagging else "No")
            if config.enable_pos_tagging:
                summary.add_row("   ├─ Model Size", config.pos_model_size)
                summary.add_row("   ├─ Tag Type", config.pos_tag_type)
                summary.add_row("   └─ Auto-download", "Yes" if config.pos_auto_download else "No")
        else:
            summary.add_row("Phase 3", "[yellow]Disabled[/yellow]")
        
        summary.add_row("Mode", "Non-interactive" if config.non_interactive else "Interactive")
        console.print(summary)
    else:
        print("\n=== Pipeline Configuration ===")
        print(f"Phase 1: {'Enabled' if config.run_phase1 else 'Disabled'}")
        if config.run_phase1:
            print(f"  Input Directory: {config.input_dir}")
            print(f"  Output Directory: {config.output_dir}")
            print(f"  Enhancement Level: {config.enhancement_level} - {ENHANCEMENT_OPTIONS[config.enhancement_level]['name']}")
            print(f"  Debug Mode: {'Yes' if config.debug_mode else 'No'}")
            print(f"  Multiprocessing: {'Yes (' + str(config.max_workers) + ' workers)' if config.max_workers else 'No'}")
            print(f"  Verbose Logging: {'Yes' if config.verbose else 'No'}")
        
        print(f"Phase 2: {'Enabled' if config.run_phase2 else 'Disabled'}")
        if config.run_phase2:
            print(f"  Transcription Output: {config.transcription_output_dir}")
            print(f"  Language: {config.language}")
            print(f"  Model: {config.model}")
            print(f"  Diarization: {'Yes' if config.use_diarization else 'No'}")
            if config.use_diarization:
                print(f"    Precision: {config.diarization_precision}")
                print(f"    Threshold: {config.diarization_threshold}")
                print(f"    Profiles: {', '.join(config.speaker_profiles) if config.speaker_profiles else 'None'}")
            else:
                print(f"    Silence Threshold: {config.silence_threshold} dB")
                print(f"    Min Silence Length: {config.min_silence_len} ms")
        
        print(f"Phase 3: {'Enabled' if config.run_phase3 else 'Disabled'}")
        if config.run_phase3:
            print(f"  Alignment Language: {config.alignment_language.capitalize()}")
            print(f"  Syllabification: {'Yes' if config.enable_syllabification else 'No'}")
            if config.enable_syllabification:
                print(f"    Dictionary: {'Yes' if config.syllabification_use_dictionary else 'No'}")
                if config.syllabification_use_dictionary:
                    print(f"      Path: {config.syllabification_dict_dir}")
                print(f"    Acoustic: {'Yes' if config.syllabification_use_acoustic else 'No'}")
                if config.syllabification_use_acoustic:
                    print(f"      Threshold: {config.syllabification_threshold}")
                    print(f"      Frame: {config.syllabification_frame_size}")
                    print(f"      Hop: {config.syllabification_hop_size}")
                if config.syllabification_use_dictionary and config.syllabification_use_acoustic:
                    print(f"    Fallback: {config.syllabification_fallback}")
            print(f"  POS Tagging: {'Yes' if config.enable_pos_tagging else 'No'}")
            if config.enable_pos_tagging:
                print(f"    Model Size: {config.pos_model_size}")
                print(f"    Tag Type: {config.pos_tag_type}")
                print(f"    Auto-download: {'Yes' if config.pos_auto_download else 'No'}")
        
        print(f"Mode: {'Non-interactive' if config.non_interactive else 'Interactive'}")


# ============================================================================
# PHASE EXECUTION
# ============================================================================

def run_phase(phase_num: int, config: PipelineConfig) -> bool:
    """
    Run a specific phase of the pipeline by calling the corresponding script.
    Returns True if successful, False otherwise.
    """
    if phase_num not in PHASE_SCRIPTS:
        logger.error(f"Unknown phase: {phase_num}")
        return False
    
    script_name = PHASE_SCRIPTS[phase_num]
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    # Build command arguments based on phase
    cmd = [
        sys.executable,  # Use the same Python interpreter
        str(script_path)
    ]
    
    if phase_num == 1:
        # Phase 1: Preprocessing arguments - EXACTLY as before
        cmd.extend([
            "--input-dir", config.input_dir,
            "--output-dir", config.output_dir,
            "--enhancement-level", str(config.enhancement_level)
        ])
        
        if config.debug_mode:
            cmd.append("--debug")
        
        if config.verbose:
            cmd.append("--verbose")
        
        if config.max_workers:
            cmd.extend(["--max-workers", str(config.max_workers)])
        
        # NO --batch-mode flag here - Phase 1 doesn't have it!
        
    elif phase_num == 2:
        # Phase 2: Transcription arguments
        cmd.extend([
            "--input-dir", config.output_dir,  # Use preprocessed audio from phase 1
            "--output-dir", config.transcription_output_dir,
            "--language", config.language,
            "--model", config.model,
            "--batch-mode"  # Only Phase 2 gets batch-mode
        ])
        
        if config.use_diarization:
            cmd.append("--use-diarization")
            cmd.extend(["--diarization-precision", config.diarization_precision])
            cmd.extend(["--diarization-threshold", str(config.diarization_threshold)])
            
            if config.speaker_profiles:
                cmd.extend(["--speaker-profiles"] + config.speaker_profiles)
        else:
            cmd.append("--no-diarization")
            cmd.extend(["--silence-threshold", str(config.silence_threshold)])
            cmd.extend(["--min-silence-len", str(config.min_silence_len)])
        
        if config.verbose:
            cmd.append("--verbose")
    
    elif phase_num == 3:
        # Phase 3: Alignment & Annotation arguments
        cmd.extend([
            "--textgrid-dir", str(Path(config.transcription_output_dir) / "textgrids"),
            "--audio-dir", config.output_dir,
            "--output-dir", "./aligned_textgrids",
            "--language", config.alignment_language
        ])
        
        # Syllabification arguments
        if config.enable_syllabification:
            cmd.append("--enable-syllabification")
            cmd.extend(["--syllabification-dict-dir", config.syllabification_dict_dir])
            
            if config.syllabification_use_dictionary:
                cmd.append("--syllabification-use-dictionary")
            else:
                cmd.append("--syllabification-no-dictionary")
            
            if config.syllabification_use_acoustic:
                cmd.append("--syllabification-use-acoustic")
                cmd.extend(["--syllabification-threshold", str(config.syllabification_threshold)])
                cmd.extend(["--syllabification-frame-size", str(config.syllabification_frame_size)])
                cmd.extend(["--syllabification-hop-size", str(config.syllabification_hop_size)])
            else:
                cmd.append("--syllabification-no-acoustic")
            
            cmd.extend(["--syllabification-fallback", config.syllabification_fallback])
        
        # POS tagging arguments
        if config.enable_pos_tagging:
            cmd.append("--enable-pos-tagging")
            cmd.extend(["--pos-model-size", config.pos_model_size])
            cmd.extend(["--pos-tag-type", config.pos_tag_type])
            
            if config.pos_auto_download:
                cmd.append("--pos-auto-download")
            else:
                cmd.append("--pos-no-auto-download")
        
        # Non-interactive mode
        cmd.append("--non-interactive")
        
        if config.verbose:
            cmd.append("--verbose")
    
    # Log the command
    logger.info(f"Running phase {phase_num}: {' '.join(cmd)}")
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]▶ Executing Phase {phase_num}: {script_name}[/bold cyan]")
        if phase_num == 1 and config.enhancement_level == 3:
            console.print("[yellow]Note: Experimental level will prompt for SpeechBrain model selection if needed[/yellow]")
    
    # Run the subprocess
    start_time = time.time()
    try:
        # Capture output to show progress
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False  # Let output go directly to console
        )
        elapsed = time.time() - start_time
        logger.info(f"Phase {phase_num} completed in {elapsed:.2f}s")
        
        if RICH_AVAILABLE:
            console.print(f"[bold green]Phase {phase_num} completed successfully[/bold green]")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"Phase {phase_num} failed after {elapsed:.2f}s with exit code {e.returncode}")
        
        if RICH_AVAILABLE:
            console.print(f"[bold red]Phase {phase_num} failed[/bold red]")
        
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Phase {phase_num} encountered an error: {e}")
        
        if RICH_AVAILABLE:
            console.print(f"[bold red]Phase {phase_num} encountered an error[/bold red]")
        
        return False


# ============================================================================
# MAIN PIPELINE CONTROLLER
# ============================================================================

def parse_arguments():
    """Parse command line arguments for the pipeline"""
    parser = argparse.ArgumentParser(description='ASR Pipeline Master Controller')
    
    parser.add_argument('--config', '-c', type=str, default='./pipeline_config.json',
                        help='Configuration file path')
    
    # Phase selection
    parser.add_argument('--no-phase1', action='store_true',
                        help='Skip phase 1 (preprocessing)')
    parser.add_argument('--no-phase2', action='store_true',
                        help='Skip phase 2 (transcription)')
    parser.add_argument('--no-phase3', action='store_true',
                        help='Skip phase 3 (alignment and annotation)')
    
    # Phase 1 arguments
    parser.add_argument('--input-dir', '-i', type=str,
                        help='Input directory for original audio (overrides config)')
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Output directory for preprocessed audio (overrides config)')
    parser.add_argument('--enhancement-level', '-e', type=int, choices=[0, 1, 2, 3],
                        help='Enhancement level (0=None, 1=Gentle, 2=Aggressive, 3=Experimental)')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--max-workers', '-w', type=int,
                        help='Maximum number of parallel workers for preprocessing')
    
    # Phase 2 arguments
    parser.add_argument('--transcription-output', type=str,
                        help='Output directory for transcriptions (overrides config)')
    parser.add_argument('--language', '-l', type=str, 
                        choices=['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'ru', 'ja', 'zh', 'ar', 'hi', 'ko', 'auto'],
                        help='Language code for transcription')
    parser.add_argument('--model', '-m', type=str, 
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size')
    parser.add_argument('--use-diarization', action='store_true',
                        help='Enable speaker diarization')
    parser.add_argument('--no-diarization', action='store_true',
                        help='Disable speaker diarization')
    parser.add_argument('--diarization-precision', type=str, 
                        choices=['low', 'medium', 'high'],
                        help='Diarization precision level')
    parser.add_argument('--diarization-threshold', type=float,
                        help='Diarization matching threshold (0.0-1.0)')
    parser.add_argument('--speaker-profiles', type=str, nargs='+',
                        help='Speaker profile names to use')
    parser.add_argument('--silence-threshold', type=int,
                        help='Silence threshold in dB for segmentation (when diarization disabled)')
    parser.add_argument('--min-silence-len', type=int,
                        help='Minimum silence length in ms for segmentation (when diarization disabled)')
    
    # Phase 3 arguments
    parser.add_argument('--alignment-language', type=str,
                        choices=['english', 'spanish', 'french', 'german', 'italian', 
                                'portuguese', 'dutch', 'russian', 'mandarin', 'japanese', 
                                'korean', 'arabic'],
                        help='Language for MFA alignment')
    
    # Syllabification arguments
    parser.add_argument('--enable-syllabification', action='store_true',
                        help='Enable syllabification after alignment')
    parser.add_argument('--disable-syllabification', dest='enable_syllabification', 
                        action='store_false', help='Disable syllabification')
    parser.add_argument('--syllabification-dict-dir', type=str,
                        help='Directory for syllable dictionaries')
    parser.add_argument('--syllabification-use-dictionary', action='store_true',
                        help='Use dictionary lookup for syllabification')
    parser.add_argument('--syllabification-no-dictionary', dest='syllabification_use_dictionary', 
                        action='store_false', help='Disable dictionary lookup for syllabification')
    parser.add_argument('--syllabification-use-acoustic', action='store_true',
                        help='Use acoustic analysis for syllabification')
    parser.add_argument('--syllabification-no-acoustic', dest='syllabification_use_acoustic', 
                        action='store_false', help='Disable acoustic analysis for syllabification')
    parser.add_argument('--syllabification-threshold', type=float,
                        help='Peak detection threshold for acoustic syllabification')
    parser.add_argument('--syllabification-frame-size', type=int,
                        help='Frame size for acoustic analysis')
    parser.add_argument('--syllabification-hop-size', type=int,
                        help='Hop size for acoustic analysis')
    parser.add_argument('--syllabification-fallback', type=str,
                        choices=['dictionary_first', 'acoustic_first'],
                        help='Fallback order for syllabification methods')
    
    # POS tagging arguments
    parser.add_argument('--enable-pos-tagging', action='store_true',
                        help='Enable POS tagging after alignment')
    parser.add_argument('--disable-pos-tagging', dest='enable_pos_tagging', 
                        action='store_false', help='Disable POS tagging')
    parser.add_argument('--pos-model-size', type=str, choices=['sm', 'md', 'lg'],
                        help='Size of spaCy model for POS tagging')
    parser.add_argument('--pos-tag-type', type=str, choices=['universal', 'fine'],
                        help='Type of POS tags to use')
    parser.add_argument('--pos-auto-download', action='store_true',
                        help='Auto-download missing spaCy models')
    parser.add_argument('--pos-no-auto-download', dest='pos_auto_download', 
                        action='store_false', help='Disable auto-download of spaCy models')
    
    # General
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--non-interactive', action='store_true',
                        help='Run in fully non-interactive mode (use defaults or command line args)')
    
    return parser.parse_args()


def main():
    """Main pipeline controller"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize config
    config = PipelineConfig()
    config.non_interactive = args.non_interactive
    
    # Load config file if it exists
    if os.path.exists(args.config):
        config.load(args.config)
    
    # Override with command line arguments if provided
    # Phase selection
    if args.no_phase1:
        config.run_phase1 = False
    if args.no_phase2:
        config.run_phase2 = False
    if args.no_phase3:
        config.run_phase3 = False
    
    # Phase 1 overrides
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.enhancement_level is not None:
        config.enhancement_level = args.enhancement_level
    if args.debug:
        config.debug_mode = True
    if args.max_workers:
        config.max_workers = args.max_workers
    
    # Phase 2 overrides
    if args.transcription_output:
        config.transcription_output_dir = args.transcription_output
    if args.language:
        config.language = args.language
    if args.model:
        config.model = args.model
    
    # Diarization settings
    if args.use_diarization:
        config.use_diarization = True
    if args.no_diarization:
        config.use_diarization = False
    if args.diarization_precision:
        config.diarization_precision = args.diarization_precision
    if args.diarization_threshold is not None:
        config.diarization_threshold = args.diarization_threshold
    if args.speaker_profiles:
        config.speaker_profiles = args.speaker_profiles
    
    # Segmentation settings (when diarization disabled)
    if args.silence_threshold is not None:
        config.silence_threshold = args.silence_threshold
    if args.min_silence_len is not None:
        config.min_silence_len = args.min_silence_len
    
    # Phase 3 overrides
    if args.alignment_language:
        config.alignment_language = args.alignment_language
    
    # Syllabification overrides
    if hasattr(args, 'enable_syllabification') and args.enable_syllabification is not None:
        config.enable_syllabification = args.enable_syllabification
    if args.syllabification_dict_dir:
        config.syllabification_dict_dir = args.syllabification_dict_dir
    if hasattr(args, 'syllabification_use_dictionary') and args.syllabification_use_dictionary is not None:
        config.syllabification_use_dictionary = args.syllabification_use_dictionary
    if hasattr(args, 'syllabification_use_acoustic') and args.syllabification_use_acoustic is not None:
        config.syllabification_use_acoustic = args.syllabification_use_acoustic
    if args.syllabification_threshold is not None:
        config.syllabification_threshold = args.syllabification_threshold
    if args.syllabification_frame_size is not None:
        config.syllabification_frame_size = args.syllabification_frame_size
    if args.syllabification_hop_size is not None:
        config.syllabification_hop_size = args.syllabification_hop_size
    if args.syllabification_fallback:
        config.syllabification_fallback = args.syllabification_fallback
    
    # POS tagging overrides
    if hasattr(args, 'enable_pos_tagging') and args.enable_pos_tagging is not None:
        config.enable_pos_tagging = args.enable_pos_tagging
    if args.pos_model_size:
        config.pos_model_size = args.pos_model_size
    if args.pos_tag_type:
        config.pos_tag_type = args.pos_tag_type
    if hasattr(args, 'pos_auto_download') and args.pos_auto_download is not None:
        config.pos_auto_download = args.pos_auto_download
    
    # General
    if args.verbose:
        config.verbose = True
    
    # Determine if we're in interactive mode
    interactive_mode = not config.non_interactive
    
    # Interactive configuration if needed
    if interactive_mode:
        # Show current config
        console.print("\n[bold]Current Configuration:[/bold]")
        display_config_summary(config)
        
        # Ask if user wants to modify
        if RICH_AVAILABLE:
            modify = Confirm.ask("\nModify configuration?", default=False)
            if modify:
                config = get_user_config(config)
        else:
            modify = input("\nModify configuration? [y/N]: ").strip().lower() == 'y'
            if modify:
                config = get_user_config(config)
        
        # Save configuration
        if RICH_AVAILABLE:
            if Confirm.ask("Save this configuration for next time?", default=True):
                config.save(args.config)
        else:
            save_input = input("Save configuration? [y/n, default: y]: ").strip().lower()
            if save_input != 'n':
                config.save(args.config)
    else:
        # Non-interactive mode - just show the config
        logger.info("Running in non-interactive mode")
        display_config_summary(config)
        
        # Auto-save in non-interactive mode
        config.save(args.config)
    
    # Confirm before proceeding (skip in non-interactive mode)
    if interactive_mode:
        if RICH_AVAILABLE:
            proceed = Confirm.ask("\nProceed with pipeline execution?", default=True)
            if not proceed:
                console.print("[yellow]Pipeline execution cancelled.[/yellow]")
                return
        else:
            proceed = input("\nProceed with pipeline execution? [Y/n]: ").strip().lower()
            if proceed == 'n':
                print("Pipeline execution cancelled.")
                return
    
    # Run phases
    console.print("\n[bold cyan]════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]      STARTING PIPELINE EXECUTION       [/bold cyan]")
    console.print("[bold cyan]════════════════════════════════════════[/bold cyan]\n")
    
    start_time = time.time()
    phase_results = {}
    
    # Phase 1: Preprocessing
    if config.run_phase1:
        phase_results[1] = run_phase(1, config)
    else:
        logger.info("Skipping phase 1")
        phase_results[1] = True
    
    # Phase 2: Transcription
    if config.run_phase2 and phase_results.get(1, False):
        # Ensure phase 1 output directory exists if phase 1 was skipped
        if not config.run_phase1:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        phase_results[2] = run_phase(2, config)
    elif config.run_phase2:
        logger.error("Cannot run phase 2: phase 1 failed or was skipped")
        phase_results[2] = False
    
    # Phase 3: Alignment & Annotation
    if config.run_phase3 and phase_results.get(2, False):
        # Ensure Phase 2 output exists
        textgrid_dir = Path(config.transcription_output_dir) / "textgrids"
        if textgrid_dir.exists():
            phase_results[3] = run_phase(3, config)
        else:
            logger.error(f"Cannot run phase 3: TextGrid directory not found at {textgrid_dir}")
            phase_results[3] = False
    elif config.run_phase3:
        logger.error("Cannot run phase 3: phase 2 failed or was skipped")
        phase_results[3] = False
    
    total_elapsed = time.time() - start_time
    
    # Display final summary
    console.print("\n[bold cyan]════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]         PIPELINE EXECUTION SUMMARY      [/bold cyan]")
    console.print("[bold cyan]════════════════════════════════════════[/bold cyan]\n")
    
    if RICH_AVAILABLE:
        summary = Table(title="Phase Results", box=None)
        summary.add_column("Phase", style="cyan")
        summary.add_column("Script", style="white")
        summary.add_column("Status", style="bold")
        
        for phase_num in sorted(PHASE_SCRIPTS.keys()):
            status = phase_results.get(phase_num)
            script = PHASE_SCRIPTS[phase_num]
            
            if status is True:
                status_text = "[green]Success[/green]"
            elif status is False:
                status_text = "[red]Failed[/red]"
            else:
                status_text = "[yellow]Skipped[/yellow]"
            
            # Only show phases that were configured to run
            should_run = getattr(config, f"run_phase{phase_num}", True)
            if should_run:
                summary.add_row(f"Phase {phase_num}", script, status_text)
        
        console.print(summary)
        console.print(f"\n[bold]Total execution time:[/bold] {total_elapsed:.2f} seconds")
        
        # Check if all enabled phases succeeded
        all_success = True
        for phase_num in PHASE_SCRIPTS.keys():
            if getattr(config, f"run_phase{phase_num}", True):
                if not phase_results.get(phase_num, False):
                    all_success = False
                    break
        
        if all_success:
            console.print("\n[bold green]Pipeline completed successfully![/bold green]")
        else:
            console.print("\n[bold yellow]Pipeline completed with issues[/bold yellow]")
    else:
        print("\n=== Phase Results ===")
        for phase_num in sorted(PHASE_SCRIPTS.keys()):
            status = phase_results.get(phase_num)
            script = PHASE_SCRIPTS[phase_num]
            status_text = "Success" if status is True else "Failed" if status is False else "Skipped"
            
            should_run = getattr(config, f"run_phase{phase_num}", True)
            if should_run:
                print(f"Phase {phase_num} ({script}): {status_text}")
        
        print(f"\nTotal execution time: {total_elapsed:.2f} seconds")
    
    # Save final configuration
    config.save(args.config)
    logger.info(f"Final configuration saved to {args.config}")
    
    # Determine exit code
    exit_code = 0
    for phase_num in PHASE_SCRIPTS.keys():
        if getattr(config, f"run_phase{phase_num}", True):
            if not phase_results.get(phase_num, False):
                exit_code = 1
                break
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
