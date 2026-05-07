"""
03_annotate.py - Alignment & Annotation Hub
@author: taylosh
Created on 15 Nov 2025
Last edited on 6 May 2026

Main alignment engine for the overhauled ASR pipeline.
- Automatically generates MFA corpus from Phase 2 TextGrids.
- Uses C-accelerated audio_segment_engine for high-speed chunking (with Python fallback).
- Performs Segment-to-Phoneme alignment via Montreal Forced Aligner.
- Integrated subscripts: libs/syllabify.py and libs/tag.py.
- Preservation of original filenames with "_aligned" suffix.
"""
"""
03_annotate.py - Alignment & Annotation Hub
@author: taylosh
Created on Nov 15 2025
Last edited on Mar 30 2026

Main alignment engine for the overhauled ASR pipeline.
- Automatically generates MFA corpus from Phase 2 TextGrids.
- Uses C-accelerated audio_segment_engine for high-speed chunking (with Python fallback).
- Performs Segment-to-Phoneme alignment via Montreal Forced Aligner.
- Integrated subscripts: libs/syllabify.py, libs/tag.py, and libs/intonalyze_A.py.
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

def normalize_mfa_model_name(model_input: str) -> str:
    """
    Normalize MFA model name.
    If it already ends with '_mfa', return as-is.
    Otherwise, append '_mfa'.
    """
    model_input = model_input.strip()
    if model_input.endswith('_mfa'):
        return model_input
    else:
        return f"{model_input}_mfa"
    
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
    parser.add_argument('--syllabification-use-rules', action='store_true', default=True,
                    help='Enable rule-based proofreading for syllabification')
    parser.add_argument('--syllabification-no-rules', dest='syllabification_use_rules', 
                        action='store_false', help='Disable rule-based proofreading')
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
    parser.add_argument('--enable-intonation', action='store_true',
                        help='Enable intonation analysis (Break Index assignment) - Spanish only')
    
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
    
    # Intonation arguments
    parser.add_argument('--intonation-clitics-path', type=str, default=None,
                        help='Path to custom clitics CSV file (Spanish only)')
    
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
    
    # Normalize the model name (don't double-add _mfa)
    model_name = normalize_mfa_model_name(language)
    
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
    """Display numbered language menu and get user selection with full model management."""
    
    # First, get list of available MFA models on the system
    available_models = []
    try:
        result = subprocess.run(["mfa", "model", "list", "acoustic"], 
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout:
            available_models = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    except Exception as e:
        logger.debug(f"Could not list MFA models: {e}")
    
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
        12: {"name": "Arabic", "code": "arabic"},
        13: {"name": "Manual entry (enter MFA model name)", "code": "manual"},
        14: {"name": "Browse available MFA models", "code": "browse"},
        15: {"name": "Enter custom model path", "code": "custom_path"}
    }
    
    if RICH_AVAILABLE:
        # Rich menu display
        table = Table(title="MFA Model Selection")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Language / Model Source", style="green")
        table.add_column("Description", style="dim")
        
        for num, lang_info in LANGUAGE_OPTIONS.items():
            if num == 13:
                table.add_row(str(num), lang_info["name"], "type any valid MFA model name")
            elif num == 14:
                table.add_row(str(num), lang_info["name"], "select from models on your system")
            elif num == 15:
                table.add_row(str(num), lang_info["name"], "use full filesystem path to custom model")
            else:
                table.add_row(str(num), lang_info["name"], lang_info["code"])
        
        console.print(table)
        choice = IntPrompt.ask("Select model option", default=1)
    else:
        # Fallback console menu
        print("\nMFA Model Selection:")
        print("-" * 60)
        for num, lang_info in LANGUAGE_OPTIONS.items():
            if num == 13:
                print(f"  {num}. {lang_info['name']}")
            elif num == 14:
                print(f"  {num}. {lang_info['name']}")
            elif num == 15:
                print(f"  {num}. {lang_info['name']}")
            else:
                print(f"  {num}. {lang_info['name']} ({lang_info['code']})")
        print("-" * 60)
        
        try:
            choice = int(input("\nSelect model option (default: 1 - English): ").strip() or "1")
        except ValueError:
            choice = 1
    
    # ============================================
    # OPTION 13: Manual entry (model name)
    # ============================================
    if choice == 13:
        if RICH_AVAILABLE:
            console.print("\n[yellow]Manual Model Name Entry[/yellow]")
            console.print("[dim]Enter any valid MFA model name. Examples:[/dim]")
            console.print("  - [cyan]english_mfa[/cyan] (standard English model)")
            console.print("  - [cyan]spanish_mfa[/cyan] (standard Spanish model)")
            console.print("  - [cyan]french_mfa[/cyan] (standard French model)")
            console.print("  - [cyan]mandarin_mfa[/cyan] (Mandarin Chinese model)")
            console.print("\n[dim]Custom models you've downloaded or trained:[/dim]")
            console.print("  - [cyan]my_custom_english_model[/cyan]")
            console.print("  - [cyan]librispeech_english[/cyan]")
            
            manual_code = Prompt.ask("Enter MFA model name")
        else:
            print("\nManual Model Name Entry")
            print("Examples: english_mfa, spanish_mfa, my_custom_model")
            manual_code = input("Enter MFA model name: ").strip()
        
        if manual_code:
            if RICH_AVAILABLE:
                console.print(f"[green]Using manual model: {manual_code}[/green]")
                console.print(f"[dim]MFA will look for this model in its default models directory[/dim]")
            else:
                print(f"Using manual model: {manual_code}")
            return manual_code  # Return as-is, will be normalized later
        else:
            if RICH_AVAILABLE:
                console.print("[yellow]No model entered, falling back to English[/yellow]")
            else:
                print("No model entered, falling back to English")
            return "english"  # Normalized to english_mfa later
    
    # ============================================
    # OPTION 14: Browse available MFA models
    # ============================================
    if choice == 14:
        if not available_models:
            if RICH_AVAILABLE:
                console.print("[red]No MFA models found on this system.[/red]")
                console.print("[yellow]You can download models using:[/yellow]")
                console.print("  [cyan]mfa model download acoustic english_mfa[/cyan]")
                console.print("  [cyan]mfa model download dictionary english_mfa[/cyan]")
                console.print("\nOr use manual entry (option 13) to specify a model name to download.")
            else:
                print("No MFA models found on this system.")
                print("Download models with: mfa model download acoustic english_mfa")
            
            # Offer to download a standard model
            if RICH_AVAILABLE:
                download_choice = Confirm.ask("Download a standard model now?", default=False)
            else:
                download_choice = input("Download a standard model now? (y/N): ").lower() == 'y'
            
            if download_choice:
                if RICH_AVAILABLE:
                    standard_model = Prompt.ask("Enter model name to download", default="english_mfa")
                else:
                    standard_model = input("Enter model name to download [english_mfa]: ").strip() or "english_mfa"
                
                try:
                    console.print(f"[dim]Downloading {standard_model}...[/dim]" if RICH_AVAILABLE else f"Downloading {standard_model}...")
                    subprocess.run(["mfa", "model", "download", "acoustic", standard_model], check=True)
                    subprocess.run(["mfa", "model", "download", "dictionary", standard_model], check=True)
                    console.print(f"[green]Successfully downloaded {standard_model}[/green]" if RICH_AVAILABLE else f"Successfully downloaded {standard_model}")
                    return standard_model  # Return as-is
                except subprocess.CalledProcessError as e:
                    console.print(f"[red]Failed to download {standard_model}[/red]" if RICH_AVAILABLE else f"Failed to download {standard_model}")
                    return "english"
            else:
                return "english"
        
        # Display available models
        if RICH_AVAILABLE:
            model_table = Table(title="Available MFA Models on Your System")
            model_table.add_column("#", style="cyan", no_wrap=True)
            model_table.add_column("Model Name", style="green")
            model_table.add_column("Type", style="dim")
            
            for idx, model in enumerate(available_models, 1):
                # Try to guess if it's a standard model
                if model.endswith("_mfa"):
                    model_type = "standard"
                else:
                    model_type = "custom"
                model_table.add_row(str(idx), model, model_type)
            
            console.print(model_table)
            console.print("[dim]Enter the number of the model to use, or type a custom name[/dim]")
            model_choice = Prompt.ask("Select model", default="1")
        else:
            print("\nAvailable MFA Models on Your System:")
            for idx, model in enumerate(available_models, 1):
                print(f"  {idx}. {model}")
            model_choice = input("\nSelect model (number) or type custom name [1]: ").strip() or "1"
        
        # Check if input is a number (select from list) or custom name
        try:
            idx = int(model_choice) - 1
            if 0 <= idx < len(available_models):
                selected_model = available_models[idx]
                if RICH_AVAILABLE:
                    console.print(f"[green]Selected: {selected_model}[/green]")
                else:
                    print(f"Selected: {selected_model}")
                return selected_model  # Return as-is
            else:
                raise ValueError("Index out of range")
        except ValueError:
            # Not a number or invalid index - treat as custom model name
            if model_choice.strip():
                if RICH_AVAILABLE:
                    console.print(f"[green]Using custom model name: {model_choice}[/green]")
                else:
                    print(f"Using custom model name: {model_choice}")
                return model_choice.strip()  # Return as-is
            else:
                if RICH_AVAILABLE:
                    console.print("[yellow]No model selected, falling back to English[/yellow]")
                else:
                    print("No model selected, falling back to English")
                return "english"  # Normalized later
    
    # ============================================
    # OPTION 15: Enter custom model path
    # ============================================
    if choice == 15:
        if RICH_AVAILABLE:
            console.print("\n[yellow]Custom Model Path Entry[/yellow]")
            console.print("[dim]Enter the full filesystem path to your custom MFA model.[/dim]")
            console.print("[dim]MFA expects a directory containing:[/dim]")
            console.print("  - [cyan].pt[/cyan] or [cyan].pt.ckpt[/cyan] acoustic model file")
            console.print("  - [cyan].dict[/cyan] dictionary file")
            console.print("\n[dim]Examples:[/dim]")
            console.print("  - [cyan]/home/user/models/my_spanish_model[/cyan]")
            console.print("  - [cyan]C:\\Users\\Name\\mfa_models\\custom_english[/cyan]")
            console.print("  - [cyan]./models/mfa/catalan_model[/cyan]")
            
            custom_path = Prompt.ask("Enter full path to custom model directory")
        else:
            print("\nCustom Model Path Entry")
            print("Enter the full filesystem path to your custom MFA model directory.")
            print("Examples: /home/user/models/my_model, ./models/custom_model")
            custom_path = input("Enter path: ").strip()
        
        if custom_path:
            # Validate the path exists
            path_obj = Path(custom_path)
            if path_obj.exists():
                if RICH_AVAILABLE:
                    console.print(f"[green]Using custom model path: {custom_path}[/green]")
                    console.print(f"[dim]Note: MFA will use this exact path. Make sure the model files are valid.[/dim]")
                else:
                    print(f"Using custom model path: {custom_path}")
                return str(custom_path)  # Return path as-is
            else:
                if RICH_AVAILABLE:
                    console.print(f"[yellow]Warning: Path '{custom_path}' does not exist.[/yellow]")
                    use_anyway = Confirm.ask("Use it anyway? (MFA will fail if invalid)", default=False)
                else:
                    use_anyway = input(f"Warning: Path '{custom_path}' does not exist. Use anyway? (y/N): ").lower() == 'y'
                
                if use_anyway:
                    return str(custom_path)
                else:
                    if RICH_AVAILABLE:
                        console.print("[yellow]Falling back to English[/yellow]")
                    else:
                        print("Falling back to English")
                    return "english"
        else:
            if RICH_AVAILABLE:
                console.print("[yellow]No path entered, falling back to English[/yellow]")
            else:
                print("No path entered, falling back to English")
            return "english"
    
    # ============================================
    # STANDARD OPTIONS (1-12)
    # ============================================
    if choice not in LANGUAGE_OPTIONS:
        if RICH_AVAILABLE:
            console.print(f"[yellow]Invalid choice {choice}, using English[/yellow]")
        else:
            print(f"Invalid choice {choice}, using English")
        return "english"  # Return base name, will be normalized to english_mfa
    
    selected = LANGUAGE_OPTIONS[choice]
    model_base = selected["code"]  # This is "spanish", not "spanish_mfa"
    
    if RICH_AVAILABLE:
        console.print(f"[green]Selected: {selected['name']} → {model_base}_mfa[/green]")
    else:
        print(f"Selected: {selected['name']} → {model_base}_mfa")
    
    return model_base  # Return just "spanish" - normalization will add _mfa once

# ============================================================================
# ENHANCED CONFIGURATION PROMPTS
# ============================================================================

def configure_syllabification() -> Dict[str, Any]:
    """Present syllabification configuration options to user."""
    config = {
        'enabled': False,
        'use_rules': True,                    # NEW - rule-based proofreading
        'dictionary_path': "./models/syllable_dicts",
        'acoustic_threshold': 0.3,
        'frame_size': 256,
        'hop_size': 128
    }
    
    if not RICH_AVAILABLE:
        # Fallback text prompts
        print("\n--- SYLLABIFICATION CONFIGURATION ---")
        enable = input("Enable syllabification? (y/N): ").lower() == 'y'
        if not enable:
            return {'enabled': False}
        
        config['enabled'] = True
        
        # Dictionary path (optional to change)
        dict_path = input(f"Dictionary path [{config['dictionary_path']}]: ").strip()
        if dict_path:
            config['dictionary_path'] = dict_path
        
        # Rule-based proofreading toggle
        use_rules = input("Enable rule-based proofreading? (Y/n): ").lower() != 'n'
        config['use_rules'] = use_rules
        
        # Acoustic parameters
        print("\n--- Acoustic Analysis Parameters ---")
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
        
        return config
    
    # Rich interactive configuration
    console.print(Panel("[bold cyan]Syllabification Configuration[/bold cyan]"))
    
    config['enabled'] = Confirm.ask("Enable syllabification?", default=False)
    if not config['enabled']:
        return {'enabled': False}
    
    # Dictionary path (still configurable)
    config['dictionary_path'] = Prompt.ask(
        "Dictionary path", 
        default=config['dictionary_path']
    )
    
    # Rule-based proofreading toggle
    console.print("\n[dim]Rule-based proofreading applies linguistic rules to correct syllable errors[/dim]")
    config['use_rules'] = Confirm.ask("Enable rule-based proofreading?", default=True)
    
    # Acoustic parameters
    console.print("\n[bold]Acoustic Analysis Parameters[/bold]")
    console.print("[dim]Used to refine syllable timing boundaries[/dim]")
    
    config['acoustic_threshold'] = FloatPrompt.ask(
        "Peak detection threshold (0.0-1.0, lower = more sensitive)", 
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
    
    # Display summary
    console.print("\n[green]Syllabification Configuration Summary:[/green]")
    console.print(f"  Dictionary path: {config['dictionary_path']}")
    console.print(f"  Rule-based proofreading: {'enabled' if config['use_rules'] else 'disabled'}")
    console.print(f"  Acoustic threshold: {config['acoustic_threshold']}")
    console.print(f"  Frame size: {config['frame_size']}")
    console.print(f"  Hop size: {config['hop_size']}")
    console.print("\n[dim]Note: Dictionary lookup + acoustic refinement are always used in pipeline order[/dim]")
    
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

def configure_intonation(language: str, syllabification_enabled: bool = False) -> Dict[str, Any]:
    """
    Present intonation configuration options to user.
    Only offered if language is Spanish AND syllabification is enabled.
    """
    config = {
        'enabled': False,
        'language': language,
        'clitics_path': None
    }
    
    # Only offer if language is Spanish AND syllabification is enabled
    if language != 'spanish':
        if RICH_AVAILABLE:
            console.print(f"[dim]Intonation analysis is only available for Spanish (current language: {language}). Skipping.[/dim]")
        else:
            print(f"Intonation analysis is only available for Spanish (current language: {language}). Skipping.")
        return config
    
    if not syllabification_enabled:
        if RICH_AVAILABLE:
            console.print(f"[dim]Intonation analysis requires syllabification to be enabled. Skipping.[/dim]")
        else:
            print(f"Intonation analysis requires syllabification to be enabled. Skipping.")
        return config
    
    if not RICH_AVAILABLE:
        # Fallback text prompts
        print("\n--- INTONATION ANALYSIS CONFIGURATION ---")
        enable = input("Enable intonation analysis (Break Index assignment)? (y/N): ").lower() == 'y'
        if not enable:
            return {'enabled': False}
        
        config['enabled'] = True
        
        clitics_path = input("Custom clitics file path (leave empty for default): ").strip()
        if clitics_path:
            config['clitics_path'] = clitics_path
        
        return config
    
    # Rich interactive configuration
    console.print(Panel("[bold cyan]Intonation Analysis Configuration[/bold cyan]"))
    console.print("[dim]Break Index assignment for Spanish prosodic analysis[/dim]")
    console.print("[dim]Note: This requires syllabification to provide syllable boundaries[/dim]")
    
    config['enabled'] = Confirm.ask("Enable intonation analysis (Break Index assignment)?", default=False)
    if not config['enabled']:
        return {'enabled': False}
    
    config['clitics_path'] = Prompt.ask(
        "Custom clitics file path (leave empty for default)",
        default=""
    )
    if not config['clitics_path']:
        config['clitics_path'] = None
    
    # Display summary
    console.print("\n[green]Intonation Configuration Summary:[/green]")
    console.print(f"  Language: Spanish")
    if config['clitics_path']:
        console.print(f"  Custom clitics: {config['clitics_path']}")
    else:
        console.print(f"  Clitics: Default (models/intonation/clitics.csv)")
    
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
    # Normalize the model name (don't double-add _mfa)
    model_name = normalize_mfa_model_name(language)
    
    # Check if model exists first
    if not check_mfa_model(model_name, auto_download):
        logger.error(f"MFA model for {model_name} not available")
        console.print(f"[red]MFA model for {model_name} not available. Run 'mfa model download {model_name}' to install.[/red]" if RICH_AVAILABLE else f"MFA model for {model_name} not available")
        return False
    
    # Check if corpus directory has files
    wav_files = list(corpus_dir.glob("*.wav"))
    if not wav_files:
        logger.error(f"No WAV files found in {corpus_dir}")
        return False
    
    console.print(f"Found {len(wav_files)} chunks to align" if RICH_AVAILABLE else f"Found {len(wav_files)} chunks to align")
    
    try:
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
            relative_dir = Path(".")  # Define relative_dir as current directory
            output_subdir = output_dir
        
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Clean filename: remove '_preprocessed' if present, then add '_aligned'
        clean_stem = family.audio_path.stem
        if clean_stem.endswith('_preprocessed'):
            clean_stem = clean_stem[:-13]  # Remove '_preprocessed' (13 characters)
        
        # ============================================
        # SAVE TO aligned_textgrids/ with _aligned suffix
        # ============================================
        aligned_output_path = output_subdir / f"{clean_stem}_aligned.TextGrid"
        new_tg.save(str(aligned_output_path), format="long_textgrid", includeBlankSpaces=True)
        
        # ============================================
        # ALSO SAVE COPY TO final_textgrids/ WITHOUT _aligned suffix
        # ============================================
        final_textgrids_dir = Path("./final_textgrids") / relative_dir
        final_output_path = final_textgrids_dir / f"{clean_stem}.TextGrid"
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save a copy to final_textgrids (stripping the _aligned suffix)
        new_tg.save(str(final_output_path), format="long_textgrid", includeBlankSpaces=True)
        
        # DEBUG: Log what tiers were saved
        logger.info(f"Saved aligned TextGrid with {len(new_tg.tierNames)} tiers: {aligned_output_path}")
        logger.info(f"Saved copy to final_textgrids: {final_output_path}")
        for tier_name in new_tg.tierNames:
            tier = new_tg.getTier(tier_name)
            if hasattr(tier, 'entries'):
                logger.info(f"  - {tier_name}: {len(tier.entries)} intervals")
            elif hasattr(tier, 'points'):
                logger.info(f"  - {tier_name}: {len(tier.points)} points")
                
        if RICH_AVAILABLE:
            console.print(f"Saved aligned TextGrid: {aligned_output_path}")
            console.print(f"[dim]Also saved copy to: {final_output_path}[/dim]")
        else:
            print(f"Saved aligned TextGrid: {aligned_output_path}")
            print(f"Also saved copy to: {final_output_path}")
        
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
            'language': language,
            'use_dictionary': True,                          # Always True
            'use_acoustic': True,                            # Always True
            'use_rules': config.get('use_rules', True),      # NEW
            'dict_dir': config.get('dictionary_path', "./models/syllable_dicts"),
            'threshold': config.get('acoustic_threshold', 0.3),
            'frame_size': config.get('frame_size', 256),
            'hop_size': config.get('hop_size', 128)
        }
        
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

def run_intonation_sub(aligned_dir: Path, language: str, config: Dict):
    """Hook for libs/intonalyze_A.py with full configuration."""
    if not config.get('enabled', False):
        console.print("Intonation analysis skipped (disabled in configuration)" if RICH_AVAILABLE else "Intonation analysis skipped")
        return
    
    # Double-check language is Spanish
    if language != 'spanish':
        console.print("[yellow]Intonation analysis is only available for Spanish. Skipping.[/yellow]" if RICH_AVAILABLE else "Intonation analysis is only available for Spanish. Skipping.")
        return
    
    try:
        # Import from libs/ directory - use correct filename
        from libs.intonalyze_A import run_intonation
        
        console.print("\nRunning intonation analysis (Break Index assignment)..." if RICH_AVAILABLE else "\nRunning intonation analysis...")
        
        stats = run_intonation(
            aligned_dir=str(aligned_dir),
            language=language,
            clitics_path=config.get('clitics_path')
        )
        
        # Display results
        if RICH_AVAILABLE:
            console.print(f"\n[green]Intonation analysis complete: {stats['success']} files processed[/green]")
        else:
            print(f"\nIntonation analysis complete: {stats['success']} files processed")
        
    except ImportError as e:
        logger.error(f"Intonation module not found: {e}")
        console.print("[red]Error: libs/intonalyze_A.py not found. Skipping intonation analysis.[/red]" if RICH_AVAILABLE else "Error: libs/intonalyze_A.py not found")
    except Exception as e:
        logger.error(f"Error in intonation analysis: {e}")
        console.print(f"[red]Error in intonation analysis: {e}[/red]" if RICH_AVAILABLE else f"Error in intonation analysis: {e}")

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
        # ====================================================================
        # INTERACTIVE MODE
        # ====================================================================
        
        # 1. Basic inputs - interactive
        lang = select_language()
        
        # 2. Get directory paths from user
        tg_in = select_textgrid_directory()
        audio_in = select_audio_directory()
        
        # 3. Get full configurations for ALL subscripts (grouped together)
        console.print("\n[bold cyan]Post-Alignment Annotation Configuration[/bold cyan]" if RICH_AVAILABLE else "\n=== POST-ALIGNMENT ANNOTATION CONFIGURATION ===")
        
        # Ask about all annotation options in sequence
        syllabify_config = configure_syllabification()
        intonation_config = configure_intonation(lang, syllabify_config.get('enabled', False))
        pos_config = configure_pos_tagging()
        
        # Define paths
        out_dir = Path("./aligned_textgrids")
        mfa_auto_download = True  # Default to True in interactive mode
        
    else:
        # ====================================================================
        # NON-INTERACTIVE MODE
        # ====================================================================
        
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
            'use_rules': getattr(args, 'syllabification_use_rules', True),  # NEW
            'dictionary_path': args.syllabification_dict_dir,
            'acoustic_threshold': args.syllabification_threshold,
            'frame_size': args.syllabification_frame_size,
            'hop_size': args.syllabification_hop_size
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
        
        # Build intonation config from arguments
        intonation_config = {
            'enabled': args.enable_intonation,
            'clitics_path': args.intonation_clitics_path
        }
        
        # Validate directories
        if not tg_in.exists():
            console.print(f"[red]Error: TextGrid directory not found: {tg_in}[/red]" if RICH_AVAILABLE else f"Error: TextGrid directory not found: {tg_in}")
            return
        
        if not audio_in.exists():
            console.print(f"[red]Error: Audio directory not found: {audio_in}[/red]" if RICH_AVAILABLE else f"Error: Audio directory not found: {audio_in}")
            return

    # ====================================================================
    # COMMON WORKFLOW (both interactive and non-interactive)
    # ====================================================================
    
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
    run_intonation_sub(out_dir, lang, intonation_config)  # Will only run if language is Spanish and enabled

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
            print(f"  Dictionary Path: {syllabify_config['dictionary_path']}")
            print(f"  Rules: {syllabify_config.get('use_rules')}")
            print(f"  Acoustic Threshold: {syllabify_config['acoustic_threshold']}")
            print(f"  Frame/Hop: {syllabify_config['frame_size']}/{syllabify_config['hop_size']}")
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
        
        # Intonation details
        if intonation_config.get('enabled', False) and lang == 'spanish':
            summary.add_row("", "")
            summary.add_row("[green]Intonation Analysis[/green]", "ENABLED")
            if intonation_config.get('clitics_path'):
                summary.add_row("  Custom Clitics", intonation_config['clitics_path'])
            else:
                summary.add_row("  Clitics", "Default (models/intonation/clitics.csv)")
        elif lang != 'spanish' and intonation_config.get('enabled', False):
            summary.add_row("", "")
            summary.add_row("[yellow]Intonation Analysis[/yellow]", "SKIPPED (Spanish only)")
        else:
            summary.add_row("", "")
            summary.add_row("[yellow]Intonation Analysis[/yellow]", "DISABLED")
        
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
        print(f"\nIntonation Analysis: {'ENABLED' if intonation_config.get('enabled') and lang == 'spanish' else 'DISABLED'}")
        if intonation_config.get('enabled') and lang == 'spanish':
            if intonation_config.get('clitics_path'):
                print(f"  Custom Clitics: {intonation_config['clitics_path']}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    main()
