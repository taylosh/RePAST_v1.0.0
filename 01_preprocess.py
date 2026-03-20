"""
01_preprocess.py - Audio Preparation & Standardization 
@author: taylosh
Created on Nov 15 2025
Last edited on Mar 15 2026

Foundational preparation script for the ASR Pipeline:
- Resamples to 16kHz (if needed)
- Stereo-to-Mono conversion (if needed)
- Peak Normalization to 0.8 (with headroom)
- Converts to 16-bit PCM with dither
- Calls enhance.py for levels 1-3 (enhancement only)
- Preserves directory structure with "_preprocessed" suffix

Can run in two modes:
  - Standalone: Interactive prompts for input/output directories
  - Pipeline: Called by 99_main.py with command-line arguments
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp

# Path resolution for consolidated libs/ and bin/
project_root = Path(__file__).parent
libs_path = project_root / "libs"
if str(libs_path) not in sys.path:
    sys.path.insert(0, str(libs_path))

# Import the Consolidated Accelerator Wrapper (for basic operations only)
try:
    from wrap_accel import get_c_acceleration_wrapper
    wrapper = get_c_acceleration_wrapper(enable_c_accel=True, enable_gpu_accel=True)
    ACCEL_READY = wrapper.available
except ImportError:
    ACCEL_READY = False

# Import enhance logic - this is the ONLY enhancement code called
try:
    from libs.enhance import run_enhancement_logic, initialize_experimental
    ENHANCE_SCRIPT_READY = True
except ImportError:
    ENHANCE_SCRIPT_READY = False

# Rich progress and console integration
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.prompt import Prompt, IntPrompt, Confirm
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    # Fallback console if Rich not available
    class FallbackConsole:
        def print(self, *args, **kwargs): print(*args)
        def input(self, prompt): return input(prompt)
    console = FallbackConsole()

# Enhancement level mapping (for menu only - actual logic in enhance.py)
ENHANCEMENT_OPTIONS = {
    0: {"name": "None", "desc": "Standardize Only (No Enhancement)", "value": "none"},
    1: {"name": "Gentle", "desc": "Call enhance.py - HPF + LUFS", "value": "gentle"},
    2: {"name": "Standard", "desc": "Call enhance.py - Gentle + NR + EQ", "value": "standard"},
    3: {"name": "Experimental", "desc": "Call enhance.py - AI Models (may prompt for model selection)", "value": "experimental"}
}

def select_enhancement_level(choice: Optional[int] = None, pipeline_mode: bool = False) -> str:
    """
    Present enhancement level selection menu to user.
    If choice is provided (from args), use that instead of prompting.
    """
    if choice is not None:
        # Use provided argument
        if choice in ENHANCEMENT_OPTIONS:
            selected = ENHANCEMENT_OPTIONS[choice]
            # Warn if experimental in pipeline mode without pre-selected model
            if choice == 3 and pipeline_mode:
                console.print("[yellow]Note: Experimental level in pipeline mode will use default model[/yellow]" if RICH_AVAILABLE else "Note: Experimental level in pipeline mode will use default model")
            console.print(f"[green]Using enhancement level: {selected['name']}[/green]" if RICH_AVAILABLE else f"Using enhancement level: {selected['name']}")
            return selected["value"]
        else:
            console.print(f"[yellow]Invalid choice {choice}, using None[/yellow]" if RICH_AVAILABLE else f"Invalid choice {choice}, using None")
            return "none"
    
    # Otherwise prompt interactively
    if RICH_AVAILABLE:
        # Rich menu display
        table = Table(title="Enhancement Levels (Handled by enhance.py)")
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Level", style="green")
        table.add_column("Description", style="white")
        
        for num, level_info in ENHANCEMENT_OPTIONS.items():
            table.add_row(str(num), level_info["name"], level_info["desc"])
        
        console.print(table)
        choice = IntPrompt.ask("Select enhancement level", default=0)
    else:
        # Fallback console menu
        print("\nEnhancement Levels:")
        for num, level_info in ENHANCEMENT_OPTIONS.items():
            print(f" {num}. {level_info['name']} - {level_info['desc']}")
        try:
            choice = int(input("\nSelect enhancement [0-3, default: 0]: ").strip() or "0")
        except ValueError:
            choice = 0
    
    # Validate choice
    if choice not in ENHANCEMENT_OPTIONS:
        console.print(f"[yellow]Invalid choice {choice}, using None[/yellow]" if RICH_AVAILABLE else f"Invalid choice {choice}, using None")
        return "none"
    
    selected = ENHANCEMENT_OPTIONS[choice]
    console.print(f"[green]Selected: {selected['name']} - {selected['desc']}[/green]" if RICH_AVAILABLE else f"Selected: {selected['name']}")
    return selected["value"]

def setup_logging(verbose: bool = False):
    """Setup logging with optional verbose mode"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "preprocessing.log"
    
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.FileHandler(log_file)]
    if RICH_AVAILABLE:
        handlers.append(RichHandler(markup=True, show_path=False))
    else:
        handlers.append(logging.StreamHandler())
        
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)
    return logging.getLogger(__name__)

logger = setup_logging()


# ============================================================================
# STANDARDIZATION FUNCTIONS (PURE - NO ENHANCEMENT)
# ============================================================================

def ensure_1d_float32(audio_data: np.ndarray) -> np.ndarray:
    """
    Ensure audio data is 1-dimensional float32 array.
    Critical for C module compatibility.
    """
    # Convert to float32 if needed
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Ensure 1D array
    if audio_data.ndim > 1:
        # Try to squeeze first (removes dimensions of size 1)
        audio_data = audio_data.squeeze()
        
        if audio_data.ndim > 1:
            # If still multi-dimensional, flatten it
            logger.debug(f"Array is {audio_data.ndim}D after squeeze, flattening")
            audio_data = audio_data.flatten()
    
    return audio_data


def stereo_to_mono(audio_data: np.ndarray) -> np.ndarray:
    """
    Convert stereo to mono with phase-aware handling.
    PURE STANDARDIZATION - no enhancement.
    """
    if audio_data.ndim == 1:
        return ensure_1d_float32(audio_data)
    
    if audio_data.ndim == 2:
        # Check if stereo (channels, samples) format
        if audio_data.shape[0] == 2:
            # Calculate correlation between channels
            correlation = np.corrcoef(audio_data[0], audio_data[1])[0, 1]
            logger.debug(f"Channel correlation: {correlation:.3f}")
            
            if correlation > 0.5:  # Channels are similar/in phase
                # Safe to average
                mono_audio = np.mean(audio_data, axis=0)
                logger.debug("Using averaged mono (channels in phase)")
            else:  # Channels are out of phase or very different
                # Take left channel only to avoid phase cancellation
                mono_audio = audio_data[0]
                logger.debug("Using left channel only (channels out of phase)")
            
            return ensure_1d_float32(mono_audio)
        else:
            # Not stereo, just return as 1D
            return ensure_1d_float32(audio_data)
    
    return ensure_1d_float32(audio_data)


def normalize_peak(audio_data: np.ndarray, target_peak: float = 0.8) -> np.ndarray:
    """
    Peak normalization with headroom - only normalizes if needed.
    PURE STANDARDIZATION - no enhancement.
    """
    audio_data = ensure_1d_float32(audio_data)
    
    # Calculate current peak
    current_peak = np.max(np.abs(audio_data))
    
    # Only normalize if outside acceptable range
    # If already between 0.5 and 0.95, leave as is
    # If too quiet (< 0.5) or too loud (> 0.95), normalize
    if current_peak < 0.5 or current_peak > 0.95:
        logger.debug(f"Current peak {current_peak:.3f} outside acceptable range, normalizing to {target_peak}")
        
        if ACCEL_READY and hasattr(wrapper.audio_basic, 'normalize'):
            audio_data = wrapper.audio_basic.normalize(audio_data, target_peak)
            logger.debug(f"Normalized to {target_peak} (C-Accelerated)")
        else:
            if current_peak > 0:
                audio_data = audio_data * (target_peak / current_peak)
                logger.debug(f"Normalized to {target_peak} peak (Python)")
    else:
        logger.debug(f"Current peak {current_peak:.3f} within acceptable range, skipping normalization")
    
    return audio_data


def convert_to_pcm16(audio_data: np.ndarray) -> np.ndarray:
    """
    Convert float audio to PCM16 with dither.
    PURE STANDARDIZATION - no enhancement.
    """
    # Ensure audio is 1D float32
    audio_data = ensure_1d_float32(audio_data)
    
    # Add TPDF dither before conversion to reduce quantization distortion
    dither = np.random.uniform(-0.5, 0.5, audio_data.shape) / 32767
    audio_dithered = audio_data + dither
    
    # Clip to valid range and convert
    return np.clip(audio_dithered * 32767, -32768, 32767).astype(np.int16)


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def preprocess_audio_file(input_path: Path, output_path: Path, enhancement_level: str, 
                          debug_mode: bool = False) -> bool:
    """
    Standardize audio format.
    
    Processing order:
    1. Load with target sample rate (handles resampling)
    2. Convert stereo to mono if needed
    3. Call enhance.py if enhancement level > 0
    4. Peak normalize to 0.8
    5. Convert to 16-bit PCM with dither
    6. Save
    """
    try:
        import librosa
        import soundfile as sf
        
        # Create output directory first
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # DEBUG: Create debug directory if needed
        if debug_mode:
            debug_dir = output_path.parent / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            debug_original = debug_dir / f"{input_path.stem}_original.wav"
            import shutil
            shutil.copy2(str(input_path), str(debug_original))
            logger.debug(f"Debug: Saved original to {debug_original}")
        
        # 1. Load with target sample rate (resamples if needed)
        # librosa with mono=False returns (channels, samples) for stereo
        audio, original_sr = librosa.load(
            str(input_path), 
            sr=16000, 
            mono=False, 
            res_type='kaiser_best'  # Best quality for final pipeline
        )
        logger.info(f"Loaded {input_path.name}: SR={original_sr}Hz, Channels={audio.shape[0] if audio.ndim > 1 else 1}, Duration={audio.shape[-1]/original_sr:.2f}s")
        
        # Track processing steps
        processing_steps = []
        
        # Check if resampling was needed
        if original_sr != 16000:
            processing_steps.append(f"Resampled: {original_sr}→16kHz")
        else:
            processing_steps.append("Already 16kHz")
        
        # DEBUG: Save after loading/resampling
        if debug_mode:
            debug_resampled = debug_dir / f"{input_path.stem}_resampled.wav"
            
            # Prepare audio for saving - soundfile expects (samples, channels)
            if audio.ndim > 1:
                audio_for_write = audio.T if audio.shape[0] < audio.shape[1] else audio
            else:
                audio_for_write = audio
            
            audio_resampled_int16 = np.clip(audio_for_write * 32767, -32768, 32767).astype(np.int16)
            
            try:
                sf.write(str(debug_resampled), audio_resampled_int16, 16000, subtype='PCM_16')
                logger.debug(f"Debug: Saved resampled to {debug_resampled}")
            except Exception as e:
                logger.error(f"Failed to save debug file: {e}")
        
        # 2. Convert stereo to mono if needed
        if audio.ndim > 1 and audio.shape[0] == 2:
            logger.info(f"Converting {input_path.name} from stereo to mono")
            audio = stereo_to_mono(audio)
            
            if ACCEL_READY:
                processing_steps.append("Stereo→Mono (C-Accelerated with phase check)")
            else:
                processing_steps.append("Stereo→Mono (Python with phase check)")
            
            # DEBUG: Save after stereo-to-mono
            if debug_mode:
                debug_mono = debug_dir / f"{input_path.stem}_mono.wav"
                audio_mono_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                sf.write(str(debug_mono), audio_mono_int16, 16000, subtype='PCM_16')
                logger.debug(f"Debug: Saved mono to {debug_mono}")
        else:
            if audio.ndim > 1:
                audio = audio[0]
            processing_steps.append("Already mono")
        
        # 3. ENHANCEMENT - DELEGATED TO ENHANCE.PY
        if enhancement_level != "none" and ENHANCE_SCRIPT_READY:
            logger.info(f"Calling enhance.py for level {enhancement_level} enhancement...")
            audio_before_enhance = audio.copy() if debug_mode else None
            
            # Call enhancement - model already initialized at this point
            audio = run_enhancement_logic(audio, sr=16000, level=enhancement_level)
            
            processing_steps.append(f"Enhanced: {enhancement_level}")
            
            # DEBUG: Save enhancement comparison
            if debug_mode and audio_before_enhance is not None:
                debug_before = debug_dir / f"{input_path.stem}_before_enhance.wav"
                debug_after = debug_dir / f"{input_path.stem}_after_enhance.wav"
                
                audio_before_int16 = np.clip(audio_before_enhance * 32767, -32768, 32767).astype(np.int16)
                audio_after_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                
                sf.write(str(debug_before), audio_before_int16, 16000, subtype='PCM_16')
                sf.write(str(debug_after), audio_after_int16, 16000, subtype='PCM_16')
                logger.debug(f"Debug: Saved enhancement comparison")
        else:
            if enhancement_level != "none" and not ENHANCE_SCRIPT_READY:
                logger.warning(f"Enhance level {enhancement_level} requested but enhance.py not available")
        
        # 4. Peak Normalization (after enhancement)
        audio = normalize_peak(audio, target_peak=0.8)
        processing_steps.append("Normalized to 0.8")
        
        # DEBUG: Save after normalization
        if debug_mode:
            debug_normalized = debug_dir / f"{input_path.stem}_normalized.wav"
            audio_norm_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            sf.write(str(debug_normalized), audio_norm_int16, 16000, subtype='PCM_16')
            logger.debug(f"Debug: Saved normalized to {debug_normalized}")
        
        # 5. Convert to 16-bit PCM with dither
        audio_pcm = convert_to_pcm16(audio)
        
        # 6. Save with robust error handling
        try:
            sf.write(str(output_path), audio_pcm, 16000, subtype='PCM_16', format='WAV')
        except Exception as e1:
            logger.debug(f"Primary save failed: {e1}")
            try:
                sf.write(str(output_path), audio_pcm, 16000, format='WAV')
            except Exception as e2:
                logger.debug(f"Secondary save failed: {e2}")
                sf.write(str(output_path), audio_pcm, 16000)
        
        # Log processing summary
        logger.info(f"Processed: {input_path.name} → {', '.join(processing_steps)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to preprocess {input_path.name}: {e}", exc_info=True)
        return False


def process_directory(input_dir: str, output_dir: str, enhancement_level: str = "none", 
                     debug_mode: bool = False, max_workers: Optional[int] = None,
                     pipeline_mode: bool = False) -> Dict[str, int]:
    """
    Process all WAV files in a directory.
    This function can be called by 99_main.py for pipeline integration.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    wav_files = list(input_path.rglob("*.wav"))
    
    if not wav_files:
        logger.error(f"No WAV files found in {input_dir}")
        return {'success': 0, 'error': 0, 'total': 0}

    stats = {'success': 0, 'error': 0, 'total': len(wav_files)}
    
    logger.info(f"Processing {len(wav_files)} files with enhancement level: {enhancement_level}")
    
    # Determine if we should use multiprocessing
    use_mp = max_workers is not None and max_workers > 1 and len(wav_files) > 10
    
    if use_mp:
        # Multiprocessing implementation
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        def process_file_wrapper(file_tuple):
            f, rel_path, out_path, enh_level, dbg = file_tuple
            target_file = out_path / rel_path.parent / f"{f.stem}_preprocessed.wav"
            success = preprocess_audio_file(f, target_file, enh_level, dbg)
            return success
        
        # Prepare arguments
        file_args = [(f, f.relative_to(input_path), output_path, enhancement_level, debug_mode) 
                     for f in wav_files]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file_wrapper, args) for args in file_args]
            
            if RICH_AVAILABLE:
                with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), 
                            BarColumn(), MofNCompleteColumn()) as progress:
                    task = progress.add_task("Preprocessing...", total=len(futures))
                    
                    for future in as_completed(futures):
                        if future.result():
                            stats['success'] += 1
                        else:
                            stats['error'] += 1
                        progress.advance(task)
            else:
                for i, future in enumerate(as_completed(futures)):
                    if future.result():
                        stats['success'] += 1
                    else:
                        stats['error'] += 1
                    if (i + 1) % 10 == 0:
                        logger.info(f"Progress: {i + 1}/{len(wav_files)} files processed")
    else:
        # Sequential processing
        if RICH_AVAILABLE:
            with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), 
                        BarColumn(), MofNCompleteColumn()) as progress:
                task = progress.add_task("Preprocessing...", total=len(wav_files))
                
                for f in wav_files:
                    rel_path = f.relative_to(input_path)
                    target_file = output_path / rel_path.parent / f"{f.stem}_preprocessed.wav"
                    
                    if preprocess_audio_file(f, target_file, enhancement_level, debug_mode):
                        stats['success'] += 1
                    else:
                        stats['error'] += 1
                    progress.advance(task)
        else:
            for i, f in enumerate(wav_files):
                rel_path = f.relative_to(input_path)
                target_file = output_path / rel_path.parent / f"{f.stem}_preprocessed.wav"
                logger.info(f"Processing ({i + 1}/{len(wav_files)}): {f.name}")
                
                if preprocess_audio_file(f, target_file, enhancement_level, debug_mode):
                    stats['success'] += 1
                else:
                    stats['error'] += 1
    
    return stats


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments for pipeline mode"""
    parser = argparse.ArgumentParser(description='Audio preprocessing for ASR pipeline')
    
    parser.add_argument('--input-dir', '-i', type=str, 
                        help='Input directory containing WAV files')
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Output directory for preprocessed files')
    parser.add_argument('--enhancement-level', '-e', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Enhancement level (0=None, 1=Gentle, 2=Standard, 3=Experimental)')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug mode (saves intermediate files)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--max-workers', '-w', type=int, default=None,
                        help='Maximum number of parallel workers (for multiprocessing)')
    
    return parser.parse_args()


def main():
    """Main function - handles both standalone and pipeline modes"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Reconfigure logging if verbose
    if args.verbose:
        global logger
        logger = setup_logging(verbose=True)
    
    # Determine if we're in pipeline mode (args provided) or standalone mode
    pipeline_mode = args.input_dir is not None
    
    if pipeline_mode:
        # PIPELINE MODE - use command line arguments
        input_dir = args.input_dir
        output_dir = args.output_dir if args.output_dir else "./preprocessed_audio"
        enhancement_level = select_enhancement_level(args.enhancement_level, pipeline_mode=True)
        debug_mode = args.debug
        max_workers = args.max_workers
        
        logger.info("Running in PIPELINE mode")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Enhancement level: {enhancement_level}")
        logger.info(f"Debug mode: {debug_mode}")
        if max_workers:
            logger.info(f"Max workers: {max_workers}")
        
    else:
        # STANDALONE MODE - interactive prompts
        logger.info("Running in STANDALONE mode")
        
        default_in = "./original_audio"
        default_out = "./preprocessed_audio"
        
        if RICH_AVAILABLE:
            input_dir = Prompt.ask("Input directory", default=default_in)
            output_dir = Prompt.ask("Output directory", default=default_out)
        else:
            input_dir = input(f"Input directory [default: {default_in}]: ").strip() or default_in
            output_dir = input(f"Output directory [default: {default_out}]: ").strip() or default_out
        
        # Get enhancement level from interactive menu
        enhancement_level = select_enhancement_level(pipeline_mode=False)
        
        # Debug mode option
        debug_mode = False
        if RICH_AVAILABLE:
            debug_mode = Confirm.ask("Enable debug mode? (saves intermediate files)", default=False)
        else:
            debug_input = input(f"Enable debug mode? (saves intermediate files) [y/n, default: n]: ").strip().lower()
            debug_mode = debug_input == "y"
        
        max_workers = None  # Use default sequential processing for standalone
    
    # CRITICAL: For experimental level, initialize model BEFORE any processing starts
    if enhancement_level == "experimental" and ENHANCE_SCRIPT_READY:
        console.print("\n[bold yellow]--- ESPnet Model Setup ---[/bold yellow]")
        console.print("You've selected Experimental enhancement level.")
        console.print("Please select a model before processing begins.\n")
        
        # Call initialization function from enhance.py
        success = initialize_experimental()
        
        if not success:
            console.print("[red]ESPnet initialization failed. Continuing without enhancement.[/red]")
            enhancement_level = "none"  # Fall back to no enhancement
    
    # Process the directory
    stats = process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        enhancement_level=enhancement_level,
        debug_mode=debug_mode,
        max_workers=max_workers,
        pipeline_mode=pipeline_mode
    )
    
    # Display summary
    if RICH_AVAILABLE:
        summary = Table(title="Preprocessing Summary", box=None)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="bold")
        summary.add_row("Files Found", str(stats['total']))
        summary.add_row("Successfully Processed", str(stats['success']))
        summary.add_row("Errors encountered", str(stats['error']))
        summary.add_row("C-Acceleration Used", "Yes" if ACCEL_READY else "No")
        if ACCEL_READY:
            summary.add_row("GPU Active", "Yes" if wrapper.gpu_backend.gpu_is_available() else "No")
        summary.add_row("Enhancement Level", enhancement_level.title())
        summary.add_row("Debug Mode", "Enabled" if debug_mode else "Disabled")
        summary.add_row("Mode", "Pipeline" if pipeline_mode else "Standalone")
        console.print(summary)
        console.print(f"\n[bold green]Preprocessing complete. Outputs in {output_dir}[/bold green]\n")
    else:
        print(f"\n=== Preprocessing Summary ===")
        print(f"Files Found: {stats['total']}")
        print(f"Successfully Processed: {stats['success']}")
        print(f"Errors encountered: {stats['error']}")
        print(f"C-Acceleration Used: {'Yes' if ACCEL_READY else 'No'}")
        if ACCEL_READY:
            print(f"GPU Active: {'Yes' if wrapper.gpu_backend.gpu_is_available() else 'No'}")
        print(f"Enhancement Level: {enhancement_level.title()}")
        print(f"Debug Mode: {'Enabled' if debug_mode else 'Disabled'}")
        print(f"Mode: {'Pipeline' if pipeline_mode else 'Standalone'}")
        print(f"\nPreprocessing complete. Outputs in {output_dir}\n")
    
    # Return stats for pipeline integration
    return stats


if __name__ == "__main__":
    main()
