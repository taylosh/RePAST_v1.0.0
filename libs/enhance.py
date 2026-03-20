"""
enhance.py - Tiered Audio Enhancement Hub
@author: taylosh
Created on Nov 15 2025
Last edited on Mar 15 2026

Advanced enhancement script providing three distinct levels of processing:
- Gentle: Foundational signal leveling (HPF, LUFS, Clipping Prevention).
- Standard: Conservative noise reduction and subtle EQ (single method).
- Experimental: AI-driven enhancement using SpeechBrain models.
"""

import os
import sys
import json
import logging
import time
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Path resolution for consolidated libs/ and bin/
project_root = Path(__file__).parent
libs_path = project_root / "libs"
if str(libs_path) not in sys.path:
    sys.path.insert(0, str(libs_path))

# Import Consolidated Accelerator Wrapper
try:
    from wrap_accel import get_c_acceleration_wrapper
    wrapper = get_c_acceleration_wrapper(enable_c_accel=True, enable_gpu_accel=True)
    ACCEL_READY = wrapper.available
except ImportError:
    ACCEL_READY = False

# Dependencies check
try:
    import noisereduce as nr
    import pyloudnorm as pyln
    from scipy import signal
    import librosa
    DEPENDENCIES_OK = True
except ImportError:
    DEPENDENCIES_OK = False

# SpeechBrain for experimental level
try:
    import torch
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

# Rich console for interactive model selection
try:
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt, IntPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    class FallbackConsole:
        def print(self, *args): print(*args)
        def input(self, prompt): return input(prompt)
    console = FallbackConsole()

def setup_logging():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "enhancement.log"
    
    level = logging.INFO
    handlers = [logging.FileHandler(log_file)]
    handlers.append(logging.StreamHandler())
        
    logging.basicConfig(level=level, format='%(message)s', handlers=handlers)
    return logging.getLogger(__name__)

logger = setup_logging()


# ============================================================================
# SPEECHBRAIN MODEL MANAGER
# ============================================================================

class SpeechBrainModelManager:
    """Manages SpeechBrain model discovery and selection."""
    
    # Known working SpeechBrain enhancement models
    KNOWN_MODELS = [
        {
            "id": 1,
            "name": "sepformer-whamr-enhancement",
            "full_id": "speechbrain/sepformer-whamr-enhancement",
            "description": "SepFormer for WHAMR! enhancement (noise + reverberation)",
            "size": "~300MB"
        },
        {
            "id": 2,
            "name": "sepformer-wham-enhancement",
            "full_id": "speechbrain/sepformer-wham-enhancement",
            "description": "SepFormer for WHAM! enhancement (noise only)",
            "size": "~300MB"
        },
        {
            "id": 3,
            "name": "metricgan-plus-voicebank",
            "full_id": "speechbrain/metricgan-plus-voicebank",
            "description": "MetricGAN+ for speech enhancement (VoiceBank)",
            "size": "~200MB"
        },
        {
            "id": 4,
            "name": "mtlface-voicebank",
            "full_id": "speechbrain/mtlface-voicebank",
            "description": "Multi-task learning for enhancement",
            "size": "~200MB"
        },
        {
            "id": 5,
            "name": "sepformer-wsj02mix",
            "full_id": "speechbrain/sepformer-wsj02mix",
            "description": "SepFormer for WSJ0-2mix separation",
            "size": "~300MB"
        }
    ]
    
    def __init__(self, models_dir: str = "./models/speechbrain"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache = {}
        
    def get_downloaded_models(self) -> List[Dict]:
        """List all downloaded models with their info"""
        downloaded = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                info_file = model_dir / "model_info.json"
                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                        downloaded.append({
                            "id": len(downloaded) + 1,
                            "name": info.get("model_name", model_dir.name),
                            "path": str(model_dir),
                            "full_id": info.get("full_id", "unknown")
                        })
                    except:
                        pass
        
        return downloaded
    
    def select_model_interactive(self) -> Optional[Dict]:
        """
        Interactive model selection menu.
        Shows downloaded models first, then known models.
        """
        downloaded = self.get_downloaded_models()
        
        console.print("\n[bold cyan]SpeechBrain Model Selection[/bold cyan]")
        console.print("Select a model for speech enhancement.\n")
        
        while True:
            if downloaded:
                table = Table(title="Downloaded Models")
                table.add_column("ID", style="cyan")
                table.add_column("Model Name", style="green")
                table.add_column("Path", style="white")
                
                for model in downloaded:
                    table.add_row(str(model["id"]), model["name"], model["path"])
                
                console.print(table)
                console.print("\n[bold]Options:[/bold]")
                console.print("  0: Browse available models")
                
                try:
                    choice = IntPrompt.ask("Select model to use", default=1)
                    
                    if choice == 0:
                        selected = self.select_known_model()
                        if selected:
                            return selected
                        continue
                    elif 1 <= choice <= len(downloaded):
                        return downloaded[choice - 1]
                    else:
                        console.print("[red]Invalid choice. Please try again.[/red]")
                except:
                    console.print("[red]Invalid input. Please try again.[/red]")
            
            else:
                console.print("[yellow]No downloaded models found.[/yellow]")
                selected = self.select_known_model()
                if selected:
                    return selected
                
                retry = Prompt.ask("Try again?", choices=["y", "n"], default="y")
                if retry != "y":
                    return None
    
    def select_known_model(self) -> Optional[Dict]:
        """Show known models and let user select one."""
        
        table = Table(title="Available SpeechBrain Models")
        table.add_column("ID", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Description", style="white")
        table.add_column("Size", style="yellow")
        
        for model in self.KNOWN_MODELS:
            table.add_row(
                str(model["id"]),
                model["name"],
                model["description"],
                model["size"]
            )
        
        console.print(table)
        console.print("\n[bold]0: Cancel[/bold]")
        
        try:
            choice = IntPrompt.ask("Select model to download", default=1)
            
            if choice == 0:
                return None
            
            for model in self.KNOWN_MODELS:
                if model["id"] == choice:
                    return model
            
            console.print("[red]Invalid model ID.[/red]")
            return None
            
        except:
            console.print("[red]Invalid input.[/red]")
            return None


# ============================================================================
# SPEECHBRAIN ENHANCER
# ============================================================================

class SpeechBrainEnhancer:
    """Handles SpeechBrain model loading and inference."""
    
    def __init__(self, models_dir: str = "./models/speechbrain"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_manager = SpeechBrainModelManager(models_dir)
        self.model = None
        self.initialized = False
        self.current_model_info = None
    
    def initialize_interactive(self) -> bool:
        """Interactive model selection and initialization."""
        try:
            from speechbrain.pretrained import SepformerSeparation
            
            selected = self.model_manager.select_model_interactive()
            
            if not selected:
                logger.error("No model selected")
                console.print("[red]No model selected. Enhancement cancelled.[/red]")
                return False
            
            return self.initialize_with_model(selected)
            
        except ImportError as e:
            logger.error(f"SpeechBrain not installed: {e}")
            console.print("[red]SpeechBrain not installed. Run: pip install speechbrain[/red]")
            return False
        except Exception as e:
            logger.error(f"SpeechBrain initialization failed: {e}")
            console.print(f"[red]SpeechBrain initialization failed: {e}[/red]")
            return False
    
    def initialize_with_model(self, model_info: Dict) -> bool:
        """Initialize with a specific model."""
        try:
            from speechbrain.pretrained import SepformerSeparation
            
            full_id = model_info["full_id"]
            model_name = model_info["name"]
            model_path = self.models_dir / model_name.replace('/', '_')
            
            console.print(f"[cyan]Loading model {full_id}...[/cyan]")
            logger.info(f"Loading model {full_id}...")
            
            # SpeechBrain automatically caches models
            self.model = SepformerSeparation.from_hparams(
                source=full_id,
                savedir=model_path,
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            
            # Save model info
            save_info = {
                "model_name": model_name,
                "full_id": full_id,
                "download_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(model_path / "model_info.json", "w") as f:
                json.dump(save_info, f, indent=2)
            
            self.initialized = True
            self.current_model_info = model_info
            console.print(f"[green]Model loaded successfully[/green]")
            logger.info(f"Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    def enhance(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Perform speech enhancement."""
        if not self.initialized or not self.model:
            logger.warning("Model not initialized, returning original")
            return audio
        
        try:
            import torch
            
            logger.info(f"Running SpeechBrain enhancement...")
            
            # Convert to torch tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio).float()
            
            # FIX: Use separate_batch instead of enhance_batch
            enhanced = self.model.separate_batch(audio_tensor.unsqueeze(0))
            
            # Convert back to numpy
            result = enhanced.squeeze().cpu().numpy()
            
            # Ensure output shape matches input
            if result.shape != audio.shape:
                logger.warning(f"Output shape mismatch")
                if len(result) > len(audio):
                    result = result[:len(audio)]
                else:
                    result = np.pad(result, (0, len(audio) - len(result)))
            
            logger.info(f"SpeechBrain enhancement complete")
            return result
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return audio


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def needs_hpf(audio: np.ndarray, sr: int, threshold_db: float = -40) -> bool:
    """Detect if HPF is needed by checking energy below 80Hz."""
    spec = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr)
    
    low_idx = freqs < 80
    
    if not np.any(low_idx):
        return False
    
    low_energy = np.sum(spec[low_idx, :])
    total_energy = np.sum(spec)
    
    if total_energy == 0:
        return False
    
    energy_ratio_db = 10 * np.log10(low_energy / total_energy + 1e-10)
    
    return energy_ratio_db > threshold_db


def needs_noise_reduction(audio: np.ndarray, sr: int) -> Tuple[bool, float]:
    """Detect if noise reduction is needed by estimating SNR."""
    noise_sample = audio[:int(0.1 * sr)]
    signal_sample = audio[int(0.1 * sr):int(0.5 * sr)]
    
    if len(noise_sample) == 0 or len(signal_sample) == 0:
        return False, 40
    
    noise_power = np.mean(noise_sample**2) + 1e-10
    signal_power = np.mean(signal_sample**2) + 1e-10
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    return snr_db < 25, snr_db


def needs_eq(audio: np.ndarray, sr: int) -> bool:
    """Detect if EQ might help by analyzing spectral balance."""
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    
    return mean_centroid < 800


# ============================================================================
# AUDIO ENHANCER CLASS
# ============================================================================

class AudioEnhancer:
    """Independent engines for Gentle, Standard, and Experimental levels."""
    
    def __init__(self, level: str = "gentle"):
        self.level = level
        self.ai_engine = None
    
    def apply_gentle(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Level 1: HPF Rumble Removal + LUFS Normalization"""
        logger.info("Level 1: Applying Gentle enhancement")
        
        if needs_hpf(audio, sr):
            nyquist = sr / 2
            b, a = signal.butter(2, 80 / nyquist, btype='high')
            audio = signal.filtfilt(b, a, audio)
            logger.info("  Applied HPF at 80Hz (rumble detected)")
        else:
            logger.info("  No significant rumble detected, skipping HPF")
        
        meter = pyln.Meter(sr)
        try:
            loudness = meter.integrated_loudness(audio)
            if abs(loudness + 23) > 5:
                audio = pyln.normalize.loudness(audio, loudness, -23.0)
                logger.info(f"  LUFS normalization: {loudness:.1f} -> -23.0")
            else:
                logger.info(f"  LUFS already within range ({loudness:.1f}), skipping")
        except Exception as e:
            logger.warning(f"  LUFS normalization failed: {e}")
        
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio = np.tanh(audio * 0.98)
            logger.info(f"  Applied soft limiting (peak was {peak:.3f})")
        
        return audio
    
    def apply_standard(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Level 2: Single noise reduction method + subtle EQ"""
        logger.info("Level 2: Applying Standard enhancement")
        
        needs_nr, snr_db = needs_noise_reduction(audio, sr)
        
        if needs_nr:
            logger.info(f"  Low SNR detected ({snr_db:.1f}dB), applying noise reduction")
            
            if snr_db < 15:
                nr_strength = 0.3 if snr_db < 10 else 0.2
                try:
                    audio = nr.reduce_noise(
                        y=audio, 
                        sr=sr, 
                        stationary=True, 
                        prop_decrease=nr_strength,
                        n_std_thresh_stationary=1.2
                    )
                    logger.info(f"  Applied Python noise reduction (strength={nr_strength})")
                except Exception as e:
                    logger.warning(f"  Python noise reduction failed: {e}")
            
            else:
                if ACCEL_READY:
                    peak = np.max(np.abs(audio))
                    threshold = 0.03 * peak if peak > 0 else 0.02
                    audio = wrapper.audio_enhance.remove_noise(audio, threshold=threshold, algorithm=0)
                    logger.info(f"  Applied C noise reduction (threshold={threshold:.4f})")
                else:
                    audio = nr.reduce_noise(
                        y=audio, 
                        sr=sr, 
                        stationary=True, 
                        prop_decrease=0.25
                    )
                    logger.info("  Applied Python noise reduction (fallback)")
        else:
            logger.info(f"  Good SNR ({snr_db:.1f}dB), skipping noise reduction")
        
        if needs_eq(audio, sr):
            if ACCEL_READY:
                audio = wrapper.audio_enhance.apply_equalization(audio, treble_boost=1.02)
                logger.info("  Applied subtle treble boost")
            else:
                logger.info("  EQ requested but C-acceleration not available, skipping")
        else:
            logger.info("  Spectral balance good, skipping EQ")
        
        return audio
    
    def apply_experimental(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Level 3: SpeechBrain AI enhancement with pre-initialized model"""
        logger.info("Level 3: Applying Experimental enhancement (SpeechBrain)")
        
        global _enhancer_instance
        
        if _enhancer_instance and _enhancer_instance.initialized:
            audio = _enhancer_instance.enhance(audio, sr)
            logger.info("  Applied SpeechBrain AI enhancement")
        else:
            logger.warning("  SpeechBrain not available, returning unprocessed audio")
        
        return audio


# ============================================================================
# GLOBAL STATE FOR EXPERIMENTAL MODE
# ============================================================================

_enhancer_instance = None
_model_selected = False


def initialize_experimental() -> bool:
    """
    Initialize experimental enhancement by selecting a SpeechBrain model.
    Called BEFORE any processing starts to avoid UI conflicts.
    """
    global _enhancer_instance, _model_selected
    
    console.print("\n[bold yellow]--- SpeechBrain Model Setup ---[/bold yellow]")
    console.print("You've selected Experimental enhancement level.")
    console.print("Please select a model before processing begins.\n")
    
    temp_enhancer = SpeechBrainEnhancer()
    success = temp_enhancer.initialize_interactive()
    
    if success:
        _enhancer_instance = temp_enhancer
        _model_selected = True
        console.print("[green]Model ready! Starting audio processing...[/green]\n")
        return True
    else:
        console.print("[red]SpeechBrain initialization failed. Continuing without enhancement.[/red]")
        return False


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_enhancement_logic(audio_buffer: np.ndarray, sr: int = 16000, level: str = "none") -> np.ndarray:
    """Main hook called by 01_preprocess.py."""
    global _enhancer_instance, _model_selected
    
    if level == "none":
        return audio_buffer
    
    if not DEPENDENCIES_OK and level != "none":
        logger.warning("Required dependencies not installed")
        return audio_buffer
    
    if level == "experimental":
        if _enhancer_instance and _enhancer_instance.initialized:
            return _enhancer_instance.enhance(audio_buffer, sr)
        else:
            logger.warning("SpeechBrain not initialized, returning unprocessed audio")
            return audio_buffer
    
    enhancer = AudioEnhancer(level=level)
    
    try:
        start_time = time.time()
        
        if level == "gentle":
            result = enhancer.apply_gentle(audio_buffer, sr)
        elif level == "standard":
            result = enhancer.apply_standard(audio_buffer, sr)
        else:
            result = audio_buffer
        
        elapsed = time.time() - start_time
        logger.info(f"{level} enhancement completed in {elapsed:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Enhancement failed for level {level}: {e}")
        return audio_buffer


if __name__ == "__main__":
    logger.info("Testing enhancement module...")
    test_audio = np.random.randn(16000) * 0.1
    test_sr = 16000
    
    for level in ["gentle", "standard", "experimental"]:
        logger.info(f"Testing level: {level}")
        result = run_enhancement_logic(test_audio, test_sr, level)
        logger.info(f"  Input shape: {test_audio.shape}, Output shape: {result.shape}")
