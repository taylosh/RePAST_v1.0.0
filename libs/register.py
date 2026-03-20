"""
register.py - Speaker Profile Registration & Management
@author: taylosh
Created on Mar 16 2026
Last edited on Mar 16 2026

Standalone speaker profile registration tool:
- Uses ECAPA-TDNN for vocal fingerprinting
- Creates new speaker profiles or updates existing ones
- Simple flattened list of audio files for selection
- Returns to main menu after each operation
- Saves profiles to ./models/embeddings/ for use by diarization pipeline
"""

import os
import sys
import json
import time
import torch
import numpy as np
import gc
from pathlib import Path
from typing import List, Optional, Dict, Tuple

# Path resolution - register.py is in libs/, so project root is parent
project_root = Path(__file__).parent.parent
libs_path = project_root / "libs"
if str(libs_path) not in sys.path:
    sys.path.insert(0, str(libs_path))

# Define paths relative to project root
PREPROCESSED_AUDIO_DIR = project_root / "preprocessed_audio"
EMBEDDINGS_DIR = project_root / "models" / "embeddings"

# Import accelerator wrapper
try:
    from wrap_accel import get_c_acceleration_wrapper
    wrapper = get_c_acceleration_wrapper(enable_c_accel=True, enable_gpu_accel=True)
    ACCEL_READY = wrapper.available
except ImportError:
    ACCEL_READY = False

# Rich console for nice output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    class FallbackConsole:
        def print(self, *args, **kwargs): print(*args)
    console = FallbackConsole()

# ============================================================================
# SECTION 1: ECAPA EMBEDDING EXTRACTOR
# ============================================================================

class ECAPAEmbeddingExtractor:
    """ECAPA-TDNN speaker embedding extractor for profile creation."""
    def __init__(self):
        self.model = None
        self.initialized = False
        self.device = "cuda" if (ACCEL_READY and wrapper.gpu_backend.gpu_is_available()) else "cpu"

    def initialize(self):
        """Load the ECAPA model."""
        try:
            from espnet2.bin.spk_inference import Speech2Embedding
            console.print(f"[dim]Loading ECAPA-TDNN model on {self.device}...[/dim]")
            self.model = Speech2Embedding.from_pretrained(
                model_tag="espnet/voxcelebs12_ecapa", 
                device=self.device
            )
            self.initialized = True
            console.print("[green]ECAPA model loaded successfully[/green]")
        except ImportError:
            console.print("[red]Error: espnet not installed. Please install with: pip install espnet[/red]")
            raise
        except Exception as e:
            console.print(f"[red]Error loading ECAPA model: {e}[/red]")
            raise

    def extract(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract embedding from audio data."""
        if not self.initialized:
            self.initialize()
        return self.model(audio_data).squeeze(0).cpu().numpy().astype(np.float32)
    
    def unload(self):
        """Unload model to free memory."""
        if self.model is not None:
            self.model = None
            self.initialized = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            console.print("[dim]ECAPA model unloaded, memory freed[/dim]")

# ============================================================================
# SECTION 2: PROFILE LIBRARY MANAGEMENT
# ============================================================================

class ProfileLibrary:
    """Manages speaker profiles in the embeddings directory."""
    def __init__(self):
        self.library_path = EMBEDDINGS_DIR
        self.library_path.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> List[str]:
        """Return list of existing profile names."""
        return sorted([f.stem for f in self.library_path.glob("*.json")])

    def profile_exists(self, name: str) -> bool:
        """Check if a profile already exists."""
        return (self.library_path / f"{name}.json").exists()

    def load_profile(self, name: str) -> np.ndarray:
        """Load an existing profile embedding."""
        with open(self.library_path / f"{name}.json", 'r') as f:
            data = json.load(f)
            return np.array(data['embedding'], dtype=np.float32)

    def save_profile(self, name: str, embedding: np.ndarray):
        """Save a new speaker profile."""
        embedding = embedding.astype(np.float32)
        output = {
            "speaker_name": name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding": embedding.tolist()
        }
        with open(self.library_path / f"{name}.json", 'w') as f:
            json.dump(output, f, indent=2)
        console.print(f"[green]Profile saved: {name}[/green]")

    def update_profile(self, name: str, new_embedding: np.ndarray, weight: float = 0.5):
        """
        Update an existing profile by blending with new embedding.
        weight: How much weight to give to the new embedding (0.0-1.0)
               Default 0.5 (equal weight to existing and new)
        """
        if not self.profile_exists(name):
            console.print(f"[red]Profile {name} does not exist[/red]")
            return False
        
        # Load existing embedding
        existing = self.load_profile(name)
        
        # Blend embeddings (weighted average)
        blended = (weight * new_embedding + (1 - weight) * existing).astype(np.float32)
        
        # Save updated profile
        with open(self.library_path / f"{name}.json", 'r') as f:
            data = json.load(f)
        
        data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["embedding"] = blended.tolist()
        
        with open(self.library_path / f"{name}.json", 'w') as f:
            json.dump(data, f, indent=2)
        
        console.print(f"[green]Profile updated: {name} (blend weight: {weight})[/green]")
        return True

    def delete_profile(self, name: str) -> bool:
        """Delete an existing profile."""
        profile_path = self.library_path / f"{name}.json"
        if profile_path.exists():
            profile_path.unlink()
            console.print(f"[yellow]Profile deleted: {name}[/yellow]")
            return True
        return False

# ============================================================================
# SECTION 3: AUDIO FILE MANAGEMENT
# ============================================================================

def get_audio_files() -> List[Path]:
    """Get flattened list of all preprocessed audio files."""
    if not PREPROCESSED_AUDIO_DIR.exists():
        console.print(f"[yellow]Directory {PREPROCESSED_AUDIO_DIR} not found[/yellow]")
        return []
    
    all_files = list(PREPROCESSED_AUDIO_DIR.rglob("*_preprocessed.wav"))
    return sorted(all_files)  # Sort for consistent display

def display_audio_files(audio_files: List[Path]):
    """Display flattened list of audio files with numbers."""
    if not audio_files:
        console.print("[yellow]No audio files found[/yellow]")
        return
    
    if RICH_AVAILABLE:
        table = Table(title="Available Audio Files")
        table.add_column("#", style="cyan", justify="right")
        table.add_column("File Path", style="green")
        table.add_column("Size", style="dim", justify="right")
        
        for i, file in enumerate(audio_files, 1):
            rel_path = file.relative_to(project_root)
            size_mb = file.stat().st_size / (1024 * 1024)
            table.add_row(str(i), str(rel_path), f"{size_mb:.1f} MB")
        
        console.print(table)
    else:
        print("\nAvailable Audio Files:")
        for i, file in enumerate(audio_files, 1):
            rel_path = file.relative_to(project_root)
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {i}. {rel_path} ({size_mb:.1f} MB)")

def select_audio_file(audio_files: List[Path]) -> Optional[Path]:
    """Let user select an audio file from the flattened list."""
    if not audio_files:
        return None
    
    display_audio_files(audio_files)
    
    if RICH_AVAILABLE:
        choice = Prompt.ask("\nSelect audio file number (or 0 to cancel)", default="0")
    else:
        choice = input("\nSelect audio file number (or 0 to cancel): ").strip() or "0"
    
    try:
        idx = int(choice) - 1
        if idx == -1:  # 0 entered
            return None
        if 0 <= idx < len(audio_files):
            return audio_files[idx]
        else:
            console.print(f"[red]Invalid number: {choice}[/red]")
            return None
    except ValueError:
        console.print("[red]Invalid input[/red]")
        return None

# ============================================================================
# SECTION 4: EMBEDDING EXTRACTION
# ============================================================================

def extract_embedding_from_file(audio_path: Path, extractor: ECAPAEmbeddingExtractor,
                                max_duration_seconds: int = 1800, 
                                segment_duration: int = 10) -> Optional[np.ndarray]:
    """
    Extract a speaker embedding from an audio file.
    Processes in chunks to handle long files efficiently.
    """
    import soundfile as sf
    
    try:
        # Get audio info
        with sf.SoundFile(audio_path) as f:
            sr = f.samplerate
            total_frames = f.frames
            total_duration = total_frames / sr
        
        console.print(f"\n[dim]Audio duration: {total_duration:.1f} seconds[/dim]")
        
        # Calculate chunks
        chunk_samples = int(segment_duration * sr)
        max_samples = int(max_duration_seconds * sr)
        
        total_possible_chunks = total_frames // chunk_samples
        max_chunks = max_samples // chunk_samples
        num_chunks = min(total_possible_chunks, max_chunks)
        
        if num_chunks == 0:
            num_chunks = 1
            chunk_samples = total_frames
        
        embedding_sum = None
        embedding_count = 0
        
        # Process each chunk with progress indicator
        if RICH_AVAILABLE:
            from rich.progress import Progress
            with Progress() as progress:
                task = progress.add_task("[cyan]Extracting embeddings...", total=num_chunks)
                
                for i in range(num_chunks):
                    start_sample = i * chunk_samples
                    end_sample = min((i + 1) * chunk_samples, total_frames)
                    
                    # Load just this chunk
                    with sf.SoundFile(audio_path) as f:
                        f.seek(start_sample)
                        chunk = f.read(end_sample - start_sample)
                    
                    # Initialize extractor for this chunk if needed
                    if not extractor.initialized:
                        extractor.initialize()
                    
                    # Extract embedding for this chunk
                    chunk_embedding = extractor.extract(chunk)
                    
                    # Accumulate
                    if embedding_sum is None:
                        embedding_sum = chunk_embedding
                    else:
                        embedding_sum += chunk_embedding
                    embedding_count += 1
                    
                    # Update progress
                    progress.update(task, advance=1)
                    
                    # Clean up chunk
                    del chunk, chunk_embedding
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            # Simple version without progress bar
            for i in range(num_chunks):
                start_sample = i * chunk_samples
                end_sample = min((i + 1) * chunk_samples, total_frames)
                
                console.print(f"  [dim]Processing chunk {i+1}/{num_chunks}...[/dim]")
                
                # Load just this chunk
                with sf.SoundFile(audio_path) as f:
                    f.seek(start_sample)
                    chunk = f.read(end_sample - start_sample)
                
                # Initialize extractor for this chunk if needed
                if not extractor.initialized:
                    extractor.initialize()
                
                # Extract embedding for this chunk
                chunk_embedding = extractor.extract(chunk)
                
                # Accumulate
                if embedding_sum is None:
                    embedding_sum = chunk_embedding
                else:
                    embedding_sum += chunk_embedding
                embedding_count += 1
                
                # Clean up chunk
                del chunk, chunk_embedding
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Average all chunk embeddings
        if embedding_count > 0:
            final_embedding = (embedding_sum / embedding_count).astype(np.float32)
            console.print(f"[green]Extracted embedding from {embedding_count} chunks[/green]")
            return final_embedding
        else:
            console.print("[red]No embeddings extracted[/red]")
            return None
        
    except Exception as e:
        console.print(f"[red]Error processing audio: {e}[/red]")
        return None

# ============================================================================
# SECTION 5: SPEAKER OPERATIONS
# ============================================================================

def register_new_speaker(library: ProfileLibrary, extractor: ECAPAEmbeddingExtractor):
    """Register a new speaker."""
    console.print(Panel("[bold cyan]Register New Speaker[/bold cyan]"))
    
    # Get speaker name
    while True:
        if RICH_AVAILABLE:
            name = Prompt.ask("Enter new speaker name").strip()
        else:
            name = input("Enter new speaker name: ").strip()
        
        if not name:
            console.print("[red]Name cannot be empty[/red]")
            continue
        
        if library.profile_exists(name):
            overwrite = Confirm.ask(f"[yellow]Profile '{name}' already exists. Overwrite?[/yellow]")
            if overwrite:
                break
        else:
            break
    
    # Get audio files
    audio_files = get_audio_files()
    if not audio_files:
        console.print("[red]No audio files available[/red]")
        return
    
    # Select audio file
    selected_file = select_audio_file(audio_files)
    if not selected_file:
        console.print("[yellow]Cancelled[/yellow]")
        return
    
    console.print(f"\n[green]Selected: {selected_file.relative_to(project_root)}[/green]")
    
    # Extract embedding
    embedding = extract_embedding_from_file(selected_file, extractor)
    
    if embedding is not None:
        library.save_profile(name, embedding)
        console.print(f"[bold green]✓ Speaker '{name}' registered successfully![/bold green]")
    else:
        console.print("[red]Failed to extract embedding[/red]")

def update_existing_speaker(library: ProfileLibrary, extractor: ECAPAEmbeddingExtractor):
    """Update an existing speaker's profile."""
    console.print(Panel("[bold cyan]Update Existing Speaker[/bold cyan]"))
    
    # Get list of existing profiles
    profiles = library.list_profiles()
    if not profiles:
        console.print("[yellow]No existing profiles found[/yellow]")
        return
    
    # Display profiles
    if RICH_AVAILABLE:
        table = Table(title="Existing Speakers")
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Speaker Name", style="green")
        
        for i, name in enumerate(profiles, 1):
            table.add_row(str(i), name)
        
        console.print(table)
    else:
        print("\nExisting Speakers:")
        for i, name in enumerate(profiles, 1):
            print(f"  {i}. {name}")
    
    # Select speaker
    if RICH_AVAILABLE:
        choice = Prompt.ask("\nSelect speaker number (or 0 to cancel)", default="0")
    else:
        choice = input("\nSelect speaker number (or 0 to cancel): ").strip() or "0"
    
    try:
        idx = int(choice) - 1
        if idx == -1:
            return
        if 0 <= idx < len(profiles):
            speaker_name = profiles[idx]
        else:
            console.print(f"[red]Invalid number: {choice}[/red]")
            return
    except ValueError:
        console.print("[red]Invalid input[/red]")
        return
    
    # Get blend weight - DEFAULT CHANGED TO 0.5
    if RICH_AVAILABLE:
        weight_input = Prompt.ask(
            "Blend weight for new samples (0.0-1.0, higher = more influence from this file)",
            default="0.5"  # Changed from 0.3 to 0.5
        )
    else:
        weight_input = input("Blend weight (0.0-1.0) [default: 0.5]: ").strip() or "0.5"  # Changed from 0.3 to 0.5
    
    try:
        weight = float(weight_input)
        weight = max(0.0, min(1.0, weight))
    except ValueError:
        weight = 0.5  # Changed from 0.3 to 0.5
    
    # Get audio files
    audio_files = get_audio_files()
    if not audio_files:
        console.print("[red]No audio files available[/red]")
        return
    
    # Select audio file
    selected_file = select_audio_file(audio_files)
    if not selected_file:
        console.print("[yellow]Cancelled[/yellow]")
        return
    
    console.print(f"\n[green]Selected: {selected_file.relative_to(project_root)}[/green]")
    
    # Extract embedding
    embedding = extract_embedding_from_file(selected_file, extractor)
    
    if embedding is not None:
        library.update_profile(speaker_name, embedding, weight)
        console.print(f"[bold green]✓ Speaker '{speaker_name}' updated successfully![/bold green]")
    else:
        console.print("[red]Failed to extract embedding[/red]")

# ============================================================================
# SECTION 6: PROFILE MANAGEMENT UTILITIES
# ============================================================================

def list_all_profiles(library: ProfileLibrary):
    """Display all existing profiles with details."""
    profiles = library.list_profiles()
    
    if not profiles:
        console.print("[yellow]No profiles found[/yellow]")
        return
    
    if RICH_AVAILABLE:
        table = Table(title="Registered Speaker Profiles")
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Speaker Name", style="green")
        table.add_column("Profile File", style="dim")
        table.add_column("Last Updated", style="yellow")
        
        for i, name in enumerate(profiles, 1):
            profile_path = library.library_path / f"{name}.json"
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(profile_path.stat().st_mtime))
            table.add_row(str(i), name, f"{name}.json", mtime)
        
        console.print(table)
    else:
        print("\nRegistered Speaker Profiles:")
        for i, name in enumerate(profiles, 1):
            print(f"  {i}. {name}")
    
    console.print(f"\nTotal profiles: {len(profiles)}")

def inspect_profile(library: ProfileLibrary):
    """View details of a specific profile."""
    profiles = library.list_profiles()
    
    if not profiles:
        console.print("[yellow]No profiles to inspect[/yellow]")
        return
    
    list_all_profiles(library)
    
    if RICH_AVAILABLE:
        choice = Prompt.ask("\nEnter profile number to inspect (or 0 to cancel)", default="0")
    else:
        choice = input("\nEnter profile number to inspect (or 0 to cancel): ").strip() or "0"
    
    try:
        idx = int(choice) - 1
        if idx == -1:
            return
        if 0 <= idx < len(profiles):
            name = profiles[idx]
            with open(library.library_path / f"{name}.json", 'r') as f:
                data = json.load(f)
            
            console.print(f"\n[bold cyan]Profile: {name}[/bold cyan]")
            console.print(f"  Created: {data.get('created_at', 'Unknown')}")
            console.print(f"  Last Updated: {data.get('last_updated', 'Unknown')}")
            console.print(f"  Embedding dimension: {len(data['embedding'])}")
            console.print(f"  First 10 values: {data['embedding'][:10]}")
        else:
            console.print(f"[red]Invalid selection: {choice}[/red]")
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        console.print(f"[red]Error reading profile: {e}[/red]")

def delete_profile_interactive(library: ProfileLibrary):
    """Interactive profile deletion."""
    profiles = library.list_profiles()
    
    if not profiles:
        console.print("[yellow]No profiles to delete[/yellow]")
        return
    
    list_all_profiles(library)
    
    if RICH_AVAILABLE:
        choice = Prompt.ask("\nEnter profile number to delete (or 0 to cancel)", default="0")
    else:
        choice = input("\nEnter profile number to delete (or 0 to cancel): ").strip() or "0"
    
    try:
        idx = int(choice) - 1
        if idx == -1:
            return
        if 0 <= idx < len(profiles):
            name = profiles[idx]
            if RICH_AVAILABLE:
                confirm = Confirm.ask(f"Delete profile '{name}'?")
            else:
                confirm_input = input(f"Delete profile '{name}'? (y/N): ").strip().lower()
                confirm = confirm_input in ['y', 'yes']
            
            if confirm:
                library.delete_profile(name)
        else:
            console.print(f"[red]Invalid selection: {choice}[/red]")
    except ValueError:
        console.print("[red]Invalid input[/red]")

# ============================================================================
# SECTION 7: MAIN MENU
# ============================================================================

def main():
    """Main menu for profile management."""
    
    # ASCII Art Header
    console.print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     SPEAKER PROFILE REGISTRATION - ECAPA-TDNN           ║
    ║     Create, update, and manage voice profiles           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize library and extractor once
    library = ProfileLibrary()
    extractor = ECAPAEmbeddingExtractor()
    
    try:
        while True:
            if RICH_AVAILABLE:
                console.print("\n[bold cyan]Main Menu:[/bold cyan]")
                console.print("  1. Register new speaker")
                console.print("  2. Update existing speaker")
                console.print("  3. List existing profiles")
                console.print("  4. Inspect a profile")
                console.print("  5. Delete a profile")
                console.print("  6. Exit")
                
                choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5", "6"], default="1")
            else:
                print("\nMain Menu:")
                print("  1. Register new speaker")
                print("  2. Update existing speaker")
                print("  3. List existing profiles")
                print("  4. Inspect a profile")
                print("  5. Delete a profile")
                print("  6. Exit")
                
                choice = input("Select option [default: 1]: ").strip() or "1"
            
            if choice == "1":
                register_new_speaker(library, extractor)
            elif choice == "2":
                update_existing_speaker(library, extractor)
            elif choice == "3":
                list_all_profiles(library)
            elif choice == "4":
                inspect_profile(library)
            elif choice == "5":
                delete_profile_interactive(library)
            elif choice == "6":
                console.print("[yellow]Exiting...[/yellow]")
                break
            
            # Pause before returning to menu
            if choice in ["1", "2", "3", "4", "5"]:
                if RICH_AVAILABLE:
                    input("\nPress Enter to continue...")
                else:
                    input("\nPress Enter to continue...")
    
    finally:
        # Always unload model on exit
        extractor.unload()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)
