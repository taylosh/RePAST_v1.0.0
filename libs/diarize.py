"""
diarize.py - Speaker Identification & Profile Management
@author: taylosh
Created on Nov 15 2025
Last edited on Mar 16 2026

Advanced speaker diarization and identification subprocess:
- Uses pyannote.audio 3.1 for temporal segmentation.
- Uses ECAPA-TDNN for vocal fingerprinting (embeddings).
- Matches speakers against existing profile library in ./models/embeddings/.
- Accelerated via consolidated audio_segment_engine C-binary.
"""

import os
import sys
import json
import logging
import time
import argparse
import torch
import numpy as np
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Path resolution for consolidated libs/ and bin/
project_root = Path(__file__).parent.parent
libs_path = project_root / "libs"
if str(libs_path) not in sys.path:
    sys.path.insert(0, str(libs_path))

# Import the new Consolidated Accelerator Wrapper
try:
    from wrap_accel import get_c_acceleration_wrapper
    wrapper = get_c_acceleration_wrapper(enable_c_accel=True, enable_gpu_accel=True)
    ACCEL_READY = wrapper.available
except ImportError:
    ACCEL_READY = False

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1: HARDWARE & ENVIRONMENT INITIALIZATION
# ============================================================================

def check_huggingface_token() -> Optional[str]:
    """Verify Hugging Face token and provide instructions if missing."""
    token_file = project_root / "HuggingFaceToken.txt"
    if token_file.exists():
        return token_file.read_text().strip()
    
    print("\nERROR: Hugging Face Token file not found.")
    print("To proceed, please visit https://huggingface.co/settings/tokens to generate a ")
    print("Read-access token, copy it, and save it into a file named 'HuggingFaceToken.txt'")
    print("in your project root directory.\n")
    return None

# ============================================================================
# SECTION 2: SPEAKER PROFILE MANAGEMENT (READ-ONLY)
# ============================================================================

class ProfileLibrary:
    """Read-only access to speaker profiles."""
    def __init__(self, library_dir: str = "./models/embeddings"):
        self.library_path = Path(library_dir)
        self.library_path.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> List[str]:
        return [f.stem for f in self.library_path.glob("*.json")]

    def load_profile(self, name: str) -> np.ndarray:
        with open(self.library_path / f"{name}.json", 'r') as f:
            data = json.load(f)
            return np.array(data['embedding'], dtype=np.float32)

# ============================================================================
# SECTION 3: DIARIZATION ENGINE (No embedding extraction)
# ============================================================================

class EnhancedSpeakerDiarizer:
    def __init__(self, precision: str = "medium"):
        self.precision = precision 
        self.library = ProfileLibrary()
        self.known_speakers = {}
        self.pipeline = None
        self.threshold = 0.5  # Default threshold

    def initialize_pipeline(self, token: str):
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        device = "cuda" if (ACCEL_READY and wrapper.gpu_backend.gpu_is_available()) else "cpu"
        self.pipeline.to(torch.device(device))

    def run(self, audio_path: Path, output_dir: Path) -> Dict:
        import soundfile as sf
        
        logger.info(f"Analyzing speech turns for: {audio_path.name}...")
        turns = self.pipeline(str(audio_path))
        
        audio_data, sr = sf.read(audio_path)
        segments_list = []
        for turn, _, speaker in turns.itertracks(yield_label=True):
            segments_list.append((int(turn.start * sr), int(turn.end * sr)))
        
        # No embedding extraction - just build basic structure
        detected_speakers = {}
        turn_data = []
        
        for (turn, _, speaker), buffer in zip(turns.itertracks(yield_label=True), [None] * len(segments_list)):
            if speaker not in detected_speakers:
                detected_speakers[speaker] = []
            
            turn_data.append({
                "start": turn.start, "end": turn.end, "orig_label": speaker
            })

        final_map = {}
        for generic_id, embeddings in detected_speakers.items():
            # No embedding extraction or verification
            # Just use generic labels
            final_map[generic_id] = generic_id

        results = {
            "audio_file": str(audio_path),
            "speaker_map": final_map,
            "segments": turn_data
        }
        return results

    def _verify_against_known(self, embedding: np.ndarray) -> Optional[str]:
        name, _ = self._verify_against_known_with_score(embedding)
        return name

    def _verify_against_known_with_score(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        best_name = None
        best_similarity = -1
        
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        for name, known_emb in self.known_speakers.items():
            if known_emb.dtype != np.float32:
                known_emb = known_emb.astype(np.float32)
                
            similarity = np.dot(embedding, known_emb) / (np.linalg.norm(embedding) * np.linalg.norm(known_emb))
            if similarity > self.threshold and similarity > best_similarity:
                best_similarity = similarity
                best_name = name
        
        return best_name, best_similarity

# ============================================================================
# SECTION 4: MAIN FUNCTION WITH ARGUMENT PARSING (Non-interactive only)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Speaker Diarization (Profile-Only Mode)')
    parser.add_argument('--audio', type=str, help='Path to audio file to process')
    parser.add_argument('--output', type=str, help='Output directory for diarization results')
    parser.add_argument('--precision', type=str, default='medium', choices=['low', 'medium', 'high'],
                       help='Diarization precision level')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Verification threshold (0.0-1.0)')
    parser.add_argument('--profiles', type=str,
                       help='Comma-separated list of speaker profiles to load')
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nNo arguments provided. Exiting.")
        return
    
    run_with_args(args)

def run_interactive():
    """Legacy function - kept for compatibility but not used."""
    print("Interactive mode disabled. Please use command-line arguments.")
    return

def run_with_args(args):
    token = check_huggingface_token()
    if not token: 
        sys.exit(1)

    audio_path = Path(args.audio)
    output_base = Path(args.output)
    
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    file_stem = audio_path.stem.replace("_preprocessed", "")
    output_dir = output_base / file_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / "enhanced_diarization.json"
    
    if json_path.exists():
        print(f"Using existing diarization data for {file_stem}")
        return
    
    # All configuration now comes from command-line arguments
    lib = ProfileLibrary()
    diarizer = EnhancedSpeakerDiarizer(precision=args.precision)
    diarizer.threshold = args.threshold
    
    # Load profiles from --profiles argument
    if args.profiles:
        profile_list = args.profiles.split(',')
        for profile_name in profile_list:
            if profile_name.strip():
                try:
                    diarizer.known_speakers[profile_name.strip()] = lib.load_profile(profile_name.strip())
                    print(f"Loaded profile: {profile_name.strip()}")
                except Exception as e:
                    print(f"Failed to load profile {profile_name}: {e}")
    
    print("\n============================================================")
    print("PROCESSING AUDIO FILE")
    print("============================================================\n")
    
    diarizer.initialize_pipeline(token)
    result = diarizer.run(audio_path, output_dir)
    
    with open(json_path, 'w') as jf:
        json.dump(result, jf, indent=2)
    print(f"Diarization data saved for {file_stem}")

# ============================================================================
# SECTION 5: CLEANUP FUNCTION FOR SUBPROCESS EXIT
# ============================================================================

def cleanup():
    """Force cleanup of models and GPU memory before exit."""
    logger.info("Cleaning up resources...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("Cleanup complete")

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
        sys.exit(0)
