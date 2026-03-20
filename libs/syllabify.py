"""
syllabify.py - Accelerated Multilingual Syllabification
@author: taylosh
Created on Nov 22 2025
Last edited on Mar 16 2026

Subprocess for the Annotation Phase:
- Adds syllable tiers to aligned TextGrids using linguistic and acoustic data.
- Accelerated energy envelope and peak detection (C-Domain Spectral Analysis).
- Language support: English, Spanish, French, German.
- Designed as a functional call from 03_annotate.py.
"""

import os
import sys
import logging
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from praatio import textgrid

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

logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1: LINGUISTIC & ACOUSTIC ENGINES
# ============================================================================

class SyllableDictionaryManager:
    """Manages linguistic syllable lookups."""
    def __init__(self, dict_dir: str = "./models/syllable_dicts"):
        self.dict_dir = Path(dict_dir)
        self.dict_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}  # Simple cache for lookups

    def lookup_word(self, word: str, language: str) -> Optional[List[str]]:
        """Dictionary-based syllable division."""
        # Check cache first
        cache_key = f"{language}:{word.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try C-accelerated lookup if available
        if ACCEL_READY and hasattr(wrapper, 'syllabifier'):
            try:
                result = wrapper.syllabifier.lookup_words([word], language=language)
                if result and len(result) > 0:
                    # Handle different return formats
                    if isinstance(result[0], list):
                        syllables = result[0]
                    else:
                        syllables = result
                    
                    # Cache and return
                    self.cache[cache_key] = syllables
                    return syllables
            except Exception as e:
                logger.debug(f"C-accelerated dictionary lookup failed: {e}")
        
        # Try file-based dictionary lookup
        dict_file = self.dict_dir / f"{language}_syllables.txt"
        if dict_file.exists():
            try:
                with open(dict_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ':' in line:
                            parts = line.strip().split(':')
                            word_entry, syll = parts[0], parts[1]
                            if word_entry.lower() == word.lower():
                                syllables = syll.split('-')
                                self.cache[cache_key] = syllables
                                return syllables
            except Exception as e:
                logger.debug(f"File-based dictionary lookup failed: {e}")
        
        return None

class AcousticSyllableAnalyzer:
    """Uses C-Acceleration to find syllable peaks in audio."""
    
    def __init__(self, threshold: float = 0.3, frame_size: int = 256, hop_size: int = 128):
        self.threshold = threshold
        self.frame_size = frame_size
        self.hop_size = hop_size
    
    def detect_syllable_boundaries(self, audio_data: np.ndarray, word_timing: Tuple[float, float], 
                                  sr: int) -> Optional[List[Tuple]]:
        if not ACCEL_READY:
            return None

        try:
            # 1. Compute Energy Envelopes via C-Domain
            if hasattr(wrapper, 'syllabifier') and hasattr(wrapper.syllabifier, 'compute_energy_envelopes'):
                energy_data, energy_lengths = wrapper.syllabifier.compute_energy_envelopes(
                    audio_data.reshape(1, -1).astype(np.float32), 
                    frame_size=self.frame_size, 
                    hop_size=self.hop_size
                )
            else:
                # Fallback to Python implementation
                return self._python_energy_detection(audio_data, word_timing, sr)
            
            # 2. Detect Peaks via C-Domain
            if hasattr(wrapper.syllabifier, 'find_syllable_peaks'):
                peaks, peak_counts = wrapper.syllabifier.find_syllable_peaks(
                    energy_data, energy_lengths, peak_threshold=self.threshold
                )
            else:
                return self._python_peak_detection(energy_data, energy_lengths, word_timing, sr)
            
            # 3. Convert peaks to time boundaries
            if peak_counts > 0:
                return self._peaks_to_time(peaks[:peak_counts], word_timing, energy_lengths[0])
            return [(word_timing[0], word_timing[1], word_timing[2])]  # Fallback to whole word
            
        except Exception as e:
            logger.debug(f"Acoustic analysis failed: {e}")
            return None
    
    def _python_energy_detection(self, audio_data: np.ndarray, word_timing: Tuple[float, float], sr: int) -> List[Tuple]:
        """Python fallback for energy-based syllable detection."""
        # Simple RMS energy calculation
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i+frame_length]
            rms = np.sqrt(np.mean(frame**2))
            energy.append(rms)
        
        # Simple peak picking
        if len(energy) < 2:
            return [(word_timing[0], word_timing[1], word_timing[2])]
        
        peaks = []
        for i in range(1, len(energy)-1):
            if energy[i] > energy[i-1] and energy[i] > energy[i+1] and energy[i] > self.threshold * np.max(energy):
                time = word_timing[0] + (i * hop_length / sr)
                peaks.append(time)
        
        # Convert peaks to intervals
        if len(peaks) < 2:
            return [(word_timing[0], word_timing[1], word_timing[2])]
        
        intervals = []
        for i in range(len(peaks)):
            start = peaks[i]
            end = peaks[i+1] if i+1 < len(peaks) else word_timing[1]
            intervals.append((start, end, word_timing[2]))
        
        return intervals
    
    def _python_peak_detection(self, energy_data, energy_lengths, word_timing, sr):
        """Python fallback for peak detection."""
        return None

    def _peaks_to_time(self, peaks, timing, n_frames):
        start, end = timing[0], timing[1]
        if n_frames <= 1:
            return [(start, end, timing[2])]
        
        frame_dur = (end - start) / n_frames
        intervals = []
        
        # Convert peak indices to time intervals
        peak_times = [start + p * frame_dur for p in peaks]
        peak_times.sort()
        
        if not peak_times:
            return [(start, end, timing[2])]
        
        # Create intervals around each peak
        for i, peak_time in enumerate(peak_times):
            interval_start = peak_time - frame_dur/2
            interval_end = peak_time + frame_dur/2
            
            # Adjust boundaries
            if i == 0:
                interval_start = start
            if i == len(peak_times) - 1:
                interval_end = end
            else:
                next_peak = peak_times[i+1]
                interval_end = min(interval_end, (peak_time + next_peak)/2)
            
            intervals.append((interval_start, interval_end, timing[2]))
        
        return intervals

# ============================================================================
# SECTION 2: ANNOTATION LOGIC
# ============================================================================

class SyllableAnnotator:
    def __init__(self, 
                 language: str = "english",
                 use_dictionary: bool = True,
                 use_acoustic: bool = True,
                 dictionary_path: str = "./models/syllable_dicts",
                 acoustic_threshold: float = 0.3,
                 frame_size: int = 256,
                 hop_size: int = 128,
                 fallback_order: str = "dictionary_first"):
        
        self.language = language
        self.use_dictionary = use_dictionary
        self.use_acoustic = use_acoustic
        self.fallback_order = fallback_order  # 'dictionary_first' or 'acoustic_first'
        
        self.dicts = SyllableDictionaryManager(dict_dir=dictionary_path)
        self.acoustic = AcousticSyllableAnalyzer(
            threshold=acoustic_threshold,
            frame_size=frame_size,
            hop_size=hop_size
        )

    def process_file(self, tg_path: Path, audio_path: Path):
        """Add syllable tiers to a specific TextGrid."""
        try:
            import soundfile as sf
            
            logger.info(f"Processing: {tg_path.name}")
            tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
            audio_data, sr = sf.read(audio_path)
            
            new_tg = textgrid.Textgrid()
            new_tg.minTimestamp, new_tg.maxTimestamp = tg.minTimestamp, tg.maxTimestamp

            # Process speakers hierarchically: phrases -> words -> syllables -> phones
            speakers = [n.replace("_words", "") for n in tg.tierNames if n.endswith("_words")]
            
            for speaker in speakers:
                # Copy Phrase tier if it exists
                try:
                    new_tg.addTier(tg.getTier(f"{speaker}_phrases"))
                except Exception:
                    pass
                
                # Copy Words tier
                words_tier = tg.getTier(f"{speaker}_words")
                new_tg.addTier(words_tier)
                
                # Generate Syllable Tier
                syll_intervals = []
                word_count = 0
                syll_count = 0
                
                for entry in words_tier.entries:
                    if not entry.label.strip(): 
                        continue
                    
                    word_count += 1
                    word_label = entry.label.strip()
                    syllables = None
                    
                    # Determine order based on fallback setting
                    if self.fallback_order == "dictionary_first":
                        # Try dictionary first
                        if self.use_dictionary:
                            syllables = self.dicts.lookup_word(word_label, self.language)
                        
                        # Then try acoustic if dictionary failed and acoustic is enabled
                        if not syllables and self.use_acoustic:
                            word_buf = audio_data[int(entry.start*sr):int(entry.end*sr)]
                            if len(word_buf) > 0:
                                acoustic_result = self.acoustic.detect_syllable_boundaries(
                                    word_buf, (entry.start, entry.end, word_label), sr
                                )
                                if acoustic_result:
                                    syllables = [s[2] for s in acoustic_result]  # Extract labels
                                    # Use acoustic intervals directly
                                    for interval in acoustic_result:
                                        syll_intervals.append(interval)
                                    syll_count += len(acoustic_result)
                                    continue  # Skip the default interval creation
                    
                    else:  # acoustic_first
                        # Try acoustic first
                        if self.use_acoustic:
                            word_buf = audio_data[int(entry.start*sr):int(entry.end*sr)]
                            if len(word_buf) > 0:
                                acoustic_result = self.acoustic.detect_syllable_boundaries(
                                    word_buf, (entry.start, entry.end, word_label), sr
                                )
                                if acoustic_result:
                                    syllables = [s[2] for s in acoustic_result]
                                    for interval in acoustic_result:
                                        syll_intervals.append(interval)
                                    syll_count += len(acoustic_result)
                                    continue
                        
                        # Then try dictionary if acoustic failed and dictionary is enabled
                        if not syllables and self.use_dictionary:
                            syllables = self.dicts.lookup_word(word_label, self.language)
                    
                    # If we have dictionary syllables (not from acoustic), map them to timeline
                    if syllables and isinstance(syllables, list):
                        # Ensure syllables are strings
                        flat_syllables = []
                        for s in syllables:
                            if isinstance(s, str):
                                flat_syllables.append(s.upper())
                            elif isinstance(s, (list, tuple)):
                                flat_syllables.extend([str(item).upper() for item in s])
                            else:
                                flat_syllables.append(str(s).upper())
                        
                        if flat_syllables:
                            # Distribute word duration evenly across syllables
                            dur = (entry.end - entry.start) / len(flat_syllables)
                            for i, s in enumerate(flat_syllables):
                                syll_intervals.append((
                                    entry.start + i * dur,
                                    entry.start + (i + 1) * dur,
                                    s
                                ))
                            syll_count += len(flat_syllables)
                    else:
                        # No syllables found - use whole word as one syllable
                        syll_intervals.append((entry.start, entry.end, word_label.upper()))
                        syll_count += 1
                
                # Create syllable tier
                syll_tier = textgrid.IntervalTier(
                    f"{speaker}_syllables", 
                    syll_intervals, 
                    tg.minTimestamp, 
                    tg.maxTimestamp
                )
                new_tg.addTier(syll_tier)
                logger.info(f"  Added {speaker}_syllables tier with {syll_count} intervals from {word_count} words")
                
                # Add phones tier if it exists
                try:
                    new_tg.addTier(tg.getTier(f"{speaker}_phones"))
                except Exception:
                    # Phones tier might not exist yet
                    pass

            # Save overwrite
            new_tg.save(str(tg_path), format="long_textgrid", includeBlankSpaces=True)
            logger.info(f"  Successfully syllabified: {tg_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to syllabify {tg_path.name}: {e}")
            return False

# ============================================================================
# MAIN HOOK FOR 03_ANNOTATE.PY
# ============================================================================

def run_syllabification(
    aligned_dir: str, 
    audio_dir: str, 
    language: str = "english",
    use_dictionary: bool = True,
    use_acoustic: bool = True,
    dict_dir: str = "./models/syllable_dicts",
    threshold: float = 0.3,
    frame_size: int = 256,
    hop_size: int = 128,
    fallback_order: str = "dictionary_first"
):
    """Entry point called by the main annotation script."""
    logger.info(f"Starting syllabification for {aligned_dir}")
    logger.info(f"  Language: {language}")
    logger.info(f"  Dictionary: {'enabled' if use_dictionary else 'disabled'} (path: {dict_dir})")
    logger.info(f"  Acoustic: {'enabled' if use_acoustic else 'disabled'} (threshold: {threshold}, frame: {frame_size}, hop: {hop_size})")
    logger.info(f"  Fallback order: {fallback_order}")
    
    annotator = SyllableAnnotator(
        language=language,
        use_dictionary=use_dictionary,
        use_acoustic=use_acoustic,
        dictionary_path=dict_dir,
        acoustic_threshold=threshold,
        frame_size=frame_size,
        hop_size=hop_size,
        fallback_order=fallback_order
    )
    
    # Find all aligned TextGrids recursively
    aligned_path = Path(aligned_dir)
    tg_files = list(aligned_path.rglob("*_aligned.TextGrid"))
    
    # Filter out temp directories
    tg_files = [f for f in tg_files if 'temp_' not in str(f) and 'mfa_' not in str(f)]
    
    logger.info(f"Found {len(tg_files)} aligned TextGrids to process")
    
    count = 0
    for tg_f in tg_files:
        # Match audio file maintaining directory structure
        rel_path = tg_f.relative_to(aligned_path)
        original_stem = tg_f.name.replace("_aligned.TextGrid", "")
        
        # Try multiple audio locations
        audio_candidates = [
            Path(audio_dir) / rel_path.parent / f"{original_stem}_preprocessed.wav",
            Path(audio_dir) / rel_path.parent / f"{original_stem}.wav",
            Path(audio_dir) / f"{original_stem}_preprocessed.wav",
            Path(audio_dir) / f"{original_stem}.wav"
        ]
        
        audio_f = None
        for candidate in audio_candidates:
            if candidate.exists():
                audio_f = candidate
                break
        
        if audio_f and audio_f.exists():
            if annotator.process_file(tg_f, audio_f):
                count += 1
        else:
            logger.warning(f"Skipping {tg_f.name}: Audio not found (tried {len(audio_candidates)} locations)")

    logger.info(f"Syllabification complete. {count} files annotated.")

if __name__ == "__main__":
    # Small test block if run standalone
    pass
