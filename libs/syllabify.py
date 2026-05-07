"""
syllabify.py - Accelerated Multilingual Syllabification
@author: taylosh
Created on 22 Nov 2025
Last edited on 6 May 2026

Subprocess for the Annotation Phase:
- Adds syllable tiers to aligned TextGrids using linguistic and acoustic data.
- Three-stage pipeline: Dictionary -> Acoustic -> Universal Vowel Rules -> Language Rules
- Language support: English, Spanish, French, German.
- Designed as a functional call from 03_annotate.py.
"""

import os
import sys
import logging
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
# VOWEL SETS PER LANGUAGE (for universal rules)
# ============================================================================

VOWEL_SETS = {
    'english': {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 
                'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX'},
    'spanish': {'A', 'E', 'I', 'O', 'U'},
    'french': {'A', 'E', 'I', 'O', 'U', 'Y', 'EU', 'OU', 'AU', 'EI'},
    'german': {'A', 'E', 'I', 'O', 'U', 'Ä', 'Ö', 'Ü', 'AU', 'EI', 'EU'}
}

# ============================================================================
# SECTION 1: LINGUISTIC & ACOUSTIC ENGINES
# ============================================================================

class SyllableDictionaryManager:
    """Manages linguistic syllable lookups."""
    def __init__(self, dict_dir: str = "./models/syllable_dicts"):
        self.dict_dir = Path(dict_dir)
        self.dict_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}

    def lookup_word(self, word: str, language: str) -> Optional[List[str]]:
        """Dictionary-based syllable division."""
        cache_key = f"{language}:{word.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
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


class SyllableRuleManager:
    """Manages rule-based syllabification proofreading (language-specific)."""
    
    def __init__(self, dict_dir: str = "./models/syllable_dicts"):
        self.dict_dir = Path(dict_dir)
        self.rules_cache = {}
        
    def load_rules(self, language: str) -> List[Dict]:
        """Load language-specific rules from file."""
        cache_key = f"{language}_rules"
        if cache_key in self.rules_cache:
            return self.rules_cache[cache_key]
        
        rules_file = self.dict_dir / f"{language}_rules.txt"
        rules = []
        
        if not rules_file.exists():
            logger.debug(f"No rules file found at {rules_file}")
            self.rules_cache[cache_key] = rules
            return rules
        
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    rules.append({'raw': line, 'line_num': line_num})
            logger.info(f"Loaded {len(rules)} rules from {rules_file}")
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
        
        self.rules_cache[cache_key] = rules
        return rules
    
    def apply_language_rules(self, 
                              syll_intervals: List[Tuple[float, float, str]],
                              word_intervals: List[Dict],
                              language: str,
                              audio_data: np.ndarray = None,
                              sr: int = None) -> List[Tuple[float, float, str]]:
        """
        Apply language-specific rules to reassign consonants to correct syllables.
        For now, returns original (stubbed for future implementation).
        """
        rules = self.load_rules(language)
        
        if not rules:
            return syll_intervals
        
        # TODO: Implement language-specific consonant reassignment rules
        # For now, just pass through
        
        return syll_intervals


class AcousticSyllableAnalyzer:
    """Uses C-Acceleration to find syllable peaks in audio."""
    
    def __init__(self, threshold: float = 0.3, frame_size: int = 256, hop_size: int = 128):
        self.threshold = threshold
        self.frame_size = frame_size
        self.hop_size = hop_size
    
    def detect_syllable_boundaries(self, audio_data: np.ndarray, word_timing: Tuple[float, float, str], 
                                  sr: int) -> Optional[List[Tuple]]:
        """Detect syllable boundaries using acoustic energy peaks."""
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
                return None
            
            # 3. Convert peaks to time boundaries
            if peak_counts > 0:
                return self._peaks_to_time(peaks[:peak_counts], word_timing, energy_lengths[0])
            return [(word_timing[0], word_timing[1], word_timing[2])]
            
        except Exception as e:
            logger.debug(f"Acoustic analysis failed: {e}")
            return None
    
    def _python_energy_detection(self, audio_data: np.ndarray, word_timing: Tuple[float, float, str], sr: int) -> List[Tuple]:
        """Python fallback for energy-based syllable detection."""
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        energy = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i+frame_length]
            rms = np.sqrt(np.mean(frame**2))
            energy.append(rms)
        
        if len(energy) < 2:
            return [(word_timing[0], word_timing[1], word_timing[2])]
        
        peaks = []
        for i in range(1, len(energy)-1):
            if energy[i] > energy[i-1] and energy[i] > energy[i+1] and energy[i] > self.threshold * np.max(energy):
                time = word_timing[0] + (i * hop_length / sr)
                peaks.append(time)
        
        if len(peaks) < 2:
            return [(word_timing[0], word_timing[1], word_timing[2])]
        
        intervals = []
        for i in range(len(peaks)):
            start = peaks[i]
            end = peaks[i+1] if i+1 < len(peaks) else word_timing[1]
            intervals.append((start, end, word_timing[2]))
        
        return intervals

    def _peaks_to_time(self, peaks, timing, n_frames):
        start, end = timing[0], timing[1]
        if n_frames <= 1:
            return [(start, end, timing[2])]
        
        frame_dur = (end - start) / n_frames
        intervals = []
        
        peak_times = [start + p * frame_dur for p in peaks]
        peak_times.sort()
        
        if not peak_times:
            return [(start, end, timing[2])]
        
        for i, peak_time in enumerate(peak_times):
            interval_start = peak_time - frame_dur/2
            interval_end = peak_time + frame_dur/2
            
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
                 use_rules: bool = True,
                 dictionary_path: str = "./models/syllable_dicts",
                 acoustic_threshold: float = 0.3,
                 frame_size: int = 256,
                 hop_size: int = 128):
        
        self.language = language
        self.use_dictionary = use_dictionary
        self.use_acoustic = use_acoustic
        self.use_rules = use_rules
        
        self.dicts = SyllableDictionaryManager(dict_dir=dictionary_path)
        self.acoustic = AcousticSyllableAnalyzer(
            threshold=acoustic_threshold,
            frame_size=frame_size,
            hop_size=hop_size
        )
        self.rules = SyllableRuleManager(dict_dir=dictionary_path)

    def _apply_universal_vowel_rules(self, 
                                      syll_intervals: List[Tuple[float, float, str]],
                                      phones_tier,
                                      word_start: float,
                                      word_end: float) -> List[Tuple[float, float, str]]:
        """
        Apply universal vowel rules to ensure:
        1. Each syllable has exactly one vowel
        2. Syllable boundaries align with phone boundaries
        3. Labels are concatenated IPA (no spaces, no capitalization changes)
        """
        vowels = VOWEL_SETS.get(self.language, VOWEL_SETS['english'])
        
        # Collect all phones with timing
        phone_list = []
        for entry in phones_tier.entries:
            phone_label = entry.label.strip()
            if not phone_label:
                continue
            phone_list.append({
                'label': phone_label,  # Preserve original case/IPA
                'start': entry.start,
                'end': entry.end,
                'is_vowel': phone_label.upper() in vowels
            })
        
        if not phone_list:
            return syll_intervals
        
        # Group phones by which syllable interval they belong to
        # First, assign each phone to a syllable
        phone_to_syllable = {}
        for i, (s_start, s_end, s_label) in enumerate(syll_intervals):
            for p in phone_list:
                if p['start'] >= s_start - 0.001 and p['end'] <= s_end + 0.001:
                    phone_to_syllable[id(p)] = i
        
        # Check each syllable for vowel count
        corrected_intervals = []
        vowel_assigned = set()
        
        for i, (s_start, s_end, s_label) in enumerate(syll_intervals):
            # Get phones in this syllable
            phones_in_syll = [p for p in phone_list 
                              if p['start'] >= s_start - 0.001 and p['end'] <= s_end + 0.001]
            
            vowel_count = sum(1 for p in phones_in_syll if p['is_vowel'])
            vowel_phones = [p for p in phones_in_syll if p['is_vowel']]
            
            if vowel_count == 1:
                # Perfect - use phone concatenation as label (preserving original case)
                new_label = "".join([p['label'] for p in phones_in_syll])
                
                # Check vowel not already assigned
                for vp in vowel_phones:
                    vowel_key = (vp['start'], vp['end'])
                    if vowel_key in vowel_assigned:
                        logger.warning(f"Vowel {vp['label']} already assigned to another syllable!")
                    else:
                        vowel_assigned.add(vowel_key)
                
                corrected_intervals.append((s_start, s_end, new_label))
                
            elif vowel_count == 0:
                # No vowel - this syllable should be merged with adjacent
                logger.debug(f"Syllable has no vowel, merging with next")
                # Merge with next syllable (will be handled in post-processing)
                if i + 1 < len(syll_intervals):
                    # Extend current to include next
                    merged_start = s_start
                    merged_end = syll_intervals[i+1][1]
                    merged_phones = [p for p in phone_list 
                                      if p['start'] >= merged_start - 0.001 and p['end'] <= merged_end + 0.001]
                    merged_label = "".join([p['label'] for p in merged_phones])
                    corrected_intervals.append((merged_start, merged_end, merged_label))
                    # Skip the next interval
                    vowel_assigned.add(id(syll_intervals[i+1]))
                else:
                    # Last syllable with no vowel - just use what we have
                    new_label = "".join([p['label'] for p in phones_in_syll])
                    corrected_intervals.append((s_start, s_end, new_label))
                    
            else:  # vowel_count > 1
                # Multiple vowels - split into multiple syllables
                logger.debug(f"Syllable has {vowel_count} vowels, splitting")
                
                # Find boundaries between vowels
                vowel_positions = []
                for p in phones_in_syll:
                    if p['is_vowel']:
                        vowel_positions.append(p)
                
                for j, vp in enumerate(vowel_positions):
                    # Find start boundary
                    if j == 0:
                        v_start = s_start
                    else:
                        # Start after previous vowel's end
                        prev_vowel = vowel_positions[j-1]
                        # Calculate desired split time (midpoint)
                        desired_start = (prev_vowel['end'] + vp['start']) / 2
                        # Snap to nearest phone boundary (walk backward)
                        v_start = self._snap_to_phone_boundary(desired_start, phones_in_syll)
                    
                    # Find end boundary
                    if j == len(vowel_positions) - 1:
                        v_end = s_end
                    else:
                        next_vowel = vowel_positions[j+1]
                        # Calculate desired split time (midpoint)
                        desired_end = (vp['end'] + next_vowel['start']) / 2
                        # Snap to nearest phone boundary (walk backward)
                        v_end = self._snap_to_phone_boundary(desired_end, phones_in_syll)
                    
                    # Get phones in this sub-syllable
                    sub_phones = [p for p in phones_in_syll 
                                if p['start'] >= v_start - 0.001 and p['end'] <= v_end + 0.001]
                    sub_label = "".join([p['label'] for p in sub_phones])
                    
                    vowel_key = (vp['start'], vp['end'])
                    if vowel_key not in vowel_assigned:
                        vowel_assigned.add(vowel_key)
                        corrected_intervals.append((v_start, v_end, sub_label))
        
        return corrected_intervals

    def _snap_to_phone_boundary(self, time: float, phone_list: List[Dict]) -> float:
        """
        Snap a time to the nearest phone boundary (walking backward).
        If time exactly matches a boundary, return it.
        Otherwise, walk backward to find the closest phone start time.
        """
        # Check if time exactly matches any phone start or end
        for phone in phone_list:
            if abs(time - phone['start']) < 0.001 or abs(time - phone['end']) < 0.001:
                return time
        
        # Walk backward to find the nearest phone start
        for phone in reversed(phone_list):
            if phone['start'] <= time:
                return phone['start']
        
        # Fallback: return original time
        return time    
    
    def _refine_with_acoustic_timing(self, dictionary_syllables: List[str], 
                                      acoustic_result: Optional[List[Tuple]],
                                      word_entry, word_buf: np.ndarray, 
                                      sr: int) -> List[Tuple[float, float, str]]:
        """Refine syllable timing using acoustic detection."""
        if not acoustic_result or len(acoustic_result) != len(dictionary_syllables):
            # No acoustic refinement - even distribution
            dur = (word_entry.end - word_entry.start) / len(dictionary_syllables)
            intervals = []
            for i, syl in enumerate(dictionary_syllables):
                intervals.append((
                    word_entry.start + i * dur,
                    word_entry.start + (i + 1) * dur,
                    syl  # Preserve original dictionary label (already correct case)
                ))
            return intervals
        
        # Use acoustic timing with dictionary labels
        intervals = []
        for i, (start, end, _) in enumerate(acoustic_result):
            label = dictionary_syllables[i] if i < len(dictionary_syllables) else "?"
            intervals.append((start, end, label))
        
        return intervals

    def _create_acoustic_syllables(self, acoustic_result: Optional[List[Tuple]],
                                    word_entry, word_label: str,
                                    word_buf: np.ndarray, 
                                    sr: int) -> List[Tuple[float, float, str]]:
        """Create syllables for non-dictionary words using acoustic detection."""
        if not acoustic_result:
            return [(word_entry.start, word_entry.end, word_label)]
        
        intervals = []
        for i, (start, end, _) in enumerate(acoustic_result):
            provisional_label = f"S{i+1}"
            intervals.append((start, end, provisional_label))
        
        return intervals

    def process_file(self, tg_path: Path, audio_path: Path):
        """Add syllable tiers to a specific TextGrid."""
        try:
            import soundfile as sf
            
            logger.info(f"Processing: {tg_path.name}")
            tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
            audio_data, sr = sf.read(audio_path)
            
            new_tg = textgrid.Textgrid()
            new_tg.minTimestamp, new_tg.maxTimestamp = tg.minTimestamp, tg.maxTimestamp

            # Process speakers hierarchically
            speakers = [n.replace("_words", "") for n in tg.tierNames if n.endswith("_words")]
            
            for speaker in speakers:
                # Copy existing tiers
                try:
                    new_tg.addTier(tg.getTier(f"{speaker}_phrases"))
                except Exception:
                    pass
                
                # Copy Words tier
                words_tier = tg.getTier(f"{speaker}_words")
                new_tg.addTier(words_tier)
                
                # Copy Phones tier (needed for universal vowel rules)
                phones_tier_exists = False
                try:
                    phones_tier = tg.getTier(f"{speaker}_phones")
                    new_tg.addTier(phones_tier)
                    phones_tier_exists = True
                except Exception:
                    logger.warning(f"No phones tier found for {speaker}, cannot apply vowel rules")
                
                # Generate initial syllable intervals
                all_word_intervals = []
                syll_intervals = []
                word_count = 0
                
                for entry in words_tier.entries:
                    if not entry.label.strip(): 
                        continue
                    
                    word_count += 1
                    word_label = entry.label.strip()
                    
                    all_word_intervals.append({
                        'start': entry.start,
                        'end': entry.end,
                        'label': word_label,
                        'intervals': None
                    })
                    
                    # STAGE 1: DICTIONARY LOOKUP
                    dictionary_syllables = None
                    if self.use_dictionary:
                        dict_result = self.dicts.lookup_word(word_label.lower(), self.language)
                        if dict_result:
                            dictionary_syllables = dict_result
                    
                    # STAGE 2: ACOUSTIC ANALYSIS
                    start_sample = int(entry.start * sr)
                    end_sample = int(entry.end * sr)
                    word_buf = audio_data[start_sample:end_sample]
                    acoustic_result = None
                    
                    if self.use_acoustic and len(word_buf) > 0:
                        acoustic_result = self.acoustic.detect_syllable_boundaries(
                            word_buf, (entry.start, entry.end, word_label), sr
                        )
                    
                    # COMBINE DICTIONARY + ACOUSTIC
                    if dictionary_syllables:
                        refined_intervals = self._refine_with_acoustic_timing(
                            dictionary_syllables, acoustic_result, entry, word_buf, sr
                        )
                    else:
                        refined_intervals = self._create_acoustic_syllables(
                            acoustic_result, entry, word_label, word_buf, sr
                        )
                    
                    all_word_intervals[-1]['intervals'] = refined_intervals
                    
                    for interval in refined_intervals:
                        syll_intervals.append(interval)
                
                # STAGE 3: UNIVERSAL VOWEL RULES
                if phones_tier_exists and syll_intervals:
                    syll_intervals = self._apply_universal_vowel_rules(
                        syll_intervals, phones_tier, tg.minTimestamp, tg.maxTimestamp
                    )
                    logger.debug(f"  Applied universal vowel rules for {speaker}")
                
                # STAGE 4: LANGUAGE-SPECIFIC RULES
                if self.use_rules and syll_intervals:
                    syll_intervals = self.rules.apply_language_rules(
                        syll_intervals, all_word_intervals, self.language, audio_data, sr
                    )
                    logger.debug(f"  Applied language-specific rules for {speaker}")
                
                # Create syllable tier
                if syll_intervals:
                    syll_tier = textgrid.IntervalTier(
                        f"{speaker}_syllables", 
                        syll_intervals, 
                        tg.minTimestamp, 
                        tg.maxTimestamp
                    )
                    new_tg.addTier(syll_tier)
                    logger.info(f"  Added {speaker}_syllables tier with {len(syll_intervals)} intervals from {word_count} words")
                
                # Add phones tier back if it exists
                try:
                    new_tg.addTier(tg.getTier(f"{speaker}_phones"))
                except Exception:
                    pass

            # SAVE TO aligned_textgrids/
            new_tg.save(str(tg_path), format="long_textgrid", includeBlankSpaces=True)
            
            # ALSO SAVE COPY TO final_textgrids/
            relative_path = tg_path.relative_to(Path("./aligned_textgrids"))
            final_filename = relative_path.name.replace("_aligned.TextGrid", ".TextGrid")
            final_path = Path("./final_textgrids") / relative_path.parent / final_filename
            final_path.parent.mkdir(parents=True, exist_ok=True)
            new_tg.save(str(final_path), format="long_textgrid", includeBlankSpaces=True)
            
            logger.info(f"  Successfully syllabified: {tg_path.name}")
            logger.info(f"  Also saved copy to: {final_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to syllabify {tg_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
    use_rules: bool = True,
    dict_dir: str = "./models/syllable_dicts",
    threshold: float = 0.3,
    frame_size: int = 256,
    hop_size: int = 128
):
    """Entry point called by the main annotation script."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info(f"Starting syllabification for {aligned_dir}")
    logger.info(f"  Language: {language}")
    logger.info(f"  Dictionary: {'enabled' if use_dictionary else 'disabled'} (path: {dict_dir})")
    logger.info(f"  Acoustic: {'enabled' if use_acoustic else 'disabled'} (threshold: {threshold})")
    logger.info(f"  Rules: {'enabled' if use_rules else 'disabled'} (path: {dict_dir})")
    
    annotator = SyllableAnnotator(
        language=language,
        use_dictionary=use_dictionary,
        use_acoustic=use_acoustic,
        use_rules=use_rules,
        dictionary_path=dict_dir,
        acoustic_threshold=threshold,
        frame_size=frame_size,
        hop_size=hop_size
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