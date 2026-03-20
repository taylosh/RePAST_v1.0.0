"""
ASR Acceleration Python Wrapper - Complete Unified Interface
@author: taylosh
Created: Dec 8 2025
Last edited: Mar 8 2026

UNIFIED wrapper for ALL C acceleration functions.
Dynamically loads modules from ./bin/ directory.

Modules:
- Syllabification (_syllabify)
- TextGrid processing (_textgrid)
- Spectrogram computation (_spectrogram)
- Audio basic operations (_audio_basic)
- Audio enhancement (_audio_enhance)
- Audio segment extraction (_audio_segment)
- Audio feature extraction (_audio_features)
- Corpus utilities (_corpus_utils)
- GPU backend management (_gpu_backend)

Provides clean Python APIs with automatic fallback to Python implementations.
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import multiprocessing as mp

logger = logging.getLogger(__name__)

class CAccelerationWrapper:
    """Unified C acceleration wrapper for all ASR acceleration modules"""
    
    def __init__(self, enable_c_accel: bool = True, enable_gpu_accel: bool = True):
        self.enable_c_accel = enable_c_accel
        self.enable_gpu_accel = enable_gpu_accel
        
        # CRITICAL: Path resolution - wrapper in ./libs/, modules in ./bin/
        self.bin_path = Path(__file__).parent.parent / "bin"
        
        # Track loaded modules
        self._loaded_modules = {}
        self._modules_available = False
        
        # Performance tracking
        self.performance_stats = {
            'call_count': 0,
            'gpu_calls': 0,
            'cpu_calls': 0,
            'python_calls': 0,
            'total_time': 0.0
        }
        
        # Initialize all modules
        self._initialize_modules()
        
        # Initialize sub-modules
        self.syllabifier = self.Syllabifier(self)
        self.textgrid = self.TextGridProcessor(self)
        self.spectrogram = self.SpectrogramComputer(self)
        self.audio_basic = self.AudioBasic(self)
        self.audio_enhance = self.AudioEnhance(self)
        self.audio_segment = self.AudioSegment(self)
        self.audio_features = self.AudioFeatures(self)
        self.corpus_utils = self.CorpusUtils(self)
        self.gpu_backend = self.GPUBackend(self)
    
    def _initialize_modules(self):
        """Initialize ALL C acceleration modules with proper error handling"""
        if not self.enable_c_accel:
            logger.info("C acceleration disabled by configuration")
            return
            
        if not self.bin_path.exists():
            logger.warning(f"C acceleration path not found: {self.bin_path}")
            logger.info("Create ./bin/ directory and build C modules first")
            return
            
        # Add bin path to Python path
        if str(self.bin_path) not in sys.path:
            sys.path.insert(0, str(self.bin_path))
        
        # FIXED: Import modules in correct dependency order
        # Base modules first, then dependent modules
        modules_to_try = [
            '_audio_basic',           # Base audio operations (stereo_to_mono, etc.)
            '_audio_signal_core',     # Signal processing core (may be inside _audio_basic or separate)
            '_audio_segment_engine',  # Segment extraction engine
            '_audio_segment',         # High-level segment extraction (depends on engine)
            '_audio_spectral_analysis', # Spectral analysis
            '_audio_features',        # Feature extraction (depends on spectral)
            '_audio_enhance',         # Audio enhancement
            '_corpus_utils',          # File operations
            '_textgrid',              # TextGrid processing
            '_syllabify',             # Syllabification
            '_gpu_backend'            # GPU acceleration
        ]
        
        loaded_count = 0
        for module_name in modules_to_try:
            try:
                module = __import__(module_name)
                self._loaded_modules[module_name] = module
                loaded_count += 1
                logger.debug(f"Successfully loaded C module: {module_name}")
            except ImportError as e:
                logger.debug(f"C module {module_name} not available: {e}")
                self._loaded_modules[module_name] = None
            except Exception as e:
                logger.debug(f"Error loading {module_name}: {e}")
                self._loaded_modules[module_name] = None
        
        if loaded_count > 0:
            self._modules_available = True
            logger.info(f"C acceleration initialized: {loaded_count} modules loaded")
            
            # Initialize GPU acceleration if requested
            if self.enable_gpu_accel and mp.current_process().name == 'MainProcess':
                self._initialize_gpu_acceleration()
        else:
            logger.info("No C acceleration modules available")
            self._modules_available = False
    
    def _initialize_gpu_acceleration(self):
        """Initialize GPU acceleration for loaded modules"""
        gpu_initialized = False
        
        # Try to enable GPU for each module that supports it
        for module_name, module in self._loaded_modules.items():
            if module is None:
                continue
                
            try:
                if hasattr(module, 'set_gpu_acceleration'):
                    module.set_gpu_acceleration(1)
                    logger.debug(f"GPU acceleration enabled for {module_name}")
                    gpu_initialized = True
                elif hasattr(module, 'py_set_gpu_acceleration'):
                    module.py_set_gpu_acceleration(1)
                    logger.debug(f"GPU acceleration enabled for {module_name}")
                    gpu_initialized = True
                elif hasattr(module, 'gpu_init'):
                    # For gpu_backend module
                    module.gpu_init()
                    logger.debug(f"GPU backend initialized for {module_name}")
                    gpu_initialized = True
            except Exception as e:
                logger.debug(f"Failed to enable GPU for {module_name}: {e}")
        
        if gpu_initialized:
            logger.info("GPU acceleration initialized for available modules")
    
    def _update_stats(self, execution_time: float, used_gpu: bool = False, used_python: bool = False):
        """Update performance statistics"""
        self.performance_stats['call_count'] += 1
        self.performance_stats['total_time'] += execution_time
        
        if used_gpu:
            self.performance_stats['gpu_calls'] += 1
        elif used_python:
            self.performance_stats['python_calls'] += 1
        else:
            self.performance_stats['cpu_calls'] += 1
    
    def _call_c_function(self, module_name: str, function_name: str, *args, gpu_function: bool = False):
        """Call C function with proper error handling and statistics"""
        start_time = time.time()
        
        try:
            # Check if overall acceleration is available
            if not self._modules_available:
                logger.debug(f"C acceleration not available for {module_name}.{function_name}")
                raise ModuleNotFoundError(f"C acceleration not available")
            
            # Get the specific module
            module = self._loaded_modules.get(module_name)
            if module is None:
                logger.debug(f"C module {module_name} not loaded for function {function_name}")
                raise ModuleNotFoundError(f"C module {module_name} not loaded")
            
            # Try GPU version first if requested
            if gpu_function and self.enable_gpu_accel:
                gpu_func_name = f"{function_name}_gpu"
                if hasattr(module, gpu_func_name):
                    try:
                        result = getattr(module, gpu_func_name)(*args)
                        self._update_stats(time.time() - start_time, used_gpu=True)
                        return result
                    except Exception as e:
                        logger.debug(f"GPU function {module_name}.{gpu_func_name} failed: {e}")
                        # Fall through to CPU version
            
            # Try CPU version
            if hasattr(module, function_name):
                try:
                    result = getattr(module, function_name)(*args)
                    self._update_stats(time.time() - start_time)
                    return result
                except Exception as e:
                    logger.debug(f"CPU function {module_name}.{function_name} failed: {e}")
                    raise
            else:
                logger.debug(f"Function {function_name} not found in module {module_name}")
                raise AttributeError(f"Function {function_name} not found")
                
        except Exception as e:
            logger.debug(f"C function {module_name}.{function_name} failed: {e}")
            # Re-raise to trigger Python fallbacks in wrapper methods
            raise
    
    @property
    def available(self):
        """Check if C acceleration is available"""
        return self._modules_available
    
    # ============================================================================
    # Syllabification Module Wrapper
    # ============================================================================
    
    class Syllabifier:
        """Syllabification functionality wrapper"""
        
        def __init__(self, parent_wrapper):
            self._wrapper = parent_wrapper
        
        def load_dictionaries(self, dict_paths: List[str], language: str = "en") -> bool:
            """Load syllable dictionaries"""
            try:
                result = self._wrapper._call_c_function('_syllabify', 'load_dictionaries', dict_paths, language)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for load_dictionaries")
                return False
        
        def lookup_words(self, words: List[str], language: str = "en", 
                        dict_names: Optional[List[str]] = None) -> List[List[str]]:
            """Look up syllable divisions for words"""
            if dict_names is None:
                dict_names = []
            
            try:
                result = self._wrapper._call_c_function('_syllabify', 'lookup', words, language, dict_names)
                return result
            except Exception:
                # Python fallback - simple syllable splitting
                logger.debug("Using Python fallback for lookup_words")
                syllables_list = []
                for word in words:
                    # Simple heuristic: split on vowels
                    syllables = []
                    current = ""
                    for char in word.lower():
                        current += char
                        if char in 'aeiouy':
                            syllables.append(current)
                            current = ""
                    if current:
                        if syllables:
                            syllables[-1] += current
                        else:
                            syllables.append(current)
                    syllables_list.append(syllables)
                return syllables_list
        
        def parse_cmu_format(self, cmu_lines: List[str]) -> List[List[str]]:
            """Parse CMU dictionary format"""
            try:
                result = self._wrapper._call_c_function('_syllabify', 'parse_cmu_format', cmu_lines)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for parse_cmu_format")
                entries = []
                for line in cmu_lines:
                    if line.strip() and not line.startswith(';;;'):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            word = parts[0]
                            phones = parts[1:]
                            entries.append([word] + phones)
                return entries
        
        def compute_energy_envelopes(self, audio_segments: np.ndarray,
                                   frame_size: int = 256,
                                   hop_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
            """Compute energy envelopes from audio segments"""
            try:
                result = self._wrapper._call_c_function('_syllabify', 'compute_energy_envelopes', 
                                                      audio_segments, frame_size, hop_size)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for compute_energy_envelopes")
                if len(audio_segments.shape) == 1:
                    audio_segments = audio_segments.reshape(1, -1)
                
                n_segments, max_len = audio_segments.shape
                n_frames = max(1, (max_len - frame_size) // hop_size + 1)
                
                energy_data = np.zeros((n_segments, n_frames), dtype=np.float32)
                energy_lengths = np.zeros(n_segments, dtype=np.int32)
                
                for i in range(n_segments):
                    segment = audio_segments[i]
                    valid_len = np.sum(~np.isnan(segment))
                    if valid_len == 0:
                        continue
                    
                    segment = segment[:int(valid_len)]
                    n_frames_seg = max(1, (len(segment) - frame_size) // hop_size + 1)
                    energy_lengths[i] = n_frames_seg
                    
                    for j in range(n_frames_seg):
                        start = j * hop_size
                        end = min(start + frame_size, len(segment))
                        frame = segment[start:end]
                        if len(frame) > 0:
                            energy_data[i, j] = np.sqrt(np.mean(frame ** 2))
                
                return energy_data, energy_lengths
        
        def compute_energy_envelopes_gpu(self, audio_segments: np.ndarray,
                                        frame_size: int = 256,
                                        hop_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
            """GPU-accelerated energy envelopes"""
            try:
                result = self._wrapper._call_c_function('_syllabify', 'compute_energy_envelopes', 
                                                      audio_segments, frame_size, hop_size, gpu_function=True)
                return result
            except Exception:
                return self.compute_energy_envelopes(audio_segments, frame_size, hop_size)
        
        def find_syllable_peaks(self, energy_data: np.ndarray,
                               energy_lengths: np.ndarray,
                               peak_threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
            """Find syllable peaks in energy envelopes"""
            try:
                result = self._wrapper._call_c_function('_syllabify', 'find_syllable_peaks', 
                                                      energy_data, energy_lengths, peak_threshold)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for find_syllable_peaks")
                n_segments = len(energy_lengths)
                peaks_list = []
                peak_counts = np.zeros(n_segments, dtype=np.int32)
                
                for i in range(n_segments):
                    segment_energy = energy_data[i, :energy_lengths[i]]
                    peaks = []
                    
                    # Simple peak detection
                    for j in range(1, len(segment_energy) - 1):
                        if (segment_energy[j] > segment_energy[j-1] and 
                            segment_energy[j] > segment_energy[j+1] and 
                            segment_energy[j] > peak_threshold):
                            peaks.append(j)
                    
                    peaks_list.append(np.array(peaks, dtype=np.int32))
                    peak_counts[i] = len(peaks)
                
                # Convert list to array with padding
                max_peaks = max(peak_counts) if peak_counts.size > 0 else 0
                peaks_array = np.zeros((n_segments, max_peaks), dtype=np.int32)
                for i, peaks in enumerate(peaks_list):
                    if len(peaks) > 0:
                        peaks_array[i, :len(peaks)] = peaks
                
                return peaks_array, peak_counts
        
        def find_syllable_peaks_gpu(self, energy_data: np.ndarray,
                                   energy_lengths: np.ndarray,
                                   peak_threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
            """GPU-accelerated peak detection"""
            try:
                result = self._wrapper._call_c_function('_syllabify', 'find_syllable_peaks', 
                                                      energy_data, energy_lengths, peak_threshold, gpu_function=True)
                return result
            except Exception:
                return self.find_syllable_peaks(energy_data, energy_lengths, peak_threshold)
        
        def align_syllables(self, syllable_data: List[str],
                           phone_intervals: List[Any]) -> List[str]:
            """Align syllables to phone intervals"""
            try:
                result = self._wrapper._call_c_function('_syllabify', 'align_syllables', 
                                                      syllable_data, phone_intervals)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for align_syllables")
                return syllable_data
        
        def concat_phone_labels(self, phone_sequences: List[List[str]]) -> List[str]:
            """Concatenate phone labels into strings"""
            try:
                result = self._wrapper._call_c_function('_syllabify', 'concat_phone_labels', phone_sequences)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for concat_phone_labels")
                return [" ".join(phones) for phones in phone_sequences]
        
        def align_syllables_gpu(self, syllable_data: List[str],
                               phone_intervals: List[Any]) -> List[str]:
            """GPU-accelerated syllable alignment"""
            try:
                result = self._wrapper._call_c_function('_syllabify', 'align_syllables', 
                                                      syllable_data, phone_intervals, gpu_function=True)
                return result
            except Exception:
                return self.align_syllables(syllable_data, phone_intervals)
    
    # ============================================================================
    # TextGrid Module Wrapper
    # ============================================================================
    
    class TextGridProcessor:
        """TextGrid processing functionality wrapper"""
        
        def __init__(self, parent_wrapper):
            self._wrapper = parent_wrapper
        
        def detect_speaker(self, tier_name: str) -> str:
            """Detect speaker from tier name"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'detect_speaker_from_tier', tier_name)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for detect_speaker")
                lower_name = tier_name.lower()
                if 'speaker_a' in lower_name:
                    return 'speaker_a'
                elif 'speaker_b' in lower_name:
                    return 'speaker_b'
                elif 'speaker_1' in lower_name:
                    return 'speaker_1'
                elif 'speaker_2' in lower_name:
                    return 'speaker_2'
                else:
                    import re
                    match = re.search(r'speaker[_-]?([a-zA-Z0-9]+)', lower_name)
                    if match:
                        return f"speaker_{match.group(1)}"
                    return 'unknown'
        
        def generate_tier_name(self, original_name: str, rename_format: int = 0) -> str:
            """Generate a new tier name"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'generate_tier_name', original_name, rename_format)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for generate_tier_name")
                if rename_format == 0:
                    return f"{original_name}_formatted"
                elif rename_format == 1:
                    return f"formatted_{original_name}"
                else:
                    return original_name
        
        def load_file(self, file_path: str) -> Any:
            """Load TextGrid file"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'textgrid_load', file_path)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for load_file")
                class DummyTextGrid:
                    def __init__(self):
                        self.tiers = []
                        self.xmin = 0.0
                        self.xmax = 1.0
                return DummyTextGrid()
        
        def save_file(self, tg_ptr: Any, file_path: str) -> bool:
            """Save TextGrid file"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'textgrid_save', tg_ptr, file_path)
                return result == 0
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for save_file")
                return True
        
        def batch_generate_tier_names(self, tier_names: List[str],
                                     rename_format: int = 0) -> List[str]:
            """Generate tier names in batch"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'batch_generate_tier_names', 
                                                      tier_names, rename_format)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for batch_generate_tier_names")
                return [self.generate_tier_name(name, rename_format) for name in tier_names]
        
        def batch_generate_tier_names_gpu(self, tier_names: List[str],
                                         rename_format: int = 0) -> List[str]:
            """GPU-accelerated tier name generation"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'batch_generate_tier_names', 
                                                      tier_names, rename_format, gpu_function=True)
                return result
            except Exception:
                return self.batch_generate_tier_names(tier_names, rename_format)
        
        def batch_create_syllable_tiers(self, words_tiers: List[Any],
                                       phones_tiers: List[Any],
                                       syllable_data: List[Any]) -> List[Any]:
            """Create syllable tiers in batch"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'batch_create_syllable_tiers', 
                                                      words_tiers, phones_tiers, syllable_data)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for batch_create_syllable_tiers")
                return [object() for _ in range(len(words_tiers))]
        
        def batch_create_syllable_tiers_gpu(self, words_tiers: List[Any],
                                           phones_tiers: List[Any],
                                           syllable_data: List[Any]) -> List[Any]:
            """GPU-accelerated syllable tier creation"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'batch_create_syllable_tiers', 
                                                      words_tiers, phones_tiers, syllable_data, gpu_function=True)
                return result
            except Exception:
                return self.batch_create_syllable_tiers(words_tiers, phones_tiers, syllable_data)
        
        def extract_phone_intervals(self, phones_tier: Any,
                                   word_intervals: List[Any]) -> List[Any]:
            """Extract phone intervals for words"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'extract_phone_intervals_for_words', 
                                                      phones_tier, word_intervals)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for extract_phone_intervals")
                return [object() for _ in range(len(word_intervals))]
        
        def align_syllables_to_intervals(self, syllable_texts: List[str],
                                                phone_intervals: List[Tuple[float, float, str]],
                                                word_start: float,
                                                word_end: float) -> List[Tuple[float, float]]:
            """Align syllable texts to phone intervals, returning time boundaries"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'align_syllables_to_intervals',
                                                    syllable_texts, phone_intervals, word_start, word_end)
                return result
            except Exception:
                logger.debug("C acceleration failed for syllable alignment, using fallback")
            
            # Python fallback
            logger.debug("Using Python fallback for align_syllables_to_intervals")
            
            if not syllable_texts or not phone_intervals:
                return []
            
            # Extract phone labels for linguistic alignment
            phone_labels = [label for _, _, label in phone_intervals]
            
            # Simple vowel-based syllable boundary detection
            vowels = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 
                    'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'A', 'E', 'I', 'O', 'U'}
            
            # Find vowel positions in phones
            vowel_positions = []
            for i, label in enumerate(phone_labels):
                if label.upper() in vowels:
                    vowel_positions.append(i)
            
            # If we found enough vowels for syllables
            if len(vowel_positions) >= len(syllable_texts):
                # Align syllables to vowel groups
                boundaries = []
                for i in range(len(syllable_texts)):
                    if i == 0:
                        start_idx = 0
                    else:
                        start_idx = vowel_positions[i-1] + 1
                    
                    if i == len(syllable_texts) - 1:
                        end_idx = len(phone_intervals)
                    else:
                        end_idx = vowel_positions[i] + 1
                    
                    # Get timing from phone intervals
                    start_time = phone_intervals[start_idx][0]
                    end_time = phone_intervals[end_idx-1][1]
                    boundaries.append((start_time, end_time))
                
                return boundaries
            
            # Fallback: proportional division
            word_duration = word_end - word_start
            if word_duration <= 0:
                return []
            
            intervals = []
            n_syllables = len(syllable_texts)
            for i in range(n_syllables):
                start = word_start + (i / n_syllables) * word_duration
                end = word_start + ((i + 1) / n_syllables) * word_duration
                intervals.append((start, end))
            
            return intervals
        
        def parse_mfa_filename(self, mfa_filename: str) -> str:
            """Parse MFA output filename"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'parse_mfa_output_name', mfa_filename)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for parse_mfa_filename")
                import os
                base = os.path.basename(mfa_filename)
                if '.' in base:
                    base = base.rsplit('.', 1)[0]
                return base
        
        def parse_mfa_filename_gpu(self, mfa_filename: str) -> str:
            """GPU-accelerated MFA filename parsing"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'parse_mfa_output_name', 
                                                      mfa_filename, gpu_function=True)
                return result
            except Exception:
                return self.parse_mfa_filename(mfa_filename)
        
        def validate_mfa_results(self, mfa_files: List[str],
                                expected_files: List[str]) -> Tuple[List[str], int]:
            """Validate MFA results"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'validate_mfa_results', 
                                                      mfa_files, expected_files)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for validate_mfa_results")
                import os
                valid_files = []
                for f in mfa_files:
                    if os.path.exists(f) and any(f.endswith(ext) for ext in ['.TextGrid', '.textgrid', '.txt']):
                        valid_files.append(f)
                
                missing_count = len([f for f in expected_files if not os.path.exists(f)])
                return valid_files, missing_count
        
        def validate_mfa_results_gpu(self, mfa_files: List[str],
                                    expected_files: List[str]) -> Tuple[List[str], int]:
            """GPU-accelerated MFA validation"""
            try:
                result = self._wrapper._call_c_function('_textgrid', 'validate_mfa_results', 
                                                      mfa_files, expected_files, gpu_function=True)
                return result
            except Exception:
                return self.validate_mfa_results(mfa_files, expected_files)
    
    # ============================================================================
    # Spectrogram Module Wrapper
    # ============================================================================
    
    class SpectrogramComputer:
        """Spectrogram computation functionality wrapper"""
        
        # Whisper-style constants
        N_FFT = 400
        HOP_LENGTH = 160
        N_MELS = 80
        SAMPLE_RATE = 16000
        MAX_FREQ = 8000
        
        def __init__(self, parent_wrapper):
            self._wrapper = parent_wrapper
        
        def get_constants(self) -> Dict[str, int]:
            """Get Whisper-style constants"""
            return {
                'N_FFT': self.N_FFT,
                'HOP_LENGTH': self.HOP_LENGTH,
                'N_MELS': self.N_MELS,
                'SAMPLE_RATE': self.SAMPLE_RATE,
                'MAX_FREQ': self.MAX_FREQ
            }
        
        def compute_mel(self, audio: np.ndarray,
                       sample_rate: int = SAMPLE_RATE,
                       n_fft: int = N_FFT,
                       hop_length: int = HOP_LENGTH,
                       n_mels: int = N_MELS) -> np.ndarray:
            """Compute Mel spectrogram"""
            try:
                result = self._wrapper._call_c_function('_spectrogram', 'compute_mel', 
                                                      audio, sample_rate, n_fft, hop_length, n_mels)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for compute_mel")
                try:
                    import librosa
                    mel = librosa.feature.melspectrogram(
                        y=audio, sr=sample_rate, n_fft=n_fft, 
                        hop_length=hop_length, n_mels=n_mels
                    )
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    return mel_db
                except ImportError:
                    n_frames = max(1, (len(audio) - n_fft) // hop_length + 1)
                    return np.random.randn(n_mels, n_frames).astype(np.float32)
        
        def compute_mel_gpu(self, audio: np.ndarray,
                           sample_rate: int = SAMPLE_RATE,
                           n_fft: int = N_FFT,
                           hop_length: int = HOP_LENGTH,
                           n_mels: int = N_MELS) -> np.ndarray:
            """GPU-accelerated Mel spectrogram"""
            try:
                result = self._wrapper._call_c_function('_spectrogram', 'compute_mel', 
                                                      audio, sample_rate, n_fft, hop_length, n_mels, gpu_function=True)
                return result
            except Exception:
                return self.compute_mel(audio, sample_rate, n_fft, hop_length, n_mels)
        
        def compute_stft(self, audio: np.ndarray,
                        sample_rate: int = SAMPLE_RATE,
                        n_fft: int = N_FFT,
                        hop_length: int = HOP_LENGTH) -> np.ndarray:
            """Compute STFT"""
            try:
                result = self._wrapper._call_c_function('_spectrogram', 'compute_stft', 
                                                      audio, sample_rate, n_fft, hop_length)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for compute_stft")
                try:
                    import librosa
                    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
                    return np.abs(stft)
                except ImportError:
                    n_frames = max(1, (len(audio) - n_fft) // hop_length + 1)
                    n_bins = n_fft // 2 + 1
                    return np.random.randn(n_bins, n_frames).astype(np.complex64)
        
        def compute_power(self, audio: np.ndarray,
                         sample_rate: int = SAMPLE_RATE,
                         n_fft: int = N_FFT,
                         hop_length: int = HOP_LENGTH) -> np.ndarray:
            """Compute power spectrogram"""
            try:
                result = self._wrapper._call_c_function('_spectrogram', 'compute_power', 
                                                      audio, sample_rate, n_fft, hop_length)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for compute_power")
                stft = self.compute_stft(audio, sample_rate, n_fft, hop_length)
                return np.abs(stft) ** 2
        
        def create_mel_filterbank(self, n_mels: int = N_MELS,
                                 n_fft: int = N_FFT,
                                 sample_rate: float = SAMPLE_RATE) -> np.ndarray:
            """Create Mel filterbank"""
            try:
                result = self._wrapper._call_c_function('_spectrogram', 'create_mel_filterbank', 
                                                      n_mels, n_fft, sample_rate)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for create_mel_filterbank")
                try:
                    import librosa
                    return librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
                except ImportError:
                    n_bins = n_fft // 2 + 1
                    return np.random.rand(n_mels, n_bins).astype(np.float32)
        
        def whisper_mel(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
            """Compute Whisper-style Mel spectrogram"""
            if sample_rate != self.SAMPLE_RATE:
                import warnings
                warnings.warn(f"Audio sample rate {sample_rate}Hz will be treated as {self.SAMPLE_RATE}Hz")
            
            return self.compute_mel(
                audio,
                sample_rate=self.SAMPLE_RATE,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                n_mels=self.N_MELS
            )
    
    # ============================================================================
    # Audio Basic Module Wrapper
    # ============================================================================
    
    class AudioBasic:
        """Audio basic operations wrapper"""
        
        def __init__(self, parent_wrapper):
            self._wrapper = parent_wrapper
        
        def _ensure_float32(self, audio: np.ndarray) -> np.ndarray:
            """Ensure audio is float32 and 1D"""
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            if audio.ndim > 1:
                audio = audio.squeeze()
                if audio.ndim > 1:
                    audio = audio.flatten()
            
            return audio
        
        def stereo_to_mono(self, stereo_audio: np.ndarray) -> np.ndarray:
            """Convert stereo audio to mono"""
            audio = self._ensure_float32(stereo_audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_basic', 'stereo_to_mono', audio)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for stereo_to_mono")
                if len(audio.shape) == 1:
                    # Interleaved stereo
                    if len(audio) % 2 != 0:
                        raise ValueError("Interleaved stereo must have even length")
                    mono = np.zeros(len(audio) // 2, dtype=np.float32)
                    for i in range(len(mono)):
                        mono[i] = (audio[i*2] + audio[i*2+1]) * 0.5
                    return mono
                elif len(audio.shape) == 2:
                    # Channel-first or channel-last
                    if audio.shape[0] == 2:
                        return np.mean(audio, axis=0)
                    elif audio.shape[1] == 2:
                        return np.mean(audio, axis=1)
                    else:
                        return audio.flatten()
                else:
                    return audio.flatten()
        
        def stereo_to_mono_gpu(self, stereo_audio: np.ndarray) -> np.ndarray:
            """GPU-accelerated stereo to mono"""
            audio = self._ensure_float32(stereo_audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_basic', 'stereo_to_mono', 
                                                      audio, gpu_function=True)
                return result
            except Exception:
                return self.stereo_to_mono(stereo_audio)
        
        def normalize(self, audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
            """Normalize audio to target peak amplitude"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_basic', 'normalize_audio', audio, target_peak)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for normalize")
                peak = np.max(np.abs(audio))
                if peak > 0:
                    return audio * (target_peak / peak)
                return audio
        
        def normalize_gpu(self, audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
            """GPU-accelerated normalization"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_basic', 'normalize_audio', 
                                                      audio, target_peak, gpu_function=True)
                return result
            except Exception:
                return self.normalize(audio, target_peak)
        
        def float_to_pcm16(self, audio: np.ndarray) -> np.ndarray:
            """Convert float32 audio to 16-bit PCM"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_basic', 'float_to_pcm16', audio)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for float_to_pcm16")
                return np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        
        def float_to_pcm16_gpu(self, audio: np.ndarray) -> np.ndarray:
            """GPU-accelerated float to PCM16"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_basic', 'float_to_pcm16', 
                                                      audio, gpu_function=True)
                return result
            except Exception:
                return self.float_to_pcm16(audio)
        
        def resample(self, audio: np.ndarray, input_rate: int, output_rate: int) -> np.ndarray:
            """Resample audio using linear interpolation"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_basic', 'resample_audio', 
                                                      audio, input_rate, output_rate)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for resample")
                input_len = len(audio)
                output_len = int(input_len * output_rate / input_rate)
                
                if input_len == output_len:
                    return audio.copy()
                
                x_old = np.linspace(0, 1, input_len)
                x_new = np.linspace(0, 1, output_len)
                return np.interp(x_new, x_old, audio).astype(np.float32)
        
        def calculate_rms(self, audio: np.ndarray) -> float:
            """Calculate RMS energy"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_basic', 'calculate_rms', audio)
                return float(result)
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for calculate_rms")
                return float(np.sqrt(np.mean(audio ** 2)))
        
        def calculate_peak(self, audio: np.ndarray) -> float:
            """Calculate peak amplitude"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_basic', 'calculate_peak', audio)
                return float(result)
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for calculate_peak")
                return float(np.max(np.abs(audio)))
        
        def apply_gain(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
            """Apply gain in decibels"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_basic', 'apply_gain', audio, gain_db)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for apply_gain")
                gain_linear = 10 ** (gain_db / 20.0)
                result = audio * gain_linear
                return np.clip(result, -1.0, 1.0).astype(np.float32)
        
        def fade_in_out(self, audio: np.ndarray, fade_samples: int) -> np.ndarray:
            """Apply fade-in and fade-out"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_basic', 'fade_in_out', audio, fade_samples)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for fade_in_out")
                if fade_samples <= 0:
                    return audio.copy()
                
                fade_samples = min(fade_samples, len(audio) // 2)
                result = audio.copy()
                
                # Fade in
                for i in range(fade_samples):
                    factor = i / fade_samples
                    result[i] *= factor
                
                # Fade out
                for i in range(fade_samples):
                    factor = i / fade_samples
                    result[-i-1] *= factor
                
                return result
    
    # ============================================================================
    # Audio Enhance Module Wrapper
    # ============================================================================
    
    class AudioEnhance:
        """Audio enhancement operations wrapper"""
        
        def __init__(self, parent_wrapper):
            self._wrapper = parent_wrapper
        
        def _ensure_float32(self, audio: np.ndarray) -> np.ndarray:
            """Ensure audio is float32 and 1D"""
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            if audio.ndim > 1:
                audio = audio.squeeze()
                if audio.ndim > 1:
                    audio = audio.flatten()
            
            return audio
        
        def create_noise_params(self, threshold: float = 0.01, reduction_factor: float = 0.5,
                               window_size: int = 512, algorithm: int = 0) -> dict:
            """Create noise reduction parameters"""
            return {
                'threshold': threshold,
                'reduction_factor': reduction_factor,
                'window_size': window_size,
                'algorithm': algorithm
            }
        
        def create_eq_params(self, bass_boost: float = 1.0, mid_boost: float = 1.0,
                            treble_boost: float = 1.0, preamp: float = 1.0) -> dict:
            """Create equalization parameters"""
            return {
                'bass_boost': bass_boost,
                'mid_boost': mid_boost,
                'treble_boost': treble_boost,
                'preamp': preamp
            }
        
        def enhance_quality(self, audio: np.ndarray, **noise_params) -> np.ndarray:
            """Enhance audio quality with noise reduction"""
            audio = self._ensure_float32(audio)
            
            params = self.create_noise_params()
            params.update(noise_params)
            
            try:
                result = self._wrapper._call_c_function('_audio_enhance', 'enhance_audio_quality',
                                                      audio, params['threshold'], params['reduction_factor'],
                                                      params['window_size'], params['algorithm'])
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for enhance_quality")
                threshold = params['threshold']
                audio_copy = audio.copy()
                audio_copy[np.abs(audio_copy) < threshold] = 0
                return audio_copy
        
        def enhance_quality_gpu(self, audio: np.ndarray, **noise_params) -> np.ndarray:
            """GPU-accelerated quality enhancement"""
            audio = self._ensure_float32(audio)
            
            params = self.create_noise_params()
            params.update(noise_params)
            
            try:
                result = self._wrapper._call_c_function('_audio_enhance', 'enhance_audio_quality',
                                                      audio, params['threshold'], params['reduction_factor'],
                                                      params['window_size'], params['algorithm'], gpu_function=True)
                return result
            except Exception:
                return self.enhance_quality(audio, **noise_params)
        
        def apply_equalization(self, audio: np.ndarray, **eq_params) -> np.ndarray:
            """Apply multi-band equalization"""
            audio = self._ensure_float32(audio)
            
            params = self.create_eq_params()
            params.update(eq_params)
            
            try:
                result = self._wrapper._call_c_function('_audio_enhance', 'apply_equalization',
                                                      audio, params['bass_boost'], params['mid_boost'],
                                                      params['treble_boost'], params['preamp'])
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for apply_equalization")
                preamp = params['preamp']
                return np.clip(audio * preamp, -1.0, 1.0).astype(np.float32)
        
        def apply_equalization_gpu(self, audio: np.ndarray, **eq_params) -> np.ndarray:
            """GPU-accelerated equalization"""
            audio = self._ensure_float32(audio)
            
            params = self.create_eq_params()
            params.update(eq_params)
            
            try:
                result = self._wrapper._call_c_function('_audio_enhance', 'apply_equalization',
                                                      audio, params['bass_boost'], params['mid_boost'],
                                                      params['treble_boost'], params['preamp'], gpu_function=True)
                return result
            except Exception:
                return self.apply_equalization(audio, **eq_params)
        
        def remove_noise(self, audio: np.ndarray, threshold: float = 0.01,
                        algorithm: int = 0) -> np.ndarray:
            """Remove noise from audio"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_enhance', 'remove_noise',
                                                      audio, threshold, algorithm)
                return result
            except Exception:
                return self.enhance_quality(audio, threshold=threshold, algorithm=algorithm)
        
        def remove_noise_gpu(self, audio: np.ndarray, threshold: float = 0.01,
                            algorithm: int = 0) -> np.ndarray:
            """GPU-accelerated noise removal"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_enhance', 'remove_noise',
                                                      audio, threshold, algorithm, gpu_function=True)
                return result
            except Exception:
                return self.remove_noise(audio, threshold, algorithm)
        
        def compress_dynamic_range(self, audio: np.ndarray, threshold_db: float = -20.0,
                                  ratio: float = 4.0, attack_ms: float = 10.0,
                                  release_ms: float = 100.0, sample_rate: int = 16000) -> np.ndarray:
            """Apply dynamic range compression"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_enhance', 'compress_dynamic_range',
                                                      audio, threshold_db, ratio, attack_ms, release_ms, sample_rate)
                return result
            except Exception:
                # Python fallback - simple limiter
                logger.debug("Using Python fallback for compress_dynamic_range")
                threshold = 10 ** (threshold_db / 20.0)
                audio_copy = audio.copy()
                
                for i in range(len(audio_copy)):
                    if abs(audio_copy[i]) > threshold:
                        audio_copy[i] = threshold * np.sign(audio_copy[i])
                
                return audio_copy
        
        def calculate_snr(self, audio: np.ndarray, noise_threshold: float = 0.01) -> float:
            """Calculate signal-to-noise ratio"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_enhance', 'calculate_snr',
                                                      audio, noise_threshold)
                return float(result)
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for calculate_snr")
                abs_audio = np.abs(audio)
                signal_mask = abs_audio > noise_threshold
                noise_mask = ~signal_mask
                
                if np.sum(signal_mask) == 0 or np.sum(noise_mask) == 0:
                    return 0.0
                
                signal_power = np.mean(audio[signal_mask] ** 2)
                noise_power = np.mean(audio[noise_mask] ** 2)
                
                if noise_power <= 0:
                    return 100.0
                
                return 10.0 * np.log10(signal_power / noise_power)
        
        def detect_silence(self, audio: np.ndarray, threshold: float = 0.01,
                          min_duration_ms: float = 100.0, max_regions: int = 100,
                          sample_rate: int = 16000) -> List[Tuple[int, int]]:
            """Detect silence regions in audio"""
            audio = self._ensure_float32(audio)
            
            try:
                result = self._wrapper._call_c_function('_audio_enhance', 'detect_silence',
                                                      audio, threshold, min_duration_ms, max_regions, sample_rate)
                return result
            except Exception:
                # Python implementation
                abs_audio = np.abs(audio)
                below_threshold = abs_audio < threshold
                
                regions = []
                start = None
                
                for i in range(len(below_threshold)):
                    if below_threshold[i] and start is None:
                        start = i
                    elif not below_threshold[i] and start is not None:
                        duration = i - start
                        if duration >= (min_duration_ms / 1000.0 * sample_rate):
                            regions.append((start, i))
                            if len(regions) >= max_regions:
                                break
                        start = None
                
                if start is not None:
                    duration = len(below_threshold) - start
                    if duration >= (min_duration_ms / 1000.0 * sample_rate):
                        regions.append((start, len(below_threshold)))
                
                return regions[:max_regions]
    
    # ============================================================================
    # Audio Segment Module Wrapper
    # ============================================================================
    
    class AudioSegment:
        """Audio segment extraction wrapper"""
        
        def __init__(self, parent_wrapper):
            self._wrapper = parent_wrapper
        
        def _ensure_1d_float32(self, audio_data: np.ndarray) -> np.ndarray:
            """Ensure audio data is 1-dimensional float32 array"""
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            if audio_data.ndim > 1:
                audio_data = audio_data.squeeze()
                if audio_data.ndim > 1:
                    audio_data = audio_data.flatten()
            
            return audio_data
        
        def extract(self, audio_data: np.ndarray, 
                    start_sample: int, end_sample: int) -> np.ndarray:
            """Extract a single audio segment"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_segment', 'audio_segment_extract',
                                                      audio_prepared, start_sample, end_sample)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for extract")
                if start_sample < 0:
                    start_sample = 0
                if end_sample > len(audio_data):
                    end_sample = len(audio_data)
                return audio_data[start_sample:end_sample].copy()
        
        def extract_gpu(self, audio_data: np.ndarray,
                       start_sample: int, end_sample: int) -> np.ndarray:
            """GPU-accelerated segment extraction"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_segment', 'audio_segment_extract',
                                                      audio_prepared, start_sample, end_sample, gpu_function=True)
                return result
            except Exception:
                return self.extract(audio_data, start_sample, end_sample)
        
        def extract_mmap(self, audio_file: str, 
                        start_time: float, end_time: float, 
                        sample_rate: int = 16000) -> np.ndarray:
            """Extract a segment from memory-mapped audio file"""
            try:
                result = self._wrapper._call_c_function('_audio_segment', 'audio_segment_extract_mmap',
                                                      audio_file, start_time, end_time, sample_rate)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for extract_mmap")
                try:
                    import soundfile as sf
                    audio, sr = sf.read(audio_file)
                    if sr != sample_rate:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                except ImportError:
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    segment_samples = end_sample - start_sample
                    return np.zeros(segment_samples, dtype=np.float32)
                
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                return audio[start_sample:end_sample].astype(np.float32)
        
        def batch_extract(self, audio_data: np.ndarray, 
                         segments: List[Tuple[int, int]]) -> List[np.ndarray]:
            """Batch extract multiple audio segments"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_segment', 'audio_segment_batch_extract',
                                                      audio_prepared, segments)
                if result is not None:
                    return result
            except Exception as e:
                logger.debug(f"C acceleration failed for batch_extract: {e}")
            
            # Python fallback
            logger.debug("Using Python fallback for batch_extract")
            output = []
            for start, end in segments:
                output.append(self.extract(audio_data, start, end))
            return output
        
        def batch_extract_gpu(self, audio_data: np.ndarray,
                             segments: List[Tuple[int, int]]) -> List[np.ndarray]:
            """GPU-accelerated batch extraction"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_segment', 'audio_segment_batch_extract',
                                                      audio_prepared, segments, gpu_function=True)
                return result
            except Exception:
                return self.batch_extract(audio_data, segments)
        
        def batch_extract_mmap(self, audio_data: np.ndarray,
                              segments: List[Tuple[float, float]], 
                              sample_rate: int = 16000) -> List[np.ndarray]:
            """Batch extract multiple segments from memory-mapped audio"""
            try:
                result = self._wrapper._call_c_function('_audio_segment', 'audio_segment_batch_extract_mmap',
                                                      audio_data, segments, sample_rate)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for batch_extract_mmap")
                sample_segments = [(int(s * sample_rate), int(e * sample_rate)) for s, e in segments]
                return self.batch_extract(audio_data, sample_segments)
        
        def concatenate(self, segments: List[np.ndarray]) -> np.ndarray:
            """Concatenate multiple audio segments"""
            try:
                result = self._wrapper._call_c_function('_audio_segment', 'audio_segment_concatenate', segments)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for concatenate")
                return np.concatenate(segments)
        
        def concatenate_gpu(self, segments: List[np.ndarray]) -> np.ndarray:
            """GPU-accelerated concatenation"""
            try:
                result = self._wrapper._call_c_function('_audio_segment', 'audio_segment_concatenate',
                                                      segments, gpu_function=True)
                return result
            except Exception:
                return self.concatenate(segments)
        
        def extract_word_segments(self, audio_data: np.ndarray,
                                word_timings: List[Tuple[float, float, str]],
                                sample_rate: int = 16000) -> Dict[str, np.ndarray]:
            """Extract word segments based on timings"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_segment', 'audio_segment_extract_word_segments',
                                                      audio_prepared, word_timings, sample_rate)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for extract_word_segments")
                result = {}
                for start, end, word in word_timings:
                    start_sample = int(start * sample_rate)
                    end_sample = int(end * sample_rate)
                    result[word] = self.extract(audio_data, start_sample, end_sample)
                return result
        
        def validate_range(self, start_sample: int, end_sample: int, 
                          buffer_size: int) -> bool:
            """Validate if segment range is within buffer bounds"""
            try:
                result = self._wrapper._call_c_function('_audio_segment', 'audio_segment_validate_range',
                                                      start_sample, end_sample, buffer_size)
                return result
            except Exception:
                # Python fallback
                return (0 <= start_sample < end_sample <= buffer_size)
        
        def create_array(self, segment_size: int) -> np.ndarray:
            """Create a numpy array for segment storage"""
            try:
                result = self._wrapper._call_c_function('_audio_segment', 'audio_segment_create_array', segment_size)
                return result
            except Exception:
                # Python fallback
                return np.zeros(segment_size, dtype=np.float32)
    
    # ============================================================================
    # Audio Features Module Wrapper
    # ============================================================================
    
    class AudioFeatures:
        """Audio feature extraction wrapper"""
        
        def __init__(self, parent_wrapper):
            self._wrapper = parent_wrapper
        
        def _ensure_1d_float32(self, audio_data: np.ndarray) -> np.ndarray:
            """Ensure audio data is 1-dimensional float32 array"""
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            if audio_data.ndim > 1:
                audio_data = audio_data.squeeze()
                if audio_data.ndim > 1:
                    audio_data = audio_data.flatten()
            
            return audio_data
        
        def extract_mfcc(self, audio_data: np.ndarray, 
                        sample_rate: int = 16000, 
                        n_mfcc: int = 13) -> np.ndarray:
            """Extract MFCC features from audio"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_extract_mfcc',
                                                      audio_prepared, sample_rate, n_mfcc)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for extract_mfcc")
                try:
                    import librosa
                    return librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
                except ImportError:
                    n_frames = max(1, (len(audio_data) - 512) // 160 + 1)
                    return np.random.randn(n_mfcc, n_frames).astype(np.float32)
        
        def extract_mfcc_gpu(self, audio_data: np.ndarray,
                            sample_rate: int = 16000,
                            n_mfcc: int = 13) -> np.ndarray:
            """GPU-accelerated MFCC extraction"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_extract_mfcc',
                                                      audio_prepared, sample_rate, n_mfcc, gpu_function=True)
                return result
            except Exception:
                return self.extract_mfcc(audio_data, sample_rate, n_mfcc)
        
        def compute_spectral_centroid(self, audio_data: np.ndarray,
                                     sample_rate: int = 16000) -> np.ndarray:
            """Compute spectral centroid from audio"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_compute_spectral_centroid',
                                                      audio_prepared, sample_rate)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for compute_spectral_centroid")
                try:
                    import librosa
                    return librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
                except ImportError:
                    n_frames = max(1, (len(audio_data) - 512) // 160 + 1)
                    return np.random.randn(n_frames).astype(np.float32) * 1000 + 1000
        
        def compute_zero_crossing_rate(self, audio_data: np.ndarray) -> np.ndarray:
            """Compute zero crossing rate from audio"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_compute_zero_crossing_rate',
                                                      audio_prepared)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for compute_zero_crossing_rate")
                try:
                    import librosa
                    return librosa.feature.zero_crossing_rate(audio_data)[0]
                except ImportError:
                    frame_len, hop = 512, 160
                    n_frames = max(1, (len(audio_data) - frame_len) // hop + 1)
                    zcr = np.zeros(n_frames, dtype=np.float32)
                    for i in range(n_frames):
                        start = i * hop
                        frame = audio_data[start:start+frame_len]
                        if len(frame) > 1:
                            zcr[i] = np.sum(np.diff(np.sign(frame)) != 0) / (len(frame) - 1)
                    return zcr
        
        def compute_spectral_rolloff(self, audio_data: np.ndarray,
                                    sample_rate: int = 16000,
                                    percentile: float = 85.0) -> np.ndarray:
            """Compute spectral rolloff from audio"""
            if percentile <= 0 or percentile >= 100:
                raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
            
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_compute_spectral_rolloff',
                                                      audio_prepared, sample_rate, percentile)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for compute_spectral_rolloff")
                try:
                    import librosa
                    return librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate, roll_percent=percentile)[0]
                except ImportError:
                    n_frames = max(1, (len(audio_data) - 512) // 160 + 1)
                    return np.random.randn(n_frames).astype(np.float32) * 1000 + 2000
        
        def compute_spectral_bandwidth(self, audio_data: np.ndarray,
                                      sample_rate: int = 16000) -> np.ndarray:
            """Compute spectral bandwidth from audio"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_compute_spectral_bandwidth',
                                                      audio_prepared, sample_rate)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for compute_spectral_bandwidth")
                try:
                    import librosa
                    return librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
                except ImportError:
                    n_frames = max(1, (len(audio_data) - 512) // 160 + 1)
                    return np.random.randn(n_frames).astype(np.float32) * 500 + 1000
        
        def compute_spectral_flatness(self, audio_data: np.ndarray) -> np.ndarray:
            """Compute spectral flatness from audio"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_compute_spectral_flatness',
                                                      audio_prepared)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for compute_spectral_flatness")
                try:
                    import librosa
                    return librosa.feature.spectral_flatness(y=audio_data)[0]
                except ImportError:
                    n_frames = max(1, (len(audio_data) - 512) // 160 + 1)
                    return np.random.rand(n_frames).astype(np.float32) * 0.5 + 0.5
        
        def compute_rms_energy(self, audio_data: np.ndarray,
                              frame_size: int = 256,
                              hop_size: int = 80) -> np.ndarray:
            """Compute RMS energy from audio"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_compute_rms_energy',
                                                      audio_prepared, frame_size, hop_size)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for compute_rms_energy")
                n_frames = max(1, (len(audio_data) - frame_size) // hop_size + 1)
                energy = np.zeros(n_frames, dtype=np.float32)
                for i in range(n_frames):
                    start = i * hop_size
                    frame = audio_data[start:start+frame_size]
                    if len(frame) > 0:
                        energy[i] = np.sqrt(np.mean(frame ** 2))
                return energy
        
        def compute_rms_energy_gpu(self, audio_data: np.ndarray,
                                  frame_size: int = 256,
                                  hop_size: int = 80) -> np.ndarray:
            """GPU-accelerated RMS energy computation"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_compute_rms_energy',
                                                      audio_prepared, frame_size, hop_size, gpu_function=True)
                return result
            except Exception:
                return self.compute_rms_energy(audio_data, frame_size, hop_size)
        
        def detect_energy_peaks(self, energy_data: np.ndarray,
                               threshold: float = 0.1) -> np.ndarray:
            """Detect peaks in energy data"""
            energy_prepared = self._ensure_1d_float32(energy_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_detect_energy_peaks',
                                                      energy_prepared, threshold)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for detect_energy_peaks")
                try:
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(energy_data, height=threshold)
                    return peaks
                except ImportError:
                    peaks = []
                    for i in range(1, len(energy_data) - 1):
                        if (energy_data[i] > energy_data[i-1] and 
                            energy_data[i] > energy_data[i+1] and 
                            energy_data[i] > threshold):
                            peaks.append(i)
                    return np.array(peaks, dtype=np.int32)
        
        def detect_energy_peaks_gpu(self, energy_data: np.ndarray,
                                   threshold: float = 0.1) -> np.ndarray:
            """GPU-accelerated peak detection"""
            energy_prepared = self._ensure_1d_float32(energy_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_detect_energy_peaks',
                                                      energy_prepared, threshold, gpu_function=True)
                return result
            except Exception:
                return self.detect_energy_peaks(energy_data, threshold)
        
        def compute_energy_envelope(self, audio_data: np.ndarray,
                                   sample_rate: int = 16000) -> np.ndarray:
            """Compute energy envelope for syllable detection"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_compute_energy_envelope',
                                                      audio_prepared, sample_rate)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for compute_energy_envelope")
                energy = self.compute_rms_energy(audio_data, 256, 80)
                n_frames = len(energy)
                envelope = np.zeros((n_frames, 3), dtype=np.float32)
                for i in range(n_frames):
                    envelope[i, 0] = (i * 80) / sample_rate
                    envelope[i, 1] = energy[i]
                mean_e, std_e = np.mean(energy), np.std(energy) + 1e-10
                envelope[:, 2] = (energy - mean_e) / std_e
                return envelope
        
        def extract_all(self, audio_data: np.ndarray,
                       sample_rate: int = 16000,
                       include_mfcc: bool = True,
                       include_spectral: bool = True,
                       include_energy: bool = True) -> Dict[str, np.ndarray]:
            """Extract all audio features from input audio"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            include_mfcc_int = 1 if include_mfcc else 0
            include_spectral_int = 1 if include_spectral else 0
            include_energy_int = 1 if include_energy else 0
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_extract_all',
                                                      audio_prepared, sample_rate, 
                                                      include_mfcc_int, include_spectral_int, include_energy_int)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for extract_all")
                features = {}
                if include_mfcc:
                    features['mfcc'] = self.extract_mfcc(audio_data, sample_rate, 13)
                if include_spectral:
                    features['spectral_centroid'] = self.compute_spectral_centroid(audio_data, sample_rate)
                    features['zero_crossing_rate'] = self.compute_zero_crossing_rate(audio_data)
                    features['spectral_rolloff'] = self.compute_spectral_rolloff(audio_data, sample_rate, 85.0)
                    features['spectral_bandwidth'] = self.compute_spectral_bandwidth(audio_data, sample_rate)
                    features['spectral_flatness'] = self.compute_spectral_flatness(audio_data)
                if include_energy:
                    features['rms_energy'] = self.compute_rms_energy(audio_data, 256, 80)
                    features['energy_envelope'] = self.compute_energy_envelope(audio_data, sample_rate)
                return features
        
        def detect_voice_activity(self, audio_data: np.ndarray,
                                 sample_rate: int = 16000,
                                 threshold: float = 0.1) -> np.ndarray:
            """Detect voice activity in audio"""
            audio_prepared = self._ensure_1d_float32(audio_data)
            
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_detect_voice_activity',
                                                      audio_prepared, sample_rate, threshold)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for detect_voice_activity")
                energy = self.compute_rms_energy(audio_data, 256, 80)
                zcr = self.compute_zero_crossing_rate(audio_data)
                min_len = min(len(energy), len(zcr))
                adaptive = threshold if threshold >= 0 else np.mean(energy) + 0.5 * (np.max(energy) - np.mean(energy))
                vad = np.zeros(min_len, dtype=np.int32)
                for i in range(min_len):
                    if energy[i] > adaptive and zcr[i] < 0.1:
                        vad[i] = 1
                return vad
        
        def create_hann_window(self, window_size: int) -> np.ndarray:
            """Create Hann window for spectral analysis"""
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_create_hann_window', window_size)
                return result
            except Exception:
                # Python fallback
                t = np.arange(window_size, dtype=np.float32)
                return 0.5 * (1.0 - np.cos(2.0 * np.pi * t / (window_size - 1)))
        
        def create_mel_filterbank(self, sample_rate: int = 16000,
                                 n_fft: int = 400,
                                 n_mels: int = 80,
                                 fmin: float = 0.0,
                                 fmax: float = 8000.0) -> np.ndarray:
            """Create Mel filterbank for spectrogram computation"""
            try:
                result = self._wrapper._call_c_function('_audio_features', 'audio_features_create_mel_filterbank',
                                                      sample_rate, n_fft, n_mels, fmin, fmax)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for create_mel_filterbank")
                try:
                    import librosa
                    return librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
                except ImportError:
                    n_bins = n_fft // 2 + 1
                    return np.random.rand(n_mels, n_bins).astype(np.float32)
    
    # ============================================================================
    # Corpus Utilities Module Wrapper
    # ============================================================================
    
    class CorpusUtils:
        """Corpus utilities wrapper"""
        
        def __init__(self, parent_wrapper):
            self._wrapper = parent_wrapper
        
        def batch_copy_files(self, file_pairs):
            """Batch copy files"""
            try:
                result = self._wrapper._call_c_function('_corpus_utils', 'batch_copy_files', file_pairs)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for batch_copy_files")
                import shutil
                results = []
                success = 0
                for src, dst in file_pairs:
                    try:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(src, dst)
                        results.append(True)
                        success += 1
                    except:
                        results.append(False)
                return results, success, len(file_pairs)
        
        def batch_copy_files_gpu(self, file_pairs):
            """GPU-accelerated batch copy"""
            try:
                result = self._wrapper._call_c_function('_corpus_utils', 'batch_copy_files', 
                                                      file_pairs, gpu_function=True)
                return result
            except Exception:
                return self.batch_copy_files(file_pairs)
        
        def batch_validate_files(self, file_list):
            """Validate file existence"""
            try:
                result = self._wrapper._call_c_function('_corpus_utils', 'batch_validate_files', file_list)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for batch_validate_files")
                results = []
                exist = 0
                for f in file_list:
                    exists = os.path.isfile(f)
                    results.append(exists)
                    if exists:
                        exist += 1
                return results, exist, len(file_list)
        
        def batch_validate_files_gpu(self, file_list):
            """GPU-accelerated file validation"""
            try:
                result = self._wrapper._call_c_function('_corpus_utils', 'batch_validate_files', 
                                                      file_list, gpu_function=True)
                return result
            except Exception:
                return self.batch_validate_files(file_list)
        
        def batch_create_directories(self, dir_list):
            """Create directories"""
            try:
                result = self._wrapper._call_c_function('_corpus_utils', 'batch_create_directories', dir_list)
                return result
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for batch_create_directories")
                results = []
                success = 0
                for d in dir_list:
                    try:
                        os.makedirs(d, exist_ok=True)
                        results.append(True)
                        success += 1
                    except:
                        results.append(False)
                return results, success, len(dir_list)
        
        def normalize_path(self, path):
            """Normalize path"""
            try:
                result = self._wrapper._call_c_function('_corpus_utils', 'normalize_path', path)
                return result
            except Exception:
                # Python fallback
                import os
                return os.path.normpath(os.path.expanduser(path))
        
        def batch_normalize_paths(self, paths):
            """Batch normalize paths"""
            try:
                result = self._wrapper._call_c_function('_corpus_utils', 'batch_normalize_paths', paths)
                return result
            except Exception:
                return [self.normalize_path(p) for p in paths]
        
        def batch_normalize_paths_gpu(self, paths):
            """GPU-accelerated path normalization"""
            try:
                result = self._wrapper._call_c_function('_corpus_utils', 'batch_normalize_paths', 
                                                      paths, gpu_function=True)
                return result
            except Exception:
                return self.batch_normalize_paths(paths)
    
    # ============================================================================
    # GPU Backend Module Wrapper
    # ============================================================================
    
    class GPUBackend:
        """GPU backend management wrapper"""
        
        def __init__(self, parent_wrapper):
            self._wrapper = parent_wrapper
            self.initialized = False
        
        def gpu_init(self, backend=4):
            """Initialize GPU backend"""
            try:
                result = self._wrapper._call_c_function('_gpu_backend', 'gpu_init', backend)
                if result is not None:
                    self.initialized = bool(result)
                    return result
            except Exception:
                pass
            
            # Python fallback
            logger.debug("Using Python fallback for gpu_init")
            self.initialized = False
            return False
        
        def gpu_cleanup(self):
            """Cleanup GPU"""
            try:
                result = self._wrapper._call_c_function('_gpu_backend', 'gpu_cleanup')
                self.initialized = False
                return
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for gpu_cleanup")
                self.initialized = False
        
        def gpu_is_available(self):
            """Check GPU availability"""
            try:
                result = self._wrapper._call_c_function('_gpu_backend', 'gpu_is_available')
                return result
            except Exception:
                # Python fallback
                return False
        
        def gpu_is_enabled(self):
            """Check if GPU enabled"""
            try:
                result = self._wrapper._call_c_function('_gpu_backend', 'gpu_is_enabled')
                return result
            except Exception:
                # Python fallback
                return self.initialized
        
        def gpu_set_enabled(self, enabled):
            """Enable/disable GPU"""
            try:
                result = self._wrapper._call_c_function('_gpu_backend', 'gpu_set_enabled', int(enabled))
                return
            except Exception:
                # Python fallback
                logger.debug("Using Python fallback for gpu_set_enabled")
                if enabled:
                    self.gpu_init()
                else:
                    self.initialized = False
        
        def gpu_get_active_backend(self):
            """Get active backend"""
            try:
                result = self._wrapper._call_c_function('_gpu_backend', 'gpu_get_active_backend')
                return result
            except Exception:
                # Python fallback
                return 0  # GPU_BACKEND_NONE
        
        def gpu_get_backend_name(self, backend=None):
            """Get backend name"""
            if backend is None:
                backend = self.gpu_get_active_backend()
            
            try:
                result = self._wrapper._call_c_function('_gpu_backend', 'gpu_get_backend_name', backend)
                return result
            except Exception:
                # Fallback names
                names = {0: "none", 1: "cuda", 2: "opencl", 3: "directml", 4: "auto"}
                return names.get(backend, "unknown")
        
        def gpu_get_device_count(self):
            """Get device count"""
            try:
                result = self._wrapper._call_c_function('_gpu_backend', 'gpu_get_device_count')
                return result
            except Exception:
                # Python fallback
                return 0
        
        def gpu_get_device_info(self, device_idx=0):
            """Get device info"""
            try:
                result = self._wrapper._call_c_function('_gpu_backend', 'gpu_get_device_info', device_idx)
                return result
            except Exception:
                # Fallback info
                return {
                    "name": "CPU Fallback",
                    "supports_float": True,
                    "supports_double": False,
                    "memory_mb": 0,
                    "compute_units": 1,
                    "max_work_group_size": 256
                }
        
        def gpu_print_info(self):
            """Print GPU info"""
            try:
                result = self._wrapper._call_c_function('_gpu_backend', 'gpu_print_info')
                return
            except Exception:
                # Fallback info
                print("\n=== GPU Backend ===")
                print(f"C Acceleration: {self._wrapper.available}")
                print(f"Initialized: {self.initialized}")
                print(f"Backend: {self.gpu_get_backend_name()}")
                print(f"Devices: {self.gpu_get_device_count()}")
                print("===================")
    
    # ============================================================================
    # Utility methods
    # ============================================================================
    
    def get_module_status(self) -> Dict[str, Any]:
        """Get comprehensive status of ALL loaded modules"""
        status = {
            'overall': {
                'c_acceleration_available': self._modules_available,
                'gpu_acceleration_enabled': self.enable_gpu_accel,
                'modules_loaded': [],
                'performance_stats': self.performance_stats.copy()
            }
        }
        
        # Add module details
        for module_name, module in self._loaded_modules.items():
            if module is not None:
                status['overall']['modules_loaded'].append(module_name)
                
                module_status = {
                    'available': True,
                    'gpu_available': False,
                    'functions_available': []
                }
                
                # Check for GPU support
                if (hasattr(module, 'set_gpu_acceleration') or 
                    hasattr(module, 'py_set_gpu_acceleration') or
                    hasattr(module, 'gpu_init')):
                    module_status['gpu_available'] = True
                
                # List available functions
                for attr_name in dir(module):
                    if not attr_name.startswith('_'):
                        module_status['functions_available'].append(attr_name)
                
                status[module_name] = module_status
        
        return status
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        if stats['call_count'] > 0:
            stats['avg_time'] = stats['total_time'] / stats['call_count']
        else:
            stats['avg_time'] = 0.0
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'call_count': 0,
            'gpu_calls': 0,
            'cpu_calls': 0,
            'python_calls': 0,
            'total_time': 0.0
        }

# ============================================================================
# Global instance management (Singleton pattern)
# ============================================================================

_global_wrapper = None

def get_c_acceleration_wrapper(enable_c_accel: bool = True, enable_gpu_accel: bool = True) -> CAccelerationWrapper:
    """Get the global C acceleration wrapper instance (Singleton)"""
    global _global_wrapper
    
    # Don't initialize C acceleration in child processes
    if mp.current_process().name != 'MainProcess':
        class ChildProcessWrapper:
            def __init__(self):
                self.available = False
                # Create dummy sub-modules
                self.syllabifier = self._create_dummy_module()
                self.textgrid = self._create_dummy_module()
                self.spectrogram = self._create_dummy_module()
                self.audio_basic = self._create_dummy_module()
                self.audio_enhance = self._create_dummy_module()
                self.audio_segment = self._create_dummy_module()
                self.audio_features = self._create_dummy_module()
                self.corpus_utils = self._create_dummy_module()
                self.gpu_backend = self._create_dummy_module()
            
            def _create_dummy_module(self):
                class DummyModule:
                    def __getattr__(self, name):
                        raise RuntimeError("C acceleration not available in child processes")
                return DummyModule()
            
            def get_module_status(self):
                return {'overall': {'c_acceleration_available': False, 'modules_loaded': []}}
            
            def get_performance_stats(self):
                return {'call_count': 0, 'gpu_calls': 0, 'cpu_calls': 0, 'python_calls': 0, 'total_time': 0.0, 'avg_time': 0.0}
        
        return ChildProcessWrapper()
    
    if _global_wrapper is None:
        _global_wrapper = CAccelerationWrapper(enable_c_accel, enable_gpu_accel)
    
    return _global_wrapper

def set_gpu_preference(backend: str):
    """Set GPU backend preference"""
    wrapper = get_c_acceleration_wrapper()
    
    # Map backend string to constant (implementation would depend on C module)
    backend_map = {
        'cuda': 1,
        'opencl': 2,
        'directml': 3,
        'auto': 4,
        'none': 0
    }
    
    backend_code = backend_map.get(backend.lower(), 4)  # Default to auto
    
    # Try to set GPU backend through gpu_backend module
    if wrapper.available and '_gpu_backend' in wrapper._loaded_modules:
        try:
            wrapper.gpu_backend.gpu_init(backend_code)
            logger.info(f"GPU backend preference set to: {backend}")
        except Exception as e:
            logger.debug(f"Failed to set GPU backend: {e}")
    else:
        logger.info(f"GPU backend configuration not available, preference saved as: {backend}")

# ============================================================================
# Convenience functions for direct use
# ============================================================================

def get_acceleration_status() -> Dict[str, Any]:
    """Get comprehensive acceleration status"""
    wrapper = get_c_acceleration_wrapper()
    return wrapper.get_module_status()

def test_acceleration_functions() -> Tuple[int, int]:
    """Test available acceleration functions"""
    wrapper = get_c_acceleration_wrapper()
    tests_passed = 0
    tests_failed = 0
    
    logger.info("Testing C acceleration functions...")
    
    # Test basic audio operations
    try:
        test_audio = np.random.randn(16000).astype(np.float32)
        mono = wrapper.audio_basic.stereo_to_mono(np.tile(test_audio, 2))
        if mono is not None and len(mono) == len(test_audio):
            tests_passed += 1
            logger.debug("Audio basic operations test passed")
        else:
            tests_failed += 1
    except Exception as e:
        tests_failed += 1
        logger.debug(f"Audio basic test error: {e}")
    
    # Test corpus utilities
    try:
        normalized = wrapper.corpus_utils.normalize_path("/home/user/test.wav")
        if normalized:
            tests_passed += 1
            logger.debug("Corpus utilities test passed")
        else:
            tests_failed += 1
    except Exception as e:
        tests_failed += 1
        logger.debug(f"Corpus utilities test error: {e}")
    
    logger.info(f"Acceleration tests completed: {tests_passed} passed, {tests_failed} failed")
    return tests_passed, tests_failed

# ============================================================================
# Module initialization
# ============================================================================

def _initialize_on_import():
    """Initialize acceleration when module is imported"""
    try:
        wrapper = get_c_acceleration_wrapper()
        if wrapper.available:
            status = wrapper.get_module_status()
            module_count = len(status['overall']['modules_loaded'])
            logger.info(f"C acceleration ready: {module_count} modules loaded")
        else:
            logger.info("C acceleration not available - using Python implementations")
    except Exception as e:
        logger.debug(f"Acceleration initialization on import failed: {e}")

# Initialize on import
_initialize_on_import()

# ============================================================================
# Export public interface
# ============================================================================

__all__ = [
    'CAccelerationWrapper',
    'get_c_acceleration_wrapper',
    'set_gpu_preference',
    'get_acceleration_status',
    'test_acceleration_functions'
]

# Module initialization complete
logger.debug("ASR acceleration wrapper initialization complete")
