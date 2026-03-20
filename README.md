RePAST: Revised Pipeline for Accessible Speech Transcription (v1.0.0)
================================================================================
Comprehensive User Guide & Documentation
--------------------------------------------------------------------------------
This pipeline is a modular, high-performance toolkit designed for academic researchers requiring high-fidelity transcription, speaker diarization, and linguistic annotation. It prioritizes data privacy through local-first processing and accessibility via an interactive, terminal-based user interface.
--------------------------------------------------------------------------------
Table of Contents
Introduction & Privacy
Phase 0: Environment Initialization (00_setup,py)
The Master Controller (99_main.py)
Phase 1: Audio Preprocessing (01_preprocess.py)
Phase 2: Transcription & Diarization (01_transcribe.py)
Phase 3: Alignment & Annotation (03_annotate.py)
Speaker Registration Tool (libs/register.py)
Hardware Acceleration & C-Modules
Citing this Work
Acknowledgement and Privacy Notice
--------------------------------------------------------------------------------
1. Introduction & Privacy
This pipeline is built to ensure that sensitive research data never leaves your local machine.
Local Processing: All engines, including OpenAI Whisper, MFA, and spaCy, run locally.
PII Management: Biometric speaker fingerprints (embeddings) are stored in ./models/embeddings/. Researchers should handle these files as sensitive Personal Identifiable Information.
--------------------------------------------------------------------------------
2. Phase 0: Environment Initialization (00_setup.py)
Before running the pipeline, use this script to gatekeep your environment and verify hardware.
What it does:
Enforces execution within the dedicated transcriber Conda environment.
Builds the necessary directory structures for all phases.
Hardware Verification: Uses the Unified GPU Backend to detect NVIDIA (CUDA) or AMD (OpenCL) hardware, or sets up a CPU fallback.
Hugging Face Management: Interactively manages your Hugging Face token (required for diarization models).
Reproducibility: Generates a requirements-lock.txt to ensure exact dependency versions.
--------------------------------------------------------------------------------
3. The Master Controller (99_main.py)
For users wanting a "hands-off" experience, the Master Controller orchestrates Phases 1 through 3 in a single linear workflow.
How to use: Run python 99_main.py.
Options: You can interactively configure the entire pipeline (enhancement levels, Whisper models, languages, and annotation types) before execution begins.
--------------------------------------------------------------------------------
4. Phase 1: Audio Preprocessing (01_preprocess.py)
This phase standardizes disparate audio files into a format optimal for ASR.
Standardization (Required): Automatically resamples to 16kHz, converts to Mono, and applies Peak Normalization to 0.8 with 16-bit PCM dither.
Enhancement Options:
Level 0 (None): Pure standardization only.
Level 1 (Gentle): High-Pass Filtering (HPF) and LUFS normalization.
Level 2 (Standard): Gentle level + noise reduction and subtle EQ.
Level 3 (Experimental): AI-driven enhancement using SpeechBrain models (e.g., SepFormer, MetricGAN+).
--------------------------------------------------------------------------------
5. Phase 2: Transcription & Diarization (02_transcribe.py)
This hub converts preprocessed audio into text and identifies who is speaking.
Transcription: Powered by OpenAI Whisper.
Model Sizes: Choose from tiny, base, small, medium, or large based on your VRAM and accuracy needs.
Languages: Supports 13 major languages and "Auto-detect".
Diarization: Uses pyannote.audio to identify speaker turns.
Precision Levels: Low (Fastest), Medium (Balanced), or High (Highest timing accuracy).
Speaker Profiles: Matches voices against the local fingerprint library in ./models/embeddings/.
--------------------------------------------------------------------------------
6. Phase 3: Alignment & Annotation (03_annotate.py)
The final phase aligns text to audio at the phoneme level and adds linguistic metadata.
Alignment: Uses the Montreal Forced Aligner (MFA) to map words and phones to specific timestamps.
Annotation Options:
Syllabification: Adds syllable tiers using linguistic dictionaries or C-accelerated acoustic peak detection.
POS Tagging: Uses spaCy to add Part-of-Speech tiers (e.g., Nouns, Verbs) in "Universal" or "Fine" formats.
--------------------------------------------------------------------------------
7. Speaker Registration Tool (libs/register.py)
A standalone "satellite" utility for managing your speaker database.
How it works: Extract a vocal fingerprint from an audio file using ECAPA-TDNN.
Functionality: Register new speakers or update existing profiles in your local library to improve diarization accuracy in Phase 2.
--------------------------------------------------------------------------------
8. Hardware Acceleration & C-Modules
To maintain high speeds during complex tasks (like spectral analysis or batch segmenting), the pipeline uses custom C-accelerated modules located in src/.
Modules include:
_audio_segment_engine: For high-speed temporal slicing and alignment.
_audio_signal_core: For waveform transforms and filtering.
_audio_spectral_analysis: For frequency-domain transforms (STFT, Mel-spectrograms).
Automatic Fallback: If the C-modules or GPU are unavailable, the pipeline automatically switches to pure Python implementations to ensure the task is still completed.
--------------------------------------------------------------------------------
9. Citing this Work
If you use this pipeline in your research, please cite it as follows:
Taylor, S. P. (2026). RePAST: Revised Accessible Pipeline for Automated Transcription. [Software]. https://doi.org/10.5281/zenodo.19140616
Reproducibility Note: It is recommended that you include your requirements-lock.txt and the specific version of the _preprocessed audio files when publishing your methodology.
--------------------------------------------------------------------------------
ACKNOWLEDGEMENT & THIRD-PARTY CITATIONS
While the core logic, Unified GPU Backend, and C-accelerated engines in this repository are original works, RePAST functions as a high-performance orchestration hub for several industry-standard and academic tools. It acknowledges and credits the following foundational technologies:
* Transcription: Powered by OpenAI Whisper for high-fidelity speech-to-text inference.
* Diarization: Utilizes pyannote.audio (v3.1) for temporal segmentation and ECAPA-TDNN for vocal fingerprinting and speaker identification.
* Alignment: Employs the Montreal Forced Aligner (MFA) for segment-to-phoneme mapping.
* Linguistic Tagging: Integrated with spaCy for automated grammatical and Part-of-Speech categorization.
* Audio Enhancement: Uses SpeechBrain (including SepFormer and MetricGAN+ models) and the noisereduce library for AI-driven signal cleaning.
* Model Hosting: Hugging Face provides the secure infrastructure for distributing pre-trained model weights.

For academic transparency and reproducibility, the exact versions of these dependencies used in this release are documented in the requirements-lock.txt file generated during initialization.
--------------------------------------------------------------------------------
Privacy & Security Notice
Although these external engines are utilized, the RePAST architecture ensures that all data processing is strictly local-first. Audio files and transcripts are processed entirely on the researcher's local CPU or GPU and are never uploaded to cloud services or external servers.
Connectivity is strictly limited to the initial authentication and download of model weights via Hugging Face; once the environment is initialized, the pipeline operates as a standalone tool. This design protects participant Personally Identifiable Information (PII) and maintains compliance with IRB, GDPR, and HIPAA protocols regarding the handling of sensitive biometric data.
