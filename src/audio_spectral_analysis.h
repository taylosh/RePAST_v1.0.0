/**
 * @file audio_spectral_analysis.h
 * @author taylosh
 * Created on Nov 16 2025
 * Last edited on Feb 28 2026
 * 
 * Consolidated Acoustic Analysis Hub for the overhauled ASR Pipeline.
 * Merges spectral transforms, feature extraction, and acoustic syllabification
 * into a single high-performance frequency-domain engine.
 *
 * Features:
 * - High-resolution STFT and Power Spectrogram generation.
 * - Whisper-style Mel filterbanks and Mel-spectrogram transforms.
 * - Comprehensive feature extraction: MFCCs, Centroid, Rolloff, and Flatness.
 * - Acoustic syllabification: RMS Energy Envelopes and Peak Detection.
 * - Adaptive Voice Activity Detection (VAD).
 * - Active GPU orchestration for parallel spectral transforms.
 */

#ifndef AUDIO_SPECTRAL_ANALYSIS_H
#define AUDIO_SPECTRAL_ANALYSIS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Spectral Configuration Constants --- */
#define DEFAULT_N_FFT 512
#define DEFAULT_HOP_LENGTH 160
#define WHISPER_N_MELS 80

/* --- Data Structures --- */

typedef struct {
    float* bins;
    int n_mels;
    int n_freqs;
    int sample_rate;
} MelFilterbank;

typedef struct {
    float* values;
    int n_frames;
    int n_features;
} SpectralFeatures;

/* --- Core Spectral Transforms (CPU & GPU) --- */
int compute_stft(const float* audio, size_t length, int n_fft, int hop, float* output);
int compute_stft_gpu(const float* audio, size_t length, int n_fft, int hop, float* output);

int compute_mel_spectrogram(const float* audio, size_t length, int sample_rate, int n_fft, int hop, int n_mels, float* output);
int compute_mel_spectrogram_gpu(const float* audio, size_t length, int sample_rate, int n_fft, int hop, int n_mels, float* output);

/* --- Feature Extraction --- */
int extract_mfccs(const float* audio, size_t length, int sample_rate, int n_mfcc, float* output);
float compute_spectral_centroid(const float* magnitude_spectrum, int n_freqs, int sample_rate);
float compute_spectral_rolloff(const float* magnitude_spectrum, int n_freqs, int sample_rate, float percentile);
float compute_spectral_flatness(const float* magnitude_spectrum, int n_freqs);

/* --- Acoustic Syllabification --- */
int compute_energy_envelope(const float* audio, size_t length, int frame_size, int hop_size, float* output);
int detect_spectral_peaks(const float* energy_data, int length, float threshold, int* peak_indices);

/* --- Voice Activity Detection --- */
int detect_voice_activity(const float* audio, size_t length, int sample_rate, float threshold, int* vad_mask);

/* --- Python C-API Interface --- */
PyObject* py_compute_mel(PyObject* self, PyObject* args);
PyObject* py_extract_features(PyObject* self, PyObject* args);
PyObject* py_compute_syllable_envelope(PyObject* self, PyObject* args);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_SPECTRAL_ANALYSIS_H */
