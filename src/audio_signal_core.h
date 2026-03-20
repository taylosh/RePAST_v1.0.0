/**
 * @file audio_signal_core.h
 * @author taylosh
 * Created on Nov 16 2025
 * Last edited on Feb 28 2026
 * 
 * Consolidated Signal Foundations module for the overhauled ASR Pipeline.
 * Merges basic waveform transformations and enhancement operations into 
 * a single high-performance domain.
 *
 * Features:
 * - Centralized AudioBuffer management.
 * - Basic transforms: stereo-to-mono, normalization, resampling, and gain.
 * - Enhancement: noise reduction, equalization, and dynamic range compression.
 * - Integrated GPU acceleration with active memory management and synchronization.
 * - Signal analysis: RMS, Peak, SNR, and silence detection.
 */

#ifndef AUDIO_SIGNAL_CORE_H
#define AUDIO_SIGNAL_CORE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Data Structures --- */

typedef struct {
    float* data;
    size_t samples;
    int sample_rate;
    int channels;
} AudioBuffer;

typedef struct {
    float threshold;
    float reduction_factor;
    int window_size;
    int algorithm; /* 0=soft, 1=hard, 2=spectral_approx */
} NoiseReductionParams;

typedef struct {
    float bass_boost;
    float mid_boost;
    float treble_boost;
    float preamp;
} EqualizationParams;

/* --- Memory Management --- */
AudioBuffer* create_audio_buffer(size_t samples, int sample_rate, int channels);
void free_audio_buffer(AudioBuffer* buffer);

/* --- Core Waveform Transforms (CPU & GPU) --- */
int stereo_to_mono(const float* input, size_t samples, float* output);
int stereo_to_mono_gpu(const float* input, size_t samples, float* output);

int normalize_audio(float* data, size_t samples, float target_peak);
int normalize_audio_gpu(float* data, size_t samples, float target_peak);

int resample_audio(const float* input, size_t in_samples, int in_rate, float* output, size_t out_samples, int out_rate);
int apply_gain(float* data, size_t samples, float gain_db);
int fade_in_out(float* data, size_t samples, size_t fade_samples);

/* --- Enhancement Operations (CPU & GPU) --- */
int remove_noise(float* data, size_t samples, const NoiseReductionParams* params);
int remove_noise_gpu(float* data, size_t samples, const NoiseReductionParams* params);

int apply_equalization(float* data, size_t samples, const EqualizationParams* params);
int apply_equalization_gpu(float* data, size_t samples, const EqualizationParams* params);

int compress_dynamic_range(float* data, size_t samples, float threshold_db, float ratio, float attack_ms, float release_ms, int rate);

/* --- Signal Analysis Utilities --- */
float calculate_rms(const float* data, size_t samples);
float calculate_peak(const float* data, size_t samples);
float calculate_snr(const float* data, size_t samples, float noise_threshold);
int detect_silence(const float* data, size_t samples, float threshold, size_t min_duration, int* regions, size_t max_regions);

/* --- Python C-API Interface --- */
PyObject* py_stereo_to_mono(PyObject* self, PyObject* args);
PyObject* py_normalize(PyObject* self, PyObject* args);
PyObject* py_remove_noise(PyObject* self, PyObject* args);
PyObject* py_apply_equalization(PyObject* self, PyObject* args);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_SIGNAL_CORE_H */
