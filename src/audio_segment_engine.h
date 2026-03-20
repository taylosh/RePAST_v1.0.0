/**
 * @file audio_segment_engine.h
 * @author taylosh
 * Created on Nov 16 2025
 * Last edited on Feb 28 2026
 * 
 * Consolidated Segment & Alignment engine for the overhauled ASR Pipeline.
 * Merges time-domain slicing, concatenation, and hierarchical alignment
 * algorithms into a single high-performance domain.
 *
 * Features:
 * - Single and batch segment extraction (Host/Device).
 * - High-speed segment concatenation with zero-copy optimizations.
 * - Temporal alignment: mapping linguistic syllables to phone intervals.
 * - Hierarchical tier mapping: Phones -> Syllables -> Words.
 * - Memory-mapped file support for large corpus processing.
 */

#ifndef AUDIO_SEGMENT_ENGINE_H
#define AUDIO_SEGMENT_ENGINE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Core Segment Operations --- */
float* extract_segment(const float* input, long start, long end, size_t* out_len);
int extract_segment_gpu(const float* input, size_t total_samples, long start, long end, float* output);

PyObject* batch_extract_segments(PyObject* self, PyObject* args);
PyObject* batch_extract_segments_gpu(PyObject* self, PyObject* args);

/* --- Concatenation Logic --- */
float* concatenate_segments(const float** segments, const size_t* lengths, int count, size_t* total_len);
PyObject* py_concatenate_segments(PyObject* self, PyObject* args);

/* --- Temporal Alignment Algorithms --- */
PyObject* align_syllables_to_intervals(PyObject* self, PyObject* args);
PyObject* extract_phone_intervals_for_words(PyObject* self, PyObject* args);

/* --- Word-Level Slicing --- */
PyObject* extract_word_segments(PyObject* self, PyObject* args);

/* --- Memory Management and Validation --- */
int validate_segment_range(long start, long end, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_SEGMENT_ENGINE_H */
