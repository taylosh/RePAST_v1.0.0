/**
 * @file textgrid.h
 * @author taylosh
 * Created on Nov 16 2025
 * Last edited on Feb 28 2026
 * 
 * Consolidated TextGrid processing module for ASR acceleration.
 * Provides optimized structures and functions for TextGrid parsing, 
 * tier management, speaker detection, and MFA alignment processing.
 *
 * Features:
 * - High-speed TextGrid file parsing and serialization.
 * - Automatic speaker identification from tier metadata.
 * - Hierarchical tier management for phrases, words, and syllables.
 * - Syllable-to-interval alignment logic for fine-grained annotation.
 * - GPU-aware entry points for batch processing large datasets.
 */

#ifndef TEXTGRID_H
#define TEXTGRID_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* TextGrid Structure Types for C-level manipulation */
typedef struct {
    char* name;
    int interval_count;
    float* start_times;
    float* end_times;
    char** labels;
    char speaker_label;
} TextGridTier;

typedef struct {
    char* file_path;
    float xmin;
    float xmax;
    int tier_count;
    TextGridTier** tiers;
} TextGridFile;

/* --- Tier Operations --- */
char detect_speaker_from_tier(const char* tier_name);
char* generate_tier_name(const char* original_name, int rename_format);
TextGridTier* tier_create(const char* name, char speaker_label);
void tier_free(TextGridTier* tier);

/* --- File I/O Operations --- */
TextGridFile* textgrid_load(const char* file_path);
int textgrid_save(TextGridFile* tg, const char* file_path);
void textgrid_free(TextGridFile* tg);

/* --- Python C-API Batch Operations --- */
PyObject* batch_generate_tier_names(PyObject* self, PyObject* args);
PyObject* batch_generate_tier_names_gpu(PyObject* self, PyObject* args);
PyObject* batch_create_syllable_tiers(PyObject* self, PyObject* args);
PyObject* batch_create_syllable_tiers_gpu(PyObject* self, PyObject* args);

/* --- Alignment and Interval Processing --- */
PyObject* extract_phone_intervals_for_words(PyObject* self, PyObject* args);
PyObject* align_syllables_to_intervals(PyObject* self, PyObject* args);

/* --- MFA Output Validation --- */
PyObject* parse_mfa_output_name(PyObject* self, PyObject* args);
PyObject* parse_mfa_output_name_gpu(PyObject* self, PyObject* args);
PyObject* validate_mfa_results(PyObject* self, PyObject* args);
PyObject* validate_mfa_results_gpu(PyObject* self, PyObject* args);

#ifdef __cplusplus
}
#endif

#endif /* TEXTGRID_H */
