/**
 * @file corpus_utils.h
 * @author taylosh
 * Created on Dec 4 2025
 * Last edited on Feb 28 2026
 * 
 * Corpus utilities module for the overhauled ASR Pipeline.
 * Provides high-performance file system operations, path processing,
 * and batch management with integrated multi-backend GPU support.
 *
 * Features:
 * - High-speed binary file copying with recursive directory creation.
 * - Cross-platform path normalization (Windows, Linux, WSL).
 * - Accelerated batch file validation and existence checking.
 * - GPU-aware entry points for large-scale dataset management.
 */

#ifndef CORPUS_UTILS_H
#define CORPUS_UTILS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Core File Operations --- */
int fast_file_copy(const char* source, const char* dest);
int create_directory_recursive(const char* path);
int file_exists(const char* path);
long get_file_size(const char* path);
char* get_file_extension(const char* path);

/* --- Python C-API Batch Operations --- */
PyObject* batch_copy_files(PyObject* self, PyObject* args);
PyObject* batch_copy_files_gpu(PyObject* self, PyObject* args);
PyObject* batch_validate_files(PyObject* self, PyObject* args);
PyObject* batch_validate_files_gpu(PyObject* self, PyObject* args);
PyObject* batch_create_directories(PyObject* self, PyObject* args);

/* --- Path Processing --- */
PyObject* normalize_path(PyObject* self, PyObject* args);
PyObject* batch_normalize_paths(PyObject* self, PyObject* args);
PyObject* batch_normalize_paths_gpu(PyObject* self, PyObject* args);

#ifdef __cplusplus
}
#endif

#endif /* CORPUS_UTILS_H */
