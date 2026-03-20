/**
 * @file corpus_utils.c
 * @author taylosh
 * Created on Dec 4 2025
 * Last edited on Feb 28 2026
 *
 * Corpus utilities implementation for ASR pipeline acceleration.
 * Handles high-performance file operations and path processing with
 * hardware-aware batch management and cross-platform support.
 *
 * UPDATED: Integrated with active GPU backend for context synchronization.
 * FIXED: Resolved WSL path mapping for batch normalization.
 * FIXED: Implemented safety checks for recursive directory creation.
 */
#include "corpus_utils.h"
#include "gpu_backend.h"
#include <errno.h>
#include <ctype.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <unistd.h>
#endif

/* ===== CORE FILE SYSTEM OPERATIONS ===== */
int create_directory_recursive(const char* path) {
    if (!path || strlen(path) == 0) return -1;
    char temp[1024]; 
    char* p = NULL;
    size_t len;
    snprintf(temp, sizeof(temp), "%s", path);
    len = strlen(temp);
    if (temp[len - 1] == '/' || temp[len - 1] == '\\') temp[len - 1] = 0;
    for (p = temp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char sep = *p;
            *p = 0;
            if (mkdir(temp, 0755) != 0 && errno != EEXIST) return -1;
            *p = sep;
        }
    }
    return (mkdir(temp, 0755) != 0 && errno != EEXIST) ? -1 : 0;
}

int fast_file_copy(const char* source, const char* dest) {
    if (!source || !dest) return -1;
    FILE *src = fopen(source, "rb");
    if (!src) return -1;
    FILE *dst = fopen(dest, "wb");
    if (!dst) {
        fclose(src);
        return -1;
    }
    char buffer;
    size_t bytes;
    while ((bytes = fread(&buffer, 1, sizeof(buffer), src)) > 0) {
        if (fwrite(&buffer, 1, bytes, dst) != bytes) {
            fclose(src);
            fclose(dst);
            return -1;
        }
    }
    fclose(src);
    return fclose(dst);
}

int file_exists(const char* path) {
    struct stat st;
    return (stat(path, &st) == 0 && S_ISREG(st.st_mode));
}

long get_file_size(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return (long)st.st_size;
}

char* get_file_extension(const char* path) {
    const char* dot = strrchr(path, '.');
    if (!dot || dot == path) return strdup("");
    return strdup(dot + 1);
}

/* ===== BATCH OPERATIONS WITH GPU SYNCHRONIZATION ===== */
PyObject* batch_copy_files(PyObject* self, PyObject* args) {
    PyObject* file_pairs;
    if (!PyArg_ParseTuple(args, "O", &file_pairs)) return NULL;
    if (!PyList_Check(file_pairs)) {
        PyErr_SetString(PyExc_TypeError, "Expected list of (src, dst) tuples");
        return NULL;
    }
    Py_ssize_t n = PyList_Size(file_pairs);
    PyObject* results = PyList_New(n);
    int success_count = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *pair = PyList_GetItem(file_pairs, i);
        const char *src = PyUnicode_AsUTF8(PyTuple_GetItem(pair, 0));
        const char *dst = PyUnicode_AsUTF8(PyTuple_GetItem(pair, 1));
        /* Ensure parent directory exists before copy */
        char* dst_dir = strdup(dst);
        char* last = strrchr(dst_dir, '/');
        if (!last) last = strrchr(dst_dir, '\\');
        if (last) {
            *last = 0;
            create_directory_recursive(dst_dir);
        }
        free(dst_dir);
        int res = (fast_file_copy(src, dst) == 0);
        PyList_SetItem(results, i, PyBool_FromLong(res));
        if (res) success_count++;
    }
    return Py_BuildValue("Oii", results, success_count, (int)n);
}

PyObject* batch_copy_files_gpu(PyObject* self, PyObject* args) {
    /* File I/O is CPU bound, but we sync the GPU context to ensure data stability */
    if (is_gpu_available()) {
        gpu_backend_synchronize();
    }
    return batch_copy_files(self, args);
}

PyObject* batch_validate_files(PyObject* self, PyObject* args) {
    PyObject* file_list;
    if (!PyArg_ParseTuple(args, "O", &file_list)) return NULL;
    Py_ssize_t n = PyList_Size(file_list);
    PyObject* results = PyList_New(n);
    int exist_count = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        const char* path = PyUnicode_AsUTF8(PyList_GetItem(file_list, i));
        int exists = file_exists(path);
        PyList_SetItem(results, i, PyBool_FromLong(exists));
        if (exists) exist_count++;
    }
    return Py_BuildValue("Oii", results, exist_count, (int)n);
}

PyObject* batch_validate_files_gpu(PyObject* self, PyObject* args) {
    if (is_gpu_available()) {
        gpu_backend_synchronize();
    }
    return batch_validate_files(self, args);
}

PyObject* batch_create_directories(PyObject* self, PyObject* args) {
    PyObject* dir_list;
    if (!PyArg_ParseTuple(args, "O", &dir_list)) return NULL;
    Py_ssize_t n = PyList_Size(dir_list);
    PyObject* results = PyList_New(n);
    int count = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        const char* path = PyUnicode_AsUTF8(PyList_GetItem(dir_list, i));
        int res = (create_directory_recursive(path) == 0);
        PyList_SetItem(results, i, PyBool_FromLong(res));
        if (res) count++;
    }
    return Py_BuildValue("Oii", results, count, (int)n);
}

/* ===== PATH PROCESSING ===== */
PyObject* normalize_path(PyObject* self, PyObject* args) {
    const char* path;
    if (!PyArg_ParseTuple(args, "s", &path)) return NULL;
    char norm[4096];
    strncpy(norm, path, 4095);
    norm[4095] = '\0';
    for (int i = 0; norm[i]; i++) {
        if (norm[i] == '\\') norm[i] = '/';
    }
    #ifdef linux
    /* Handle Windows Drive Mapping in WSL (C: -> /mnt/c) */
    if (norm[1] == ':' && isalpha((unsigned char)norm[0])) {
        char wsl_path[4096];
        snprintf(wsl_path, sizeof(wsl_path), "/mnt/%c%s", tolower((unsigned char)norm[0]), norm + 2);
        return PyUnicode_FromString(wsl_path);
    }
    #endif
    return PyUnicode_FromString(norm);
}

PyObject* batch_normalize_paths(PyObject* self, PyObject* args) {
    PyObject* path_list;
    if (!PyArg_ParseTuple(args, "O", &path_list)) return NULL;
    Py_ssize_t n = PyList_Size(path_list);
    PyObject* result = PyList_New(n);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PyList_GetItem(path_list, i);
        PyObject* args_tuple = PyTuple_Pack(1, item);
        PyObject* normalized = normalize_path(self, args_tuple);
        Py_DECREF(args_tuple);
        PyList_SetItem(result, i, normalized);
    }
    return result;
}

PyObject* batch_normalize_paths_gpu(PyObject* self, PyObject* args) {
    if (is_gpu_available()) {
        gpu_backend_synchronize();
    }
    return batch_normalize_paths(self, args);
}

/* ===== PYTHON MODULE INITIALIZATION ===== */
static PyMethodDef CorpusUtilsMethods[] = {
    {"batch_copy_files", batch_copy_files, METH_VARARGS, "Copy files in batch"},
    {"batch_copy_files_gpu", batch_copy_files_gpu, METH_VARARGS, "GPU-aware batch copy"},
    {"batch_validate_files", batch_validate_files, METH_VARARGS, "Validate existence in batch"},
    {"batch_validate_files_gpu", batch_validate_files_gpu, METH_VARARGS, "GPU-aware validation"},
    {"batch_create_directories", batch_create_directories, METH_VARARGS, "Recursive mkdir in batch"},
    {"normalize_path", normalize_path, METH_VARARGS, "Normalize file path format"},
    {"batch_normalize_paths", batch_normalize_paths, METH_VARARGS, "Normalize list of paths"},
    {"batch_normalize_paths_gpu", batch_normalize_paths_gpu, METH_VARARGS, "GPU-aware normalization"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef corpusutilsmodule = {
    PyModuleDef_HEAD_INIT, "_corpus_utils", "High-performance corpus management", -1, CorpusUtilsMethods
};

PyMODINIT_FUNC PyInit__corpus_utils(void) {
    return PyModule_Create(&corpusutilsmodule);
}
