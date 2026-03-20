/**
 * @file textgrid.c
 * @author taylosh
 * Created on Nov 16 2025
 * Last edited on Feb 28 2026
 * 
 * TextGrid processing implementation for the overhauled ASR pipeline.
 * Handles the heavy lifting of parsing Praat TextGrid files and performing
 * linguistic-to-temporal alignment for the annotation stage.
 *
 * UPDATED: Integrated with centralized GPU context for synchronization.
 * FIXED: Refined syllable-to-phone interval mapping logic.
 * FIXED: Improved speaker detection heuristics for multi-speaker files.
 */

#include "textgrid.h"
#include "gpu_backend.h"
#include <ctype.h>
#include <math.h>

/* ===== TIER OPERATIONS ===== */

char detect_speaker_from_tier(const char* tier_name) {
    if (!tier_name || strlen(tier_name) == 0) return 'A';

    char speaker_char = 'A';
    int found_b = 0, found_c = 0;
    const char* p = tier_name;

    while (*p) {
        char c = tolower((unsigned char)*p);
        if (c == 'b' && (p == tier_name || !isalnum((unsigned char)*(p-1)))) found_b = 1;
        else if (c == 'c' && (p == tier_name || !isalnum((unsigned char)*(p-1)))) found_c = 1;

        if (strncmp(p, "speaker_a", 9) == 0 || strncmp(p, "speaker_0", 9) == 0) {
            speaker_char = 'A'; break;
        } else if (strncmp(p, "speaker_b", 9) == 0 || strncmp(p, "speaker_1", 9) == 0) {
            speaker_char = 'B'; break;
        } else if (strncmp(p, "speaker_c", 9) == 0 || strncmp(p, "speaker_2", 9) == 0) {
            speaker_char = 'C'; break;
        }
        p++;
    }

    if (speaker_char == 'A') {
        if (found_b) speaker_char = 'B';
        else if (found_c) speaker_char = 'C';
    }
    return speaker_char;
}

char* generate_tier_name(const char* original_name, int rename_format) {
    if (!original_name) return NULL;
    char* new_name = malloc(256);
    if (!new_name) return NULL;

    const char* type_str = "unknown";
    char speaker_char = detect_speaker_from_tier(original_name);
    
    if (strstr(original_name, "phrase") || strstr(original_name, "utterance")) type_str = "phrases";
    else if (strstr(original_name, "word")) type_str = "words";
    else if (strstr(original_name, "phone") || strstr(original_name, "segment")) type_str = "phones";

    if (rename_format == 1) {
        snprintf(new_name, 256, "%c_%s", speaker_char, type_str);
    } else {
        snprintf(new_name, 256, "mfa_%s", original_name);
    }
    return new_name;
}

/* ===== TEXTGRID FILE I/O ===== */

TextGridFile* textgrid_load(const char* file_path) {
    if (!file_path) return NULL;
    FILE* file = fopen(file_path, "r");
    if (!file) return NULL;

    TextGridFile* tg = calloc(1, sizeof(TextGridFile));
    tg->file_path = strdup(file_path);
    
    char line;
    int tier_idx = -1;
    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "xmin = ")) sscanf(line, " xmin = %f", &tg->xmin);
        else if (strstr(line, "xmax = ")) sscanf(line, " xmax = %f", &tg->xmax);
        else if (strstr(line, "size = ")) {
            sscanf(line, " size = %d", &tg->tier_count);
            tg->tiers = calloc(tg->tier_count, sizeof(TextGridTier*));
        } else if (strstr(line, "class = \"IntervalTier\"")) {
            tier_idx++;
            if (tier_idx < tg->tier_count) {
                tg->tiers[tier_idx] = calloc(1, sizeof(TextGridTier));
            }
        }
    }
    fclose(file);
    return tg;
}

int textgrid_save(TextGridFile* tg, const char* file_path) {
    if (!tg || !file_path) return -1;
    FILE* file = fopen(file_path, "w");
    if (!file) return -1;

    fprintf(file, "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n\n");
    fprintf(file, "xmin = %f\nxmax = %f\ntiers? <exists>\nsize = %d\nitem []:\n", 
            tg->xmin, tg->xmax, tg->tier_count);

    for (int i = 0; i < tg->tier_count; i++) {
        TextGridTier* tier = tg->tiers[i];
        fprintf(file, "    item [%d]:\n        class = \"IntervalTier\"\n", i + 1);
        fprintf(file, "        name = \"%s\"\n        xmin = %f\n        xmax = %f\n", 
                tier->name, tg->xmin, tg->xmax);
        fprintf(file, "        intervals: size = %d\n", tier->interval_count);
        for (int j = 0; j < tier->interval_count; j++) {
            fprintf(file, "        intervals [%d]:\n            xmin = %f\n            xmax = %f\n            text = \"%s\"\n",
                    j + 1, tier->start_times[j], tier->end_times[j], tier->labels[j]);
        }
    }
    fclose(file);
    return 0;
}

/* ===== SYLLABLE ALIGNMENT LOGIC ===== */

PyObject* align_syllables_to_intervals(PyObject* self, PyObject* args) {
    PyObject *syllable_texts, *phone_intervals;
    float word_start, word_end;

    if (!PyArg_ParseTuple(args, "OOff", &syllable_texts, &phone_intervals, &word_start, &word_end)) return NULL;
    if (!PyList_Check(syllable_texts)) return PyList_New(0);

    Py_ssize_t num_syllables = PyList_Size(syllable_texts);
    PyObject* result = PyList_New(num_syllables);
    double word_duration = (double)word_end - (double)word_start;

    Py_ssize_t num_phones = PyList_Check(phone_intervals) ? PyList_Size(phone_intervals) : 0;

    if (num_phones >= num_syllables && num_syllables > 0) {
        /* Distribute phones to syllables */
        int phones_per = (int)(num_phones / num_syllables);
        int remainder = (int)(num_phones % num_syllables);
        int phone_idx = 0;

        for (int i = 0; i < num_syllables; i++) {
            int count = phones_per + (i < remainder ? 1 : 0);
            PyObject* first = PyList_GetItem(phone_intervals, phone_idx);
            PyObject* last = PyList_GetItem(phone_intervals, phone_idx + count - 1);
            
            double start = PyFloat_AsDouble(PyTuple_GetItem(first, 0));
            double end = PyFloat_AsDouble(PyTuple_GetItem(last, 1));
            PyList_SetItem(result, i, Py_BuildValue("(dd)", start, end));
            phone_idx += count;
        }
    } else {
        /* Proportional division fallback */
        double dur = word_duration / (double)num_syllables;
        for (int i = 0; i < num_syllables; i++) {
            double s = word_start + (i * dur);
            double e = (i == num_syllables - 1) ? word_end : s + dur;
            PyList_SetItem(result, i, Py_BuildValue("(dd)", s, e));
        }
    }
    return result;
}

/* ===== GPU AWARE WRAPPERS ===== */

PyObject* batch_generate_tier_names_gpu(PyObject* self, PyObject* args) {
    if (is_gpu_available()) gpu_backend_synchronize();
    return batch_generate_tier_names(self, args);
}

PyObject* validate_mfa_results_gpu(PyObject* self, PyObject* args) {
    if (is_gpu_available()) gpu_backend_check_error("MFA Validation");
    return validate_mfa_results(self, args);
}

/* ===== PYTHON MODULE INITIALIZATION ===== */

static PyMethodDef TextGridMethods[] = {
    {"batch_generate_tier_names", batch_generate_tier_names, METH_VARARGS, "Rename tiers in batch"},
    {"batch_generate_tier_names_gpu", batch_generate_tier_names_gpu, METH_VARARGS, "GPU-aware tier renaming"},
    {"align_syllables_to_intervals", align_syllables_to_intervals, METH_VARARGS, "Align syllables to phone intervals"},
    {"validate_mfa_results_gpu", validate_mfa_results_gpu, METH_VARARGS, "GPU-aware MFA validation"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef textgridmodule = {
    PyModuleDef_HEAD_INIT, "_textgrid", "TextGrid Acceleration Module", -1, TextGridMethods
};

PyMODINIT_FUNC PyInit__textgrid(void) {
    return PyModule_Create(&textgridmodule);
}
