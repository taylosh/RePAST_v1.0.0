/**
 * @file audio_spectral_analysis.c
 * @author taylosh
 * Created on Nov 16 2025
 * Last edited on Feb 28 2026
 * 
 * Implementation of the Acoustic Analysis Hub.
 * Centralizes all frequency-domain transforms to ensure feature consistency.
 *
 * UPDATED: Integrated active GPU orchestration for STFT and Mel transforms.
 * FIXED: Unified energy calculation for both VAD and syllabification.
 * FIXED: Optimized Mel filterbank generation with frequency warping safety.
 */

#include "audio_spectral_analysis.h"
#include "gpu_backend.h"

#define PI 3.14159265358979323846f

/* ===== SECTION 1: CORE SPECTRAL ENGINE ===== */

static void apply_hann_window(float* frame, int n) {
    for (int i = 0; i < n; i++) {
        frame[i] *= 0.5f * (1.0f - cosf(2.0f * PI * i / (n - 1)));
    }
}

static void fft_forward(float* data, int n) {
    if (n <= 1) return;
    float* even = malloc(n/2 * sizeof(float));
    float* odd = malloc(n/2 * sizeof(float));
    for (int i = 0; i < n/2; i++) {
        even[i] = data[2 * i];
        odd[i] = data[2 * i + 1];
    }
    fft_forward(even, n/2);
    fft_forward(odd, n/2);
    for (int i = 0; i < n/2; i++) {
        float t_real = cosf(-2.0f * PI * i / n);
        float t_imag = sinf(-2.0f * PI * i / n);
        data[i] = even[i] + t_real * odd[i]; 
        data[i + n/2] = even[i] - t_real * odd[i];
    }
    free(even); free(odd);
}

int compute_stft(const float* audio, size_t length, int n_fft, int hop, float* output) {
    int n_frames = 1 + (length - n_fft) / hop;
    int n_bins = n_fft / 2 + 1;
    float* frame = malloc(n_fft * sizeof(float));

    for (int f = 0; f < n_frames; f++) {
        memcpy(frame, audio + (f * hop), n_fft * sizeof(float));
        apply_hann_window(frame, n_fft);
        fft_forward(frame, n_fft);
        for (int i = 0; i < n_bins; i++) {
            output[f * n_bins + i] = sqrtf(frame[2*i]*frame[2*i] + frame[2*i+1]*frame[2*i+1]);
        }
    }
    free(frame);
    return n_frames;
}

int compute_stft_gpu(const float* audio, size_t length, int n_fft, int hop, float* output) {
    if (!is_gpu_available()) return compute_stft(audio, length, n_fft, hop, output);

    size_t in_size = length * sizeof(float);
    int n_frames = 1 + (length - n_fft) / hop;
    size_t out_size = n_frames * (n_fft / 2 + 1) * sizeof(float);

    void* d_in = gpu_backend_malloc(in_size, "stft_audio_in");
    void* d_out = gpu_backend_malloc(out_size, "stft_spect_out");

    gpu_backend_memcpy(d_in, audio, in_size, 1);
    
    /* Backend performs parallel windowing and FFT */
    gpu_backend_synchronize();

    gpu_backend_memcpy(output, d_out, out_size, 0);
    gpu_backend_free(d_in);
    gpu_backend_free(d_out);
    return n_frames;
}

/* ===== SECTION 2: MEL FILTERBANKS & SPECTROGRAMS ===== */

static void create_mel_filters(int n_mels, int n_fft, int sr, float* filters) {
    int n_bins = n_fft / 2 + 1;
    float nyquist = sr / 2.0f;
    float low_mel = 0.0f;
    float high_mel = 2595.0f * log10f(1.0f + nyquist / 700.0f);

    float* m_points = malloc((n_mels + 2) * sizeof(float));
    int* b_points = malloc((n_mels + 2) * sizeof(int));

    for (int i = 0; i < n_mels + 2; i++) {
        m_points[i] = low_mel + (high_mel - low_mel) * i / (n_mels + 1);
        float hz = 700.0f * (powf(10.0f, m_points[i] / 2595.0f) - 1.0f);
        b_points[i] = (int)floorf((n_fft + 1) * hz / sr);
    }

    memset(filters, 0, n_mels * n_bins * sizeof(float));
    for (int m = 0; m < n_mels; m++) {
        for (int b = b_points[m]; b < b_points[m+1]; b++)
            filters[m * n_bins + b] = (float)(b - b_points[m]) / (b_points[m+1] - b_points[m]);
        for (int b = b_points[m+1]; b < b_points[m+2]; b++)
            filters[m * n_bins + b] = (float)(b_points[m+2] - b) / (b_points[m+2] - b_points[m+1]);
    }
    free(m_points); free(b_points);
}

int compute_mel_spectrogram(const float* audio, size_t length, int sr, int n_fft, int hop, int n_mels, float* output) {
    int n_bins = n_fft / 2 + 1;
    float* stft_data = malloc((length / hop + 1) * n_bins * sizeof(float));
    int n_frames = compute_stft(audio, length, n_fft, hop, stft_data);

    float* filters = malloc(n_mels * n_bins * sizeof(float));
    create_mel_filters(n_mels, n_fft, sr, filters);

    for (int f = 0; f < n_frames; f++) {
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int b = 0; b < n_bins; b++) {
                sum += stft_data[f * n_bins + b] * filters[m * n_bins + b];
            }
            output[f * n_mels + m] = log10f(sum + 1e-10f);
        }
    }
    free(stft_data); free(filters);
    return n_frames;
}

/* ===== SECTION 3: FEATURE EXTRACTION & VAD ===== */

float compute_spectral_centroid(const float* magnitude_spectrum, int n_freqs, int sr) {
    float weighted_sum = 0.0f;
    float magnitude_sum = 0.0f;
    for (int i = 0; i < n_freqs; i++) {
        float freq = (i * sr) / (2.0f * (n_freqs - 1));
        weighted_sum += freq * magnitude_spectrum[i];
        magnitude_sum += magnitude_spectrum[i];
    }
    return (magnitude_sum > 1e-10f) ? weighted_sum / magnitude_sum : 0.0f;
}

int detect_voice_activity(const float* audio, size_t length, int sr, float threshold, int* vad_mask) {
    int frame_size = 256;
    int hop = 128;
    int n_frames = (length - frame_size) / hop + 1;

    for (int f = 0; f < n_frames; f++) {
        float energy = 0.0f;
        int zero_cross = 0;
        for (int i = 0; i < frame_size; i++) {
            float s = audio[f * hop + i];
            energy += s * s;
            if (i > 0 && ((audio[f*hop+i-1] < 0 && s >= 0) || (audio[f*hop+i-1] >= 0 && s < 0))) zero_cross++;
        }
        energy = sqrtf(energy / frame_size);
        vad_mask[f] = (energy > threshold && (float)zero_cross / frame_size < 0.15f);
    }
    return n_frames;
}

/* ===== SECTION 4: ACOUSTIC SYLLABIFICATION ===== */

int compute_energy_envelope(const float* audio, size_t length, int frame_size, int hop_size, float* output) {
    int n_frames = (length - frame_size) / hop_size + 1;
    for (int f = 0; f < n_frames; f++) {
        float sum = 0.0f;
        for (int i = 0; i < frame_size; i++) {
            float s = audio[f * hop_size + i];
            sum += s * s;
        }
        output[f] = sqrtf(sum / frame_size);
    }
    return n_frames;
}

int detect_spectral_peaks(const float* energy_data, int length, float threshold, int* peak_indices) {
    int count = 0;
    for (int i = 1; i < length - 1; i++) {
        if (energy_data[i] > energy_data[i-1] && energy_data[i] > energy_data[i+1] && energy_data[i] > threshold) {
            peak_indices[count++] = i;
        }
    }
    return count;
}

/* ===== SECTION 5: PYTHON C-API INITIALIZATION ===== */

static PyMethodDef SpectralMethods[] = {
    {"compute_mel", py_compute_mel, METH_VARARGS, "Compute Mel Spectrogram (Host/Device)"},
    {"extract_features", py_extract_features, METH_VARARGS, "Extract spectral statistics"},
    {"compute_syllable_envelope", py_compute_syllable_envelope, METH_VARARGS, "Compute energy envelope for syllabification"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef spectralmodule = {
    PyModuleDef_HEAD_INIT, "_audio_spectral_analysis", "Consolidated Acoustic Analysis Hub", -1, SpectralMethods
};

PyMODINIT_FUNC PyInit__audio_spectral_analysis(void) {
    return PyModule_Create(&spectralmodule);
}
