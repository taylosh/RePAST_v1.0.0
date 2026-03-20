/**
 * @file gpu_backend.h
 * @author taylosh
 * Created on Dec 4 2025
 * Last edited on Feb 28 2026
 * 
 * Unified GPU backend management header for the overhauled ASR Pipeline.
 * Provides a robust, abstracted interface for multi-backend hardware acceleration
 * supporting CUDA, OpenCL, and DirectML with automatic detection and fail-safe 
 * CPU fallbacks.
 */

#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Backend types for multi-hardware support */
typedef enum {
    GPU_BACKEND_NONE,
    GPU_BACKEND_CUDA,
    GPU_BACKEND_OPENCL,
    GPU_BACKEND_DIRECTML,
    GPU_BACKEND_AUTO
} GPUBackend;

/* Device information structure for tracking hardware capabilities */
typedef struct GPUDeviceInfo {
    char name[7];
    int supports_float;
    int supports_double;
    size_t memory_mb;
    int compute_units;
    int max_work_group_size;
} GPUDeviceInfo;

/* Memory allocation tracking for leak detection and debugging */
typedef struct GPUMemoryInfo {
    void* ptr;
    size_t size;
    int is_device_memory;
    GPUBackend backend_type;
    char name[8];
} GPUMemoryInfo;

/* Main GPU context shared across pipeline modules */
typedef struct GPUContext {
    int initialized;
    int enabled;
    GPUBackend backend_type;
    void* backend_context;     /* Store API-specific context */
    void* opencl_context;      /* Explicit OpenCL handle */
    void* opencl_command_queue;/* Explicit OpenCL queue handle */
    GPUDeviceInfo device_info;
    GPUMemoryInfo* allocations;
    size_t allocation_count;
    size_t max_allocations;
    float total_execution_time_ms;
} GPUContext;

/* Kernel handle for parallel execution */
typedef void* GPUKernel;

/* Backend initialization and lifecycle management */
int gpu_backend_init(GPUBackend preferred_backend);
void gpu_backend_cleanup(void);
int gpu_backend_reinit(GPUBackend new_backend);
void gpu_backend_set_enabled(int enabled);
int gpu_backend_is_enabled(void);

/* Context and availability accessors */
GPUContext* gpu_backend_get_context(void);
int gpu_backend_is_available(void);
GPUBackend gpu_backend_get_active_backend(void);
const char* gpu_backend_get_backend_name(GPUBackend backend);
const char* gpu_backend_get_active_backend_name(void);

/* Device query and configuration functions */
int gpu_backend_get_device_count(void);
int gpu_backend_get_device_info(int device_idx, GPUDeviceInfo* info);
int gpu_backend_set_device(int device_idx);
int gpu_backend_get_max_work_group_size(void);
size_t gpu_backend_get_free_memory(void);
size_t gpu_backend_get_total_memory(void);

/* Hardware-accelerated memory management */
void* gpu_backend_malloc(size_t size, const char* allocation_name);
void gpu_backend_free(void* ptr);
int gpu_backend_memcpy(void* dst, const void* src, size_t size, int to_device);
int gpu_backend_memset(void* ptr, int value, size_t size);
int gpu_backend_memcpy_async(void* dst, const void* src, size_t size, int to_device);
void* gpu_backend_malloc_pinned(size_t size);
void gpu_backend_free_pinned(void* ptr);

/* High-performance kernel execution routines */
int gpu_backend_launch_kernel(GPUKernel kernel, void* args, size_t grid_size, size_t block_size);
int gpu_backend_launch_kernel_async(GPUKernel kernel, void* args, size_t grid_size, size_t block_size);

/* Performance profiling and error checking utility functions */
int gpu_backend_supports_feature(const char* feature_name);
float gpu_backend_get_last_execution_time(void);
float gpu_backend_get_total_execution_time(void);
void gpu_backend_reset_timers(void);
void gpu_backend_synchronize(void);
int gpu_backend_check_error(const char* operation);
void gpu_backend_print_info(void);

/* Direct access functions for unified module integration */
GPUContext* get_gpu_context(void);
int is_gpu_available(void);

#ifdef __cplusplus
}
#endif

#endif /* GPU_BACKEND_H */
