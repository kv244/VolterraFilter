/**
 * @file VolterraKernels.cuh
 * @brief CUDA device kernels for the Pruned Volterra / Memory Polynomial Filter.
 *
 * TWO ROLES
 * ---------
 *  1. Batch offline processing  – processBlockCUDA()
 *     Processes an entire audio buffer on the GPU.  Each CUDA thread handles
 *     one output sample.  The serial time-dependency (ring buffer) is broken
 *     by pre-copying the full input history into a flat look-back array before
 *     launching, which is valid for offline / non-streaming contexts.
 *
 *  2. Feature matrix construction – buildFeatureMatrixCUDA()
 *     Builds the N×(3M) regressor matrix Φ used for least-squares kernel
 *     fitting.  Each thread fills one row (one sample's worth of basis
 *     functions).  Feeds directly into cuBLAS/cuSOLVER for the solve step.
 *
 * REAL-TIME NOTE
 * --------------
 * The real-time VST3 process loop stays on the CPU (see VolterraProcessor.h).
 * CUDA is invoked only for:
 *   a) Offline batch rendering / export
 *   b) Kernel learning from reference captures
 *
 * COMPILE GUARD
 * -------------
 * Every symbol in this file is guarded by VOLTERRA_CUDA_AVAILABLE, which
 * CMakeLists.txt defines iff find_package(CUDAToolkit) succeeds.
 */

#pragma once

#ifdef VOLTERRA_CUDA_AVAILABLE

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
//  Constants mirrored on device
// ─────────────────────────────────────────────────────────────────────────────

/// Must match kMemoryLength in PrunedVolterraFilter.h
static constexpr int kCudaMemoryLength = 64;

/// Coefficients per tap (h1, h2, h3)
static constexpr int kOrders = 3;

/// Total coefficients = 3 × M
static constexpr int kTotalCoeffs = kOrders * kCudaMemoryLength;

// ─────────────────────────────────────────────────────────────────────────────
//  Device-side helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compute the three basis functions for a single scalar input sample.
 *
 * Inlined into both kernels below to avoid register spill from a device call.
 *
 * @param xs        The delayed input sample x[n-m]
 * @param[out] b1   Linear basis:     xs
 * @param[out] b2   Quadratic basis:  xs * |xs|
 * @param[out] b3   Cubic basis:      xs^3
 */
__device__ __forceinline__
void computeBases(float xs, float& b1, float& b2, float& b3)
{
    float xs_abs = fabsf(xs);
    b1 = xs;
    b2 = xs * xs_abs;    // x·|x| — signed quadratic, preserves polarity
    b3 = xs * xs * xs;   // x³   — symmetric cubic
}

// ─────────────────────────────────────────────────────────────────────────────
//  Kernel 1 – Batch offline block processing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Process N output samples in parallel given a pre-padded input array.
 *
 * LAUNCH CONFIG (suggested):
 *   dim3 block(256);
 *   dim3 grid((numSamples + 255) / 256);
 *   volterraProcessKernel<<<grid, block>>>(d_input_padded, d_output,
 *                                          d_kernels, numSamples);
 *
 * INPUT LAYOUT:
 *   d_input_padded must be a pre-allocated device array of length
 *   (numSamples + kCudaMemoryLength - 1), where the first (kCudaMemoryLength-1)
 *   elements are the history from the previous block (or zeros on first block).
 *   Element [kCudaMemoryLength - 1 + n] is x[n].
 *
 *   This layout lets thread n read x[n-m] as d_input_padded[n + M - 1 - m]
 *   without any modulo or ring-buffer logic.
 *
 * KERNEL LAYOUT (d_kernels):
 *   Interleaved: [h1[0], h2[0], h3[0], h1[1], h2[1], h3[1], ...]
 *   Matches the layout in PrunedVolterraFilter::kernels_.
 *   Loaded into __shared__ memory once per block for reuse across all taps.
 *
 * @param d_input_padded  Device pointer, length numSamples + M - 1
 * @param d_output        Device pointer, length numSamples
 * @param d_kernels       Interleaved kernel array, length 3*M
 * @param numSamples      Number of output samples to compute
 */
__global__
void volterraProcessKernel(const float* __restrict__ d_input_padded,
                                 float* __restrict__ d_output,
                           const float* __restrict__ d_kernels,
                           int numSamples)
{
    // Load the full kernel array into shared memory once per CUDA block.
    // 3 × 64 floats = 768 bytes — well within the 48KB shared limit.
    __shared__ float sh_kernels[kTotalCoeffs];

    // Cooperatively fill shared memory: each thread loads one or more elements.
    for (int i = threadIdx.x; i < kTotalCoeffs; i += blockDim.x)
        sh_kernels[i] = d_kernels[i];
    __syncthreads();

    // Global sample index handled by this thread.
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= numSamples) return;

    // Accumulate across M taps.
    // d_input_padded[n + M - 1 - m]  =  x[n - m]
    float acc = 0.0f;
    const int base = n + kCudaMemoryLength - 1;

    #pragma unroll 8   // unroll factor matches 8-wide float SIMD analogy
    for (int m = 0; m < kCudaMemoryLength; ++m)
    {
        const float xs = d_input_padded[base - m];

        float b1, b2, b3;
        computeBases(xs, b1, b2, b3);

        const float h1 = sh_kernels[3 * m + 0];
        const float h2 = sh_kernels[3 * m + 1];
        const float h3 = sh_kernels[3 * m + 2];

        acc += h1 * b1 + h2 * b2 + h3 * b3;
    }

    // Hard clamp + NaN guard (isfinite check on device).
    if (!isfinite(acc)) acc = 0.0f;
    d_output[n] = fminf(fmaxf(acc, -1.0f), 1.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Kernel 2 – Feature matrix construction (for kernel learning)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Build Φ in row-major layout for subsequent GPU transpose before cuSOLVER.
 *
 * OUTPUT MEMORY LAYOUT (row-major, [N rows × M3 cols]):
 * ───────────────────────────────────────────────────────
 * Thread n writes one complete row — the M3 = 3M basis function evaluations
 * for training sample n, stored at consecutive addresses:
 *
 *   d_Phi[n * M3 + 0*M + m]  = b1[n,m]  = x[n-m]          (linear basis, tap m)
 *   d_Phi[n * M3 + 1*M + m]  = b2[n,m]  = x[n-m]·|x[n-m]| (quadratic basis)
 *   d_Phi[n * M3 + 2*M + m]  = b3[n,m]  = x[n-m]³          (cubic basis)
 *
 * WHY ROW-MAJOR (not column-major directly):
 * ────────────────────────────────────────────
 * Writing column-major directly would require thread n to stride by N floats
 * between successive writes (one per coefficient column).  For N = 44100,
 * each write touches a different 256KB page — N coalescing groups of 1 thread
 * each, destroying memory throughput.
 *
 * Writing row-major gives fully coalesced stores: threads 0..127 write
 * d_Phi[0*M3..127*M3-1] in 127*M3/32 = ~760 transactions of 128 bytes each.
 *
 * The row-major output is transposed to column-major by cublasSgeam() in the
 * host before the cuSOLVER SGELS call.  See fitKernelsCUDA() for details.
 *
 * SOLUTION VECTOR ORDERING (after SGELS):
 * ─────────────────────────────────────────
 * After transpose, column j of the column-major matrix corresponds to row j
 * of this row-major output.  The SGELS solution vector h has grouped layout:
 *   h[0..M-1]   = h1[0..M-1]   (linear kernel)
 *   h[M..2M-1]  = h2[0..M-1]   (quadratic kernel)
 *   h[2M..3M-1] = h3[0..M-1]   (cubic kernel)
 *
 * This grouped layout is re-interleaved to [h1[0],h2[0],h3[0],h1[1],...] on
 * the host before being passed to updateKernels().
 *
 * @param d_input_padded  Padded input, length numSamples + M - 1
 * @param d_Phi           Output: row-major [N × M3], to be transposed before solve
 * @param numSamples      N — number of training samples (rows)
 */
__global__
void buildFeatureMatrixKernel(const float* __restrict__ d_input_padded,
                                    float* __restrict__ d_Phi,
                              int numSamples)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= numSamples) return;

    // Base pointer for this thread's row — M3 consecutive floats.
    // All threads in a warp write to consecutive cache lines → coalesced.
    float* row = d_Phi + n * kTotalCoeffs;

    const int base = n + kCudaMemoryLength - 1;

    #pragma unroll 8
    for (int m = 0; m < kCudaMemoryLength; ++m)
    {
        const float xs = d_input_padded[base - m];

        float b1, b2, b3;
        computeBases(xs, b1, b2, b3);

        // Grouped layout within row: all h1 taps, then h2, then h3.
        // After the cublasSgeam transpose this becomes the correct column
        // grouping that SGELS needs to produce separated h1/h2/h3 solution.
        row[0 * kCudaMemoryLength + m] = b1;
        row[1 * kCudaMemoryLength + m] = b2;
        row[2 * kCudaMemoryLength + m] = b3;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host-side CUDA helper: error checking macro
// ─────────────────────────────────────────────────────────────────────────────

#define VOLTERRA_CUDA_CHECK(call)                                          \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            /* In a JUCE plugin: log via DBG() and fall back to CPU.      \
               Never throw from a CUDA error in the audio thread.  */     \
            DBG("CUDA error in " #call ": " << cudaGetErrorString(err));  \
            return false;                                                  \
        }                                                                  \
    } while (0)

#endif // VOLTERRA_CUDA_AVAILABLE
