/**
 * @file PrunedVolterraFilter.h
 * @brief CPU-side 3rd-Order Diagonal Memory Polynomial — real-time safe.
 *
 * This class implements the pure CPU processing path.  It is always compiled
 * regardless of CUDA availability and is used:
 *   - As the exclusive processor when CUDA is absent.
 *   - As the real-time sample-accurate fallback when CUDA is present but
 *     the block size is too small to justify GPU round-trip overhead.
 *
 * See VolterraProcessor.h for the dispatch logic.
 *
 * MATHEMATICAL MODEL
 * ------------------
 *   y[n] = Σ_{m=0}^{M-1}  h1[m]·x[n-m]
 *         + Σ_{m=0}^{M-1}  h2[m]·x[n-m]·|x[n-m]|
 *         + Σ_{m=0}^{M-1}  h3[m]·x[n-m]³
 *
 * MEMORY LAYOUT (unchanged from v1, CUDA-compatible)
 * ---------------------------------------------------
 * kernels_[] is interleaved: [h1[0],h2[0],h3[0], h1[1],h2[1],h3[1], ...]
 * This matches the d_kernels_ layout on the CUDA device so that
 * VolterraProcessor can upload the same array to both paths without reformatting.
 */

#pragma once

#include <array>
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef _MSC_VER
#  define VOLTERRA_RESTRICT __restrict
#else
#  define VOLTERRA_RESTRICT __restrict__
#endif

static constexpr int kMemoryLength   = 64;
static constexpr int kMask           = kMemoryLength - 1;
static constexpr int kKernelArraySize = 3 * kMemoryLength;

static_assert((kMemoryLength & (kMemoryLength - 1)) == 0,
              "kMemoryLength must be a power of two.");

class PrunedVolterraFilter
{
public:
    PrunedVolterraFilter() noexcept
    {
        loadWarmthKernels();
        reset();
    }

    // ── Public API ────────────────────────────────────────────────────────────

    void setKernels(const float* h1, const float* h2, const float* h3) noexcept
    {
        for (int m = 0; m < kMemoryLength; ++m)
        {
            kernels_[3 * m + 0] = h1[m];
            kernels_[3 * m + 1] = h2[m];
            kernels_[3 * m + 2] = h3[m];
        }
    }

    /**
     * @brief Set kernels from the interleaved flat array (matches CUDA device layout).
     *
     * Allows VolterraProcessor to push a single interleaved array to both
     * the CPU filter and the CUDA processor without two separate reformatting steps.
     *
     * @param interleavedKernels  Array of 3*M floats in [h1[0],h2[0],h3[0],...] order.
     */
    void setKernelsInterleaved(const float* interleavedKernels) noexcept
    {
        std::memcpy(kernels_.data(), interleavedKernels,
                    kKernelArraySize * sizeof(float));
    }

    /**
     * @brief Read-only access to the interleaved kernel array.
     *
     * VolterraProcessor uses this pointer to upload kernels to the CUDA device
     * without an intermediate copy.
     */
    [[nodiscard]] const float* getKernelsInterleaved() const noexcept
    {
        return kernels_.data();
    }

    void reset() noexcept
    {
        history_.fill(0.0f);
        writeHead_ = 0;
    }

    /**
     * @brief Process a block sample-by-sample on the CPU.
     *
     * Inner loop structured for AVX2 auto-vectorisation:
     *   [v6] Double-length history buffer eliminates ring-buffer modulo from
     *   the inner tap loop — addresses are contiguous, enabling the compiler
     *   to emit stride-1 SIMD loads for the 64-tap accumulation.
     *   • Trip count is compile-time constant (64).
     *   • No loop-carried dependency on `acc`.
     *   • __restrict__ on kernelPtr signals no aliasing to the compiler.
     *   • alignas(64) on history_ and kernels_ enables aligned SIMD loads.
     */
    void processBlock(const float* inputSignal,
                            float* outputSignal,
                      int numSamples) noexcept
    {
        const float* VOLTERRA_RESTRICT kernelPtr = kernels_.data();

        for (int n = 0; n < numSamples; ++n)
        {
            // [v6] Write to both halves of the double-length buffer so the
            // inner loop reads history_[writeHead_+m] with contiguous addresses
            // and no modulo, enabling AVX2/SSE auto-vectorisation.
            const auto base = static_cast<std::size_t>(writeHead_);
            history_[base]                = inputSignal[n];
            history_[base + kMemoryLength] = inputSignal[n];

            float acc = 0.0f;

            for (int m = 0; m < kMemoryLength; ++m)
            {
                // [v6] Direct index — range [0, 2M-2], no modulo, stride-1.
                const float xs      = history_[base + static_cast<std::size_t>(m)];
                const float xs_abs  = std::fabs(xs);
                const float xs_sq   = xs * xs;

                const float h1 = kernelPtr[3 * m + 0];
                const float h2 = kernelPtr[3 * m + 1];
                const float h3 = kernelPtr[3 * m + 2];

                acc += h1 * xs;
                acc += h2 * (xs * xs_abs);
                acc += h3 * (xs_sq * xs);
            }

            // [v6] Removed [[unlikely]] C++20 attribute — project targets C++17
            // and MSVC emitted C5051 warnings for it without any code-gen benefit.
            if (!std::isfinite(acc))
            {
                reset();
                acc = 0.0f;
            }

            outputSignal[n] = std::clamp(acc, -1.0f, 1.0f);
            writeHead_ = (writeHead_ - 1) & kMask;
        }
    }

    // [v6] Efficient single-sample entry point — avoids the outer loop and
    // numSamples branch of processBlock when called sample-by-sample from
    // VolterraProcessor::processBlockCPU.
    [[nodiscard]] float processSample (float x) noexcept
    {
        const float* VOLTERRA_RESTRICT kernelPtr = kernels_.data();

        const auto base = static_cast<std::size_t>(writeHead_);
        history_[base]                = x;
        history_[base + kMemoryLength] = x;

        float acc = 0.0f;
        for (int m = 0; m < kMemoryLength; ++m)
        {
            const float xs     = history_[base + static_cast<std::size_t>(m)];
            const float xs_abs = std::fabs(xs);
            const float xs_sq  = xs * xs;

            acc += kernelPtr[3 * m + 0] * xs;
            acc += kernelPtr[3 * m + 1] * (xs * xs_abs);
            acc += kernelPtr[3 * m + 2] * (xs_sq * xs);
        }

        if (!std::isfinite(acc))
        {
            reset();
            acc = 0.0f;
        }

        writeHead_ = (writeHead_ - 1) & kMask;
        return std::clamp(acc, -1.0f, 1.0f);
    }

    [[nodiscard]] float getCoefficient(int order, int tap) const noexcept
    {
        if (tap < 0 || tap >= kMemoryLength || order < 1 || order > 3)
            return 0.0f;
        return kernels_[3 * tap + (order - 1)];
    }

private:
    // [v6] Double-length: write each sample to both halves so the inner loop
    // reads history_[writeHead_ + m] with contiguous (non-wrapping) addresses.
    alignas(64) std::array<float, 2 * kMemoryLength> history_{};
    alignas(64) std::array<float, kKernelArraySize> kernels_{};
    int writeHead_ = 0;

    void loadWarmthKernels() noexcept
    {
        constexpr float kDecayH1 = 0.88f, kDecayH2 = 0.72f, kDecayH3 = 0.65f;
        constexpr float kH1_0   =  1.0f,  kH2_0   =  0.04f, kH3_0   = -0.025f;

        float h1[kMemoryLength], h2[kMemoryLength], h3[kMemoryLength];
        float v1 = kH1_0, v2 = kH2_0, v3 = kH3_0;
        for (int m = 0; m < kMemoryLength; ++m)
        {
            h1[m] = v1; h2[m] = v2; h3[m] = v3;
            v1 *= kDecayH1; v2 *= kDecayH2; v3 *= kDecayH3;
        }
        setKernels(h1, h2, h3);
    }
};
