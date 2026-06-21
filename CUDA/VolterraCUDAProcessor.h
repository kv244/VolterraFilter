/**
 * @file VolterraCUDAProcessor.h
 * @brief Host-side CUDA wrapper — lock-free double-buffered kernel updates.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *  THE RACE CONDITION (fixed in this version)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Previous design had a single kernel buffer `d_kernels_` and a single
 *  CUDA stream.  Two CPU threads could reach it concurrently:
 *
 *    Audio thread   → processBlockCUDA()  → cudaMemcpyAsync(d_kernels_...)
 *                                         → volterraProcessKernel(d_kernels_)
 *                                         → cudaStreamSynchronize(stream_)   ← priority inversion
 *
 *    Message thread → updateKernels()     → cudaMemcpyAsync(d_kernels_...)   ← DATA RACE on stream_
 *
 *  Calling cudaMemcpyAsync on the same stream_ from two CPU threads
 *  simultaneously is undefined behaviour — the CUDA driver's internal stream
 *  state is not thread-safe to concurrent host access.  Additionally,
 *  cudaStreamSynchronize on the audio thread blocks the entire audio callback
 *  if the message thread has pushed work into that stream.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *  THE FIX: Double-buffered kernels + dedicated write stream + event fence
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  TWO separate device buffers hold kernel coefficients:
 *
 *    d_kernels_[0]  ← active buffer   (audio thread reads from this)
 *    d_kernels_[1]  ← pending buffer  (message thread writes to this)
 *
 *  TWO separate CUDA streams:
 *
 *    audioStream_   ← owned exclusively by the audio thread
 *    writeStream_   ← owned exclusively by the message thread
 *
 *  ONE cudaEvent_t fence:
 *
 *    writeCompletedEvent_  ← recorded on writeStream_ after the H→D copy.
 *                            The audio thread makes audioStream_ wait on
 *                            this event before reading the pending buffer.
 *
 *  ONE std::atomic<int>  activeIndex_  (0 or 1) — points to the current
 *    active buffer.  Written by the message thread (compare-exchange to
 *    flip 0↔1) and read by the audio thread.  No mutex needed.
 *
 *  ONE std::atomic<bool> swapPending_  — set true by the message thread
 *    after the write is enqueued; cleared by the audio thread after it has
 *    inserted the event wait and committed the index flip.
 *
 *  TIMELINE:
 *
 *   Message thread                         Audio thread
 *   ─────────────                          ─────────────
 *   cudaMemcpyAsync(d_kernels_[1-active],  (busy processing, using
 *     ..., writeStream_)                    d_kernels_[active] on
 *   cudaEventRecord(writeCompletedEvent_,   audioStream_)
 *     writeStream_)
 *   swapPending_.store(true)
 *                                          ─ between blocks ─
 *                                          if (swapPending_.load())
 *                                            cudaStreamWaitEvent(audioStream_,
 *                                              writeCompletedEvent_)
 *                                            activeIndex_ ^= 1
 *                                            swapPending_.store(false)
 *                                          launch kernel with d_kernels_[active]
 *
 *  GUARANTEES:
 *   • The audio thread never touches writeStream_ or the message thread's buffer.
 *   • The message thread never touches audioStream_ or the active buffer.
 *   • cudaStreamWaitEvent() is the only cross-stream synchronisation; it is
 *     inserted on the audio thread's own stream so it cannot cause a priority
 *     inversion (it is a GPU-side wait, not a CPU-side block).
 *   • No mutex, no spinlock, no condition variable in the audio path.
 *   • If two rapid preset changes arrive before the audio thread commits the
 *     first swap, the second write simply overwrites the pending buffer — the
 *     audio thread will pick up the most recent state, which is correct.
 *
 *  fitKernelsCUDA() uses its own temporary stream so it never touches
 *  writeStream_ or audioStream_ during the solve phase.  After the solve it
 *  calls updateKernels() to enqueue the swap via writeStream_.
 */

#pragma once

#ifdef VOLTERRA_CUDA_AVAILABLE

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <atomic>
#include <array>
#include <cstring>

#include <juce_core/juce_core.h>   // DBG(), jassert

#include "VolterraKernels.cuh"

// ─────────────────────────────────────────────────────────────────────────────
//  Helper macro — checks CUDA return codes and returns false on failure.
//  Uses DBG() from JUCE so the message appears in the JUCE debug console.
// ─────────────────────────────────────────────────────────────────────────────

#define VCUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            DBG ("CUDA error " #call ": " << cudaGetErrorString (_e));         \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define VCUDA_CHECK_VOID(call)                                                 \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess)                                                 \
            DBG ("CUDA error " #call ": " << cudaGetErrorString (_e));         \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────

class VolterraCUDAProcessor
{
public:
    // ── Construction / teardown ───────────────────────────────────────────────

    VolterraCUDAProcessor() = default;

    ~VolterraCUDAProcessor() { teardown(); }

    VolterraCUDAProcessor (const VolterraCUDAProcessor&)            = delete;
    VolterraCUDAProcessor& operator= (const VolterraCUDAProcessor&) = delete;

    // ── Initialisation ────────────────────────────────────────────────────────

    /**
     * @brief Allocate all device memory, streams, and the inter-stream fence.
     *
     * Called from prepareToPlay() on the audio thread before processing starts.
     * No other thread touches this object until initialise() returns true.
     *
     * @param maxBlockSize  Max samples per processBlockCUDA() call.
     * @param kernels       Initial interleaved kernel array (3*M floats).
     * @return true on full success; false → caller falls back to CPU path.
     */
    bool initialise (int maxBlockSize, const float* kernels) noexcept
    {
        maxBlockSize_ = maxBlockSize;
        const int paddedLen = maxBlockSize + kCudaMemoryLength - 1;

        // ── Two independent CUDA streams ──────────────────────────────────────
        //
        // audioStream_  : used exclusively by the audio thread for all
        //                 input/output transfers and kernel launches.
        // writeStream_  : used exclusively by the message thread for
        //                 kernel coefficient H→D uploads.
        //
        // cudaStreamNonBlocking prevents either stream from being
        // inadvertently serialised behind the default stream.
        VCUDA_CHECK (cudaStreamCreateWithFlags (&audioStream_, cudaStreamNonBlocking));
        VCUDA_CHECK (cudaStreamCreateWithFlags (&writeStream_, cudaStreamNonBlocking));

        // cuSOLVER and cuBLAS run on a temporary stream created per-call in
        // fitKernelsCUDA() to avoid stream contention.
        VCUDA_CHECK (cusolverDnCreate (&solverHandle_) == CUSOLVER_STATUS_SUCCESS
                     ? cudaSuccess : cudaErrorUnknown);
        VCUDA_CHECK (cublasCreate (&blasHandle_) == CUBLAS_STATUS_SUCCESS
                     ? cudaSuccess : cudaErrorUnknown);

        // ── Inter-stream fence event ──────────────────────────────────────────
        //
        // cudaEventDisableTiming skips timestamp infrastructure, making
        // cudaEventRecord() cheaper.  This event is the only cross-stream
        // synchronisation point in the entire design.
        VCUDA_CHECK (cudaEventCreateWithFlags (&writeCompletedEvent_,
                                               cudaEventDisableTiming));

        // ── Double-buffered kernel arrays on device ───────────────────────────
        //
        // d_kernels_[0] and d_kernels_[1] are allocated independently.
        // The audio thread always reads from d_kernels_[activeIndex_].
        // The message thread always writes to d_kernels_[1 - activeIndex_].
        for (int i = 0; i < 2; ++i)
        {
            VCUDA_CHECK (cudaMalloc (&d_kernels_[i], kTotalCoeffs * sizeof (float)));

            // Initialise both slots with the same starting kernel so there is
            // no stale data in the pending slot if it gets swapped in early.
            VCUDA_CHECK (cudaMemcpyAsync (d_kernels_[i], kernels,
                                          kTotalCoeffs * sizeof (float),
                                          cudaMemcpyHostToDevice, audioStream_));
        }

        // ── Audio I/O buffers ─────────────────────────────────────────────────
        VCUDA_CHECK (cudaMalloc (&d_inputPadded_, paddedLen    * sizeof (float)));
        VCUDA_CHECK (cudaMalloc (&d_output_,      maxBlockSize * sizeof (float)));
        VCUDA_CHECK (cudaMemsetAsync (d_inputPadded_, 0,
                                      paddedLen * sizeof (float), audioStream_));

        // Wait for all initialisations to complete before marking ready.
        VCUDA_CHECK (cudaStreamSynchronize (audioStream_));

        activeIndex_.store  (0,     std::memory_order_relaxed);
        swapPending_.store  (false, std::memory_order_release);
        initialised_ = true;

        DBG ("VolterraCUDAProcessor: initialised. maxBlock=" << maxBlockSize
             << "  paddedLen=" << paddedLen);
        return true;
    }

    // ── Kernel update (MESSAGE THREAD — non-blocking to audio thread) ─────────

    /**
     * @brief Enqueue a new set of kernel coefficients for the next block swap.
     *
     * THREAD: Message thread only.
     *
     * This method:
     *   1. Determines the inactive (pending) buffer index.
     *   2. Uploads coefficients to that buffer on writeStream_.
     *   3. Records writeCompletedEvent_ on writeStream_.
     *   4. Sets swapPending_ = true so the audio thread commits the flip.
     *
     * The audio thread is NEVER blocked.  Its current kernel read proceeds
     * uninterrupted on the active buffer.  The GPU-side event fence ensures
     * the new coefficients are visible to the device before the kernel launch
     * that follows the swap.
     *
     * If called a second time before the audio thread commits the first swap,
     * the pending slot is simply overwritten — the audio thread will always
     * end up with the most recent coefficients, which is correct behaviour.
     *
     * @param kernels  Interleaved kernel array, 3*M floats.  Copied before return.
     */
    bool updateKernels (const float* kernels) noexcept
    {
        if (!initialised_) return false;

        // Determine which slot is currently NOT active.
        // Use relaxed order here; we only need acquire semantics when the audio
        // thread reads swapPending_ and then reads activeIndex_.
        const int pendingSlot = 1 - activeIndex_.load (std::memory_order_relaxed);

        // Upload to the inactive slot on the dedicated write stream.
        // This call is non-blocking to the host; it returns immediately after
        // enqueueing the DMA into writeStream_.
        if (cudaMemcpyAsync (d_kernels_[pendingSlot], kernels,
                             kTotalCoeffs * sizeof (float),
                             cudaMemcpyHostToDevice, writeStream_) != cudaSuccess)
        {
            DBG ("VolterraCUDAProcessor: updateKernels H→D failed.");
            return false;
        }

        // Record the completion event on writeStream_.
        // When the audio thread waits on this event from audioStream_, the
        // GPU scheduler will stall audioStream_ only until the DMA is done —
        // not until any CPU code runs.  This is a GPU-side wait; zero CPU spin.
        if (cudaEventRecord (writeCompletedEvent_, writeStream_) != cudaSuccess)
        {
            DBG ("VolterraCUDAProcessor: cudaEventRecord failed.");
            return false;
        }

        // Signal the audio thread.  release order pairs with the acquire load
        // in processBlockCUDA() so the activeIndex_ read sees a consistent value.
        swapPending_.store (true, std::memory_order_release);
        return true;
    }

    // ── Audio processing (AUDIO THREAD — wait-free on host) ──────────────────

    /**
     * @brief Process a block of audio samples on the GPU.
     *
     * THREAD: Audio thread only.
     *
     * At the start of each block, checks whether a kernel swap is pending.
     * If so, inserts a GPU-side event wait (zero CPU cost) and commits the
     * active-index flip before launching the kernel — ensuring the kernel
     * always reads from the fully-written, most-recent coefficients.
     *
     * The audio thread never touches writeStream_ or d_kernels_[pendingSlot].
     * The message thread never touches audioStream_ or d_kernels_[activeSlot].
     *
     * @param input      Host input buffer, numSamples floats.
     * @param output     Host output buffer, numSamples floats.
     * @param numSamples Must be ≤ maxBlockSize passed to initialise().
     * @return true on success; false → caller should fall back to CPU.
     */
    bool processBlockCUDA (const float* input,
                                 float* output,
                           int    numSamples) noexcept
    {
        if (!initialised_) return false;
        jassert (numSamples <= maxBlockSize_);

        // ── 1. Commit pending kernel swap (if any) ────────────────────────────
        //
        // acquire load pairs with the release store in updateKernels() to
        // guarantee we see the updated activeIndex_ after the swap.
        if (swapPending_.load (std::memory_order_acquire))
        {
            // Insert a GPU-side wait on audioStream_ before the pending kernel
            // slot is used.  The GPU will stall audioStream_ until writeStream_
            // has finished its DMA transfer (signalled by writeCompletedEvent_).
            // This is entirely handled by the GPU scheduler — the CPU does not
            // block or spin here.
            if (cudaStreamWaitEvent (audioStream_, writeCompletedEvent_, 0)
                    != cudaSuccess)
            {
                DBG ("VolterraCUDAProcessor: cudaStreamWaitEvent failed — "
                     "skipping kernel swap this block.");
            }
            else
            {
                // Atomically flip the active index.
                // XOR with 1 flips 0↔1.  We use relaxed order because
                // the happens-before relationship is enforced by the GPU event,
                // not by the CPU atomic.
                activeIndex_.fetch_xor (1, std::memory_order_relaxed);
            }

            // Clear the flag so we don't re-apply next block.
            swapPending_.store (false, std::memory_order_release);
        }

        // Snapshot the active index once for this entire block.
        const int activeSlot = activeIndex_.load (std::memory_order_relaxed);

        // ── 2. Slide history window (D→D copy on audioStream_) ────────────────
        //
        // BUFFER LAYOUT INVARIANT (maintained across every block):
        //
        //   d_inputPadded_[0 .. M-2]             ← history tail: x[n-M+1]..x[n-1]
        //   d_inputPadded_[M-1 .. M-1+N-1]       ← current block: x[n]..x[n+N-1]
        //
        //   Total floats used = (M-1) + N  ≤  (M-1) + maxBlockSize  = allocation
        //
        // where M = kCudaMemoryLength, N = numSamples (THIS block, not maxBlockSize).
        //
        // After processing block of size N, the M-1 samples that form the
        // history for the NEXT block are the last (M-1) samples of the
        // current padded buffer, starting at index N (= the numSamples offset).
        //
        //   Correct src = d_inputPadded_ + numSamples
        //   Wrong src   = d_inputPadded_ + maxBlockSize_   ← BUG (fixed here)
        //
        // Using maxBlockSize_ instead of numSamples reads into the stale region
        // [numSamples .. maxBlockSize-1] whenever numSamples < maxBlockSize_.
        // DAWs frequently pass sub-maximal block sizes (transport loop wrap,
        // tempo-map grid boundaries, final fragment of a render), so this
        // failure mode is hit in normal production use, not just edge cases.
        //
        // The kernel reads d_inputPadded_[n + M - 1 - m] for thread n, delay m.
        // For the last thread (n = N-1) and deepest tap (m = M-1):
        //   index = N - 1 + M - 1 - (M-1) = N - 1  ← within [0, M-1+N-1]. ✓
        // For n = N-1, m = 0:
        //   index = N - 1 + M - 1 = M + N - 2       ← last written index.  ✓
        // Read range [0, M+N-2] exactly matches written range. No OOB possible.

        const int historyBytes = (kCudaMemoryLength - 1) * sizeof (float);

        VCUDA_CHECK (cudaMemcpyAsync (d_inputPadded_,
                                      d_inputPadded_ + numSamples,   // ← numSamples, not maxBlockSize_
                                      historyBytes,
                                      cudaMemcpyDeviceToDevice, audioStream_));

        // ── 3. Upload new input block (H→D on audioStream_) ───────────────────
        //
        // Always written at the fixed offset (M-1) regardless of block size,
        // because the history tail above always occupies exactly [0, M-2].
        VCUDA_CHECK (cudaMemcpyAsync (d_inputPadded_ + (kCudaMemoryLength - 1),
                                      input,
                                      numSamples * sizeof (float),
                                      cudaMemcpyHostToDevice, audioStream_));

        // ── 4. Launch kernel using the active (fully-written) kernel slot ─────
        const int threads = 256;
        const int blocks  = (numSamples + threads - 1) / threads;

        volterraProcessKernel<<<blocks, threads, 0, audioStream_>>> (
            d_inputPadded_,
            d_output_,
            d_kernels_[activeSlot],   // ← always the completed, active buffer
            numSamples);

        VCUDA_CHECK (cudaGetLastError());

        // ── 5. Download results (D→H on audioStream_) ─────────────────────────
        VCUDA_CHECK (cudaMemcpyAsync (output, d_output_,
                                      numSamples * sizeof (float),
                                      cudaMemcpyDeviceToHost, audioStream_));

        // ── 6. Synchronise audioStream_ ───────────────────────────────────────
        //
        // Necessary so the host DAW can read `output` safely after we return.
        // writeStream_ is NOT touched here — no priority inversion possible.
        VCUDA_CHECK (cudaStreamSynchronize (audioStream_));

        return true;
    }

    // ── Kernel fitting (MESSAGE THREAD — uses its own temporary stream) ───────

    /**
     * @brief Fit kernel coefficients to a reference capture via cuSOLVER SGELS.
     *
     * THREAD: Message thread only. Never called from the audio thread.
     *
     * Uses a dedicated temporary CUDA stream for the solve so that neither
     * audioStream_ nor writeStream_ is touched during the (potentially long)
     * least-squares computation.  On success, calls updateKernels() to enqueue
     * the fitted coefficients via writeStream_ for the next block swap.
     *
     * @param h_input      Padded input: (numSamples + M - 1) floats.
     * @param h_target     Target output: numSamples floats.
     * @param numSamples   Training sample count N.
     * @param h_kernelsOut Receives fitted interleaved coefficients (3*M floats).
     * @return true on success and kernel update enqueued.
     */
    bool fitKernelsCUDA (const float* h_input,
                         const float* h_target,
                         int          numSamples,
                         float*       h_kernelsOut) noexcept
    {
        if (!initialised_) return false;

        const int N  = numSamples;
        const int M3 = kTotalCoeffs;

        // Create a private stream for the entire solve — isolates the long
        // computation from both audioStream_ and writeStream_.
        cudaStream_t solveStream = nullptr;
        if (cudaStreamCreateWithFlags (&solveStream, cudaStreamNonBlocking)
                != cudaSuccess)
            return false;

        cusolverDnSetStream (solverHandle_, solveStream);
        cublasSetStream     (blasHandle_,   solveStream);

        // ── Allocate temporary device buffers ─────────────────────────────────
        //
        // d_Phi    : row-major [N × M3]   — written by buildFeatureMatrixKernel
        // d_PhiT   : column-major [N × M3] — produced by cublasSgeam transpose
        //            (same shape, different memory order; needed by cuSOLVER)
        // d_y      : [N × 1] target vector (overwritten with solution [M3 × 1])
        // d_inputFit, d_work, d_info : scratch
        float *d_Phi = nullptr, *d_PhiT = nullptr;
        float *d_y   = nullptr, *d_inputFit = nullptr;
        float *d_work = nullptr;
        int   *d_info = nullptr;

        const int paddedLen = N + kCudaMemoryLength - 1;

        bool ok = true;
        ok = ok && (cudaMalloc (&d_Phi,      (size_t)N * M3 * sizeof (float)) == cudaSuccess);
        ok = ok && (cudaMalloc (&d_PhiT,     (size_t)N * M3 * sizeof (float)) == cudaSuccess);
        ok = ok && (cudaMalloc (&d_y,        (size_t)N      * sizeof (float)) == cudaSuccess);
        ok = ok && (cudaMalloc (&d_inputFit, paddedLen      * sizeof (float)) == cudaSuccess);

        if (!ok)
        {
            DBG ("VolterraCUDAProcessor: fitKernelsCUDA allocation failed.");
            goto cleanup;
        }

        // ── Upload training data ───────────────────────────────────────────────
        cudaMemcpyAsync (d_inputFit, h_input,  paddedLen * sizeof (float),
                         cudaMemcpyHostToDevice, solveStream);
        cudaMemcpyAsync (d_y,        h_target, N         * sizeof (float),
                         cudaMemcpyHostToDevice, solveStream);

        // ── Build Φ row-major [N × M3] on device ─────────────────────────────
        //
        // Each thread n writes one row of M3 consecutive floats (coalesced).
        // Layout: row n = [b1[n,0]..b1[n,M-1], b2[n,0]..b2[n,M-1], b3[n,0]..b3[n,M-1]]
        {
            const int threads = 128;
            const int blocks  = (N + threads - 1) / threads;
            buildFeatureMatrixKernel<<<blocks, threads, 0, solveStream>>> (
                d_inputFit, d_Phi, N);

            if (cudaGetLastError() != cudaSuccess)
            {
                ok = false;
                goto cleanup;
            }
        }

        // ── Transpose Φ to column-major using cublasSgeam ─────────────────────
        //
        // cuSOLVER SGELS (and all cuBLAS/cuSOLVER routines) interpret every
        // matrix as column-major (Fortran order): element (r,c) lives at
        // offset r + c * lda.  Our d_Phi is row-major: element (r,c) lives at
        // offset r * M3 + c.  Passing d_Phi directly to SGELS would make it
        // solve Φᵀ·h = y instead of Φ·h = y — returning garbage coefficients.
        //
        // Fix: cublasSgeam performs C = α·op(A) + β·op(B) on column-major data.
        //
        // We tell cuBLAS to treat d_Phi as a column-major [M3 × N] matrix
        // (which is exactly how the same bytes look from cuBLAS's perspective
        // when the real layout is row-major [N × M3]).  Transposing [M3 × N]
        // gives [N × M3] in column-major — which is what SGELS needs.
        //
        //   A (input):  d_Phi   treated as col-major [M3 × N],  lda = M3
        //   C (output): d_PhiT  col-major [N × M3],             ldc = N
        //   op = CUBLAS_OP_T  → C = Aᵀ
        //   m = N (rows of C),  n = M3 (cols of C)
        //   α = 1.0f, β = 0.0f
        {
            const float alpha = 1.0f, beta = 0.0f;
            cublasStatus_t st = cublasSgeam (
                blasHandle_,
                CUBLAS_OP_T,    // transpose A
                CUBLAS_OP_N,    // B unused (beta=0)
                N,              // rows of output C
                M3,             // cols of output C
                &alpha,
                d_Phi,          // A: col-major [M3 × N] (same bytes as row-major [N × M3])
                M3,             // lda = M3 (leading dim of A viewed as [M3 × N])
                &beta,
                d_PhiT,         // B: ignored (beta=0), but must be non-null and same size
                N,              // ldb = N
                d_PhiT,         // C: col-major [N × M3], ldc = N
                N);             // ldc = N

            if (st != CUBLAS_STATUS_SUCCESS)
            {
                DBG ("VolterraCUDAProcessor: cublasSgeam transpose failed: " << (int)st);
                ok = false;
                goto cleanup;
            }
        }

        // ── cuSOLVER SGELS: min ‖Φ·h – y‖² ──────────────────────────────────
        //
        // d_PhiT is now [N × M3] column-major with lda = N.
        // d_y    is [N × 1] column-major (trivially), ldb = N.
        // SGELS overwrites d_y with the solution vector h of length M3.
        //
        // After the solve, d_y[0..M3-1] contains h in grouped order:
        //   d_y[0..M-1]    = h1[0..M-1]   (linear kernel, all taps)
        //   d_y[M..2M-1]   = h2[0..M-1]   (quadratic kernel, all taps)
        //   d_y[2M..3M-1]  = h3[0..M-1]   (cubic kernel, all taps)
        //
        // This grouped order matches the column grouping in d_PhiT: column j
        // of d_PhiT holds the basis values for coefficient j across all N samples.
        {
            int lwork = 0;
            cusolverDnSgels_bufferSize (solverHandle_,
                                        N,   M3, 1,      // rows, cols, nrhs
                                        d_PhiT, N,       // A, lda
                                        d_y,    N,       // B, ldb
                                        d_y,    N,       // X (solution overwrites B), ldx
                                        nullptr, &lwork);

            if (cudaMalloc (&d_work, lwork * sizeof (float)) != cudaSuccess ||
                cudaMalloc (&d_info, sizeof (int))           != cudaSuccess)
            {
                ok = false;
                goto cleanup;
            }

            int iters = 0;
            cusolverStatus_t status =
                cusolverDnSgels (solverHandle_,
                                 N,   M3, 1,
                                 d_PhiT, N,
                                 d_y,    N,
                                 d_y,    N,
                                 d_work, lwork,
                                 &iters, d_info);

            ok = (status == CUSOLVER_STATUS_SUCCESS);
            if (ok)
            {
                // Download the grouped solution vector [h1[0..M-1], h2[0..M-1], h3[0..M-1]].
                // Temporary host buffer — we need to re-interleave before handing
                // to the CPU filter and CUDA double-buffer, both of which expect
                // [h1[0], h2[0], h3[0], h1[1], h2[1], h3[1], ...].
                std::array<float, kTotalCoeffs> grouped{};
                cudaMemcpyAsync (grouped.data(), d_y,
                                 M3 * sizeof (float),
                                 cudaMemcpyDeviceToHost, solveStream);
                cudaStreamSynchronize (solveStream);

                // Re-interleave: grouped [h1..., h2..., h3...] → [h1[m], h2[m], h3[m], ...]
                // The CPU filter's kernels_ array and d_kernels_ on device both
                // use this interleaved layout for cache-local tap access.
                for (int m = 0; m < kCudaMemoryLength; ++m)
                {
                    h_kernelsOut[3 * m + 0] = grouped[0 * kCudaMemoryLength + m]; // h1[m]
                    h_kernelsOut[3 * m + 1] = grouped[1 * kCudaMemoryLength + m]; // h2[m]
                    h_kernelsOut[3 * m + 2] = grouped[2 * kCudaMemoryLength + m]; // h3[m]
                }

                DBG ("VolterraCUDAProcessor: fit complete. iters=" << iters);

                // Enqueue fitted kernels via the double-buffer swap path.
                // writeStream_ is the only stream touched here.
                updateKernels (h_kernelsOut);
            }
            else
            {
                DBG ("VolterraCUDAProcessor: cuSOLVER SGELS failed, status=" << (int)status);
            }
        }

    cleanup:
        cudaFree (d_Phi);
        cudaFree (d_PhiT);    // transposed column-major copy
        cudaFree (d_y);
        cudaFree (d_inputFit);
        cudaFree (d_work);
        cudaFree (d_info);

        cudaStreamDestroy (solveStream);

        // Restore handles to writeStream_ so they're ready for the next update.
        cusolverDnSetStream (solverHandle_, writeStream_);
        cublasSetStream     (blasHandle_,   writeStream_);

        return ok;
    }

    // ── Teardown ──────────────────────────────────────────────────────────────

    void teardown() noexcept
    {
        // [CHANGE] Removed early exit if (!initialised_) to allow cleaning up 
        // partially allocated resources if initialise() fails midway.
        
        // Drain both streams before freeing any memory, if they were created.
        // [CHANGE] Added null-checks to prevent synchronizing the global default stream.
        if (audioStream_) VCUDA_CHECK_VOID (cudaStreamSynchronize (audioStream_));
        if (writeStream_) VCUDA_CHECK_VOID (cudaStreamSynchronize (writeStream_));

        // [CHANGE] Safely null-check pointers before freeing device memory.
        if (d_inputPadded_) cudaFree (d_inputPadded_);
        if (d_output_)      cudaFree (d_output_);
        for (int i = 0; i < 2; ++i) {
            if (d_kernels_[i]) cudaFree (d_kernels_[i]);
        }

        // [CHANGE] Safely null-check event and streams before destroying them.
        if (writeCompletedEvent_) cudaEventDestroy   (writeCompletedEvent_);
        if (audioStream_)         cudaStreamDestroy  (audioStream_);
        if (writeStream_)         cudaStreamDestroy  (writeStream_);
        if (solverHandle_) cusolverDnDestroy (solverHandle_);
        if (blasHandle_)   cublasDestroy     (blasHandle_);

        audioStream_         = nullptr;
        writeStream_         = nullptr;
        writeCompletedEvent_ = nullptr;
        solverHandle_        = nullptr;
        blasHandle_          = nullptr;
        d_inputPadded_       = nullptr;
        d_output_            = nullptr;
        d_kernels_[0]        = nullptr;
        d_kernels_[1]        = nullptr;

        initialised_ = false;
        DBG ("VolterraCUDAProcessor: torn down.");
    }

    [[nodiscard]] bool isReady()     const noexcept { return initialised_; }
    [[nodiscard]] bool swapPending() const noexcept
    {
        return swapPending_.load (std::memory_order_relaxed);
    }

private:
    // ── Double-buffered kernel device arrays ──────────────────────────────────
    float* d_kernels_[2] = { nullptr, nullptr };

    // ── Audio I/O device buffers ──────────────────────────────────────────────
    float* d_inputPadded_ = nullptr;
    float* d_output_      = nullptr;

    // ── CUDA streams ──────────────────────────────────────────────────────────
    //
    // audioStream_  : audio thread exclusive — input DMA, kernel launch, output DMA
    // writeStream_  : message thread exclusive — kernel coefficient DMA
    cudaStream_t audioStream_ = nullptr;
    cudaStream_t writeStream_ = nullptr;

    // ── Inter-stream synchronisation ──────────────────────────────────────────
    //
    // Recorded on writeStream_ after each H→D coefficient copy.
    // Waited on audioStream_ before each kernel launch that follows a swap.
    // GPU-side wait only — zero CPU cost on the audio thread.
    cudaEvent_t writeCompletedEvent_ = nullptr;

    // ── cuSOLVER / cuBLAS handles (message thread only) ───────────────────────
    cusolverDnHandle_t solverHandle_ = nullptr;
    cublasHandle_t     blasHandle_   = nullptr;

    // ── Lock-free state ───────────────────────────────────────────────────────
    //
    // activeIndex_  : which slot in d_kernels_[] the audio thread reads.
    //                 0 or 1.  Written (XOR flip) by audio thread only.
    //                 Read by message thread only to find the pending slot.
    //
    // swapPending_  : true when the message thread has enqueued a new write
    //                 and is waiting for the audio thread to commit the flip.
    //                 Written by message thread (true), cleared by audio thread.
    std::atomic<int>  activeIndex_ { 0 };
    std::atomic<bool> swapPending_ { false };

    int  maxBlockSize_ = 0;
    bool initialised_  = false;
};

#endif // VOLTERRA_CUDA_AVAILABLE
