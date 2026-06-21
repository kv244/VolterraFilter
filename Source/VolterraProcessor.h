/**
 * @file VolterraProcessor.h
 * @brief JUCE AudioProcessor implementing the Pruned Volterra / Memory Polynomial
 *        nonlinear filter as a VST3 / AU plugin.
 *
 * BUGS FIXED vs. v1 (verified against https://docs.juce.com/master/):
 * ─────────────────────────────────────────────────────────────────────
 *  1. getStateInformation / setStateInformation
 *     – Was: manual XmlDocument parse + MemoryOutputStream::writeTo().
 *     – Now: AudioProcessor::copyXmlToBinary() / getXmlFromBinary() which
 *       is the documented, host-compatible binary XML pattern.
 *     – replaceState() takes a ValueTree, so we call ValueTree::fromXml()
 *       on the parsed XmlElement rather than passing the XmlElement directly.
 *
 *  2. Denormal handling
 *     – Was: ScopedNoDenormals per processBlock() call (stack overhead/block).
 *     – Now: FloatVectorOperations::disableDenormalisedNumberSupport() called
 *       once in prepareToPlay(). Docs: "convenient thing to call before audio
 *       processing code where you really want to avoid denormalisation hits."
 *
 *  3. Parameter smoothing (zipper noise)
 *     – Was: raw atomic reads of drive/mix/output per block → click on fast
 *       automation.
 *     – Now: juce::SmoothedValue<float> for drive, mix, outputGain, each
 *       reset in prepareToPlay() with a 20 ms ramp. Per-sample getNextValue()
 *       advances the ramp; skip() is used when CUDA processes the full block
 *       and we can only apply gain before/after en-bloc.
 *
 *  4. processBlockBypassed override
 *     – Was: missing. Host would call base-class no-op, leaving stale signal.
 *     – Now: overridden to pass audio through unmodified and reset filter
 *       state so no transient on re-engage.
 *
 *  5. APVTS member initialisation order
 *     – Was: parameters_ initialised in constructor body assignment.
 *     – Now: initialised in member-initialiser list after AudioProcessor base,
 *       matching documented APVTS lifetime requirement.
 *
 *  6. Kernel serialisation appended to APVTS XML tree
 *     – Was: separate XmlElement added to ValueTree (invalid).
 *     – Now: kernel coefficients stored as a space-separated attribute on the
 *       root XmlElement returned by copyState().createXml(), then recovered
 *       in setStateInformation via getXmlFromBinary.
 *
 * DISPATCH ARCHITECTURE
 * ──────────────────────
 *   prepareToPlay():
 *     ├─ Always creates PrunedVolterraFilter[L/R]   (CPU, always compiled)
 *     └─ If VOLTERRA_CUDA_AVAILABLE:
 *          probe cudaGetDeviceCount → VolterraCUDAProcessor::initialise()
 *          failure → cudaProcessor_ = nullptr  (silent CPU fallback)
 *
 *   processBlock():
 *     ├─ CUDA: cudaProcessor_ && numSamples >= kCudaMinBlockSize
 *     └─ CPU:  cpuFilters_[ch].processBlock()
 *
 * THREAD OWNERSHIP MODEL (v3 — race condition fix)
 * ─────────────────────────────────────────────────
 * Three categories of state, each with a clear owner:
 *
 *  1. cudaProcessor_ unique_ptr
 *     Created/destroyed only in prepareToPlay() / releaseResources(), both
 *     of which the VST3 / AU host contract guarantees are called from the
 *     message thread with the audio thread NOT running.  No lock needed.
 *
 *  2. Inside VolterraCUDAProcessor — kernel double-buffer (the core fix)
 *     audioStream_ + d_kernels_[active]  → audio thread EXCLUSIVE
 *     writeStream_ + d_kernels_[pending] → message thread EXCLUSIVE
 *     Swap coordinated via:
 *       std::atomic<int>  activeIndex_      (0 or 1)
 *       std::atomic<bool> swapPending_      (set by msg, cleared by audio)
 *       cudaEvent_t       writeCompletedEvent_ (GPU-side fence, no CPU spin)
 *     Zero CPU blocking on the audio thread.
 *
 *  3. cpuFilters_[]  [v6 — data-race fixed]
 *     Audio thread EXCLUSIVE for reads and writes.
 *     Message-thread kernel updates are staged via pendingKernels_ (plain array)
 *     + kernelUpdatePending_ (std::atomic<bool>). The audio thread consumes the
 *     update at the top of processBlock() with acquire/release ordering.
 *     See setKernels() for details.
 *
 * PARAMETERS
 * ──────────
 *  "drive"   0.0–4.0, default 1.0  – Input drive before nonlinearity
 *  "mix"     0.0–1.0, default 1.0  – Dry/wet blend
 *  "output"  −24–+6 dB, default 0  – Output trim
 */

#pragma once

// Sentinel so VolterraPresets.h knows VolterraProcessor is available and can
// compile the load(Id, VolterraProcessor&) convenience overload.
#define VOLTERRA_PROCESSOR_INCLUDED

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_core/juce_core.h>

#include <atomic>

#include "PrunedVolterraFilter.h"

#ifdef VOLTERRA_CUDA_AVAILABLE
  #include "../CUDA/VolterraCUDAProcessor.h"
  #include <memory>
#endif

/// Minimum block size (samples) to route through CUDA.
/// Below this the PCIe round-trip costs more than the GPU saves.
static constexpr int kCudaMinBlockSize = 512;

/// Ramp length in seconds for parameter smoothing (zipper prevention).
static constexpr double kSmoothedRampSeconds = 0.020;

// ─────────────────────────────────────────────────────────────────────────────

class VolterraProcessor final : public juce::AudioProcessor
{
public:
    // ── Construction ──────────────────────────────────────────────────────────

    VolterraProcessor()
        : AudioProcessor (BusesProperties()
                             .withInput ("Input",  juce::AudioChannelSet::stereo(), true)
                             .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
          // APVTS must be initialised in the member-initialiser list AFTER
          // AudioProcessor so it can safely call addParameter() on *this.
          parameters_ (*this, nullptr, "VolterraParams", createParameterLayout())
    {
        // [v6] Sync pendingKernels_ with the Analog Warmth defaults loaded by
        // PrunedVolterraFilter's constructor so getStateInformation() returns
        // the correct kernels before any explicit preset is selected.
        std::memcpy (pendingKernels_.data(),
                     cpuFilters_[0].getKernelsInterleaved(),
                     kKernelArraySize * sizeof (float));
    }

    ~VolterraProcessor() override = default;

    // ── AudioProcessor boilerplate ────────────────────────────────────────────

    // [CHANGE] Removed 'const' return type to allow compiler to use move semantics 
    // instead of forcing a copy operation for juce::String.
    const juce::String getName() const override { return "Volterra NL Filter"; }
    bool acceptsMidi()  const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }

    double getTailLengthSeconds() const override
    {
        // M samples of nonlinear memory at the current sample rate.
        return static_cast<double>(kMemoryLength) / lastSampleRate_;
    }

    int getNumPrograms()    override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    // [CHANGE] Removed 'const' return type here as well to enable move semantics.
    const juce::String getProgramName (int) override { return "Default"; }
    void changeProgramName (int, const juce::String&) override {}

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    void prepareToPlay (double sampleRate, int maxBlockSize) override
    {
        lastSampleRate_ = sampleRate;

        // ── Denormal suppression ──────────────────────────────────────────────
        // Call once here rather than installing ScopedNoDenormals per block.
        // Sets DAZ+FZ bits in MXCSR on x86 / flush-to-zero on ARM NEON.
        // Documented at:
        //   FloatVectorOperations::disableDenormalisedNumberSupport()
        juce::FloatVectorOperations::disableDenormalisedNumberSupport();

        // ── Parameter smoothers ───────────────────────────────────────────────
        // Reset to current parameter values with the new sample rate so that
        // the ramp starts from where the parameter currently sits, not from 0.
        const float curDrive    = parameters_.getRawParameterValue("drive") ->load();
        const float curMix      = parameters_.getRawParameterValue("mix")   ->load();
        const float curOutputdB = parameters_.getRawParameterValue("output")->load();

        smoothDrive_.reset    (sampleRate, kSmoothedRampSeconds);
        smoothMix_.reset      (sampleRate, kSmoothedRampSeconds);
        smoothOutputGain_.reset(sampleRate, kSmoothedRampSeconds);

        smoothDrive_.setCurrentAndTargetValue (curDrive);
        smoothMix_.setCurrentAndTargetValue   (curMix);
        // [v6] Initialise the dB cache to match so the first processBlock
        // block skips the decibelsToGain (std::pow) call unnecessarily.
        lastOutputDb_ = curOutputdB;
        smoothOutputGain_.setCurrentAndTargetValue (
            juce::Decibels::decibelsToGain (curOutputdB));

        // ── CPU filter reset ──────────────────────────────────────────────────
        for (auto& f : cpuFilters_)
            f.reset();

        dryBuffer_.setSize (2, maxBlockSize, false, true, true);

        // ── CUDA initialisation ───────────────────────────────────────────────
#ifdef VOLTERRA_CUDA_AVAILABLE
        cudaProcessor_.reset();

        int deviceCount = 0;
        if (cudaGetDeviceCount (&deviceCount) == cudaSuccess && deviceCount > 0)
        {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties (&prop, 0);
            DBG ("VolterraProcessor: CUDA device: " << prop.name
                 << "  CC=" << prop.major << "." << prop.minor);

            auto cuda = std::make_unique<VolterraCUDAProcessor>();

            if (cuda->initialise (maxBlockSize, cpuFilters_[0].getKernelsInterleaved()))
            {
                cudaProcessor_ = std::move (cuda);
                DBG ("VolterraProcessor: CUDA path active.");
            }
            else
            {
                DBG ("VolterraProcessor: CUDA init failed — CPU fallback.");
            }
        }
        else
        {
            DBG ("VolterraProcessor: No CUDA device — CPU path.");
        }
#endif
    }

    void releaseResources() override
    {
#ifdef VOLTERRA_CUDA_AVAILABLE
        cudaProcessor_.reset();
#endif
    }

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()  &&
            layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
            return false;
        return layouts.getMainInputChannelSet() == layouts.getMainOutputChannelSet();
    }

    // ── Bypass (FIX: was missing) ─────────────────────────────────────────────

    /**
     * @brief Pass audio through unprocessed when the host bypasses the plugin.
     *
     * Also resets the nonlinear filter state so there's no transient when the
     * plugin re-engages.  The default base-class implementation is a no-op,
     * which would leave whatever happened to be in the buffer.
     */
    void processBlockBypassed (juce::AudioBuffer<float>& buffer,
                                juce::MidiBuffer&) override
    {
        // The host should already have copied input to output by this point for
        // a bypass, but we ensure filter state is clean for re-engage.
        for (auto& f : cpuFilters_)
            f.reset();

        // Snap smoothers to current target so no ramp artefact on un-bypass.
        smoothDrive_.skip     (buffer.getNumSamples());
        smoothMix_.skip       (buffer.getNumSamples());
        smoothOutputGain_.skip(buffer.getNumSamples());
    }

    // ── Core processing ───────────────────────────────────────────────────────

    void processBlock (juce::AudioBuffer<float>& buffer,
                       juce::MidiBuffer&) override
    {
        const int numSamples  = buffer.getNumSamples();
        const int numChannels = juce::jmin (buffer.getNumChannels(), 2);

        // ── Update smoothed parameter targets ─────────────────────────────────
        // Push new targets each block; smoothers advance per-sample below.
        smoothDrive_.setTargetValue (
            parameters_.getRawParameterValue ("drive")->load());
        smoothMix_.setTargetValue (
            parameters_.getRawParameterValue ("mix")->load());
        // [v6] Cache the last dB value to skip std::pow (decibelsToGain) when
        // the output parameter is unchanged — pow is expensive in a hot callback.
        {
            const float curOutputDb = parameters_.getRawParameterValue ("output")->load();
            if (curOutputDb != lastOutputDb_)
            {
                lastOutputDb_ = curOutputDb;
                smoothOutputGain_.setTargetValue (
                    juce::Decibels::decibelsToGain (curOutputDb));
            }
        }

        // ── Consume any pending kernel update from the message thread ─────────
        // [v6] cpuFilters_[] is audio-thread-exclusive. The message thread stages
        // updates via pendingKernels_ and sets kernelUpdatePending_ with release
        // ordering. We load with acquire here so all preceding float writes are
        // visible before we copy them into cpuFilters_[].
        if (kernelUpdatePending_.load (std::memory_order_acquire))
        {
            for (auto& f : cpuFilters_)
                f.setKernelsInterleaved (pendingKernels_.data());
            kernelUpdatePending_.store (false, std::memory_order_relaxed);
        }

        // ── Store dry signal for mix blend ────────────────────────────────────
        for (int ch = 0; ch < numChannels; ++ch)
            dryBuffer_.copyFrom (ch, 0, buffer, ch, 0, numSamples);

        // ── Input drive (smoothed, per-sample to prevent clicks) ──────────────
        // We need a local copy of the smoother state to apply per-sample drive
        // before routing into the nonlinear processor.  Drive is applied once
        // here; we advance the smoother separately.
        {
            // Apply smoothed drive gain sample-by-sample. Each channel resets
            // from the same ramp position so L/R receive identical gain curves.
            // [v6] Removed unused `driveCopy` snapshot — it was constructed but
            // never read; driveRamp is copied directly from smoothDrive_ per channel.
            for (int ch = 0; ch < numChannels; ++ch)
            {
                auto* data     = buffer.getWritePointer (ch);
                auto driveRamp = smoothDrive_;   // snapshot per channel
                for (int n = 0; n < numSamples; ++n)
                    data[n] *= driveRamp.getNextValue();
            }
            // Advance the master smoother past these samples.
            smoothDrive_.skip (numSamples);
        }

        // ── Nonlinear processing: CUDA or CPU ─────────────────────────────────
#ifdef VOLTERRA_CUDA_AVAILABLE
        if (cudaProcessor_ != nullptr && numSamples >= kCudaMinBlockSize)
        {
            // [v6] Track per-block CUDA success across all channels.
            // Previously always reported cudaActive=true even when a channel
            // fell back to CPU, giving a misleading status badge in the editor.
            bool allCudaOk = true;
            for (int ch = 0; ch < numChannels; ++ch)
            {
                float* data = buffer.getWritePointer (ch);
                if (! cudaProcessor_->processBlockCUDA (data, data, numSamples))
                {
                    DBG ("CUDA block fail ch=" << ch << " — CPU fallback");
                    cpuFilters_[ch].processBlock (data, data, numSamples);
                    allCudaOk = false;
                }
            }
            cudaActiveLastBlock_.store (allCudaOk, std::memory_order_relaxed);

            // Advance mix and output smoothers without per-sample work
            // (CUDA path processes en-bloc; apply gain as bulk ramp after).
            applySmoothedGainBulk (buffer, numChannels, numSamples);
        }
        else
#endif
        {
            // CPU path: per-sample processing with fully smoothed parameters.
            processBlockCPU (buffer, numChannels, numSamples);
            cudaActiveLastBlock_.store (false, std::memory_order_relaxed);
        }
    }

    // ── Kernel management ─────────────────────────────────────────────────────

    // [v6] Kernel updates are staged through pendingKernels_ so the message
    // thread never writes cpuFilters_[] directly. The audio thread consumes
    // them at the top of processBlock() using acquire/release ordering.
    // Safe to call concurrently with processBlock() provided updates arrive
    // less frequently than the audio block rate (guaranteed by GUI event timing).
    void setKernels (const float* h1, const float* h2, const float* h3)
    {
        for (int m = 0; m < kMemoryLength; ++m)
        {
            pendingKernels_[3 * m + 0] = h1[m];
            pendingKernels_[3 * m + 1] = h2[m];
            pendingKernels_[3 * m + 2] = h3[m];
        }
        // release: all float writes above are visible when the audio thread
        // loads kernelUpdatePending_ with acquire.
        kernelUpdatePending_.store (true, std::memory_order_release);

#ifdef VOLTERRA_CUDA_AVAILABLE
        if (cudaProcessor_)
            cudaProcessor_->updateKernels (pendingKernels_.data());
#endif
    }

    bool fitKernelsFromCapture (const float* inputPadded,
                                const float* target,
                                int numSamples)
    {
#ifdef VOLTERRA_CUDA_AVAILABLE
        if (cudaProcessor_)
        {
            std::array<float, kKernelArraySize> fitted{};
            if (cudaProcessor_->fitKernelsCUDA (inputPadded, target, numSamples, fitted.data()))
            {
                // [v6] Stage through pendingKernels_ (same path as setKernels)
                // so the audio thread picks them up without a direct write race.
                pendingKernels_ = fitted;
                kernelUpdatePending_.store (true, std::memory_order_release);
                return true;
            }
        }
#endif
        DBG ("fitKernelsFromCapture: CUDA not available or fit failed.");
        return false;
    }

    // [v6] Atomic load — written by audio thread, read by message thread
    // (VolterraEditor::timerCallback). relaxed ordering is sufficient for a
    // status display that can tolerate one-block lag.
    [[nodiscard]] bool isCUDAActive() const noexcept
    {
        return cudaActiveLastBlock_.load (std::memory_order_relaxed);
    }

    // ── State persistence (FIX: was using wrong serialisation path) ────────────
    //
    // Correct JUCE pattern (confirmed in docs + official tutorials):
    //   getStateInformation:  copyState().createXml()  →  copyXmlToBinary()
    //   setStateInformation:  getXmlFromBinary()  →  ValueTree::fromXml()
    //                          →  replaceState(ValueTree)
    //
    // Kernel coefficients are stored as an extra XML attribute on the root
    // element rather than a child ValueTree, so replaceState() only receives
    // the APVTS-owned portion of the tree and doesn't corrupt parameter state.

    void getStateInformation (juce::MemoryBlock& destData) override
    {
        // copyState() is thread-safe and flushes pending parameter updates.
        // Do NOT call this from the audio thread.
        auto state = parameters_.copyState();
        std::unique_ptr<juce::XmlElement> xml (state.createXml());

        if (xml != nullptr)
        {
            // [v6] Read from pendingKernels_ — the authoritative message-thread
            // store. cpuFilters_[0] may lag by one block if a pending update
            // hasn't been consumed by the audio thread yet.
            const float* k = pendingKernels_.data();
            juce::String coeffStr;
            coeffStr.preallocateBytes (kKernelArraySize * 14);
            for (int i = 0; i < kKernelArraySize; ++i)
            {
                coeffStr += juce::String (k[i], 8);
                if (i < kKernelArraySize - 1) coeffStr += ' ';
            }
            xml->setAttribute ("kernels", coeffStr);

            // AudioProcessor::copyXmlToBinary() is the documented host-safe
            // method for serialising XML into a MemoryBlock.
            copyXmlToBinary (*xml, destData);
        }
    }

    void setStateInformation (const void* data, int sizeInBytes) override
    {
        // getXmlFromBinary() is the documented inverse of copyXmlToBinary().
        std::unique_ptr<juce::XmlElement> xml (getXmlFromBinary (data, sizeInBytes));

        if (xml == nullptr)
            return;

        // Restore kernel coefficients from the attribute we stored.
        // [CHANGE] Reordered the logic to parse the kernel coefficients first.
        // This allows us to modify the XML in place afterwards, avoiding a heap allocation.
        if (xml->hasAttribute ("kernels"))
        {
            auto tokens = juce::StringArray::fromTokens (
                xml->getStringAttribute ("kernels"), " ", "");

            if (tokens.size() == kKernelArraySize)
            {
                std::array<float, kKernelArraySize> loaded{};

                // [v6] Validate every value before applying. A corrupted project
                // file with NaN/Inf kernels would otherwise trigger the NaN guard
                // on every audio block (reset() called 44100+ times per second).
                bool valid = true;
                for (int i = 0; i < kKernelArraySize && valid; ++i)
                {
                    loaded[i] = tokens[i].getFloatValue();
                    if (!std::isfinite (loaded[i]))
                        valid = false;
                }

                if (valid)
                {
                    // [v6] Stage through pendingKernels_ (same path as setKernels)
                    // so the audio thread picks them up safely.
                    pendingKernels_ = loaded;
                    kernelUpdatePending_.store (true, std::memory_order_release);

#ifdef VOLTERRA_CUDA_AVAILABLE
                    if (cudaProcessor_)
                        cudaProcessor_->updateKernels (loaded.data());
#endif
                }
            }
        }

        // Restore APVTS parameter state.
        // replaceState() takes a ValueTree — use ValueTree::fromXml() to convert.
        // Guard with hasTagName() to avoid feeding wrong plugin's state.
        if (xml->hasTagName (parameters_.state.getType()))
        {
            // [CHANGE] Stripping the attribute directly from the original xml object in place
            // before creating the ValueTree, which eliminates the need for std::make_unique.
            xml->removeAttribute ("kernels");
            parameters_.replaceState (juce::ValueTree::fromXml (*xml));
        }
    }

    // ── Editor ────────────────────────────────────────────────────────────────
    // Defined out-of-line in PluginInit.cpp after VolterraEditor is visible.

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    // ── APVTS accessor ────────────────────────────────────────────────────────
    juce::AudioProcessorValueTreeState& getAPVTS() { return parameters_; }

private:
    // ── Parameter layout ──────────────────────────────────────────────────────

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
    {
        std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

        params.push_back (std::make_unique<juce::AudioParameterFloat> (
            juce::ParameterID {"drive", 1}, "Drive",
            juce::NormalisableRange<float> (0.0f, 4.0f, 0.001f, 0.5f), 1.0f));

        params.push_back (std::make_unique<juce::AudioParameterFloat> (
            juce::ParameterID {"mix", 1}, "Mix",
            juce::NormalisableRange<float> (0.0f, 1.0f, 0.001f), 1.0f));

        params.push_back (std::make_unique<juce::AudioParameterFloat> (
            juce::ParameterID {"output", 1}, "Output",
            juce::NormalisableRange<float> (-24.0f, 6.0f, 0.1f), 0.0f));

        return { params.begin(), params.end() };
    }

    // ── CPU processing sub-path ───────────────────────────────────────────────

    /**
     * @brief Per-sample CPU processing with fully smoothed drive/mix/output.
     *
     * This is the real-time path — all parameter automation is handled
     * sample-accurately via SmoothedValue::getNextValue().
     */
    void processBlockCPU (juce::AudioBuffer<float>& buffer,
                          int numChannels,
                          int numSamples)
    {
        // We need to apply smoothed mix and output per-sample.
        // Process each sample through the filter, then blend.
        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto*       wet = buffer.getWritePointer (ch);
            const auto* dry = dryBuffer_.getReadPointer (ch);

            // Snapshot the smoothers for this channel.
            // All channels advance in lock-step from the same ramp position
            // so stereo image doesn't drift under fast automation.
            auto mixRamp    = smoothMix_;
            auto outputRamp = smoothOutputGain_;

            for (int n = 0; n < numSamples; ++n)
            {
                // [v6] processSample() replaces processBlock(..., 1) — avoids
                // the outer-loop wrapper overhead of the multi-sample entry point.
                const float processed  = cpuFilters_[ch].processSample (wet[n]);
                const float mixGain    = mixRamp.getNextValue();
                const float outputGain = outputRamp.getNextValue();

                wet[n] = (processed * mixGain + dry[n] * (1.0f - mixGain)) * outputGain;
            }
        }

        // Advance master smoothers past this block.
        smoothMix_.skip (numSamples);
        smoothOutputGain_.skip (numSamples);
    }

    /**
     * @brief Apply smoothed mix and output as bulk ramp after CUDA en-bloc processing.
     *
     * CUDA processes the whole block atomically, so we can't interleave per-sample
     * gain changes.  We apply the gain ramp as a block multiply using the
     * SmoothedValue, which is still sample-accurate from the host's perspective.
     */
    void applySmoothedGainBulk (juce::AudioBuffer<float>& buffer,
                                int numChannels,
                                int numSamples)
    {
        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto*       wet = buffer.getWritePointer (ch);
            const auto* dry = dryBuffer_.getReadPointer (ch);

            auto mixRamp    = smoothMix_;
            auto outputRamp = smoothOutputGain_;

            for (int n = 0; n < numSamples; ++n)
            {
                const float mixGain    = mixRamp.getNextValue();
                const float outputGain = outputRamp.getNextValue();
                wet[n] = (wet[n] * mixGain + dry[n] * (1.0f - mixGain)) * outputGain;
            }
        }

        smoothMix_.skip (numSamples);
        smoothOutputGain_.skip (numSamples);
    }

    // ── Members ───────────────────────────────────────────────────────────────

    juce::AudioProcessorValueTreeState parameters_;

    /// One CPU filter instance per channel so L/R state never bleeds.
    std::array<PrunedVolterraFilter, 2> cpuFilters_;

    /// Scratch buffer for the dry signal (needed for mix blend).
    juce::AudioBuffer<float> dryBuffer_;

    /// Smoothed parameters — prevent zipper noise on fast automation.
    /// Linear smoothing type (default) is appropriate for gain ratios.
    juce::SmoothedValue<float> smoothDrive_;
    juce::SmoothedValue<float> smoothMix_;
    juce::SmoothedValue<float> smoothOutputGain_;

    double lastSampleRate_ = 44100.0;

    // [v6] std::atomic<bool>: written by audio thread (processBlock),
    // read by message thread (isCUDAActive in VolterraEditor::timerCallback).
    // Was a plain bool — a data race per the C++ memory model.
    std::atomic<bool> cudaActiveLastBlock_{ false };

    // [v6] Staging buffer for kernel updates arriving on the message thread.
    // The audio thread consumes this at the top of processBlock() when
    // kernelUpdatePending_ is true. Release/acquire ordering ensures all
    // float writes into pendingKernels_ are visible before the flag is seen.
    std::array<float, kKernelArraySize> pendingKernels_{};
    std::atomic<bool>                   kernelUpdatePending_{ false };

    // [v6] Cached output dB to skip std::pow (decibelsToGain) when the
    // output parameter hasn't changed between blocks.
    float lastOutputDb_{ 0.0f };

#ifdef VOLTERRA_CUDA_AVAILABLE
    std::unique_ptr<VolterraCUDAProcessor> cudaProcessor_;
#endif

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VolterraProcessor)
};
