# Volterra NL Filter — VST3 / AU Plugin

[![Lint](https://github.com/kv244/VolterraFilter/actions/workflows/lint.yml/badge.svg)](https://github.com/kv244/VolterraFilter/actions/workflows/lint.yml)
[![Build](https://github.com/kv244/VolterraFilter/actions/workflows/build.yml/badge.svg)](https://github.com/kv244/VolterraFilter/actions/workflows/build.yml)
[![Release](https://img.shields.io/github/v/release/kv244/VolterraFilter?include_prereleases&label=release&color=blue)](https://github.com/kv244/VolterraFilter/releases/latest)

3rd-Order Diagonal Memory Polynomial (Pruned Volterra Filter) as a JUCE VST3/AU plugin, with optional CUDA acceleration and transparent CPU fallback.

---

## File Map

```
VolterraPlugin/
├── CMakeLists.txt                   — Build system, CUDA detection, dependency tracking
├── README.md
├── installer.nsi                    — NSIS installer script (VST3 + Standalone)
├── Source/
│   ├── PluginInit.cpp               — Plugin entry point; defines createEditor()
│   ├── PrunedVolterraFilter.h       — CPU filter, real-time safe, always compiled
│   ├── VolterraProcessor.h          — JUCE AudioProcessor, CPU↔CUDA dispatch
│   ├── VolterraEditor.h             — JUCE AudioProcessorEditor + VoltLookAndFeel
│   └── VolterraPresets.h            — Six named kernel presets
└── CUDA/
    ├── VolterraKernels.cuh          — CUDA device kernels (process + feature matrix)
    └── VolterraCUDAProcessor.h      — Host-side CUDA wrapper, double-buffered kernels
```

---

## Build — with CUDA (RTX 4070 / sm_89)

```bash
cmake -B build -S . \
  -DJUCE_DIR=/path/to/JUCE \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release -j$(nproc)
```

CMake calls `find_package(CUDAToolkit)` automatically. If found:
- `VOLTERRA_CUDA_AVAILABLE` is defined for all translation units.
- CUDA sources in `CUDA/` are compiled with `nvcc`.
- cuBLAS and cuSOLVER are linked for the kernel-fitting path.

## Build — CPU only (no CUDA toolkit installed)

```bash
cmake -B build -S . \
  -DJUCE_DIR=/path/to/JUCE \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

If `find_package(CUDAToolkit)` fails, `VOLTERRA_CUDA_AVAILABLE` is **not** defined. All `#ifdef VOLTERRA_CUDA_AVAILABLE` blocks compile out. The plugin is fully functional on the CPU path with no code changes.

---

## Runtime CUDA Dispatch Policy

| Condition | Path used |
|---|---|
| `VOLTERRA_CUDA_AVAILABLE` not defined | CPU always |
| Defined, but no CUDA device at runtime | CPU always |
| Defined, device found, `initialise()` fails | CPU always |
| All above pass, **but** `numSamples < 512` | CPU (avoids PCIe overhead) |
| All above pass, `numSamples ≥ 512` | **CUDA** |

The 512-sample threshold (`kCudaMinBlockSize`) is tunable in `VolterraProcessor.h`. At 48 kHz, 512 samples = 10.67 ms — typical for offline render or large buffer sizes. DAW real-time sessions at 64–256 samples always use the CPU path, which is correct behaviour.

---

## DAW Parameters

Three automatable parameters are exposed to the host via the APVTS. They appear in the DAW's automation lanes and in the plugin's own GUI (see [GUI](#gui) below).

| ID | Range | Default | Smoothing | Description |
|---|---|---|---|---|
| `drive` | 0.0 – 4.0 | 1.0 | 20 ms ramp | Input gain applied before the nonlinearity |
| `mix` | 0.0 – 1.0 | 1.0 | 20 ms ramp | Dry/wet blend |
| `output` | −24 – +6 dB | 0 dB | 20 ms ramp | Output level trim after processing |

All three use `juce::SmoothedValue<float>` to prevent zipper noise on fast automation moves.

The filter's tonal character is controlled by the **kernel arrays** (`h1`, `h2`, `h3`), each of length 64 (192 coefficients total). These are set via `setKernels()`, loaded from a preset, or fitted from a hardware capture. They are persisted in the DAW project via `getStateInformation` / `setStateInformation`.

---

---

## GUI

`VolterraEditor.h` implements a 560 × 300 dark hardware-style UI (works in both the Standalone app and any VST3 host).

| Zone | Component | Notes |
|---|---|---|
| Header left | Plugin title + version | Static |
| Header right | Preset combo box | Loads kernel arrays on selection; reverts to "Custom" for manual edits |
| Centre | Drive / Mix / Output knobs | Rotary, `TextBoxBelow`; bound to APVTS via `SliderAttachment` |
| Footer right | CPU / CUDA badge | Polls `isCUDAActive()` at 10 Hz; green when GPU path is live |

`VoltLookAndFeel` (inner class) overrides `drawRotarySlider` with a blue gradient value arc, a pointer line, and a glow dot at the tip. The combo box and popup menu colours are also overridden to match the dark palette.

**Circular-dependency pattern:** `VolterraProcessor.h` declares `createEditor()` without defining it. `VolterraEditor.h` includes `VolterraProcessor.h`. `PluginInit.cpp` includes `VolterraEditor.h` and provides the out-of-line `createEditor()` definition — neither header includes the other.

---

## Installer

An NSIS installer script (`installer.nsi`) is included. It produces a single self-contained `.exe` that places files in their standard Windows locations.

Build the installer (requires [NSIS 3](https://nsis.sourceforge.io/) on PATH):

```bat
makensis installer.nsi
```

Output: `build\Volterra-NL-Filter-1.0.0-win64-Setup.exe`

| Section | Destination | Required |
|---|---|---|
| VST3 Plugin | `%CommonProgramFiles%\VST3\Volterra NL Filter.vst3` | Yes |
| Standalone App | `%ProgramFiles%\VoltDSP\Volterra NL Filter\` | Optional |

The installer registers itself in **Add / Remove Programs** and creates a Start Menu shortcut for the standalone app. The uninstaller removes all installed files and the registry key.

---

## Presets

Six built-in presets are defined in `VolterraPresets.h`. Load them from any non-audio thread (e.g. a button callback in the editor):

```cpp
#include "VolterraPresets.h"

// Load by enum — pushes to both CPU filter and CUDA device buffer
VolterraPresets::load(VolterraPresets::Id::MetallicKlang, processor);

// Or retrieve raw arrays (e.g. for blending two presets)
float h1[kMemoryLength], h2[kMemoryLength], h3[kMemoryLength];
VolterraPresets::build(VolterraPresets::Id::TapeSlap, h1, h2, h3);
filter.setKernels(h1, h2, h3);
```

| Preset | `Id` enum | Character |
|---|---|---|
| **Analog Warmth** | `AnalogWarmth` | Gentle tube triode — even harmonics, soft peak rounding. Default load. |
| **Tape Slap** | `TapeSlap` | Magnetic tape — long linear smear + symmetric remanence saturation |
| **Glass Harmonic** | `GlassHarmonic` | Sparse comb resonance — bowed glass / singing bowl pitched ring |
| **Metallic Klang** | `MetallicKlang` | Alternating-sign IM distortion — struck metal, inharmonic partials |
| **Sub Octave** | `SubOctave` | Strong even-harmonic fold — genuine f/2 suboctave from x·\|x\| rectification |
| **Bitecrusher** | `Bitecrusher` | Positive cubic expansion into hard clip — flat-top crunch, strong odd harmonics |

### Kernel design reference

The three kernel arrays control distinct aspects of the nonlinear character:

| Kernel | Term | Effect | Design levers |
|---|---|---|---|
| `h1[m]` | `x[n-m]` (linear) | Frequency shaping, memory length | Decay rate → tight vs. washed; alternating signs → comb/metallic resonance |
| `h2[m]` | `x[n-m]·\|x[n-m]\|` | Asymmetric saturation, even harmonics (2nd, 4th…) | Positive → tube plate curve; slow decay → cross-tap IM widens harmonic spread |
| `h3[m]` | `x[n-m]³` | Symmetric saturation, odd harmonics (3rd, 5th…) | Negative → soft clip; positive → expansion into hard limiter; alt. signs → metallic IM |

**Stability guideline:** keep `|Σ h1| + |Σ h2| + |Σ h3| ≲ 2.0` for unity-gain input. The output hard-clipper at ±1.0 and the NaN/Inf circuit-breaker (history flush) are last-resort guards, not primary limiters.

---

## Kernel Fitting (CUDA)

To capture a hardware device and fit the 192 coefficients to match it:

```cpp
// Collect input/output recording from hardware (N samples each).
// inputPadded must have M-1 zeros prepended (or previous block's tail).
std::vector<float> inputPadded(N + kMemoryLength - 1, 0.0f);
std::copy(recordedInput.begin(), recordedInput.end(),
          inputPadded.begin() + kMemoryLength - 1);

bool ok = processor.fitKernelsFromCapture(
    inputPadded.data(),
    recordedOutput.data(),
    N);

// If ok == true, processor now uses the fitted kernels.
// Kernels are automatically persisted on next getStateInformation() call.
```

The GPU builds the N×192 feature matrix Φ (`buildFeatureMatrixKernel`), transposes it to column-major via `cublasSgeam`, then solves min‖Φ·h − y‖² with `cusolverDnSgels`. The solution is re-interleaved from grouped `[h1…, h2…, h3…]` order to the plugin's interleaved `[h1[0], h2[0], h3[0], h1[1]…]` layout before being pushed to both the CPU filter and the CUDA double-buffer.

---

## Mathematical Model

```
y[n] = Σ h1[m] · x[n-m]             (linear memory)
     + Σ h2[m] · x[n-m] · |x[n-m]|  (asymmetric saturation — even harmonics)
     + Σ h3[m] · x[n-m]³             (symmetric saturation — odd harmonics)
       m = 0 .. 63
```

Dropping all off-diagonal cross-terms reduces the parameter count from O(M³) = 262,144 to O(3M) = 192, making real-time CPU processing viable while preserving the dominant nonlinear character of hardware devices.

---

## Thread Safety

| State | Owner | Mechanism |
|---|---|---|
| `cudaProcessor_` unique_ptr | Message thread (prepareToPlay / releaseResources) | VST3/AU host contract guarantees no audio thread overlap |
| `d_kernels_[active]` + `audioStream_` | Audio thread exclusively | No message thread access permitted |
| `d_kernels_[pending]` + `writeStream_` | Message thread exclusively | No audio thread access permitted |
| Active↔pending swap | Coordinated | `std::atomic<bool> swapPending_` + `cudaEvent_t` GPU-side fence |
| `cpuFilters_[]` | Audio thread exclusively (read + write) | Kernel updates staged via `pendingKernels_` + `kernelUpdatePending_` atomic |
| `cudaActiveLastBlock_` | Audio thread writes, message thread reads | `std::atomic<bool>` with relaxed ordering |

The audio thread is entirely wait-free on the host CPU. The GPU-side `cudaStreamWaitEvent` stall is handled by the GPU scheduler — the CPU audio callback returns immediately.

---

## Tested On

- JUCE 8.x, CMake 3.24+
- CUDA Toolkit 12.x, sm_89 (RTX 4070)
- Windows 11 (MSVC 19.3x), Ubuntu 22.04 (GCC 12)
- VST3 hosts: REAPER 7, Ableton Live 12

---

## Changelog

### v6 — Bug fixes and performance (2026-06-21)

**Thread safety — CPU kernel updates (critical)**
- Was: `setKernels()` wrote directly into `cpuFilters_[]` on the message thread while the audio thread read them in `processBlock()` — an unconditional data race.
- Now: `setKernels()`, `fitKernelsFromCapture()`, and `setStateInformation()` write to `pendingKernels_` (a plain `std::array`), then set `std::atomic<bool> kernelUpdatePending_` with `memory_order_release`. The audio thread loads the flag with `memory_order_acquire` at the top of each `processBlock()` call, copying the kernels into `cpuFilters_[]` only on the audio thread. `cpuFilters_[]` is now audio-thread-exclusive.

**Thread safety — CUDA active flag**
- Was: `cudaActiveLastBlock_` was a plain `bool` written by the audio thread and read by the message thread — a data race.
- Now: `std::atomic<bool>` with `store(relaxed)` on the write path and `load(relaxed)` on the read path.

**Correctness — dead variable removed**
- Removed unused `driveCopy` snapshot in the drive application block. It was constructed, never read, and left confusing dead code alongside the actual `driveRamp` copy that was used.

**Correctness — kernel validation on load**
- `setStateInformation()` now checks every loaded kernel value with `std::isfinite()` before applying it. A corrupted project file with NaN/Inf kernels would previously pass silently and trigger the NaN guard (a full `reset()`) on every audio block.

**Correctness — state persistence source**
- `getStateInformation()` now reads from `pendingKernels_` instead of `cpuFilters_[0]`. Previously, if a preset was selected and the audio thread hadn't yet consumed the pending update, the saved state would contain stale kernels.

**Correctness — CUDA partial fallback status**
- If CUDA succeeded on ch=0 but failed on ch=1, `isCUDAActive()` used to return `true` for the whole block. Now `cudaActiveLastBlock_` is only `true` when all channels succeeded via CUDA.

**Performance — per-sample processing**
- Added `PrunedVolterraFilter::processSample(float) -> float`. The CPU path in `processBlockCPU` now calls `processSample()` instead of `processBlock(..., 1)`, eliminating the outer-loop wrapper overhead for each sample.

**Performance — AVX2 vectorisation**
- `PrunedVolterraFilter::history_[]` changed from `kMemoryLength` (64 floats) to `2 × kMemoryLength` (128 floats). Each input sample is written to both halves. The inner tap loop now reads `history_[base + m]` with a contiguous, non-wrapping address range — removing the `(writeHead_ + m) & kMask` modulo that previously blocked SIMD auto-vectorisation.

**Performance — output dB caching**
- `processBlock()` now skips `Decibels::decibelsToGain()` (`std::pow`) when the output parameter is unchanged since the previous block.

---

### v5 — GUI + Installer + Windows build fixes

**GUI (`VolterraEditor.h`)**
- Added `VolterraEditor`: 560 × 300 dark hardware-style `AudioProcessorEditor`.
- Added `VoltLookAndFeel`: custom rotary knob renderer (gradient value arc, pointer, glow dot) and dark combo/popup colours.
- Three knobs (Drive, Mix, Output) bound to APVTS via `SliderAttachment`.
- Preset combo box in header; loads kernel arrays via `VolterraPresets::build()`.
- CPU / CUDA status badge in footer, refreshed at 10 Hz via `juce::Timer`.
- `createEditor()` defined out-of-line in `PluginInit.cpp` to avoid circular header dependency.

**Installer (`installer.nsi`)**
- NSIS 3 script; produces `Volterra-NL-Filter-1.0.0-win64-Setup.exe`.
- Installs VST3 bundle to `%CommonProgramFiles%\VST3` and optional Standalone to `%ProgramFiles%\VoltDSP\...`.
- Registers in Add / Remove Programs with working uninstaller.

**Windows / MSVC build fixes**
- **CUDA toolset absent:** replaced bare `enable_language(CUDA)` with `check_language(CUDA)` probe so the build falls back to CPU-only gracefully when the VS CUDA integration is not installed.
- **RTTI:** removed `/GR-` and `-fno-rtti` — JUCE 8 requires RTTI.
- **`__restrict__`:** MSVC spells this `__restrict`. Added `VOLTERRA_RESTRICT` portability macro in `PrunedVolterraFilter.h`.
- **Virtual return types:** JUCE 8 declares `virtual const String getName()` and `getProgramName()`; overrides now match with `const juce::String` return type.
- **VST3 parameter ID warning:** added `JUCE_IGNORE_VST3_MISMATCHED_PARAMETER_ID_WARNING=1` (new plugin, no VST2 predecessor).
- **Source layout:** moved all source files into `Source/` and `CUDA/` subdirectories to match CMakeLists paths.

### v4 — Presets

- Added `VolterraPresets.h` with six named kernel presets: Analog Warmth, Tape Slap, Glass Harmonic, Metallic Klang, Sub Octave, Bitecrusher.
- Added `VOLTERRA_PROCESSOR_INCLUDED` sentinel to `VolterraProcessor.h` so the `VolterraPresets::load(Id, VolterraProcessor&)` convenience overload compiles correctly when both headers are included together.
- Added `VolterraPresets.h` to `VOLTERRA_HEADER_SOURCES` in `CMakeLists.txt`.

### v3 — Bug fixes (thread safety, DSP, layout, CMake)

**Thread safety — CUDA kernel double-buffer (critical)**
- Was: single `d_kernels_` buffer and single `stream_` shared between audio and message threads. Concurrent `cudaMemcpyAsync` calls from two CPU threads on the same stream = undefined behaviour at the CUDA driver level.
- Now: `d_kernels_[2]` double-buffer. `audioStream_` is audio-thread exclusive; `writeStream_` is message-thread exclusive. Swap coordinated via `std::atomic<bool> swapPending_` + `cudaEvent_t writeCompletedEvent_` GPU-side fence. Zero CPU blocking on the audio thread.

**History tail sliding window (high priority DSP)**
- Was: `cudaMemcpyAsync(d_inputPadded_, d_inputPadded_ + maxBlockSize_, ...)` — reads stale data when `numSamples < maxBlockSize_`, which DAWs produce regularly at transport loop boundaries and render tail.
- Now: `cudaMemcpyAsync(d_inputPadded_, d_inputPadded_ + numSamples, ...)` — always shifts exactly the samples that were just processed.

**cuSOLVER matrix layout (high priority DSP)**
- Was: `d_Phi` written row-major, passed directly to `cusolverDnSgels` which expects column-major. Solved `Φᵀ·h = y` instead of `Φ·h = y` — returned garbage coefficients.
- Now: `buildFeatureMatrixKernel` writes row-major (coalesced stores). `cublasSgeam` transposes to column-major before the SGELS call. Solution vector re-interleaved from grouped `[h1…,h2…,h3…]` to interleaved `[h1[m],h2[m],h3[m],…]` on host.

**CMake dependency tracking (medium priority)**
- Was: `file(WRITE)` generated `PluginInit.cpp` at configure time in `CMAKE_BINARY_DIR`. No build-phase dependency; `.cuh` files in `target_sources` could be miscompiled by some generators.
- Now: `Source/PluginInit.cpp` is a committed source file. `set_source_files_properties(HEADER_FILE_ONLY TRUE)` applied to `.cuh` files. Source list split into `VOLTERRA_COMPILED_SOURCES` and `VOLTERRA_HEADER_SOURCES` for clarity.

### v2 — JUCE API audit

- `getStateInformation` / `setStateInformation`: replaced manual XML with `copyXmlToBinary` / `getXmlFromBinary` / `ValueTree::fromXml`.
- Denormal handling: `FloatVectorOperations::disableDenormalisedNumberSupport()` called once in `prepareToPlay()` instead of `ScopedNoDenormals` per block.
- Parameter smoothing: `juce::SmoothedValue<float>` for all three parameters (20 ms ramp).
- Added `processBlockBypassed` override: resets filter state on bypass to prevent transient on re-engage.
- Fixed wrong `PluginInit.cpp` include (`juce_audio_plugin_client` is a CMake target, not a header).
- Kernel XML isolated as a root attribute so APVTS never sees foreign data in its own tree.
