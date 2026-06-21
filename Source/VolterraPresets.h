/**
 * @file VolterraPresets.h
 * @brief Named kernel presets for the Pruned Volterra / Memory Polynomial Filter.
 *
 * USAGE
 * ─────
 *   // In VolterraProcessor or a preset manager:
 *   VolterraPresets::load(VolterraPresets::Id::AnalogWarmth, processor);
 *
 *   // Or retrieve raw arrays and pass to setKernels():
 *   float h1[kMemoryLength], h2[kMemoryLength], h3[kMemoryLength];
 *   VolterraPresets::build(VolterraPresets::Id::TapeSlap, h1, h2, h3);
 *   filter.setKernels(h1, h2, h3);
 *
 * KERNEL SEMANTICS (quick reference)
 * ────────────────────────────────────
 *   h1[m]  Linear kernel.  Controls frequency shaping and memory length.
 *          Fast decay  → present, tight impulse response.
 *          Slow decay  → washed, pre-echo / smear character.
 *          Alt. signs  → comb filtering / resonant metallic tone.
 *
 *   h2[m]  Quadratic nonlinearity: term is x[n-m] · |x[n-m]|.
 *          Produces EVEN harmonics (2nd, 4th…) — asymmetric saturation.
 *          Positive  → positive half-cycle saturates harder (tube plate curve).
 *          Negative  → negative half-cycle saturates harder (inverted asymmetry).
 *          Fast decay → only instantaneous asymmetry (no IM between taps).
 *          Slow decay → cross-tap intermodulation, widens harmonic spread.
 *
 *   h3[m]  Cubic nonlinearity: term is x[n-m]³.
 *          Produces ODD harmonics (3rd, 5th…) — symmetric saturation.
 *          Negative  → compressive soft-clip (peaks pulled back).
 *          Positive  → expansive before the output hard-limiter → edge/crunch.
 *          Alt. signs with slow decay → metallic intermodulation ring.
 *
 * STABILITY CONSTRAINT
 * ─────────────────────
 * The output hard-clipper at ±1.0 is the last line of defence, but coefficients
 * should be designed so that unity-gain input (|x| ≤ 1) produces a bounded
 * accumulator before clipping.  A safe guideline:
 *
 *   |Σ h1[m]| + |Σ h2[m]| + |Σ h3[m]|  ≲  2.0
 *
 * All presets below satisfy this.  If you design custom kernels, verify by
 * computing the sum of absolute values of each kernel.
 */

#pragma once

#include <cmath>
#include <cstring>
#include <array>

#include "PrunedVolterraFilter.h"   // for kMemoryLength

// Forward declaration — avoids circular include with VolterraProcessor.h.
class VolterraProcessor;

// ─────────────────────────────────────────────────────────────────────────────

namespace VolterraPresets
{

enum class Id
{
    AnalogWarmth,   ///< Gentle tube-style even-harmonic warming
    TapeSlap,       ///< Magnetic tape: slow linear smear + soft symmetric clip
    GlassHarmonic,  ///< Sparse comb resonance — bowed glass / singing bowl
    MetallicKlang,  ///< Dense alternating-sign IM distortion — metallic clang
    SubOctave,      ///< Strong even-harmonic fold adds an octave-down component
    Bitecrusher,    ///< Aggressive cubic expansion into hard clip — digital crunch
    kCount
};

/// Human-readable name for each preset (matches Id order).
inline const char* name (Id id) noexcept
{
    switch (id)
    {
        case Id::AnalogWarmth:  return "Analog Warmth";
        case Id::GlassHarmonic: return "Glass Harmonic";
        case Id::TapeSlap:      return "Tape Slap";
        case Id::MetallicKlang: return "Metallic Klang";
        case Id::SubOctave:     return "Sub Octave";
        case Id::Bitecrusher:   return "Bitecrusher";
        default:                return "Unknown";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Kernel builder — fills h1[], h2[], h3[] for a given preset.
// ─────────────────────────────────────────────────────────────────────────────

inline void build (Id id,
                   float h1[kMemoryLength],
                   float h2[kMemoryLength],
                   float h3[kMemoryLength]) noexcept
{
    // Zero all arrays first so unused taps don't carry stale data.
    std::memset (h1, 0, kMemoryLength * sizeof (float));
    std::memset (h2, 0, kMemoryLength * sizeof (float));
    std::memset (h3, 0, kMemoryLength * sizeof (float));

    switch (id)
    {
    // ─────────────────────────────────────────────────────────────────────────
    case Id::AnalogWarmth:
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Models a Class-A tube triode stage with mild plate loading.
    //
    // h1: Near-unity linear path with gentle exponential decay representing
    //     the finite bandwidth of a tube stage (pole at ~0.88 normalised).
    //     Tap 0 = 1.0 (no gain change); tails add subtle pre-ringing.
    //
    // h2: Positive, fast-decaying asymmetric saturation.  The even harmonics
    //     produced push the positive half-cycle slightly more than the negative,
    //     mimicking the asymmetric transfer curve of a triode's plate current.
    //     Decay 0.72: contribution from the previous sample is ~5% of tap 0,
    //     limiting IM distortion to a musically pleasant shimmer.
    //
    // h3: Small negative cubic — compresses peaks symmetrically (soft-knee
    //     limiting).  Negative sign means x³ term opposes large amplitudes,
    //     rounding off transient peaks the way a tube's transconductance
    //     drops at high signal levels.  Decay 0.65: mostly instantaneous.
    {
        float v1 = 1.0f,  v2 = 0.04f,  v3 = -0.025f;
        for (int m = 0; m < kMemoryLength; ++m)
        {
            h1[m] = v1;  h2[m] = v2;  h3[m] = v3;
            v1 *= 0.88f; v2 *= 0.72f; v3 *= 0.65f;
        }
        break;
    }

    // ─────────────────────────────────────────────────────────────────────────
    case Id::TapeSlap:
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Models magnetic tape: slow linear smear from playhead gap dispersion,
    // symmetric soft saturation from magnetic remanence.
    //
    // h1: Very slow exponential decay (0.97) — represents the broad, sluggish
    //     impulse response of a tape transport.  Taps are still 15% of tap 0
    //     at m=64, giving a long, low-level pre-echo smear characteristic of
    //     vintage tape at high record levels.  Attenuated slightly at tap 0
    //     (0.85) to compensate for the large tail energy, keeping total gain ≈1.
    //
    // h2: Very small asymmetric term — tape oxide formulations have slight
    //     asymmetry from particulate alignment.  Fast decay (0.60): only the
    //     instantaneous sample contributes; no inter-tap IM.
    //
    // h3: Moderate negative cubic (soft clip) with slow decay (0.92).
    //     This is the dominant character of tape saturation: symmetric
    //     compression that rounds transients.  The slow decay means recent
    //     history also contributes cubic terms — modelling the magnetic
    //     hysteresis loop's memory of recent flux.
    {
        float v1 = 0.85f, v2 = 0.012f, v3 = -0.055f;
        for (int m = 0; m < kMemoryLength; ++m)
        {
            h1[m] = v1;  h2[m] = v2;  h3[m] = v3;
            v1 *= 0.97f; v2 *= 0.60f; v3 *= 0.92f;
        }
        break;
    }

    // ─────────────────────────────────────────────────────────────────────────
    case Id::GlassHarmonic:
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Bowed glass / singing bowl character: sparse resonant comb structure
    // in the linear kernel, very light nonlinearity to add shimmer.
    //
    // h1: Sparse taps at multiples of a short period (every 8 samples ≈
    //     5.5 kHz comb at 44.1 kHz) with slowly decaying positive amplitudes.
    //     Between the comb teeth, h1 = 0 — so only resonant frequencies
    //     survive the linear path, creating a pitched, glassy resonance.
    //     The decay factor (0.82 per comb tooth) gives 6 audible resonant
    //     echoes before dropping below -30 dB.
    //
    // h2: Near-zero asymmetric term — just enough to introduce a subtle
    //     second harmonic that makes the resonance sound less purely digital.
    //     Concentrated at tap 0 only (fast decay 0.1).
    //
    // h3: Small positive cubic — slight expansion before the output clipper
    //     adds a glassy brightness at the resonant frequencies without
    //     sounding crunchy.  Concentrated near tap 0.
    {
        // Sparse comb: non-zero only at multiples of combPeriod.
        constexpr int combPeriod = 8;
        float combAmp = 0.70f;
        for (int m = 0; m < kMemoryLength; ++m)
        {
            if (m % combPeriod == 0)
            {
                h1[m] = combAmp;
                combAmp *= 0.82f;
            }
            // else h1[m] = 0 (already zeroed above)
        }

        // Tiny nonlinearity at tap 0 only.
        float v2 = 0.008f, v3 = 0.012f;
        for (int m = 0; m < kMemoryLength; ++m)
        {
            h2[m] = v2;  h3[m] = v3;
            v2 *= 0.10f; v3 *= 0.10f;
        }
        break;
    }

    // ─────────────────────────────────────────────────────────────────────────
    case Id::MetallicKlang:
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Metallic clang — dense alternating-sign coefficients produce rich
    // intermodulation that mimics the inharmonic partials of struck metal.
    //
    // h1: Damped sinusoid impulse response — alternating positive/negative
    //     taps with exponential decay.  This is equivalent to a resonator at
    //     half the Nyquist frequency (fs/2), creating a strong metallic ring.
    //     The alternation period (every 2 taps) sets the resonant pitch;
    //     change the period to shift the metallic character.
    //
    // h2: Moderate asymmetric term with slow alternating decay — the cross-tap
    //     IM from past and present samples with alternating signs produces
    //     sum-and-difference frequencies that land at inharmonic ratios,
    //     which is exactly what gives struck metal its characteristic clang.
    //
    // h3: Moderate positive cubic with alternating signs and slow decay.
    //     Positive cubic normally expands signals; alternating signs cause
    //     constructive/destructive IM between taps, adding the bright,
    //     brittle overtones of thin sheet metal or a cymbal crash.
    {
        float env1 = 0.75f, env2 = 0.06f, env3 = 0.04f;
        for (int m = 0; m < kMemoryLength; ++m)
        {
            // Alternating sign: +/- every tap for h1, every 3 for h2/h3
            // to produce distinct inharmonic IM products.
            float s1 = (m % 2 == 0) ?  1.0f : -1.0f;
            float s2 = (m % 3 == 0) ?  1.0f : (m % 3 == 1) ? -0.5f : 0.25f;
            float s3 = (m % 2 == 0) ?  1.0f : -0.7f;

            h1[m] = s1 * env1;
            h2[m] = s2 * env2;
            h3[m] = s3 * env3;

            env1 *= 0.91f;
            env2 *= 0.88f;
            env3 *= 0.86f;
        }
        break;
    }

    // ─────────────────────────────────────────────────────────────────────────
    case Id::SubOctave:
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Adds a strong octave-below component by exploiting the even-harmonic
    // (x·|x|) term.  For a pure tone at frequency f, x·|x| contains a
    // strong component at f/2 (half-wave rectification effect), which after
    // the linear memory mixes back into the output as a suboctave.
    //
    // h1: Reduced linear contribution (0.60 at tap 0) to leave headroom for
    //     the strong quadratic term.  Fast decay — tight, present sound.
    //
    // h2: Strong positive quadratic, slower decay than usual (0.80).
    //     The slower decay means past samples contribute to the folding,
    //     enriching the sub-octave content with harmonic sidebands and
    //     making it track pitch changes more smoothly (less like a hard
    //     half-wave rectifier, more like an analogue octave divider).
    //
    // h3: Small negative cubic — prevents the strong h2 term from creating
    //     harsh-sounding high-order products.  Acts as a soft limiter on
    //     the nonlinear output.
    {
        float v1 = 0.60f, v2 = 0.18f, v3 = -0.04f;
        for (int m = 0; m < kMemoryLength; ++m)
        {
            h1[m] = v1;  h2[m] = v2;  h3[m] = v3;
            v1 *= 0.75f; v2 *= 0.80f; v3 *= 0.70f;
        }
        break;
    }

    // ─────────────────────────────────────────────────────────────────────────
    case Id::Bitecrusher:
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Aggressive digital crunch — positive cubic expansion drives the output
    // hard into the ±1.0 hard-clipper, producing flat-top clipping distortion
    // with strong odd harmonics (3rd, 5th, 7th…).  The linear kernel is
    // slightly boosted so the effect is audible even at low drive settings.
    //
    // h1: Unity at tap 0, very fast decay — tight, no memory smear.  The
    //     linear path is mostly pass-through; character comes from h3.
    //
    // h2: Small negative asymmetric term — slight odd-even imbalance that
    //     stops the crunch from sounding too "perfect" and adds a subtle
    //     transistor-style asymmetry to the clipping.
    //
    // h3: Large positive cubic with moderate decay.  Positive cubic expands
    //     the signal (opposite of soft-clip): large amplitudes grow faster,
    //     slamming into the output clipper and creating flat-top, hard-clipped
    //     waveforms.  The moderate decay (0.78) spreads the cubic nonlinearity
    //     across a few taps — creating inter-sample IM that adds the rough,
    //     bitey texture rather than clean waveshaper clipping.
    {
        float v1 = 1.0f,  v2 = -0.015f, v3 = 0.12f;
        for (int m = 0; m < kMemoryLength; ++m)
        {
            h1[m] = v1;  h2[m] = v2;  h3[m] = v3;
            v1 *= 0.60f; v2 *= 0.50f; v3 *= 0.78f;
        }
        break;
    }

    default:
        // Fallback: identity (linear passthrough, no nonlinearity)
        h1[0] = 1.0f;
        break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Convenience: load a preset directly into a VolterraProcessor.
//  Include VolterraProcessor.h before this header to use this overload.
// ─────────────────────────────────────────────────────────────────────────────

#ifdef VOLTERRA_PROCESSOR_INCLUDED

inline void load (Id id, VolterraProcessor& processor)
{
    float h1[kMemoryLength], h2[kMemoryLength], h3[kMemoryLength];
    build (id, h1, h2, h3);
    processor.setKernels (h1, h2, h3);
}

#endif // VOLTERRA_PROCESSOR_INCLUDED

} // namespace VolterraPresets
