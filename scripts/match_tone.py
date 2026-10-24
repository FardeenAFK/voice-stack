#!/usr/bin/env python
r"""match_tone.py - make a clone's tone match its reference, by measurement not by ear.

    python match_tone.py REFERENCE.wav CLONE.wav OUT.wav [--max-gain 9] [--dry]

WHY
  Chatterbox throws away high frequency energy. Measured on a real cloned voice:
  his recording carried 15.96% of its energy above 6 kHz, the clone only 7.02% - less
  than half. That deficit is what reads as "dull" and contributes to "robotic".

WHAT IT DOES
  Compares the long-term average spectrum of clone vs reference in octave bands, and
  builds an ffmpeg equalizer chain that corrects the difference. Gains are clamped,
  because a big boost in a band the model never generated only lifts noise.

WHAT IT CANNOT FIX
  Prosody. Flat delivery (low F0 IQR) is inherited from the reference performance and
  no EQ touches it. Tone and liveliness are different problems - this one is tone.
"""
import argparse
import subprocess
import sys

import numpy as np
import librosa

# octave-ish bands across the speech range; 24 kHz audio means nothing above 12 kHz
BANDS = [(120, 250), (250, 500), (500, 1000), (1000, 2000),
         (2000, 4000), (4000, 6000), (6000, 9000), (9000, 11500)]


def band_energy(path, sr=24000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    S = np.abs(librosa.stft(y, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mean = S.mean(axis=1)
    total = mean.sum() + 1e-12
    out = []
    for lo, hi in BANDS:
        sel = (freqs >= lo) & (freqs < hi)
        out.append(float(mean[sel].sum() / total))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("clone")
    ap.add_argument("out")
    ap.add_argument("--max-gain", type=float, default=9.0,
                    help="dB ceiling per band; beyond this you are amplifying noise")
    ap.add_argument("--dry", action="store_true", help="print the filter, write nothing")
    a = ap.parse_args()

    ref, clo = band_energy(a.reference), band_energy(a.clone)
    print("%-14s %9s %9s %8s" % ("band (Hz)", "ref %", "clone %", "gain dB"))

    filters = []
    for (lo, hi), r, c in zip(BANDS, ref, clo):
        # ratio of proportions -> dB. Both are normalised, so this corrects TILT,
        # not level; overall loudness is restored by loudnorm afterwards.
        gain = 10.0 * np.log10((r + 1e-9) / (c + 1e-9))
        gain = float(np.clip(gain, -a.max_gain, a.max_gain))
        centre = int(round((lo * hi) ** 0.5))          # geometric centre
        width = hi - lo
        print("%-14s %8.2f%% %8.2f%% %+8.1f" % ("%d-%d" % (lo, hi), r * 100, c * 100, gain))
        if abs(gain) >= 0.5:                            # skip inaudible corrections
            filters.append("equalizer=f=%d:t=h:w=%d:g=%.2f" % (centre, width, gain))

    if not filters:
        print("\nno correction needed")
        return 0
    chain = ",".join(filters) + ",loudnorm=I=-20:TP=-2"
    print("\nfilter chain:\n  " + chain)
    if a.dry:
        return 0
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", a.clone,
                    "-af", chain, "-ac", "1", "-ar", "24000", a.out], check=True)

    after = band_energy(a.out)
    before_err = float(np.abs(np.log10((ref + 1e-9) / (clo + 1e-9))).mean())
    after_err = float(np.abs(np.log10((ref + 1e-9) / (after + 1e-9))).mean())
    hi_ref = ref[5:].sum() * 100
    hi_before = clo[5:].sum() * 100
    hi_after = after[5:].sum() * 100
    print("\nenergy above 4 kHz:  reference %.2f%% | before %.2f%% | after %.2f%%"
          % (hi_ref, hi_before, hi_after))
    print("mean spectral error: before %.4f -> after %.4f  (lower is closer)"
          % (before_err, after_err))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
