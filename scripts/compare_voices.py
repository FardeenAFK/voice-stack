"""Measure how close a cloned voice is to its reference - objectively, not by ear.

    python compare_voices.py REFERENCE.wav CLONE.wav [CONTROL.wav ...]

Reports, per file:
  F0 median/spread  - perceived pitch and how much it moves (monotone vs expressive)
  MFCC cosine       - timbre match: the vocal-tract fingerprint. This is the one that
                      actually says "same speaker or not".
  spectral centroid - brightness; a bright clone against a dark reference reads as
                      a different person even when pitch matches.
A CONTROL (a DIFFERENT speaker) is essential: without it a cosine of 0.9 means nothing,
because unrelated speech already scores high. The gap clone-vs-control is the real signal.
"""
import sys
import numpy as np
import librosa


def profile(path):
    y, sr = librosa.load(path, sr=24000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr,
                                 frame_length=2048, hop_length=256)
    f0v = f0[~np.isnan(f0)]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # drop coefficient 0: it tracks loudness, not timbre, and would mask real differences
    return {
        "dur": len(y) / sr,
        "f0_med": float(np.median(f0v)) if f0v.size else float("nan"),
        "f0_iqr": float(np.percentile(f0v, 75) - np.percentile(f0v, 25)) if f0v.size else float("nan"),
        "centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "vec": np.mean(mfcc[1:], axis=1),
    }


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    paths = sys.argv[1:]
    if len(paths) < 2:
        sys.exit(__doc__)
    ps = [(p, profile(p)) for p in paths]
    ref_name, ref = ps[0]

    print(f"{'file':34s} {'dur':>6s} {'F0 med':>8s} {'F0 IQR':>7s} {'bright':>8s} {'MFCC cos':>9s}")
    for name, p in ps:
        cos = "  (ref)" if p is ref else f"{cosine(ref['vec'], p['vec']):9.4f}"
        short = name.split("/")[-1].split("\\")[-1]
        print(f"{short:34s} {p['dur']:6.1f} {p['f0_med']:8.1f} {p['f0_iqr']:7.1f} "
              f"{p['centroid']:8.0f} {cos:>9s}")

    if len(ps) >= 3:
        clone = cosine(ref["vec"], ps[1][1]["vec"])
        ctrl = max(cosine(ref["vec"], p["vec"]) for _, p in ps[2:])
        print(f"\nclone {clone:.4f} vs best control {ctrl:.4f}  ->  margin {clone - ctrl:+.4f}")
        print("A clone that is genuinely the same voice should sit clearly ABOVE every control.")


if __name__ == "__main__":
    main()
