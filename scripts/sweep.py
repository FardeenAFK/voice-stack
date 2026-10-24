#!/usr/bin/env python
r"""sweep.py - drive Chatterbox DIRECTLY to reach the knobs Voicebox does not expose.

Voicebox's /generate only accepts max_chunk_chars, crossfade_ms and normalize. The
parameters that actually control delivery - exaggeration, cfg_weight, temperature -
live in the chatterbox_tts package and are never wired through. This runs the model
in-process so they can be swept.

    python sweep.py --ref REF.wav --text-file LINES.txt --out DIR

Set HF_HOME to the EXISTING cache so the 3 GB of weights are reused, never re-downloaded:
    set HF_HOME=D:\hf-cache    (any folder; keep using the same one)

Nothing here assumes an API shape: the class is discovered, generate() is inspected,
and only kwargs it really accepts are passed. Chatterbox has shipped several signatures.
"""
import argparse
import inspect
import itertools
import os
import sys
import time

# HF_HOME comes from the environment (see above); unset means the default HF cache.


def load_model(device):
    """Find whichever Chatterbox class this version ships, English or multilingual."""
    import chatterbox
    candidates = []
    try:
        from chatterbox.tts import ChatterboxTTS
        candidates.append(("ChatterboxTTS", ChatterboxTTS))
    except Exception as e:
        print("  ChatterboxTTS unavailable:", str(e)[:90])
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        candidates.append(("ChatterboxMultilingualTTS", ChatterboxMultilingualTTS))
    except Exception as e:
        print("  Multilingual unavailable:", str(e)[:90])
    if not candidates:
        sys.exit("no Chatterbox class importable")
    name, cls = candidates[0]
    print("using", name, "| package", getattr(chatterbox, "__version__", "?"))
    t0 = time.time()
    model = cls.from_pretrained(device=device)
    print("loaded in %.1fs" % (time.time() - t0))
    return name, model


def supported(fn):
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out", default="sweep_out")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    import torchaudio
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    device = a.device if torch.cuda.is_available() else "cpu"

    os.makedirs(a.out, exist_ok=True)
    text = open(a.text_file, encoding="utf-8").read().strip()
    # one paragraph is enough to hear and measure delivery; the whole script is not
    text = text.split("\n\n")[0].strip()
    print("text (%d chars): %s..." % (len(text), text[:70]))

    name, model = load_model(device)
    params = supported(model.generate)
    print("generate() accepts:", sorted(params))

    # exaggeration drives intensity; cfg_weight trades fidelity-to-reference against
    # freedom (lower = looser, usually livelier and faster). Sweep the corners.
    grid = list(itertools.product([0.3, 0.5, 0.7, 0.9], [0.3, 0.5]))
    rows = []
    for exag, cfg in grid:
        kw = {}
        if "audio_prompt_path" in params:
            kw["audio_prompt_path"] = a.ref
        if "exaggeration" in params:
            kw["exaggeration"] = exag
        if "cfg_weight" in params:
            kw["cfg_weight"] = cfg
        if "temperature" in params:
            kw["temperature"] = 0.8
        if "language_id" in params:
            kw["language_id"] = "en"
        tag = "e%02d_c%02d" % (exag * 10, cfg * 10)
        dst = os.path.join(a.out, "%s.wav" % tag)
        try:
            t0 = time.time()
            wav = model.generate(text, **kw)
            torchaudio.save(dst, wav.cpu() if hasattr(wav, "cpu") else wav, model.sr)
            print("  %s  exag=%.1f cfg=%.1f  %.1fs" % (tag, exag, cfg, time.time() - t0))
            rows.append(dst)
        except Exception as e:
            print("  %s  FAILED %s: %s" % (tag, type(e).__name__, str(e)[:110]))
    print("\nwrote %d/%d variants to %s" % (len(rows), len(grid), a.out))


if __name__ == "__main__":
    main()
