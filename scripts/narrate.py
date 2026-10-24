#!/usr/bin/env python
r"""narrate.py - production-safe narration with Voicebox/Chatterbox.

WHY THIS EXISTS
  A single long generation SILENTLY DROPS TEXT. Measured 2026-09-17 on the real
  documentary cold open: 237 words in, 213 spoken - 24 words (10.1%) never
  uttered, including a load-bearing statistic. Status reported "completed" the whole
  time. There is no error to catch, so the only defence is to verify every clip.

WHAT IT DOES
  Splits the script into sentences, generates each one on its own, transcribes each
  clip back and compares it against the sentence that was asked for, retries failures
  with a fresh seed, then concatenates with real pauses. Anything still failing is
  reported loudly and LEFT OUT rather than smuggled into the output.

    python narrate.py SCRIPT.txt --profile <profile_id> --out DIR
    python narrate.py SCRIPT.txt --profile <id> --out DIR --gap 0.45 --retries 3

VERIFY LOGIC - the part that matters
  Substitutions are usually the TRANSCRIBER's fault, not the voice's: whisper writes
  "$21.62" for "twenty-one dollars sixty-two", and hears medal/metal, hairs/heirs.
  Counting those as errors reported 82.1% when real content loss was 10.1%.
  So numbers are normalised to words on BOTH sides, and only a contiguous DELETION of
  >= MIN_DROP words counts as a real failure. Substitutions only warn.

NOTE: run the server fresh before a long job - it leaks memory between models.
    voicebox-server-cuda.exe --host 127.0.0.1 --port 8778 --data-dir %USERPROFILE%\VoiceStack\voicebox-data
"""
import argparse
import difflib
import inspect
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.request

BASE = "http://127.0.0.1:8778"
MIN_DROP = 3          # a contiguous gap this long is real content loss, not a mishearing
SIM_WARN = 0.90       # below this, flag for a human listen even when nothing was dropped

ABBREV = r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bSt)(?<!\bJr)(?<!\bSr)(?<!\bvs)(?<!\bNo)"


def split_sentences(text, hard_limit=280):
    """Sentence split that does not break on decimals, money or common abbreviations."""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)      # protect 21.62
    parts = re.split(ABBREV + r"(?<=[.!?])\s+(?=[\"'A-Z0-9])", text)
    out = []
    for p in parts:
        p = p.replace("<DOT>", ".").strip()
        if not p:
            continue
        # an over-long sentence is the exact condition that caused the drop - break it
        while len(p) > hard_limit:
            cut = p.rfind(", ", 0, hard_limit)
            if cut < hard_limit // 2:
                cut = p.rfind(" ", 0, hard_limit)
            if cut <= 0:
                break
            out.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            out.append(p)
    return out


def _n2w(n):
    from num2words import num2words
    try:
        return num2words(n)
    except Exception:
        return str(n)


def normalise(s):
    """Canonical word list. Digits become words so '$21.62' == 'twenty-one dollars sixty-two'."""
    s = unicodedata.normalize("NFKD", s).lower().replace("&", " and ")

    def money(m):
        out = _n2w(int(m.group(1))) + " dollars"
        if m.group(2) and int(m.group(2)):
            out += " " + _n2w(int(m.group(2))) + " cents"
        return " " + out + " "

    s = re.sub(r"\$(\d+)(?:\.(\d{1,2}))?", money, s)
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b",
               lambda m: " " + _n2w(int(m.group(1))) + "th ", s)
    s = re.sub(r"\b\d+\.\d+\b", lambda m: " " + _n2w(float(m.group(0))) + " ", s)
    s = re.sub(r"\b\d+\b", lambda m: " " + _n2w(int(m.group(0))) + " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    # 'twenty-seventh' vs 'twenty seventh' must match; 'and' is noise in spoken numbers
    return [w for w in s.split() if w and w != "and"]


def group_sentences(sents, per_group, max_chars=300):
    """Batch sentences so intonation can carry across a few lines.

    One-at-a-time is safest but flattest: measured F0 IQR 29.4 vs 39.2 for a single
    long pass, because an isolated sentence gets a neutral start and end contour.
    Grouping keeps us far below the ~800-char zone where text silently vanished,
    while giving the model a paragraph to shape.
    """
    if per_group <= 1:
        return [[s] for s in sents]
    groups, cur = [], []
    for s in sents:
        projected = sum(len(x) + 1 for x in cur) + len(s)
        if cur and (len(cur) >= per_group or projected > max_chars):
            groups.append(cur)
            cur = []
        cur.append(s)
    if cur:
        groups.append(cur)
    return groups


def compare(asked, heard):
    a, b = normalise(asked), normalise(heard)
    sm = difflib.SequenceMatcher(None, a, b)
    drops = [" ".join(a[i1:i2]) for tag, i1, i2, _, _ in sm.get_opcodes()
             if tag == "delete" and (i2 - i1) >= MIN_DROP]
    return sm.ratio(), drops


def _get(path, timeout=120):
    raw = urllib.request.urlopen(BASE + path, timeout=timeout).read()
    t = raw.decode("utf-8", "replace").strip()
    if t.startswith("data:"):                      # this API mixes plain JSON and SSE
        t = t.split("data:", 1)[1].strip().split("\n\n")[0]
    try:
        return json.loads(t)
    except Exception:
        return {"_raw": t[:200]}


def _post(path, payload, timeout=1800):
    req = urllib.request.Request(BASE + path, method="POST",
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps(payload).encode())
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


class VoiceboxBackend:
    """The HTTP server. Safe and simple, but it exposes NO delivery controls -
    only chunking and normalize. Locked output measured at F0 IQR 15.2."""

    def __init__(self, profile, engine="chatterbox"):
        self.profile, self.engine = profile, engine

    def speak(self, text, seed, out_path, timeout=1500):
        gid = _post("/generate", {"profile_id": self.profile, "text": text, "language": "en",
                                  "engine": self.engine, "normalize": True, "seed": seed}).get("id")
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                s = _get("/generate/%s/status" % gid)
            except Exception:
                time.sleep(4)
                continue
            st = s.get("status")
            if st == "completed":
                data = urllib.request.urlopen("%s/audio/%s" % (BASE, gid), timeout=300).read()
                open(out_path, "wb").write(data)
                return
            if st in ("failed", "error", "cancelled"):
                raise RuntimeError(s.get("error") or st)
            time.sleep(3)
        raise TimeoutError("generation timed out")


class DirectBackend:
    """Chatterbox in-process, which is the ONLY way to reach exaggeration/cfg_weight.

    Measured on a real reference: exag 0.7 / cfg 0.3 gives F0 IQR 22.1
    against Voicebox's locked 15.2, at +5.3% pitch. Do NOT push exaggeration to 0.9 -
    IQR hits 42.9 but pitch inflates 24.6% and it stops sounding like the speaker.
    MFCC identity stays ~0.987 there and will NOT warn you, so watch F0, not identity.

    The model loads once and is reused for every unit, so a long script pays the
    ~40s load a single time.
    """

    def __init__(self, ref, exaggeration, cfg_weight, temperature, device="cuda"):
        os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
        import torch
        from chatterbox.tts import ChatterboxTTS
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("WARNING: CUDA unavailable - a torch-touching install probably "
                  "replaced the CUDA build with +cpu. This will be very slow.")
        print("loading chatterbox on %s ..." % self.device, flush=True)
        t0 = time.time()
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        print("loaded in %.1fs" % (time.time() - t0), flush=True)
        self.ref, self.exag = ref, exaggeration
        self.cfg, self.temp = cfg_weight, temperature
        self.params = set(inspect.signature(self.model.generate).parameters)

    def speak(self, text, seed, out_path):
        import torchaudio
        self.torch.manual_seed(seed)
        kw = {}
        for key, val in (("audio_prompt_path", self.ref), ("exaggeration", self.exag),
                         ("cfg_weight", self.cfg), ("temperature", self.temp)):
            if key in self.params:
                kw[key] = val
        wav = self.model.generate(text, **kw)
        torchaudio.save(out_path, wav.cpu() if hasattr(wav, "cpu") else wav, self.model.sr)


def ffmpeg(args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y"] + args, check=True)


def duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip()
    return float(out or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--backend", choices=("direct", "voicebox"), default="direct",
                    help="direct = chatterbox in-process WITH the delivery knobs; "
                         "voicebox = the HTTP server, which exposes none of them")
    ap.add_argument("--profile", help="voicebox backend: the profile id")
    ap.add_argument("--ref", help="direct backend: the reference wav to clone")
    ap.add_argument("--exaggeration", type=float, default=0.7,
                    help="delivery intensity. 0.7 measured best; 0.9 inflates pitch ~25%%")
    ap.add_argument("--cfg-weight", dest="cfg_weight", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", default="narration")
    ap.add_argument("--engine", default="chatterbox")
    ap.add_argument("--gap", type=float, default=0.38, help="seconds of silence between sentences")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--group", type=int, default=1,
                    help="sentences per generation. 1 = safest but flattest; "
                         "2-3 keeps intonation across lines and is still far under "
                         "the ~800 chars where text silently vanishes")
    a = ap.parse_args()

    if a.backend == "direct" and not a.ref:
        ap.error("--backend direct needs --ref (the reference wav to clone)")
    if a.backend == "voicebox" and not a.profile:
        ap.error("--backend voicebox needs --profile")

    os.makedirs(a.out, exist_ok=True)
    text = open(a.script, encoding="utf-8").read().strip()
    sents = split_sentences(text)
    groups = group_sentences(sents, a.group)
    units = [" ".join(g) for g in groups]
    print("%d sentences -> %d units (group=%d, longest %d chars), %d words\n"
          % (len(sents), len(units), a.group,
             max(len(u) for u in units), len(text.split())))
    sents = units

    if a.backend == "direct":
        backend = DirectBackend(a.ref, a.exaggeration, a.cfg_weight, a.temperature)
        print("knobs: exaggeration=%.2f cfg_weight=%.2f temperature=%.2f\n"
              % (a.exaggeration, a.cfg_weight, a.temperature))
    else:
        backend = VoiceboxBackend(a.profile, a.engine)
        print("backend: voicebox (NO delivery controls available)\n")

    from faster_whisper import WhisperModel
    wm = WhisperModel("small.en", device="cpu", compute_type="int8")

    manifest, clean = [], []
    for i, s in enumerate(sents, 1):
        row = {"n": i, "text": s, "attempts": []}
        ok = False
        for attempt in range(a.retries):
            seed = a.seed + attempt * 977
            raw = os.path.join(a.out, "s%03d_raw.wav" % i)
            wav = os.path.join(a.out, "s%03d.wav" % i)
            try:
                backend.speak(s, seed, raw)
            except Exception as e:
                row["attempts"].append({"seed": seed, "error": str(e)[:120]})
                print("  [%3d] attempt %d ERROR %s" % (i, attempt + 1, str(e)[:70]))
                continue
            # ffmpeg can fail on perfectly good audio when the machine is out of commit
            # memory (loudnorm needs a buffer; seen at 0.2GB headroom). That is a retryable
            # condition, NOT a reason to abandon the whole script - which is what it did.
            try:
                ffmpeg(["-i", raw, "-ac", "1", "-ar", "24000",
                        "-af", "loudnorm=I=-20:TP=-2", wav])
            except subprocess.CalledProcessError as e:
                if os.path.exists(wav) and os.path.getsize(wav) == 0:
                    os.remove(wav)          # a 0-byte file would poison the concat list
                row["attempts"].append({"seed": seed, "error": "ffmpeg rc=%s" % e.returncode})
                print("  [%3d] attempt %d ffmpeg FAILED (rc=%s) - low memory?"
                      % (i, attempt + 1, e.returncode))
                continue
            segs, _ = wm.transcribe(wav, beam_size=5)
            heard = " ".join(x.text.strip() for x in segs).strip()
            ratio, drops = compare(s, heard)
            row["attempts"].append({"seed": seed, "ratio": round(ratio, 3),
                                    "drops": drops, "heard": heard})
            if not drops:
                ok = True
                row.update(file=wav, ratio=round(ratio, 3),
                           seconds=round(duration(wav), 2), warn=(ratio < SIM_WARN))
                flag = "  <- LISTEN (low similarity)" if ratio < SIM_WARN else ""
                print("  [%3d] ok   sim=%.3f %5.1fs%s" % (i, ratio, duration(wav), flag))
                clean.append(wav)
                break
            print("  [%3d] attempt %d DROPPED %s" % (i, attempt + 1, drops))
        if not ok:
            row["failed"] = True
            print("  [%3d] *** FAILED after %d attempts - EXCLUDED" % (i, a.retries))
        manifest.append(row)

    sil = os.path.join(a.out, "_gap.wav")
    ffmpeg(["-f", "lavfi", "-t", str(a.gap), "-i", "anullsrc=r=24000:cl=mono",
            "-ac", "1", "-ar", "24000", sil])
    listfile = os.path.join(a.out, "_concat.txt")
    with open(listfile, "w", encoding="utf-8") as fh:
        for k, w in enumerate(clean):
            fh.write("file '%s'\n" % os.path.abspath(w).replace("\\", "/"))
            if k != len(clean) - 1:
                fh.write("file '%s'\n" % os.path.abspath(sil).replace("\\", "/"))
    final = os.path.join(a.out, "narration.wav")
    if clean:
        ffmpeg(["-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", final])

    failed = [r["n"] for r in manifest if r.get("failed")]
    warn = [r["n"] for r in manifest if r.get("warn")]
    json.dump({"script": a.script, "backend": a.backend, "profile": a.profile,
               "ref": a.ref, "engine": a.engine, "group": a.group, "gap": a.gap,
               "knobs": ({"exaggeration": a.exaggeration, "cfg_weight": a.cfg_weight,
                          "temperature": a.temperature} if a.backend == "direct"
                         else "none - voicebox exposes no delivery controls"),
               "sentences": manifest, "failed": failed, "warn": warn,
               "output": final if clean else None},
              open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8"), indent=2)

    print("\n%d/%d sentences verified" % (len(clean), len(sents)))
    if warn:
        print("LISTEN to sentences %s - similarity below %s" % (warn, SIM_WARN))
    if failed:
        print("*** %d FAILED and are NOT in the output: %s" % (len(failed), failed))
    if clean:
        print("wrote %s  %.1fs" % (final, duration(final)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
