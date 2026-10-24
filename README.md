# Voice Stack

The free, local voice-over setup I use for YouTube narration and ads, on Windows with an NVIDIA GPU.
Everything runs on your own machine. This repo is the **setup guide plus my own helper scripts**; the apps
and models themselves come from their official sources (links below), so install those from there.

| Job | Tool | Why |
|---|---|---|
| Fast, clean narration | [Koe](https://github.com/FardeenAFK/koe) (Kokoro-82M) | 58 voices, offline, instant; a bit flat |
| Voice with emotion / your own cloned voice | [Voicebox](https://github.com/jamiepine/voicebox) running Chatterbox | Zero-shot cloning, natural pauses |
| Emotion knobs (exaggeration, cfg) | [chatterbox-tts](https://github.com/resemble-ai/chatterbox) in its own venv | Voicebox doesn't expose them |
| Checking every word was spoken | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Long generations silently drop words |
| Pace, EQ, loudness | [FFmpeg](https://ffmpeg.org/) | `atempo`, equalizer, `loudnorm` |

## The rule: whose voice

Clone **only your own voice, or a voice whose owner agreed** (a hired voice actor with a contract, a
public-domain reader). Never clone a real, identifiable person to make them say things they didn't say, and
never imitate a well-known person so it could pass as them. For "a lively narrator", use recipe 3 below:
no one gets cloned. Chatterbox embeds an inaudible watermark in every output; leave it in.

## Install

1. **Koe**: see its [README](https://github.com/FardeenAFK/koe).
2. **Voicebox**: install the app from its releases page, then let it download the Chatterbox model.
   Headless server for scripts:
   ```
   voicebox-server-cuda.exe --host 127.0.0.1 --port 8778 --data-dir %USERPROFILE%\VoiceStack\voicebox-data
   ```
3. **Direct Chatterbox** (for the emotion knobs), in a separate venv such as `%USERPROFILE%\VoiceStack\cbx`:
   ```
   python -m venv .venv
   .venv\Scripts\pip install chatterbox-tts faster-whisper num2words librosa
   ```
   **CPU-torch trap:** `chatterbox-tts` pins torch and silently swaps CUDA torch for the CPU build. Read its pin,
   then `pip install --force-reinstall torch==<pin> torchaudio==<pin> --index-url https://download.pytorch.org/whl/cu124`
   and check `python -c "import torch;print(torch.cuda.is_available())"`.
4. **FFmpeg** on PATH.

## Scripts

| Script | What it does |
|---|---|
| `scripts/narrate.py` | Generates a script in sentence groups, transcribes every clip back, re-seeds anything that dropped words, joins with real pauses. Backends: `direct` (Chatterbox with knobs) or `voicebox`. |
| `scripts/compare_voices.py` | Measures a clone against its reference: pitch, pitch movement, MFCC timbre match. Always pass a control voice too. |
| `scripts/match_tone.py` | Builds an FFmpeg EQ chain so the clone's tone matches the reference (Chatterbox loses high-end "air"). |
| `scripts/sweep.py` | Sweeps exaggeration / cfg_weight / temperature in-process to find the delivery you want. |

## Recipes

**1. Your own voice, with emotion**
1. Record an 18-25 s *performed* reference (reading with energy, not chatting). A flat reference gives a flat clone.
2. `python scripts/narrate.py SCRIPT.txt --backend direct --ref REF.wav --exaggeration 0.7 --cfg-weight 0.3 --group 3 --gap 0.42 --retries 3 --out OUT`
3. Check pace: the knobs speed speech up. Fix with FFmpeg `atempo=<target/actual>` to land 130-150 wpm.
4. `python scripts/match_tone.py REF.wav OUT/narration.wav FINAL.wav`

**2. Voicebox only (no knobs)**: `--backend voicebox --profile <id> --group 3`. Safe, about 30% flatter.

**3. A lively narrator without cloning anyone**: a Koe voice reads the script with direction, its first ~11 s
becomes the Chatterbox reference, then recipe 1 steps 2-4. Measured +14% pitch movement over the plain Kokoro take.

## Measured settings

| exaggeration | cfg_weight | effect |
|---|---|---|
| 0.5 | 0.5 | library default, safe, a bit flat |
| **0.7** | **0.3** | **best balance**: kept all of the reference's expressiveness, pitch +5% |
| 0.9 | 0.3 | maximum energy, but pitch +25% (stops sounding like the person) |

- Seed variance beats the knobs: pitch swings -6% to +18% across seeds. Generate 4 seeds per line and pick.
- Pick by transcript first, then liveliness. Picking by liveliness alone picks mistakes.

## Traps

- **Long single generations silently drop words** (24 of 237 words vanished with status "completed"). Generate in
  groups of 3 sentences and transcribe back; `narrate.py` does this.
- Short lines fail or get junk prepended. Generate them inside a longer line and cut at the silence.
- Chatterbox leaves 0.2-0.4 s tails, so keep added gaps at 0.12-0.25 s for reels.
- Write acronyms as "XAI", not "X-A-I".
- Unloading a model in Voicebox doesn't free memory: restart the server between models.
- Voicebox status endpoints stream Server-Sent Events, not JSON. Trust `/tasks/active`, not the `downloaded` flag.

## License

MIT for the scripts in this repo. Koe, Voicebox, Chatterbox, Kokoro and faster-whisper keep their own licenses.
