"""Intelligibility eval: Whisper word error rate on codec reconstructions, per bitrate, on speaker-disjoint test-clean.

  modal run eval_wer.py::wer --runs base,gan --n 500

Transcribes the original audio (the ASR floor) and each run's reconstruction from the first 1/2/4/8 quantizers,
scores against the LibriTTS normalized transcripts, prints a table and writes /runs/wer.json.
"""
import json, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None


def load_utts(root, n, workers=32):
    """n utterances spread across test-clean speakers: (waveform at 16 kHz, normalized transcript)."""
    import soundfile as sf, torch, torchaudio
    from train import SR
    files = sorted(Path(root, "test-clean").rglob("*.wav"))
    files = files[::max(1, len(files) // n)][:n]

    def one(f):
        w, fs = sf.read(f, dtype="float32")
        w = torch.from_numpy(w)
        if fs != SR:
            w = torchaudio.functional.resample(w, fs, SR)
        return w, f.with_suffix(".normalized.txt").read_text().strip()

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, files))


def basic_norm(t):
    return " ".join(re.sub(r"[^a-z' ]+", " ", t.lower()).split())


def codec_roundtrip(codec, wavs, nq, dev):
    """Full utterances through encode -> first nq quantizers -> decode; padded to the hop, trimmed back."""
    import torch, torch.nn.functional as F
    out = []
    with torch.no_grad():
        for w in wavs:
            T = len(w)
            x = F.pad(w, (0, (-T) % codec.hop)).view(1, 1, -1).to(dev)
            _, idx, _ = codec.rvq(codec.encode(x))
            out.append(codec.decode_indices(idx[..., :nq])[0, 0, :T].clamp(-1, 1).cpu())
    return out


def score(utts, transcribe, norm, load_codec, runs, nqs, dev):
    """transcribe: list of waveforms -> list of strings. load_codec: run name -> Codec. Returns {condition: wer}."""
    import jiwer
    refs = [norm(t) for _, t in utts]
    results = {"original": jiwer.wer(refs, [norm(t) for t in transcribe([w for w, _ in utts])])}
    print(f"original audio: WER {100 * results['original']:.1f}%", flush=True)
    for run in runs:
        codec = load_codec(run)
        for nq in nqs:
            recon = codec_roundtrip(codec, [w for w, _ in utts], nq, dev)
            key = f"{run} @ {codec.kbps(nq):g} kbps"
            results[key] = jiwer.wer(refs, [norm(t) for t in transcribe(recon)])
            print(f"{key}: WER {100 * results[key]:.1f}%", flush=True)
    return results


if modal:
    app = modal.App("mini-codec-wer")
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch", "torchaudio", "numpy", "soundfile", "vector-quantize-pytorch", "transformers", "jiwer", "accelerate")
             .add_local_python_source("train"))
    data_vol = modal.Volume.from_name("codec-data")
    run_vol = modal.Volume.from_name("codec-runs")

    @app.function(image=image, gpu="A10", volumes={"/data": data_vol, "/runs": run_vol}, memory=8192, timeout=3600)
    def wer(runs: str = "base", n: int = 500, nqs: str = "1,2,4,8", asr_model: str = "openai/whisper-small.en"):
        import torch
        from transformers import pipeline
        from train import Codec, SR
        dev = "cuda"
        utts = load_utts("/data/LibriTTS_R", n)
        print(f"{len(utts)} utterances, {sum(len(w) for w, _ in utts) / SR / 60:.1f} min of audio", flush=True)
        asr = pipeline("automatic-speech-recognition", model=asr_model, device=0, chunk_length_s=30)

        def norm(t):
            try:
                return asr.tokenizer.normalize(t)       # Whisper's English normalizer (numbers, spellings, punctuation)
            except Exception:
                return basic_norm(t)

        def transcribe(wavs):
            out = asr([{"raw": w.numpy(), "sampling_rate": SR} for w in wavs], batch_size=16)
            return [o["text"] for o in out]

        def load_codec(run):
            cfg = json.loads(Path(f"/runs/{run}/cfg.json").read_text())
            codec = Codec(cfg["C"], cfg["D"], n_q=cfg["n_q"], codebook=cfg["codebook"]).to(dev).eval()
            codec.load_state_dict(torch.load(f"/runs/{run}/last.pt", map_location=dev)["model"])
            return codec

        results = score(utts, transcribe, norm, load_codec, runs.split(","), [int(q) for q in nqs.split(",")], dev)
        results["_meta"] = {"n": len(utts), "asr": asr_model}
        Path("/runs/wer.json").write_text(json.dumps(results, indent=1))
        run_vol.commit()
        print(json.dumps(results, indent=1))
