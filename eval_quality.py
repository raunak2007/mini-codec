"""Objective voice-quality proxies for codec reconstructions, per bitrate, on speaker-disjoint test-clean.
No ASR involved; these track what the discriminator changes and WER does not measure.

  modal run eval_quality.py::quality --runs base,gan,gan_soft --n 500

Per condition, averaged over utterances:
  cpp      cepstral peak prominence on voiced frames (harmonic clarity; low = breathy/rough). Reported next to the original's.
  balance  mean absolute band-energy deviation from the original over 0-1, 1-2, 2-4, 4-8 kHz, in dB (0 = same spectral tilt).
  flat     spectral flatness of the 4-8 kHz band on voiced frames (1 = noise, lower = structured). Reported next to the original's.
Voiced frames are chosen on the original (top 40% by CPP) and the same frames are scored for every condition.
"""
import json
from pathlib import Path
import numpy as np

try:
    import modal
except ImportError:
    modal = None

SR, WIN, HOP, NFFT = 16000, 640, 160, 1024
BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def frame_spectra(w):                        # (T,) -> log |spectrum| (frames, NFFT/2+1) and power (frames, NFFT/2+1)
    n = (len(w) - WIN) // HOP
    f = np.stack([w[i * HOP:i * HOP + WIN] for i in range(n)]) * np.hanning(WIN)
    mag = np.abs(np.fft.rfft(f, NFFT))
    return np.log(mag + 1e-8), mag ** 2


def cpp(logmag):                             # cepstral peak prominence per frame, vectorised
    c = np.fft.irfft(logmag, NFFT)[:, :NFFT // 2]
    q = np.arange(NFFT // 2) / SR
    lo, hi = np.searchsorted(q, 1 / 400), np.searchsorted(q, 1 / 80)    # 80-400 Hz pitch range
    cw, qw = c[:, lo:hi], q[lo:hi]
    a, b = np.polyfit(qw, cw.T, 1)                                       # linear trend per frame
    i = cw.argmax(1)
    return cw.max(1) - (a * qw[i] + b)


def flatness(power):                         # spectral flatness of the 4-8 kHz band per frame
    lo = int(4000 / SR * NFFT)
    p = power[:, lo:] + 1e-12
    return np.exp(np.log(p).mean(1)) / p.mean(1)


def band_dev(power, power_ref):              # mean |dB deviation| of band energy from the reference
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    devs = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        devs.append(abs(10 * np.log10(power[:, m].mean() / (power_ref[:, m].mean() + 1e-12) + 1e-12)))
    return float(np.mean(devs))


def measure(orig, recon):
    """orig, recon: 1-D float arrays of equal length. Returns per-utterance proxies."""
    lo, po = frame_spectra(orig)
    lr, pr = frame_spectra(recon)
    c0 = cpp(lo)
    voiced = c0 > np.percentile(c0, 60)
    return dict(cpp=float(cpp(lr)[voiced].mean()), cpp_ref=float(c0[voiced].mean()),
                flat=float(flatness(pr)[voiced].mean()), flat_ref=float(flatness(po)[voiced].mean()),
                balance=band_dev(pr, po))


def summarise(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


if modal:
    app = modal.App("mini-codec-quality")
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch", "torchaudio", "numpy", "soundfile", "vector-quantize-pytorch")
             .add_local_python_source("train", "eval_wer"))
    data_vol = modal.Volume.from_name("codec-data")
    run_vol = modal.Volume.from_name("codec-runs")

    @app.function(image=image, gpu="A10", volumes={"/data": data_vol, "/runs": run_vol}, memory=8192, timeout=3600)
    def quality(runs: str = "base", n: int = 500, nqs: str = "1,2,4,8"):
        import torch
        from train import Codec
        from eval_wer import load_utts, codec_roundtrip
        dev = "cuda"
        utts = load_utts("/data/LibriTTS_R", n)
        wavs = [w for w, _ in utts]
        results = {}
        print(f"{'condition':22s} {'cpp':>6s} {'cpp_ref':>8s} {'flat':>6s} {'flat_ref':>9s} {'balance dB':>11s}", flush=True)
        for run in runs.split(","):
            cfg = json.loads(Path(f"/runs/{run}/cfg.json").read_text())
            codec = Codec(cfg["C"], cfg["D"], n_q=cfg["n_q"], codebook=cfg["codebook"]).to(dev).eval()
            codec.load_state_dict(torch.load(f"/runs/{run}/last.pt", map_location=dev)["model"])
            for nq in [int(q) for q in nqs.split(",")]:
                recon = codec_roundtrip(codec, wavs, nq, dev)
                key = f"{run} @ {codec.kbps(nq):g} kbps"
                results[key] = s = summarise([measure(w.numpy(), y.numpy()) for w, y in zip(wavs, recon)])
                print(f"{key:22s} {s['cpp']:6.3f} {s['cpp_ref']:8.3f} {s['flat']:6.3f} {s['flat_ref']:9.3f} {s['balance']:11.2f}", flush=True)
        results["_meta"] = {"n": len(utts)}
        Path("/runs/quality.json").write_text(json.dumps(results, indent=1))
        run_vol.commit()
