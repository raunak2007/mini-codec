"""Codebook utilization per quantizer on test-clean: how many of the 1024 codes each quantizer actually uses.

  modal run eval_usage.py::usage --runs base,gan_soft --n 500

Reports, per quantizer: distinct codes used across all frames, perplexity (exp of the code-histogram entropy, the
effective codebook size), and duplicate rows in the trained codebook. Writes /runs/usage.json.
"""
import json
from pathlib import Path

from eval_wer import load_utts

try:
    import modal
except ImportError:
    modal = None


def usage_stats(codec, wavs, dev):
    import torch, torch.nn.functional as F
    n_q, K = codec.rvq.num_quantizers, codec.rvq.codebook_size
    counts = torch.zeros(n_q, K, dtype=torch.long)
    with torch.no_grad():
        for w in wavs:
            x = F.pad(w, (0, (-len(w)) % codec.hop)).view(1, 1, -1).to(dev)
            _, idx, _ = codec.rvq(codec.encode(x))                  # (1, T, n_q)
            for q in range(n_q):
                counts[q] += torch.bincount(idx[0, :, q].cpu(), minlength=K)
    out = {}
    for q, layer in enumerate(codec.rvq.layers):
        p = counts[q].double() / counts[q].sum()
        ent = -(p[p > 0] * p[p > 0].log()).sum()
        embed = layer._codebook.embed.detach().flatten(0, -2)
        out[f"q{q + 1}"] = {"used": int((counts[q] > 0).sum()), "perplexity": round(float(ent.exp()), 1),
                            "duplicate_rows": int(embed.shape[0] - embed.unique(dim=0).shape[0]), "codebook_size": K}
    return out


if modal:
    app = modal.App("mini-codec-usage")
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch", "torchaudio", "numpy", "soundfile", "vector-quantize-pytorch")
             .add_local_python_source("train", "eval_wer"))
    data_vol = modal.Volume.from_name("codec-data")
    run_vol = modal.Volume.from_name("codec-runs")

    @app.function(image=image, gpu="A10", volumes={"/data": data_vol, "/runs": run_vol}, memory=8192, timeout=1800)
    def usage(runs: str = "base", n: int = 500):
        import torch
        from train import Codec
        dev = "cuda"
        wavs = [w for w, _ in load_utts("/data/LibriTTS_R", n)]
        results = {}
        for run in runs.split(","):
            cfg = json.loads(Path(f"/runs/{run}/cfg.json").read_text())
            codec = Codec(cfg["C"], cfg["D"], n_q=cfg["n_q"], codebook=cfg["codebook"]).to(dev).eval()
            codec.load_state_dict(torch.load(f"/runs/{run}/last.pt", map_location=dev)["model"])
            results[run] = usage_stats(codec, wavs, dev)
            print(f"{run}:", flush=True)
            for q, s in results[run].items():
                print(f"  {q}: {s['used']}/{s['codebook_size']} codes used, perplexity {s['perplexity']}, "
                      f"{s['duplicate_rows']} duplicate rows", flush=True)
        results["_meta"] = {"n": len(wavs)}
        Path("/runs/usage.json").write_text(json.dumps(results, indent=1))
        run_vol.commit()
