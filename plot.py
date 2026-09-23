"""Rate-distortion + training curves from a run directory.  Usage: python plot.py runs/base [runs/gan ...]
Writes <run>/curves.png for each run given; with several runs, also compare.png overlaying their final RD curves."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run):
    run = Path(run)
    evals = [json.loads(l) for l in (run / "eval.jsonl").open()]
    logs = [json.loads(l) for l in (run / "log.jsonl").open()]
    return run, evals, logs


def curves(run, evals, logs):
    final = evals[-1]
    nqs = [k for k in final if k.startswith("nq")]
    kbps = [final[k]["kbps"] for k in nqs]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

    ax[0].plot(kbps, [final[k]["mel"] for k in nqs], "o-", color="C0")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(kbps); ax[0].set_xticklabels([f"{k:g}" for k in kbps])
    ax[0].set_xlabel("kbps"); ax[0].set_ylabel("log-mel distance (lower = better)", color="C0")
    ax[0].set_title(f"rate-distortion at step {final['step']} ({final['n']} test clips)")
    a2 = ax[0].twinx(); a2.plot(kbps, [final[k]["sisnr"] for k in nqs], "s--", color="C1"); a2.set_ylabel("SI-SNR dB (higher = better)", color="C1")

    for k in nqs:
        ax[1].plot([e["step"] for e in evals], [e[k]["mel"] for e in evals], "o-", ms=3, label=f"{final[k]['kbps']:g} kbps")
    ax[1].set_ylim(0, 0.8); ax[1].set_xlabel("step"); ax[1].set_ylabel("eval log-mel distance"); ax[1].set_title("eval during training (step 0 is off the chart)"); ax[1].legend()

    steps, mel = np.array([l["step"] for l in logs]), np.array([l["mel"] for l in logs])
    w = min(20, len(mel))
    ax[2].plot(steps, mel, alpha=0.25, label="per batch (random # quantizers)")
    ax[2].plot(steps[w - 1:], np.convolve(mel, np.ones(w) / w, mode="valid"), label=f"moving average ({w})")
    ax[2].set_ylim(0, 0.8); ax[2].set_xlabel("step"); ax[2].set_ylabel("train log-mel loss"); ax[2].set_title(f"training ({np.mean([l['ms'] for l in logs]):.0f} ms/step)"); ax[2].legend()
    fig.suptitle(run.name); fig.tight_layout(); fig.savefig(run / "curves.png", dpi=150); plt.close(fig)
    print("wrote", run / "curves.png")


def compare(runs):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for run, evals, _ in runs:
        final = evals[-1]; nqs = [k for k in final if k.startswith("nq")]; kbps = [final[k]["kbps"] for k in nqs]
        ax[0].plot(kbps, [final[k]["mel"] for k in nqs], "o-", label=run.name)
        ax[1].plot(kbps, [final[k]["sisnr"] for k in nqs], "s-", label=run.name)
    for a, yl in zip(ax, ("log-mel distance (lower = better)", "SI-SNR dB (higher = better)")):
        a.set_xscale("log", base=2); a.set_xticks(kbps); a.set_xticklabels([f"{k:g}" for k in kbps]); a.set_xlabel("kbps"); a.set_ylabel(yl); a.legend()
    fig.tight_layout(); fig.savefig("compare.png", dpi=150); print("wrote compare.png")


if __name__ == "__main__":
    runs = [load(r) for r in (sys.argv[1:] or ["runs/base"])]
    for r in runs:
        curves(*r)
    if len(runs) > 1:
        compare(runs)
