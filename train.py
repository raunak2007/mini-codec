"""mini-codec: a SoundStream-style neural audio codec at 16 kHz.

conv encoder (hop 320 -> 50 frames/s) -> residual VQ with quantizer dropout -> conv decoder.
Loss: multi-scale log-mel L1 + waveform L1 + VQ commitment, plus (with --w_adv > 0) a multi-scale STFT
discriminator with hinge adversarial loss and feature matching (EnCodec-style discriminator, HiFi-GAN-style weights).
One training run gives the whole bitrate curve: eval decodes from the first 1/2/4/8 quantizers = 0.5-4 kbps.

local debug (MPS/CPU):
  python train.py --run debug --max_files 200 --steps 200 --eval_split dev-clean --eval_clips 32 --eval_every 100 --ckpt_every 100
modal (one-off CPU preprocessing, then a detached GPU run):
  modal run train.py::prep
  modal run --detach train.py::train --steps 50000 --run base
  modal run --detach train.py::train --run gan --init /runs/base/last.pt --w_adv 0.1 --steps 25000   # adversarial fine-tune
  modal app logs ap-XXXX                         # follow the run (use the app id printed at launch)
  modal volume get codec-runs base ./runs/base   # checkpoint, log.jsonl, eval.jsonl, eval_final.json, samples/
"""
import argparse, json, math, os, random, time, warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import soundfile as sf
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
from vector_quantize_pytorch import ResidualVQ

warnings.filterwarnings("ignore", message=".*mel filterbank.*")
SR = 16000


@dataclass
class Cfg:
    root: str = "data/LibriTTS_R"   # <root>/<split>/<speaker>/<chapter>/*.wav
    cache: str = "data/cache"       # resampled int16 clips, one .pt per split
    out: str = "checkpoints"        # <out>/<run>/{last.pt, log.jsonl, eval.jsonl, eval_final.json, samples/}
    run: str = "base"
    train_split: str = "dev-clean"
    eval_split: str = "test-clean"  # speaker-disjoint from dev-clean
    max_files: int = 0              # 0 = all files; 200 for local debugging
    steps: int = 50000
    batch: int = 16
    seg: int = SR                   # 1 s random crops for training
    eval_seg: int = 2 * SR          # 2 s centre crops for eval
    eval_clips: int = 512           # periodic-eval subset (spread across speakers); final eval uses everything
    lr: float = 3e-4
    warmup: int = 500
    w_mel: float = 1.0
    w_wave: float = 1.0
    w_commit: float = 1.0
    w_adv: float = 0.0              # > 0 enables the discriminator; 0.1 (with w_fm 0.3) gives the adversarial and
    w_fm: float = 0.3               # feature-matching terms roughly DAC's share of the total loss next to the mel term
    init: str = ""                  # warm-start the codec from a checkpoint (weights only, step counter restarts)
    C: int = 32                     # base channels: 32-64-128-256-512
    D: int = 128                    # latent / codebook dim
    n_q: int = 8                    # quantizers; each adds 0.5 kbps (10-bit codes at 50 Hz)
    codebook: int = 1024
    log_every: int = 50
    eval_every: int = 2500
    ckpt_every: int = 2000
    seed: int = 0
    device: str = ""                # auto: cuda > mps > cpu


# ----------------------------------------------------------------------------- model
class ResUnit(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.net = nn.Sequential(nn.ELU(), nn.Conv1d(ch, ch, 7, dilation=dil, padding=3 * dil), nn.ELU(), nn.Conv1d(ch, ch, 1))

    def forward(self, x):
        return x + self.net(x)


def res_stack(ch):
    return [ResUnit(ch, 1), ResUnit(ch, 3), ResUnit(ch, 9)]


class Codec(nn.Module):
    """Strides (2,4,5,8) -> hop 320 = 50 frames/s at 16 kHz. Lengths are preserved exactly when T % 320 == 0."""

    def __init__(self, C=32, D=128, strides=(2, 4, 5, 8), n_q=8, codebook=1024):
        super().__init__()
        ch = [C * 2 ** i for i in range(len(strides) + 1)]
        enc = [nn.Conv1d(1, ch[0], 7, padding=3)]
        for i, s in enumerate(strides):
            enc += res_stack(ch[i]) + [nn.ELU(), nn.Conv1d(ch[i], ch[i + 1], 2 * s, stride=s, padding=(s + 1) // 2)]
        enc += [nn.ELU(), nn.Conv1d(ch[-1], D, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)
        self.rvq = ResidualVQ(dim=D, num_quantizers=n_q, codebook_size=codebook, kmeans_init=True, kmeans_iters=10,
                              threshold_ema_dead_code=2, quantize_dropout=True, quantize_dropout_cutoff_index=0)
        dec = [nn.Conv1d(D, ch[-1], 7, padding=3)]
        for i, s in reversed(list(enumerate(strides))):
            dec += [nn.ELU(), nn.ConvTranspose1d(ch[i + 1], ch[i], 2 * s, stride=s, padding=(s + 1) // 2, output_padding=s % 2)]
            dec += res_stack(ch[i])
        dec += [nn.ELU(), nn.Conv1d(ch[0], 1, 7, padding=3)]
        self.decoder = nn.Sequential(*dec)
        self.hop, self.bits = math.prod(strides), math.log2(codebook)

    def encode(self, x):                     # (B,1,T) -> (B,T/hop,D)
        return self.encoder(x).transpose(1, 2)

    def decode(self, z):                     # (B,T/hop,D) -> (B,1,T)
        return self.decoder(z.transpose(1, 2))

    def forward(self, x):
        zq, idx, commit = self.rvq(self.encode(x))   # commit: (n_q,), zeros for dropped quantizers
        return self.decode(zq), commit.sum(), idx

    def decode_indices(self, idx):           # idx (B,T/hop,q) with q <= n_q: reconstruct from the first q quantizers
        return self.decode(self.rvq.get_output_from_indices(idx))

    def kbps(self, n):
        return n * self.bits * SR / self.hop / 1000


class MelLoss(nn.Module):
    """Mean over scales of L1 between log-mel spectrograms."""

    def __init__(self, ffts=(512, 1024, 2048), n_mels=64):
        super().__init__()
        self.mels = nn.ModuleList([torchaudio.transforms.MelSpectrogram(SR, n_fft=n, hop_length=n // 4, n_mels=n_mels, power=1.0) for n in ffts])

    def forward(self, y, x):
        y, x = y.squeeze(1), x.squeeze(1)
        return sum((m(y).add(1e-5).log() - m(x).add(1e-5).log()).abs().mean() for m in self.mels) / len(self.mels)


def si_snr(y, x, eps=1e-8):                  # (B,1,T) -> (B,) in dB
    y, x = y.flatten(1), x.flatten(1)
    y, x = y - y.mean(1, keepdim=True), x - x.mean(1, keepdim=True)
    s = (y * x).sum(1, keepdim=True) / (x.pow(2).sum(1, keepdim=True) + eps) * x
    return 10 * torch.log10(s.pow(2).sum(1) / ((y - s).pow(2).sum(1) + eps) + eps)


class STFTDisc(nn.Module):
    """One scale of an EnCodec-style multi-scale STFT discriminator: 2-D convs over the complex spectrogram
    laid out as (batch, 2, frames, bins); strides shrink bins then frames, dilations widen the time context."""

    def __init__(self, n_fft, C=32):
        super().__init__()
        self.n_fft, self.hop = n_fft, n_fft // 4
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        wn = nn.utils.parametrizations.weight_norm
        specs = [(2, C, (1, 1), 1), (C, C, (1, 2), 1), (C, C, (2, 2), 2), (C, C, (2, 2), 4), (C, C, (1, 1), 1)]
        self.convs = nn.ModuleList([wn(nn.Conv2d(ci, co, (3, 9), stride=st, dilation=(d, 1), padding=(d, 4))) for ci, co, st, d in specs])
        self.out = wn(nn.Conv2d(C, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):                    # (B,1,T) -> logits map, list of feature maps
        z = torch.stft(x.squeeze(1), self.n_fft, self.hop, self.n_fft, self.window, return_complex=True)
        z = torch.stack([z.real, z.imag], 1).transpose(2, 3)
        feats = []
        for c in self.convs:
            z = F.leaky_relu(c(z), 0.2)
            feats.append(z)
        return self.out(z), feats


def make_disc(ffts=(1024, 512, 256, 128)):
    return nn.ModuleList([STFTDisc(n) for n in ffts])


def disc_loss(disc, x, y):                   # hinge; y is the generator output, detached here
    y = y.detach()
    return sum(F.relu(1 - d(x)[0]).mean() + F.relu(1 + d(y)[0]).mean() for d in disc) / len(disc)


def gen_losses(disc, x, y):                  # generator hinge term + feature matching against the real features
    adv, fm = 0.0, 0.0
    for d in disc:
        with torch.no_grad():
            _, fr = d(x)
        lf, ff = d(y)
        adv = adv + F.relu(1 - lf).mean()
        fm = fm + sum((a - b).abs().mean() for a, b in zip(fr, ff))
    return adv / len(disc), fm / len(disc)


# ----------------------------------------------------------------------------- data
def _load_one(f):
    w, fs = sf.read(f, dtype="float32")
    w = torch.from_numpy(w)
    if fs != SR:
        w = torchaudio.functional.resample(w, fs, SR)
    return (w.clamp(-1, 1) * 32767).round().to(torch.int16)


def load_split(root, split, cache_dir, max_files=0, workers=32):
    """Returns a list of int16 1-D tensors at SR. Resampling from 24 kHz happens once; the result is cached.
    Reads are threaded: on a Modal volume the per-file latency dominates, not the resample."""
    files = sorted(Path(root, split).rglob("*.wav"))
    if max_files:
        files = files[:max_files]
    assert files, f"no wav files under {Path(root, split)}"
    cache = Path(cache_dir, f"{split}_{SR}_{len(files)}.pt")
    if cache.exists():
        return torch.load(cache)
    print(f"loading {len(files)} files from {split}, resampling to {SR} Hz ...", flush=True)
    clips = []
    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(files), 1000):
            clips += list(ex.map(_load_one, files[i:i + 1000]))
            print(f"  {min(i + 1000, len(files))}/{len(files)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clips, cache)
    print(f"{split}: {len(clips)} clips, {sum(len(c) for c in clips) / SR / 3600:.2f} h -> {cache}", flush=True)
    return clips


class Batcher:
    """Random 1 s crops from random utterances, sampled in the main thread (cheap: int16 slices)."""

    def __init__(self, clips, seg, batch, seed):
        self.clips = [c for c in clips if len(c) >= seg]
        self.seg, self.batch, self.rng = seg, batch, random.Random(seed)

    def __call__(self):
        x = torch.empty(self.batch, 1, self.seg)
        for i in range(self.batch):
            c = self.rng.choice(self.clips)
            s = self.rng.randrange(len(c) - self.seg + 1)
            x[i, 0] = c[s:s + self.seg].float() / 32767
        return x


def make_eval(clips, seg):                   # centre crop of every clip long enough -> int16 (N, seg)
    return torch.stack([c[(len(c) - seg) // 2:][:seg] for c in clips if len(c) >= seg])


# ----------------------------------------------------------------------------- eval / io
@torch.no_grad()
def evaluate(model, mel, eval_x, dev, nqs, bs=32):
    model.eval()
    tot = {n: [0.0, 0.0] for n in nqs}
    for i in range(0, len(eval_x), bs):
        x = (eval_x[i:i + bs].float() / 32767).unsqueeze(1).to(dev)
        _, idx, _ = model.rvq(model.encode(x))
        for n in nqs:
            y = model.decode_indices(idx[..., :n])
            tot[n][0] += mel(y, x).item() * len(x)
            tot[n][1] += si_snr(y, x).sum().item()
    model.train()
    return {f"nq{n}": {"kbps": model.kbps(n), "mel": m / len(eval_x), "sisnr": s / len(eval_x)} for n, (m, s) in tot.items()}


@torch.no_grad()
def save_samples(model, sample_x, dev, out, step, nqs):
    sdir = out / "samples"
    sdir.mkdir(exist_ok=True)
    x = (sample_x.float() / 32767).unsqueeze(1).to(dev)
    for i in range(len(x)):
        if not (sdir / f"orig_{i}.wav").exists():
            sf.write(sdir / f"orig_{i}.wav", x[i, 0].cpu().numpy(), SR)
    model.eval()
    _, idx, _ = model.rvq(model.encode(x))
    for n in nqs:
        y = model.decode_indices(idx[..., :n]).clamp(-1, 1)
        for i in range(len(y)):
            sf.write(sdir / f"step{step:06d}_nq{n}_{i}.wav", y[i, 0].cpu().numpy(), SR)
    model.train()


def save_ckpt(path, model, opt, step, disc=None, opt_d=None):
    tmp = path.with_suffix(".tmp")
    sd = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
    if disc is not None:
        sd.update(disc=disc.state_dict(), opt_d=opt_d.state_dict())
    torch.save(sd, tmp)
    tmp.replace(path)


# ----------------------------------------------------------------------------- train
def train_loop(cfg: Cfg, on_save=lambda: None):
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    dev = cfg.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True
    out = Path(cfg.out, cfg.run)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfg.json").write_text(json.dumps(asdict(cfg), indent=1))

    train_clips = load_split(cfg.root, cfg.train_split, cfg.cache, cfg.max_files)
    eval_clips = train_clips if cfg.eval_split == cfg.train_split else load_split(cfg.root, cfg.eval_split, cfg.cache, cfg.max_files)
    eval_all = make_eval(eval_clips, cfg.eval_seg)
    del eval_clips
    eval_sub = eval_all[::max(1, len(eval_all) // cfg.eval_clips)][:cfg.eval_clips] if cfg.eval_clips else eval_all

    model = Codec(cfg.C, cfg.D, n_q=cfg.n_q, codebook=cfg.codebook).to(dev)
    mel = MelLoss().to(dev)
    betas = (0.8, 0.99) if cfg.w_adv > 0 else (0.9, 0.99)   # lower momentum for adversarial training (HiFi-GAN / DAC)
    opt = torch.optim.Adam(model.parameters(), cfg.lr, betas=betas)
    disc = opt_d = None
    if cfg.w_adv > 0:
        disc = make_disc().to(dev)
        opt_d = torch.optim.Adam(disc.parameters(), cfg.lr, betas=betas)
    sample_x = eval_all[[len(eval_all) * i // 4 for i in range(4)]]   # fixed clips for the audio samples, spread over speakers
    nqs = sorted({n for n in (1, 2, 4) if n < cfg.n_q} | {cfg.n_q})
    print(f"device {dev} | params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | train clips {len(train_clips)} "
          f"({sum(len(c) for c in train_clips) / SR / 3600:.2f} h) | eval clips {len(eval_sub)}/{len(eval_all)} | "
          f"bitrates {[model.kbps(n) for n in nqs]} kbps | discriminator {'on' if disc else 'off'}", flush=True)

    def run_eval(step, x, tag=""):
        res = evaluate(model, mel, x, dev, nqs)
        with open(out / "eval.jsonl", "a") as f:
            f.write(json.dumps({"step": step, "n": len(x), **res}) + "\n")
        print(f"eval{tag} step {step} ({len(x)} clips): " +
              " | ".join(f"{k} {v['kbps']:.1f}kbps mel {v['mel']:.3f} sisnr {v['sisnr']:.1f}" for k, v in res.items()), flush=True)
        save_samples(model, sample_x, dev, out, step, sorted({min(2, cfg.n_q), cfg.n_q}))
        return res

    ckpt, step = out / "last.pt", 0
    if ckpt.exists():
        sd = torch.load(ckpt, map_location=dev)
        model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"]); step = sd["step"]
        if disc is not None and "disc" in sd:
            disc.load_state_dict(sd["disc"]); opt_d.load_state_dict(sd["opt_d"])
        print(f"resumed {ckpt} at step {step}", flush=True)
    elif cfg.init:
        model.load_state_dict(torch.load(cfg.init, map_location=dev)["model"])
        print(f"initialised codec weights from {cfg.init}", flush=True)
    batcher = Batcher(train_clips, cfg.seg, cfg.batch, cfg.seed + step)
    if step == 0:
        model.eval()
        with torch.no_grad():
            model(batcher().to(dev))         # k-means init of every codebook from one training batch (eval mode: no dropout)
        model.train()
        run_eval(0, eval_sub[:64])           # untrained baseline; also smoke-tests the eval path before a long detached run

    log = open(out / "log.jsonl", "a")
    t_last = time.time()
    while step < cfg.steps:
        lr = cfg.lr * min(1.0, (step + 1) / cfg.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / cfg.steps)))
        for g in opt.param_groups + (opt_d.param_groups if opt_d else []):
            g["lr"] = lr
        x = batcher().to(dev)
        y, commit, _ = model(x)
        if disc is not None:                                  # discriminator step first, on the detached output
            l_d = disc_loss(disc, x, y)
            opt_d.zero_grad(set_to_none=True)
            l_d.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()
            l_adv, l_fm = gen_losses(disc, x, y)
        l_mel, l_wave = mel(y, x), (y - x).abs().mean()
        loss = cfg.w_mel * l_mel + cfg.w_wave * l_wave + cfg.w_commit * commit
        if disc is not None:
            loss = loss + cfg.w_adv * l_adv + cfg.w_fm * l_fm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % cfg.log_every == 0:
            now = time.time(); ms = (now - t_last) / cfg.log_every * 1000; t_last = now
            rec = dict(step=step, loss=loss.item(), mel=l_mel.item(), wave=l_wave.item(), commit=commit.item(), lr=lr, ms=ms)
            line = f"step {step:6d} | loss {rec['loss']:.4f} mel {rec['mel']:.4f} wave {rec['wave']:.4f} commit {rec['commit']:.4f}"
            if disc is not None:
                rec.update(adv=l_adv.item(), fm=l_fm.item(), d=l_d.item())
                line += f" adv {rec['adv']:.3f} fm {rec['fm']:.3f} d {rec['d']:.3f}"
            print(line + f" | lr {lr:.2e} | {ms:.0f} ms/step | eta {(cfg.steps - step) * ms / 60000:.0f} min", flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
        if step % cfg.eval_every == 0 and step < cfg.steps:
            run_eval(step, eval_sub); t_last = time.time()
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            save_ckpt(ckpt, model, opt, step, disc, opt_d); on_save(); t_last = time.time()
    log.close()

    res = run_eval(step, eval_all, " (final, full split)")
    (out / "eval_final.json").write_text(json.dumps(res, indent=1))
    on_save()
    return res


def parse():
    p = argparse.ArgumentParser()
    for f in fields(Cfg):
        p.add_argument(f"--{f.name}", type=f.type, default=f.default)
    return Cfg(**vars(p.parse_args()))



# ----------------------------------------------------------------------------- modal
try:
    import modal
except ImportError:            # local runs don't need it
    modal = None

if modal:

    app = modal.App("mini-codec")
    image = modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install("torch", "torchaudio", "numpy", "soundfile", os.environ.get("VQ_PKG", "vector-quantize-pytorch"))
    data_vol = modal.Volume.from_name("codec-data")
    run_vol = modal.Volume.from_name("codec-runs", create_if_missing=True)
    MODAL_PATHS = dict(root="/data/LibriTTS_R", cache="/data/cache", out="/runs")


    @app.function(image=image, volumes={"/data": data_vol}, cpu=4, memory=8192, timeout=3600)
    def prep():
        """Resample both splits once on CPU and cache them in the data volume (a few minutes, a few cents)."""
        for split in ("dev-clean", "test-clean"):
            load_split(MODAL_PATHS["root"], split, MODAL_PATHS["cache"])
        data_vol.commit()


    @app.function(image=image, gpu="A10", volumes={"/data": data_vol, "/runs": run_vol}, memory=8192, timeout=8 * 3600)
    def train(steps: int = 50000, run: str = "base", batch: int = 16, lr: float = 3e-4, seed: int = 0,
              w_adv: float = 0.0, w_fm: float = 0.3, init: str = ""):
        print("gpu:", torch.cuda.get_device_name(0), flush=True)
        cfg = Cfg(**MODAL_PATHS, steps=steps, run=run, batch=batch, lr=lr, seed=seed, w_adv=w_adv, w_fm=w_fm, init=init)
        train_loop(cfg, on_save=run_vol.commit)
        data_vol.commit()   # in case the cache was built here instead of in prep()


if __name__ == "__main__":
    train_loop(parse())
