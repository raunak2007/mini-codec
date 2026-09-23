import time, torch, torch.nn as nn

def down(ci, co, s): return nn.Sequential(nn.Conv1d(ci, co, 2*s, stride=s, padding=s//2), nn.ELU())
def up(ci, co, s):   return nn.Sequential(nn.Upsample(scale_factor=s), nn.Conv1d(ci, co, 7, padding=3), nn.ELU())

class Codec(nn.Module):
    def __init__(self, C=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(1, C, 7, padding=3), down(C, 2*C, 4), down(2*C, 4*C, 4),
                                 down(4*C, 8*C, 5), down(8*C, 8*C, 4), nn.Conv1d(8*C, 256, 3, padding=1))
        self.dec = nn.Sequential(nn.Conv1d(256, 8*C, 3, padding=1), up(8*C, 8*C, 4), up(8*C, 4*C, 5),
                                 up(4*C, 2*C, 4), up(2*C, C, 4), nn.Conv1d(C, 1, 7, padding=3))
    def forward(self, x): return self.dec(self.enc(x))

def sync(dev):
    if dev == "cuda": torch.cuda.synchronize()
    elif dev == "mps": torch.mps.synchronize()

def bench(dev, batch=16, steps=20):
    m = Codec().to(dev); opt = torch.optim.Adam(m.parameters(), 1e-4)
    x = torch.randn(batch, 1, 16000, device=dev)
    print("params (M):", round(sum(p.numel() for p in m.parameters()) / 1e6, 2))
    for i in range(steps + 5):
        if i == 5: sync(dev); t0 = time.time()
        y = m(x); T = min(x.shape[-1], y.shape[-1])
        loss = (x[..., :T] - y[..., :T]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sync(dev); dt = (time.time() - t0) / steps
    print(f"{dev}: {dt*1000:.0f} ms/step at batch {batch}x1s; 50k steps = {50000*dt/3600:.1f} h")

if __name__ == "__main__":
    bench("mps" if torch.backends.mps.is_available() else "cpu")

import modal
app = modal.App("codec-bench")
@app.function(gpu="T4", image=modal.Image.debian_slim(python_version="3.11").pip_install("torch"), timeout=600)
def remote(): bench("cuda")
