"""Count torch.compile graph compilations across the five bucket shapes: must equal the number of
distinct shapes (5), not grow when shapes repeat."""
import torch, time, sys
import torch._dynamo
from tinydit.model import TinyDiT, CONFIGS
from tinydit import flow
torch._dynamo.config.recompile_limit = 64 if hasattr(torch._dynamo.config, "recompile_limit") else None
m = TinyDiT(latent_ch=32, ctx_dim=768, **CONFIGS["run1"]).cuda()
net = torch.compile(m, dynamic=False)
shapes = [(32, 32), (28, 36), (36, 28), (26, 40), (40, 26)]
from torch._dynamo.utils import counters
t0 = time.time()
for rnd in range(2):
    for H, W in shapes:
        z = torch.randn(8, 32, H, W, device="cuda"); ctx = torch.randn(8, 128, 768, device="cuda")
        msk = torch.ones(8, 128, dtype=torch.bool, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, _, _, _ = flow.loss(net, z, ctx, msk, t_mean=-1.03)
        l.backward()
        print(f"round {rnd} shape {H}x{W}: loss {l.item():.3f}  graphs compiled so far: {counters['frames']['ok']}  ({time.time()-t0:.0f}s)", flush=True)
print("RESULT compiled graphs:", counters["frames"]["ok"], "(expect 5)")
