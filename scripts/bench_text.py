"""Cost of encoding captions live with flan-t5-base, per training batch, at several max lengths."""
import time, torch, sys
sys.path.insert(0, "src")
from tinydit import text as T
tok, enc = T.load("out/models/t5")
words = ("a photo of a red bicycle leaning against a brick wall next to a blue door while two people walk past "
         "carrying paper bags and a small dog looks up at a pigeon on the window ledge above the bakery sign ") * 6
caps = [" ".join(words.split()[:k]) for k in [8, 20, 45, 90, 130]] * 52   # 260 mixed-length captions
for L in (32, 64, 128, 192):
    for _ in range(3): T.embed(tok, enc, caps[:256], L)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10): e, m = T.embed(tok, enc, caps[:256], L)
    torch.cuda.synchronize(); dt = (time.time() - t) / 10
    print(f"L={L:4d}: {dt*1000:6.1f} ms per batch of 256  (padding='max_length'; real tokens used {m.sum().item()/256/L*100:.0f}%)")
# pad-to-longest-in-batch instead of fixed max_length
b = tok(caps[:256], padding="longest", truncation=True, max_length=128, return_tensors="pt").to("cuda")
torch.cuda.synchronize(); t = time.time()
for _ in range(10):
    with torch.no_grad(): enc(input_ids=b.input_ids, attention_mask=b.attention_mask)
torch.cuda.synchronize(); print(f"pad-to-longest (this batch -> {b.input_ids.shape[1]} tokens): {(time.time()-t)/10*1000:6.1f} ms")
