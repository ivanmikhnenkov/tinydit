"""HTML rows for the report from analysis.json: one row per scorer / question."""
import json
A = json.load(open("out/reward_eval/analysis.json"))
def g(sec, key, sub=None):
    v = A.get(sec, {}).get(key); 
    if v is None: return None
    return v.get(sub) if sub else v
def f(v, d=2): return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.{d}f}"
def cls(v, lo, hi):
    if v is None or v != v: return ""
    return ' class="n good"' if v >= hi else (' class="n bad"' if v <= lo else ' class="n mid"')
ROWS = [  # label, scorer.key for: T1 real/flux, T1 real/tinydit, T2a samples, T2b hardneg all, T2c failures, T3 artifacts, T4 snr
 ("CLIP score", "cheap.clip", None, "cheap.clip", "cheap.clip", "cheap.clip", "cheap.clip"),
 ("PickScore", "cheap.pickscore", "cheap.pickscore", "cheap.pickscore", "cheap.pickscore", "cheap.pickscore", "cheap.pickscore"),
 ("HPSv2.1", "cheap.hpsv2", "cheap.hpsv2", "cheap.hpsv2", "cheap.hpsv2", "cheap.hpsv2", "cheap.hpsv2"),
 ("HPSv3 (μ)", "hpsv3.hps", "hpsv3.hps", "hpsv3.hps", "hpsv3.hps", "hpsv3.hps", "hpsv3.hps"),
 ("DINOv2 real-vs-render probe", "cheap.dino_real", None, None, "cheap.dino_real", "cheap.dino_real", "cheap.dino_real"),
]
for m, name in (("qwen35_4b", "Qwen3.5-4B"), ("qwen3vl8b", "Qwen3-VL-8B"), ("qwen35_9b", "Qwen3.5-9B"), ("qwen38_27b", "Qwen3.8-27B")):
    ROWS += [(f"{name} · \"shows the prompt?\"", None, f"{m}.align", f"{m}.align", f"{m}.align", f"{m}.align", f"{m}.align"),
             (f"{name} · \"anatomical errors?\"", f"{m}.artifact", None, None, f"{m}.artifact", f"{m}.artifact", f"{m}.artifact"),
             (f"{name} · \"real photograph?\"", f"{m}.real", None, None, f"{m}.real", f"{m}.real", f"{m}.real"),
             (f"{name} · \"AI look?\"", f"{m}.aislop", None, None, f"{m}.aislop", f"{m}.aislop", f"{m}.aislop"),
             (f"{name} · rubric integrity 1–10", f"{m}.rubric_integrity", None, None, f"{m}.rubric_integrity", f"{m}.rubric_integrity", f"{m}.rubric_integrity")]
out = []
for label, k1, k2a, k2b, k2c, k3, k4 in ROWS:
    r1 = g("realism", k1) if k1 else None
    t1a = r1.get("real_vs_flux") if r1 else None; t1b = r1.get("real_vs_tinydit") if r1 else None
    t2a = g("alignment", k2a, "samples") if k2a else None
    t2b = g("hardneg", k2b, "all") if k2b else None
    t2c = g("misalign", k2c, "auroc_detect") if k2c else None
    t3 = g("artifact", k3, "auroc_detect") if k3 else None
    sn = g("group_snr", k4) if k4 else None; t4 = sn.get("snr") if sn else None
    if all(v is None for v in (t1a, t1b, t2a, t2b, t2c, t3, t4)): continue
    out.append(f'<tr><td class="h">{label}</td><td{cls(t1a,.4,.7)}>{f(t1a)}</td><td{cls(t1b,.4,.7)}>{f(t1b)}</td><td{cls(t2a,.9,.98)}>{f(t2a,3)}</td>'
               f'<td{cls(t2b,.9,.97)}>{f(t2b,3)}</td><td{cls(t2c,.65,.85)}>{f(t2c)}</td><td{cls(t3,.6,.75)}>{f(t3)}</td><td{cls(t4,1.5,3)}>{f(t4,1)}</td></tr>')
print("\n".join(out))
# description test rows
print("<!-- describe -->")
for m, v in A.get("describe", {}).items():
    print(f'<tr><td class="h">{m}</td><td class="n">{f(v["recall_artifacts"])}</td><td class="n">{f(v["false_positive_rate"])}</td><td class="n">{f(v["auroc"])}</td></tr>')
