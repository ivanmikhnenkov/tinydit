"""Turn out/reward_eval/scores/*.json into the tables of the reward study.
Tests: realism (AUROC real vs FLUX-render, real vs tinydit), alignment (AUROC own vs mismatched caption), within-group
signal-to-noise on tinydit samples (std over 8 seeds / JPEG-90 re-encode noise), rank agreement between scorers inside a
prompt group, artifact detection (AUROC artifact vs clean on the hand-labelled subset), and speed.
    .venv/bin/python scripts/reward_eval/analyze.py            -> out/reward_eval/analysis.json + printed tables
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr
E = "out/reward_eval"; M = json.load(open(f"{E}/manifest.json"))
S = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{E}/scores/*.json") if not p.endswith("_hardneg.json")}
HN = {os.path.basename(p)[:-13]: json.load(open(p)) for p in glob.glob(f"{E}/scores/*_hardneg.json")}
LAB = json.load(open(f"{E}/labels.json")) if os.path.exists(f"{E}/labels.json") else {}
def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float); pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg): return None
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
def col(scorer, group, key):
    d = S[scorer].get(group, {}); return {f: v.get(key) for f, v in d.items() if isinstance(v, dict) and v.get(key) is not None}
# direction: +1 = higher means better/more real/aligned, -1 = higher means worse (artifact, aislop)
DIR = {"artifact": -1, "aislop": -1}
def detect_auroc(pos_bad, neg_good, key):
    """AUROC for detecting the bad class: for quality metrics bad should score lower."""
    a = auroc(pos_bad, neg_good)
    return None if a is None else (a if DIR.get(key, 1) < 0 else 1 - a)
# which keys are "higher = more real" / "higher = better aligned" / "higher = more artifacts"
REAL = {"cheap": ["pickscore", "hpsv2", "dino_real", "sat", "sharp", "contrast"], "hpsv3": ["hps"],
        "*vlm": ["real", "aislop", "rubric_realism", "rubric_integrity", "artifact"]}
ALIGN = {"cheap": [("clip", "clip_mis"), ("pickscore", "pickscore_mis"), ("hpsv2", "hpsv2_mis")], "hpsv3": [("hps", "hps_mis")], "*vlm": [("align", "align_mis"), ("rubric_alignment", None)]}
NOISE = {"cheap": [("pickscore", "pickscore_jpeg"), ("hpsv2", "hpsv2_jpeg")], "hpsv3": [("hps", "hps_jpeg")], "*vlm": [("align", "align_jpeg"), ("artifact", "artifact_jpeg"), ("real", "real_jpeg")]}
def keys_for(table, scorer): return table.get(scorer) or (table.get("*vlm", []) if scorer not in ("cheap", "hpsv3") else [])
out = {"realism": {}, "alignment": {}, "hardneg": {}, "group_snr": {}, "agreement": {}, "artifact": {}, "by_source": {}, "timing": {}}
for sc, d in HN.items():
    keys = sorted({k for v in d.values() for k in v["own"]})
    for k in keys:
        allp = [(v["own"][k], [n[k] for n in v["neg"]]) for f, v in d.items()]
        alp = [(v["own"][k], [n[k] for n in v["neg"]]) for f, v in d.items() if LAB.get(f, {}).get("align", 1) == 1]
        def pair_auroc(rows): return float(np.mean([(o > n) + 0.5 * (o == n) for o, ns in rows for n in ns]))
        out["hardneg"][f"{sc}.{k}"] = dict(all=pair_auroc(allp), aligned_only=pair_auroc(alp), n=len(allp))
for sc in S:
    if "_timing" in S[sc]: out["timing"][sc] = S[sc]["_timing"]
    for k in keys_for(REAL, sc):
        r, f, s = col(sc, "ref_real", k), col(sc, "ref_flux", k), col(sc, "samples", k)
        if k == "dino_real":       # only the held-out half of the references is a fair test
            tr = {g: {ff for ff, v in S[sc][g].items() if isinstance(v, dict) and v.get("dino_probe_train")} for g in ("ref_real", "ref_flux")}
            r = {ff: v for ff, v in r.items() if ff not in tr["ref_real"]}; f = {ff: v for ff, v in f.items() if ff not in tr["ref_flux"]}
        if r and f:
            flip = (lambda a: None if a is None else (1 - a if DIR.get(k, 1) < 0 else a))
            out["realism"][f"{sc}.{k}"] = dict(real_vs_flux=flip(auroc(list(r.values()), list(f.values()))), real_vs_tinydit=flip(auroc(list(r.values()), list(s.values()))) if s else None,
                                           flux_vs_tinydit=flip(auroc(list(f.values()), list(s.values()))) if s else None,
                                           mean_real=float(np.mean(list(r.values()))), mean_flux=float(np.mean(list(f.values()))), mean_tinydit=float(np.mean(list(s.values()))) if s else None)
    for k, km in keys_for(ALIGN, sc):
        rec = {}
        for g in ("ref_real", "ref_flux", "samples"):
            a = col(sc, g, k); b = col(sc, g, km) if km else {}
            if a and b: rec[g] = auroc([a[f] for f in a if f in b], [b[f] for f in a if f in b])
        if rec: out["alignment"][f"{sc}.{k}"] = rec
    # within-group signal on tinydit samples: std over the 8 seeds of a prompt, vs JPEG-90 noise where available
    groups = {}
    for smp in M["samples"]: groups.setdefault(smp["id"], []).append(smp["file"])
    cand = [k for k in (keys_for(REAL, sc) + [k for k, _ in keys_for(ALIGN, sc)]) if col(sc, "samples", k)]
    for k in dict.fromkeys(cand):
        v = col(sc, "samples", k)
        stds = [np.std([v[f] for f in fs if f in v]) for fs in groups.values() if sum(f in v for f in fs) >= 4]
        rng = np.std(list(v.values()))
        rec = dict(group_std=float(np.mean(stds)) if stds else None, global_std=float(rng))
        for kk, kj in keys_for(NOISE, sc):
            if kk == k:
                vj = col(sc, "samples", kj); d = [abs(v[f] - vj[f]) for f in v if f in vj]
                rec["noise_jpeg"] = float(np.mean(d)); rec["snr"] = rec["group_std"] / max(rec["noise_jpeg"], 1e-9) if rec["group_std"] else None
        out["group_snr"][f"{sc}.{k}"] = rec
    # per prompt source
    for k in dict.fromkeys(cand):
        v = col(sc, "samples", k); bysrc = {}
        for smp in M["samples"]:
            if smp["file"] in v: bysrc.setdefault(smp["source"], []).append(v[smp["file"]])
        out["by_source"][f"{sc}.{k}"] = {s_: float(np.mean(x)) for s_, x in bysrc.items()}
    # artifact detection on the labelled subset (label: 1 = artifact, 0 = clean; uncertain excluded)
    if LAB:
        for k in dict.fromkeys(cand):
            v = col(sc, "samples", k)
            pos = [v[f] for f, l in LAB.items() if l.get("label") == 1 and f in v]; neg = [v[f] for f, l in LAB.items() if l.get("label") == 0 and f in v]
            if pos and neg: out["artifact"][f"{sc}.{k}"] = dict(auroc_detect=detect_auroc(pos, neg, k), n_art=len(pos), n_clean=len(neg))
            posa = [v[f] for f, l in LAB.items() if l.get("align") == 0 and f in v]; nega = [v[f] for f, l in LAB.items() if l.get("align") == 1 and f in v]
            if posa and nega: out.setdefault("misalign", {})[f"{sc}.{k}"] = dict(auroc_detect=detect_auroc(posa, nega, k), n_fail=len(posa), n_ok=len(nega))
# free-form descriptions on the labelled set: "none"-like answers = no error flagged
import re as _re
def flagged(txt):
    """True if the description names an error. 'none', 'None.', and sentences like 'there are no obvious visual errors or
    artifacts' (the 27B's phrasing) count as not flagged; anything that goes on to list errors counts as flagged."""
    t = (txt or "").strip().lower()
    if t == "" or t.startswith("none"): return False
    head = t[:220]
    neg = _re.search(r"\b(no|without any|free of|does not (contain|have|show))\b[^.]{0,60}\b(errors?|artifacts?|issues?|problems?)", head)
    lists = _re.search(r"here are (the |some |a few )?(potential |possible |notable |several )?(visual )?(errors|artifacts|issues)|"
                       r"(contains|has|shows|exhibits) (the following|several|some|a few|multiple|notable) (visual )?(errors|artifacts|issues)|"
                       r"the following (visual )?(errors|artifacts|issues)", head)
    return not (neg and not lists)
for sc in S:
    d = S[sc].get("samples", {}); rows = [(f, LAB[f]["label"], flagged(d[f]["describe"])) for f in LAB if f in d and "describe" in d[f] and LAB[f]["label"] is not None]
    if rows:
        art = [fl for f, l, fl in rows if l == 1]; cln = [fl for f, l, fl in rows if l == 0]
        out.setdefault("describe", {})[sc] = dict(recall_artifacts=float(np.mean(art)), false_positive_rate=float(np.mean(cln)), n_art=len(art), n_clean=len(cln),
                                                 auroc=auroc([float(x) for x in art], [float(x) for x in cln]))
# rank agreement inside prompt groups between scorers (Spearman over the 8 seeds, averaged over prompts)
allk = [(sc, k) for sc in S for k in (out["group_snr"]) if k.startswith(sc + ".")]
names = sorted({k for k in out["group_snr"]})
groups = {}
for smp in M["samples"]: groups.setdefault(smp["id"], []).append(smp["file"])
def vec(name): sc, k = name.split(".", 1); return col(sc, "samples", k)
V = {n: vec(n) for n in names}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        rs = []
        for fs in groups.values():
            fs2 = [f for f in fs if f in V[a] and f in V[b]]
            if len(fs2) >= 6:
                x = [V[a][f] for f in fs2]; y = [V[b][f] for f in fs2]
                if np.std(x) > 0 and np.std(y) > 0: rs.append(spearmanr(x, y).correlation)
        if rs: out["agreement"][f"{a} ~ {b}"] = float(np.nanmean(rs))
json.dump(out, open(f"{E}/analysis.json", "w"), indent=1)
def show(title, d, keys):
    print(f"\n== {title}"); 
    for n, r in d.items(): print(f"  {n:34s} " + "  ".join(f"{k} {r[k]:.3f}" if isinstance(r.get(k), float) else f"{k} {r.get(k)}" for k in keys))
show("realism (AUROC, higher = scorer rates real higher)", out["realism"], ["real_vs_flux", "real_vs_tinydit", "flux_vs_tinydit"])
show("alignment (AUROC own caption vs mismatched)", out["alignment"], ["ref_real", "ref_flux", "samples"])
show("hard-negative alignment (P(own prompt scores above a minimally edited one))", out["hardneg"], ["all", "aligned_only", "n"])
show("within-group signal on tinydit samples", out["group_snr"], ["group_std", "noise_jpeg", "snr"])
show("artifact detection (AUROC for flagging the 21 labelled artifacts, direction-aware)", out["artifact"], ["auroc_detect", "n_art", "n_clean"])
show("alignment-failure detection (AUROC for flagging the 14 labelled failures)", out.get("misalign", {}), ["auroc_detect", "n_fail", "n_ok"])
show("free-form descriptions: flags an error? (labelled set)", out.get("describe", {}), ["recall_artifacts", "false_positive_rate", "auroc", "n_art", "n_clean"])
print("\n== rank agreement inside prompt groups (mean Spearman)"); [print(f"  {k:60s} {v:+.3f}") for k, v in out["agreement"].items()]
print("\n== by prompt source"); [print(f"  {k:28s} " + "  ".join(f"{s} {x:.3f}" for s, x in v.items())) for k, v in out["by_source"].items()]
