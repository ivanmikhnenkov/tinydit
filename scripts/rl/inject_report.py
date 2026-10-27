"""Slim out/diag/report_data.json and inject it (plus the narrative JSON) into the report template's /*DATA*/ slot.
    .venv/bin/python scripts/rl/inject_report.py <template.html> <out.html> [text.json]"""
import json, sys
D = json.load(open("out/diag/report_data.json")); tpl, out = sys.argv[1], sys.argv[2]
keep = {k: v for k, v in D["runs"].items() if k in ("rl_dgpo_compress", "rl2_dgpo_v5", "rl2_dgpo_v10") or k.startswith("diag_")}
if D["sanity"]: D["sanity"]["2_grpo_ratio_at_step0"].pop("per_step", None); D["sanity"]["4_dgpo_direction"].pop("steps", None)
P = D["passk"]
if P:
    P["grids"] = {k: v for k, v in P["grids"].items() if k in ("odd01_base", "odd01_v10_192", "odd05_base", "odd05_v10_192")}
    for jn in P["summary"]:
        for n in P["summary"][jn]: P["summary"][jn][n].pop("soft", None)
PL = D.get("passk_long")
if PL:
    for jn in PL["summary"]:
        for n in PL["summary"][jn]: PL["summary"][jn][n].pop("soft", None)
slim = {"runs": keep, "sanity": D["sanity"], "passk": P, "passk_long": PL, "band": D.get("band"), "bias": D.get("bias"), "pick": D["pick"], "text": json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}}
blob = json.dumps(slim).replace("</", "<\\/")
html = open(tpl).read().replace("/*DATA*/", blob, 1); open(out, "w").write(html)
print(f"{out}: {len(html)/1e6:.2f} MB; runs {list(keep)}; grids {list(P['grids']) if P else None}")
