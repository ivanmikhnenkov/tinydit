"""Stage-2 reward: three objectives, four terms, combined as advantages.

    r_i = w_align * A(align) + w_real * A(real) + w_qual * A(quality) + w_anat * A(anatomy)
    align    = P(yes | Qwen3-VL-8B, "does this image show '{prompt}'?")          judge study: failures 0.94, SNR 4.1
    real     = P(yes | Qwen3-VL-8B, "real photograph?")  (--real-source vlm)      0.83 real vs FLUX, SNR 2.4
               or P(real | DINOv2 probe) (--real-source probe; gameable by softening, kept as a monitor 'probe')
    quality  = HPSv3 mu (or HPSv2.1 if --quality hpsv2)                           group SNR 2.9, artifacts 0.75
    anatomy  = 1 - P(yes | Qwen3-VL-8B, "anatomical or structural errors?")      SNR 2.3, artifacts 0.74
    A(x)     = (x - mean of the prompt's group) / (std over the round + 1e-4)
    veto     : blank / noise images (pixel std < 0.02) get the group's minimum advantage. (An align < 0.02 veto was tried
               first: it hit ~18 % of samples at -2 and let the negative half of the update dominate, drifting the policy
               toward generic smooth images instead of teaching alignment.)
The four terms are near-orthogonal inside a prompt group (|rho| <= 0.2), so their advantages add. HPSv3 prefers renders;
the realism term outweighs it on purpose. Effective influence is weight x within-group spread: on base samples HPSv3 varies ~2x
more between seeds than the realism probe does, hence 0.30 real vs 0.15 quality (combined advantage SNR 3.4, judge study). Every raw term is returned for logging.
"""
from __future__ import annotations
import numpy as np, torch
from . import judges as JD

WEIGHTS = {"align": 0.40, "real": 0.30, "quality": 0.15, "anatomy": 0.15, "style": 0.15}
# style = one-sided drift guard: 0 inside the base model's range of saturation / contrast / sharpness, negative beyond its
# 90th percentile (in units of the base std). It cannot push toward grey or blurry images, only against the preference
# model's known exits (validate_reward.py: a saturated copy still ranked mid-group without it).


class Stage2Reward:
    def __init__(self, weights=WEIGHTS, quality="hpsv3", vlm="qwen3vl8b", device="cuda", batch=8, style_base="out/models/judges/style_base.json", veto_align=0.0, real_source="probe"):
        self.w = dict(weights); self.batch = batch; self.veto_align = veto_align; self.real_source = real_source
        import json; self.base = json.load(open(style_base))
        self.vlm = JD.VLM(vlm, device)
        self.quality_name = quality if self.w.get("quality", 0) > 0 else None     # weight 0: do not load the quality model at all
        if self.quality_name == "hpsv3": self.hps = JD.HPSv3(device)
        elif self.quality_name == "hpsv2":
            from .. import metrics; self.metrics = metrics
        self.probe = JD.DinoProbe(device=device)

    @torch.no_grad()
    def terms(self, images_u8, prompts):
        """raw per-sample terms -> dict of float arrays (N,)"""
        t = {"align": self.vlm.yesno(images_u8, "align", prompts, self.batch),
             "anatomy": 1 - self.vlm.yesno(images_u8, "artifact", prompts, self.batch),
             "probe": self.probe(images_u8)}                                       # always logged, a monitor
        # realism term: the VLM's "real photograph?" (hard to satisfy by softening) or the linear probe (v3-v6 showed the
        # policy buys probe score with softer, more muted images, which DINO features cannot separate from real softness)
        t["real"] = self.vlm.yesno(images_u8, "real", None, self.batch) if self.real_source == "vlm" else t["probe"]
        if self.quality_name == "hpsv3": t["quality"] = self.hps.score(images_u8, prompts, self.batch)
        elif self.quality_name == "hpsv2": t["quality"] = np.array(self.metrics.hpsv2(images_u8, prompts))
        else: t["quality"] = np.zeros(len(images_u8), dtype=np.float32)
        st = JD.pixel_stats(images_u8); t.update({f"px_{k}": v for k, v in st.items()})
        t["style"] = -sum(np.clip((st[k] - self.base[k]["p95"]) / self.base[k]["std"], 0, None)
                          + np.clip((self.base[k]["p5"] - st[k]) / self.base[k]["std"], 0, None) for k in ("sat", "contrast", "sharp"))
        pix = np.stack([im.astype(np.float32).std() / 255 for im in images_u8])
        t["veto"] = ((pix < 0.02) | (t["align"] < self.veto_align)).astype(np.float32)   # degenerate images only by default
        return t

    def advantage(self, t, groups):
        """t: terms dict | groups: (N,) int group id per sample -> combined advantage (N,) and per-term advantages"""
        N = len(groups); g = np.asarray(groups); A = {}
        for k in self.w:
            x = np.asarray(t[k], dtype=np.float64); gm = np.zeros(N)
            for gid in np.unique(g): gm[g == gid] = x[g == gid].mean()
            # the guard is already in base-std units and mostly zero: standardising it would turn a few mild exceedances
            # into huge negatives (that shifted the whole distribution down in v3/v4), so it enters group-centred only
            A[k] = (x - gm) if k == "style" else (x - gm) / (x.std() + 1e-4)
        adv = sum(self.w[k] * A[k] for k in self.w)
        if t["veto"].any():
            for gid in np.unique(g):
                m = g == gid; lo = adv[m].min()
                adv[m & (t["veto"] > 0)] = min(lo, -2.0)
        return adv, A
