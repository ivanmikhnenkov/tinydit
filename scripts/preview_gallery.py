"""preview.json (from hfpeek.py sample) -> a single self-contained HTML contact sheet."""
from __future__ import annotations
import html, json, statistics, sys

# rough T5 tokenizer rate for English prose; the training cache clips at --max-len tokens
TOK_PER_WORD, MAX_LEN = 1.35, 64
SIZES = {  # dataset -> (total images, hosting, caption source)
    "COCO recap (Recap-COCO-30K)": ("30k (of 123k COCO)", "bytes in parquet", "human + LLaVA-1.5 recaption"),
    "COCO/LVIS GPT-4V 220k": ("218k", "COCO URLs", "GPT-4V"),
    "PixMo-Cap (Ai2)": ("717k", "URL list", "human speech → LLM"),
    "PD12M (Spawning)": ("12.4M", "Spawning S3 URLs", "Florence-2"),
    "Megalith-10M + ShareCaptioner": ("10M", "Flickr URLs", "ShareCaptioner"),
    "Pexels photo-concept-bucket": ("568k", "Pexels URLs", "CogVLM"),
    "CommonCatalog CC-BY (YFCC)": ("~26M", "bytes in parquet (21 TB)", "BLIP-2"),
    "ImageNet-1k VL-enriched": ("1.28M", "bytes in parquet (gated)", "BLIP-2"),
    "LLaVA-ReCap-CC12M": ("12M", "bytes in parquet (465 GB)", "LLaVA-NeXT"),
    "PixelProse (RedCaps shard)": ("16M", "URL list", "Gemini"),
    "DenseFusion-1M (LAION)": ("1M", "URLs + zips", "multi-expert VLM"),
    "text-to-image-2M (synthetic)": ("2M", "bytes in tar (423 GB)", "generating prompt"),
    "FLUX-Reason-6M (synthetic)": ("6M", "bytes in parquet (882 GB)", "Qwen-VL multi-facet"),
    "DOCCI (Google)": ("15k", "tar.gz on GCS", "human, expert-written"),
    "SA-1B @512² + Qwen2.5-VL (BLIP3o)": ("11M", "bytes in tar (565 GB)", "Qwen2.5-VL-7B"),
    "Megalith-10M mirror (moondream mdqa)": ("5.46M (of 9.6M)", "bytes in parquet (1.06 TB)", "Moondream (short); Qwen3-VL/InternVL2 joins"),
    "Open Images + Qwen3-VL captions": ("1.76M", "bytes in parquet (580 GB)", "Qwen3-VL-30B"),
    "Pexels 2.8M @256² (i1 TFRecord)": ("2.8M", "bytes in TFRecord (269 GB)", "Qwen3-VL-30B ×5"),
}

# tag -> (label, css class). Order here is the sidebar order.
TAGS = {"core": ("Run 1 · core", "core"), "mix": ("Run 1 · mix-in", "mix"), "anchor": ("Run 1 · anchor", "core"),
        "eval": ("Eval only", "eval"), "reserve": ("Reserve for scale-up", "reserve"), "skip": ("Skip", "skip")}
VERDICTS = {
 "Open Images + Qwen3-VL captions": ("reserve", "dropped from run 1 after your review",
   "Real Flickr snapshots with honest, object-naming captions, but cluttered amateur framing rather than clean subjects. A second shard (train_9) looked the same: rifle, garden sign, cipher machine, concert crowd. Better suited to a later bucketed, higher-resolution stage than to this run."),
 "Pexels 2.8M @256² (i1 TFRecord)": ("skip", "pre-cropped to squares",
   "Same clean Pexels photos, but the i1 authors already centre-cropped them to 1:1, throwing away a third of every 3:2 frame. Since we now keep aspect ratios with buckets, take Pexels from a source that preserves the full frame instead (see the photo-concept-bucket tab)."),
 "FLUX-Reason-6M (synthetic)": ("mix", "30% · 1.2M from the Aesthetics parts, top by clarity+structure score",
   "Perfect caption fit and clean composition: exactly what teaches prompt adherence. Capped because synthetic-only data hands the model FLUX's look and failure modes, while real photos keep textures and lighting honest. Its Text and Imaginative parts are skipped."),
 "COCO/LVIS GPT-4V 220k": ("anchor", "15% · the 118k COCO-2017 train images (oversampled ~5x)",
   "The COCO images you already trust, with 66-word GPT-4V captions on top of the five human ones; the human captions are the short variant. Exact aspect check on all 118k: median 1.37 (4:3), 77% within 3:2, only 2% wider than 16:9, so with buckets almost nothing is cropped."),
 "COCO recap (Recap-COCO-30K)": ("skip", "covered by the GPT-4V captions",
   "Val2014 images with one LLaVA recaption. Superseded by the GPT-4V set above; kept on the page as a reference for what recaptioned COCO looks like."),
 "DOCCI (Google)": ("eval", "500 prompts, never trained on",
   "15k photos with expert human captions. Too small to train on and too good to waste: the external prompt set for evaluation."),
 "Megalith-10M mirror (moondream mdqa)": ("reserve", "first place to grow past 4.4M",
   "Real photography at scale with bytes on HF, and I verified that its keys join to the Qwen3-VL captions. Event- and crowd-heavy and one more format to ingest, so not in run 1."),
 "Megalith-10M + ShareCaptioner": ("skip", "use the mirror above instead",
   "Flickr URLs plus flowery 170-word captions. If Megalith is used at all, the mirror with Qwen3-VL captions is strictly better."),
 "SA-1B @512² + Qwen2.5-VL (BLIP3o)": ("reserve", "licence is research-only",
   "Licensed stock photos at a training-ready 512², but street scenes and storefronts dominate. Fine for a later scale-up, not for an object-centric first run."),
 "Pexels photo-concept-bucket": ("core", "55% · 568k, or 2.8M if you accept one gated repo",
   "Studio-clean subjects and the full frame: the Pexels CDN serves any photo resized to 640 px wide with the aspect ratio intact (tested, 14 to 65 KB each, 25 GB for all 568k). Short CogVLM captions ship with it; long Qwen3-VL captions join by Pexels id. The same photographers' 2.8M-image pool at 640 px sits in animetimm/pexels-tagger-v0-w640-ws-full, which is auto-gated: one click on Hugging Face and Pexels becomes 2.8M images."),
 "PixMo-Cap (Ai2)": ("skip", "wrong domain",
   "The most faithful captions on the page, but the images are mixed web content: screenshots, ads, collages."),
 "PD12M (Spawning)": ("skip", "wrong domain",
   "Cleanest licence anywhere, but museum objects, scans and engravings rather than photographs of everyday things."),
 "LLaVA-ReCap-CC12M": ("skip", "same web pool that failed before",
   "Better captions than the recap set that hurt last time, but the same CC12M images, and some captions start mid-sentence. Not needed with Open Images and Pexels available."),
 "PixelProse (RedCaps shard)": ("skip", "redundant, URL-only",
   "Hobbyist Reddit photos with decent Gemini captions, behind rate-limited URLs. Nothing the run-1 sources lack."),
 "CommonCatalog CC-BY (YFCC)": ("skip", "captions too weak",
   "Amateur snapshots with 10-word BLIP-2 captions. Would need recaptioning before it is useful."),
 "ImageNet-1k VL-enriched": ("skip", "captions too weak",
   "Object-centric but low resolution, and BLIP-2 captions that are often wrong."),
 "DenseFusion-1M (LAION)": ("skip", "wrong domain",
   "Product grids, posters and collages, with 30% dead links in the sample."),
 "text-to-image-2M (synthetic)": ("skip", "wrong style",
   "Anime and illustration heavy DALL-E 3 output. FLUX-Reason is the better synthetic source for photoreal."),
}
PLAN = {
 "rows": [
  ["Pexels photo-concept-bucket (full frame, 640 px)", "core", "568k, or 2.8M via the gated twin", "55%", "Qwen3-VL by Pexels id / CogVLM", "36 GB (180 GB)", "25 GB (125 GB)"],
  ["FLUX-Reason-6M, Aesthetics parts", "mix-in", "1.2M of 4.3M", "30%", "caption_detail / caption_entity", "77 GB", "~200 GB parquet"],
  ["COCO 2017 train + GPT-4V", "anchor", "118k", "15%", "GPT-4V 66 words / 5 human captions", "8 GB", "18 GB zip"],
 ],
 "totals": ["1.9M images (4.1M with the gated Pexels twin)", "121 GB of fp16 latents kept (265 GB)", "~240 GB streamed and deleted (~340 GB)"],
 "rules": [
  "Aspect ratios are kept. Five buckets of ~256 tokens each: 256x256, 288x224, 224x288, 320x208, 208x320. Each image is resized to cover its nearest bucket and only the residual is cropped: 0% for exact 4:3 or 3:2 frames, 16% of the width for a 16:9 frame; anything wider than 2:1 is dropped (COCO 2%, Pexels 1%, FLUX 0%). Every batch is a single bucket, so throughput is unchanged; the model needs no change because RoPE reads position from the grid.",
  "Per sample: 50% long caption (cut at 128 T5 tokens), 40% short caption (CogVLM sentence, human COCO caption, or caption_entity), 10% empty for classifier-free guidance. Long and short captions are encoded as separate sub-batches at their own lengths so T5 never pads short prompts to 128.",
  "Sampling weights are per-batch source probabilities, not proportional to size: COCO is seen ~5x more often than its share.",
  "Evaluation is a held-out slice of the training sources: 1,000 images per source never trained on. Their latents give the validation loss; their captions give the unseen-prompt grids, FID and reward scores. Plus one short hand-written prompt list for compositions that are not in the data.",
  "Images are never stored. Each shard or URL is streamed, bucketed, encoded to a latent (64 KB) and deleted. With compiled training at ~460 img/s, 110M samples take 2.8 days; ingest is a few hours, download-bound.",
 ],
 "questions": [
  "Pexels at 2.8M needs you to click 'agree' once on huggingface.co/datasets/animetimm/pexels-tagger-v0-w640-ws-full (auto-approved). Without it the run has 1.9M images and FLUX becomes the largest source; with it, real photos are 70% of the data. My recommendation: accept.",
  "FLUX-Reason share: 30% (my default) or lower now that it is the most 'perfect' source? It is the only source that is all 1:1, so it also feeds the square bucket.",
  "Megalith and Open Images stay in reserve for a later bucketed, higher-resolution stage. Agree?",
 ],
}


def stats(ds):
    s = ds.get("samples", [])
    if not s: return {}
    ws = sorted(min(a["w"], a["h"]) for a in s)
    words = []
    for a in s:
        c = next(iter(a["caps"].values()), "")
        words.append(len(c.split()))
    fails = ds.get("fails", [])
    return dict(short_side_med=ws[len(ws)//2], short_side_min=ws[0],
                words_med=int(statistics.median(words)) if words else 0,
                trunc_pct=round(100 * sum(1 for w in words if w * TOK_PER_WORD > MAX_LEN) / len(words)),
                fail_pct=round(100 * len(fails) / max(1, ds.get("n_tried", len(s) + len(fails)))))


def main(inp, outp):
    data = json.load(open(inp))
    rank = {tg: i for i, tg in enumerate(TAGS)}
    order = sorted([k for k in data], key=lambda k: (rank[VERDICTS.get(k, ("skip",))[0]], list(SIZES).index(k) if k in SIZES else 99))
    payload = []
    for k in order:
        d = data[k]; st = stats(d) if "samples" in d else {}
        total, hosting, capsrc = SIZES.get(k, ("?", "?", "?"))
        tag, share, comment = VERDICTS.get(k, ("skip", "", ""))
        payload.append(dict(name=k, repo=d.get("repo", ""), note=d.get("note", ""), error=d.get("error"),
                            tag=tag, tag_label=TAGS[tag][0], tag_cls=TAGS[tag][1], share=share, comment=comment,
                            total=total, hosting=hosting, capsrc=capsrc, stats=st,
                            cap_fields=list(d["samples"][0]["caps"].keys()) if d.get("samples") else [],
                            samples=[dict(t=a["thumb"], w=a["w"], h=a["h"], caps=a["caps"]) for a in d.get("samples", [])]))
    js = json.dumps(payload).replace("</", "<\\/")
    page = TEMPLATE.replace("__PLAN__", json.dumps(PLAN).replace("</", "<\\/")).replace("__DATA__", js).replace("__MAXLEN__", str(MAX_LEN)).replace("__TPW__", str(TOK_PER_WORD))
    open(outp, "w").write(page)
    print(f"wrote {outp}: {len(payload)} datasets, {sum(len(p['samples']) for p in payload)} tiles, {len(page)/1e6:.1f} MB")


TEMPLATE = r'''<title>Contact Sheets</title>
<meta name="description" content="Real samples from candidate text-to-image training sets, seen the way a 256² model will see them.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#EDEFF2; --surface:#FFFFFF; --surface-2:#F5F6F8; --line:#D3D7DD; --line-strong:#AEB5BF;
  --ink:#171B21; --ink-2:#48505B; --ink-3:#7B8390;
  --accent:#D99A2B; --accent-ink:#7A5210; --accent-soft:#FBEFD6;
  --ok:#2F8F5B; --warn:#C9811F; --bad:#C24A3A;
  --shadow:0 1px 2px rgba(23,27,33,.06),0 6px 20px rgba(23,27,33,.06);
  --sans:"Archivo",system-ui,-apple-system,"Segoe UI",sans-serif;
  --serif:"Source Serif 4",Georgia,"Times New Roman",serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#14171C; --surface:#1C2027; --surface-2:#232830; --line:#2E343D; --line-strong:#454D58;
  --ink:#E9ECEF; --ink-2:#B4BCC6; --ink-3:#7F8894;
  --accent:#F0B24E; --accent-ink:#F5C877; --accent-soft:#3A2E17;
  --ok:#5CBF87; --warn:#E5A44A; --bad:#E36A5A;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35); color-scheme:dark;}}
:root[data-theme="dark"]{
  --bg:#14171C; --surface:#1C2027; --surface-2:#232830; --line:#2E343D; --line-strong:#454D58;
  --ink:#E9ECEF; --ink-2:#B4BCC6; --ink-3:#7F8894;
  --accent:#F0B24E; --accent-ink:#F5C877; --accent-soft:#3A2E17;
  --ok:#5CBF87; --warn:#E5A44A; --bad:#E36A5A;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35); color-scheme:dark;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.45;-webkit-font-smoothing:antialiased}
a{color:inherit}
.top{display:flex;align-items:baseline;gap:20px;padding:18px 24px 14px;border-bottom:1px solid var(--line);background:var(--surface)}
.top h1{margin:0;font-size:22px;font-weight:700;font-stretch:87.5%;letter-spacing:-.01em}
.top p{margin:0;color:var(--ink-2);font-size:13.5px;max-width:64ch}
.top .spacer{flex:1}
.seg{display:inline-flex;border:1px solid var(--line-strong);border-radius:6px;overflow:hidden;font-size:12.5px}
.seg button{appearance:none;border:0;background:var(--surface);color:var(--ink-2);padding:6px 11px;cursor:pointer;font:inherit;font-weight:500}
.seg button+button{border-left:1px solid var(--line-strong)}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--surface)}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.wrap{display:grid;grid-template-columns:272px minmax(0,1fr);min-height:calc(100vh - 62px)}
@media (max-width:900px){.wrap{grid-template-columns:1fr}.side{border-right:0;border-bottom:1px solid var(--line)}}
.side{border-right:1px solid var(--line);background:var(--surface);padding:10px 0 24px}
.side .lbl{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);padding:10px 20px 6px;font-weight:600}
.side button.ds{display:block;width:100%;text-align:left;appearance:none;border:0;background:none;color:var(--ink);font:inherit;padding:9px 20px 9px 17px;border-left:3px solid transparent;cursor:pointer}
.side button.ds:hover{background:var(--surface-2)}
.side button.ds[aria-current="true"]{border-left-color:var(--accent);background:var(--surface-2)}
.side button.ds:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.side .n{font-weight:600;font-size:13.5px;display:block;line-height:1.3}
.side .m{display:block;color:var(--ink-3);font-family:var(--mono);font-size:11px;margin-top:3px}
.side .m .dot{display:inline-block;width:7px;height:7px;border-radius:50%;vertical-align:1px;margin-right:5px;background:var(--ok)}
.side .m .dot.warn{background:var(--warn)}.side .m .dot.bad{background:var(--bad)}
main{padding:22px 28px 60px;min-width:0}
.head{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px 24px;align-items:start;margin-bottom:16px}
.head h2{margin:0;font-size:26px;font-weight:700;font-stretch:87.5%;letter-spacing:-.015em;text-wrap:balance}
.head .repo{font-family:var(--mono);font-size:12px;color:var(--ink-3);margin-top:2px}
.head .note{grid-column:1/-1;margin:0;color:var(--ink-2);font-family:var(--serif);font-size:15px;max-width:72ch}
.chips{display:flex;flex-wrap:wrap;gap:8px;grid-column:1/-1}
.chip{display:inline-flex;flex-direction:column;gap:1px;padding:7px 11px;border:1px solid var(--line);border-radius:6px;background:var(--surface);min-width:96px}
.chip b{font-family:var(--mono);font-weight:500;font-size:13.5px;font-variant-numeric:tabular-nums}
.chip span{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3)}
.chip.warn b{color:var(--warn)}.chip.bad b{color:var(--bad)}.chip.ok b{color:var(--ok)}
.capsel{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--ink-2);grid-column:1/-1}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(212px,1fr));gap:14px}
figure{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:6px;overflow:hidden;box-shadow:var(--shadow);display:flex;flex-direction:column}
.ph{position:relative;aspect-ratio:1/1;background:var(--surface-2);overflow:hidden}
.ph img{width:100%;height:100%;display:block;object-fit:contain}
body[data-view="crop"] .ph img{object-fit:cover}
.ph .fr{position:absolute;left:0;bottom:0;padding:2px 6px;font-family:var(--mono);font-size:10.5px;color:var(--accent-ink);background:var(--surface);border-top-right-radius:4px;opacity:.95}
figcaption{padding:9px 11px 10px;font-family:var(--serif);font-size:13.5px;line-height:1.42;color:var(--ink);display:-webkit-box;-webkit-line-clamp:5;-webkit-box-orient:vertical;overflow:hidden;cursor:pointer;flex:1}
figcaption.open{display:block;-webkit-line-clamp:unset}
figcaption em{font-style:normal;color:var(--ink-3)}
.ft{display:flex;justify-content:space-between;padding:0 11px 9px;font-family:var(--mono);font-size:10.5px;color:var(--ink-3);font-variant-numeric:tabular-nums}
.ft .tr{color:var(--warn)}
.tag{display:inline-block;font-size:10.5px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:4px;border:1px solid transparent;vertical-align:middle}
.tag.core{background:var(--accent-soft);color:var(--accent-ink);border-color:var(--accent)}
.tag.mix{background:var(--surface-2);color:var(--accent-ink);border-color:var(--accent)}
.tag.eval{background:var(--surface-2);color:var(--ink-2);border-color:var(--line-strong)}
.tag.reserve{background:var(--surface-2);color:var(--ink-2);border-color:var(--line-strong);border-style:dashed}
.tag.skip{background:transparent;color:var(--ink-3);border-color:var(--line)}
.verdict{grid-column:1/-1;display:grid;grid-template-columns:auto minmax(0,1fr);gap:6px 14px;align-items:start;padding:12px 14px;border-left:3px solid var(--accent);background:var(--surface);border-radius:0 6px 6px 0;max-width:80ch}
.verdict .share{font-family:var(--mono);font-size:12px;color:var(--ink-2);grid-column:2}
.verdict .cmt{grid-column:1/-1;margin:0;font-family:var(--serif);font-size:15px;line-height:1.5;color:var(--ink)}
.verdict .who{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);grid-column:1/-1}
.side .grp{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);padding:14px 20px 4px;font-weight:600;border-top:1px solid var(--line);margin-top:6px}
.side .grp:first-of-type{border-top:0;margin-top:0}
.side button.ds.skip .n{color:var(--ink-2);font-weight:500}
.plan h2{margin:0 0 6px;font-size:26px;font-weight:700;font-stretch:87.5%;letter-spacing:-.015em}
.plan .lede{font-family:var(--serif);font-size:15.5px;color:var(--ink-2);max-width:72ch;margin:0 0 18px}
.plan table{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.plan th{text-align:left;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);padding:9px 12px;border-bottom:1px solid var(--line);background:var(--surface-2)}
.plan td{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
.plan tr:last-child td{border-bottom:0}
.plan td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.plan tfoot td{background:var(--surface-2);font-weight:600}
.plan .tw{overflow-x:auto;margin-bottom:22px}
.plan h3{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);margin:22px 0 8px;font-weight:600}
.plan ol,.plan ul{margin:0;padding-left:22px;font-family:var(--serif);font-size:15px;line-height:1.55;max-width:78ch}
.plan li{margin:0 0 7px}
.plan .ask li{padding:8px 12px;margin:0 0 8px;background:var(--surface);border:1px solid var(--line);border-radius:6px;list-style-position:inside}
.empty{padding:40px;border:1px dashed var(--line-strong);border-radius:8px;color:var(--ink-2);font-family:var(--serif)}
kbd{font-family:var(--mono);font-size:11px;border:1px solid var(--line-strong);border-bottom-width:2px;border-radius:4px;padding:0 5px;color:var(--ink-2)}
.hint{color:var(--ink-3);font-size:12px;margin:18px 0 0}
@media (prefers-reduced-motion:no-preference){figure{transition:transform .12s ease}figure:hover{transform:translateY(-1px)}}
</style>
<header class="top">
  <h1>Contact Sheets</h1>
  <p>Twenty real rows from one shard of each candidate training set, grouped by my verdict. Toggle to the square centre-crop to see exactly what a 256² model is trained on.</p>
  <span class="spacer"></span>
  <div class="seg" role="group" aria-label="View">
    <button id="v-full" aria-pressed="true">Full frame</button>
    <button id="v-crop" aria-pressed="false">256² centre crop</button>
  </div>
</header>
<div class="wrap">
  <nav class="side" aria-label="Datasets">
    <div class="lbl">Datasets</div>
    <div id="list"></div>
    <p class="hint" style="padding:0 20px"><kbd>←</kbd> <kbd>→</kbd> switch set · click a caption to expand</p>
  </nav>
  <main id="main"></main>
</div>
<script>
const DATA=__DATA__, MAXLEN=__MAXLEN__, TPW=__TPW__;
let cur=0, capField={};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function words(s){return s?s.trim().split(/\s+/).length:0}
function list(){
  const el=document.getElementById('list'); el.innerHTML='';
  const p=document.createElement('button'); p.className='ds'; p.setAttribute('aria-current',cur===-1);
  p.innerHTML=`<span class="n">Proposed mix for run 1</span><span class="m">3 sources · 1.9M to 4.1M images · revised</span>`;
  p.onclick=()=>showPlan(); el.appendChild(p);
  let lastTag=null;
  DATA.forEach((d,i)=>{
    if(d.tag_label!==lastTag){const g=document.createElement('div'); g.className='grp'; g.textContent=d.tag_label; el.appendChild(g); lastTag=d.tag_label;}
    const b=document.createElement('button'); b.className='ds '+d.tag_cls; b.setAttribute('aria-current',i===cur);
    const st=d.stats||{}; const f=st.fail_pct||0; const cls=d.error?'bad':(f>=40?'bad':f>=15?'warn':'');
    const okn=d.samples.length;
    b.innerHTML=`<span class="n">${esc(d.name)}</span><span class="m"><i class="dot ${cls}"></i>${d.error?'failed':okn+' ok'} · ${esc(d.total)} · ${esc(d.hosting.split(' (')[0])}</span>`;
    b.onclick=()=>show(i); el.appendChild(b);
  });
}
function show(i){
  cur=i; list(); const d=DATA[i]; const m=document.getElementById('main'); const st=d.stats||{};
  const field=capField[i]||d.cap_fields[0];
  const capOf=s=>(s.caps[field]||Object.values(s.caps).find(x=>x)||'');
  const wl=d.samples.map(s=>words(capOf(s))).sort((a,b)=>a-b); st.words_med=wl.length?wl[wl.length>>1]:0; st.trunc_pct=wl.length?Math.round(100*wl.filter(w=>w*TPW>MAXLEN).length/wl.length):0;
  let h=`<div class="head"><div><h2>${esc(d.name)}</h2><div class="repo">${esc(d.repo)}</div></div>`;
  h+=`<p class="note">${esc(d.note)}</p>`;
  h+=`<div class="verdict"><span class="tag ${d.tag_cls}">${esc(d.tag_label)}</span><span class="share">${esc(d.share)}</span><p class="cmt">${esc(d.comment)}</p><span class="who">my take</span></div>`;
  if(d.error){h+=`</div><div class="empty">Could not read this source: <code>${esc(d.error)}</code></div>`; m.innerHTML=h; return;}
  const fc=st.fail_pct||0;
  h+=`<div class="chips">
    <div class="chip"><b>${esc(d.total)}</b><span>images</span></div>
    <div class="chip"><b>${esc(d.hosting)}</b><span>hosting</span></div>
    <div class="chip"><b>${esc(d.capsrc)}</b><span>captions</span></div>
    <div class="chip"><b>${st.short_side_med}px</b><span>median short side</span></div>
    <div class="chip"><b>${st.short_side_min}px</b><span>min short side</span></div>
    <div class="chip"><b>${st.words_med}</b><span>median words</span></div>
    <div class="chip ${st.trunc_pct>50?'warn':''}"><b>${st.trunc_pct}%</b><span>over ${MAXLEN} T5 tokens</span></div>
    <div class="chip ${fc>=40?'bad':fc>=15?'warn':'ok'}"><b>${fc}%</b><span>fetch failures</span></div>
  </div>`;
  if(d.cap_fields.length>1){
    h+=`<div class="capsel">Caption field <div class="seg" role="group">`+d.cap_fields.map(f=>`<button data-f="${esc(f)}" aria-pressed="${f===field}">${esc(f)}</button>`).join('')+`</div></div>`;
  }
  h+=`</div><div class="grid">`;
  d.samples.forEach((s,k)=>{
    const c=capOf(s); const w=words(c); const tok=Math.round(w*TPW);
    h+=`<figure><div class="ph"><img loading="lazy" src="data:image/jpeg;base64,${s.t}" alt=""><span class="fr">${s.w}×${s.h}</span></div>
      <figcaption title="click to expand">${c?esc(c):'<em>(no caption)</em>'}</figcaption>
      <div class="ft"><span>#${String(k+1).padStart(2,'0')}</span><span class="${tok>MAXLEN?'tr':''}">${w} words ≈ ${tok} tok${tok>MAXLEN?' · clipped':''}</span></div></figure>`;
  });
  h+=`</div>`; m.innerHTML=h;
  m.querySelectorAll('.capsel button').forEach(b=>b.onclick=()=>{capField[i]=b.dataset.f; show(i)});
  m.querySelectorAll('figcaption').forEach(f=>f.onclick=()=>f.classList.toggle('open'));
  m.scrollIntoView?.({block:'start'});
}
const PLAN=__PLAN__;
function showPlan(){
  cur=-1; list(); const m=document.getElementById('main');
  let h=`<div class="plan"><h2>Proposed mix for run 1</h2><p class="lede">Revised after your review: three sources, aspect ratios kept with buckets instead of square crops, evaluation from held-out slices of the same sources. Everything else on this page is a reserve for scaling past this or skipped for the reason written on its tab.</p>`;
  h+=`<div class="tw"><table><thead><tr><th>Source</th><th>Role</th><th>Images used</th><th>Share of batches</th><th>Captions: long / short</th><th>Latents kept</th><th>Streamed</th></tr></thead><tbody>`;
  PLAN.rows.forEach(r=>{h+=`<tr><td><b>${esc(r[0])}</b></td><td>${esc(r[1])}</td><td class="num">${esc(r[2])}</td><td class="num">${esc(r[3])}</td><td>${esc(r[4])}</td><td class="num">${esc(r[5])}</td><td class="num">${esc(r[6])}</td></tr>`});
  h+=`</tbody><tfoot><tr><td colspan="2">Total</td><td class="num">${esc(PLAN.totals[0])}</td><td class="num">100%</td><td></td><td class="num" colspan="2">${esc(PLAN.totals[1])} · ${esc(PLAN.totals[2])}</td></tr></tfoot></table></div>`;
  h+=`<h3>Rules of the run</h3><ol>`+PLAN.rules.map(r=>`<li>${esc(r)}</li>`).join('')+`</ol>`;
  h+=`<h3>Where I want your call</h3><ul class="ask">`+PLAN.questions.map(r=>`<li>${esc(r)}</li>`).join('')+`</ul>`;
  h+=`<p class="hint">Use <kbd>→</kbd> to walk the tabs in the order of the sidebar; every tab ends its header with my verdict.</p></div>`;
  m.innerHTML=h;
}
document.getElementById('v-full').onclick=()=>setView('full');
document.getElementById('v-crop').onclick=()=>setView('crop');
function setView(v){document.body.dataset.view=v;document.getElementById('v-full').setAttribute('aria-pressed',v==='full');document.getElementById('v-crop').setAttribute('aria-pressed',v==='crop');try{localStorage.setItem('cs-view',v)}catch(e){}}
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){cur+1>=DATA.length?showPlan():show(cur+1)}if(e.key==='ArrowLeft'){cur<=0?(cur===0?showPlan():show(DATA.length-1)):show(cur-1)}});
let v='full';try{v=localStorage.getItem('cs-view')||'full'}catch(e){}
setView(v); showPlan();
</script>
'''

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
