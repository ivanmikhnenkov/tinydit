"""VLM judges over the reward-eval set, and HPSv3 re-implemented without its package.

    .venv/bin/python scripts/reward_eval/score_vlm.py --model qwen35_4b --modes align,artifact,real,aislop,rubric
    .venv/bin/python scripts/reward_eval/score_vlm.py --model hpsv3

Yes/no modes read P(yes) from the logits of the first answer token (VQAScore style), thinking disabled: one forward pass,
deterministic, continuous. `rubric` generates a JSON with three 1-10 scores; `describe` generates a free-form list of errors
(for the labelled subset). Images are upsampled to 448x448 (= HPSv3's 256*28*28 budget) before the processor.
Writes out/reward_eval/scores/<model>.json {group: {file: {metric: value}}} and a timing entry.
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, "src")
import numpy as np, torch
from PIL import Image
from tinydit.ae import hf_token
E = "out/reward_eval"; J = "out/models/judges"
MODELS = {"qwen3vl8b": ("qwen3vl8b", "qwen3vl"), "qwen35_4b": ("qwen35_4b", "qwen35"), "qwen35_9b": ("qwen35_9b", "qwen35"),
          "qwen36_35b": ("qwen36_35b", "qwen35"), "qwen38_27b": ("qwen38_27b", "qwen35"), "hpsv3": ("hpsv3", "hpsv3")}
Q = {
 "align":    "Does this image show \"{prompt}\"? Answer yes or no.",
 "artifact": "Look carefully at the animals, people and objects in this image. Are there any anatomical or structural errors, "
             "such as extra or missing limbs, a wrong number of legs, eyes or ears, merged or deformed body parts, duplicated "
             "features, or objects with impossible geometry? Answer yes or no.",
 "real":     "Is this image a real photograph taken with a camera, as opposed to an AI-generated, rendered, painted or illustrated image? Answer yes or no.",
 "aislop":   "Does this image have the typical look of AI-generated pictures: overly smooth or glossy surfaces, oversaturated colours, "
             "a plastic or airbrushed finish, or an unnaturally clean composition? Answer yes or no.",
 "rubric":   "Evaluate this image for the prompt \"{prompt}\" on three criteria, each from 1 (worst) to 10 (best): "
             "alignment (does it show what the prompt says: subjects, count, attributes, actions, setting), "
             "realism (does it look like a real photograph rather than an AI-generated or rendered image), "
             "integrity (freedom from anatomical or structural errors such as extra or missing limbs, deformed faces, impossible geometry). "
             "Answer with only a JSON object like {{\"alignment\": 7, \"realism\": 8, \"integrity\": 5}}.",
 "describe": "List every visual error or artifact in this image: wrong anatomy (extra, missing or merged limbs, heads, eyes, ears), "
             "impossible geometry, duplicated or floating parts, unreadable or garbled details. Be specific and brief. If there are none, answer: none.",
}
INSTRUCTION = ("You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. \n\n"
 "**Visual Quality:**  \nEvaluate the overall visual quality of the image. The following sub-dimensions should be considered:\n"
 "- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.\n"
 "- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.\n"
 "- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).\n"
 "- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.\n"
 "- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. \n\n"
 "**Text Alignment:**  \nAssess how well the image matches the textual prompt across the following sub-dimensions:\n"
 "- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.\n"
 "- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.\n"
 "- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.\n"
 "- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.\n"
 "- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.\n"
 "Textual prompt - {text_prompt}")
HPS_TAIL = "Please provide the overall ratings of this image: <|Reward|>\n\nEND"


def groups_of(M, which, subset=None):
    g = {"samples": [(s["file"], s["prompt"]) for s in M["samples"]],
         "ref_real": [(f"ref/real/{r['idx']:04d}.png", r["caption"]) for r in M["ref"]["real"]],
         "ref_flux": [(f"ref/flux/{r['idx']:04d}.png", r["caption"]) for r in M["ref"]["flux"]]}
    if subset:
        keep = set(json.load(open(subset)).keys())
        g = {k: [(f, p) for f, p in v if f in keep] for k, v in g.items()}
    return {k: g[k] for k in which}


def img448(path):
    return Image.open(f"{E}/{path}").convert("RGB").resize((448, 448), Image.BICUBIC)


class Judge:
    def __init__(self, name):
        d, kind = MODELS[name]; self.kind = kind; self.dir = os.path.join(J, d)
        from transformers import AutoProcessor
        if kind == "hpsv3":
            from transformers import AutoConfig, Qwen2VLForConditionalGeneration
            from safetensors import safe_open
            base = os.path.join(J, "qwen2vl_tok")     # tokenizer, image processor and config of Qwen/Qwen2-VL-7B-Instruct
            self.proc = AutoProcessor.from_pretrained(base, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
            self.proc.tokenizer.add_special_tokens({"additional_special_tokens": ["<|Reward|>"]})
            self.reward_id = self.proc.tokenizer.convert_tokens_to_ids("<|Reward|>")
            # the checkpoint's config.json says qwen2_5_vl but its vision tower is Qwen2-VL (fc1/fc2 MLP): use the Qwen2-VL config
            cfg = AutoConfig.from_pretrained(base); cfg.vocab_size = len(self.proc.tokenizer)
            if hasattr(cfg, "text_config"): cfg.text_config.vocab_size = len(self.proc.tokenizer)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.dir, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
            h = cfg.text_config.hidden_size if hasattr(cfg, "text_config") else cfg.hidden_size
            self.head = torch.nn.Sequential(torch.nn.Linear(h, 1024), torch.nn.ReLU(), torch.nn.Dropout(0.05), torch.nn.Linear(1024, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)).cuda().eval()
            with safe_open(os.path.join(self.dir, "HPSv3.safetensors"), "pt") as s:
                self.head.load_state_dict({k[len("rm_head."):]: s.get_tensor(k).float() for k in s.keys() if k.startswith("rm_head.")})
            assert len(self.proc.tokenizer) == self.model.get_input_embeddings().weight.shape[0], "vocab mismatch"
        else:
            self.proc = AutoProcessor.from_pretrained(self.dir)
            self.proc.tokenizer.padding_side = "left"
            if kind == "qwen3vl":
                from transformers import Qwen3VLForConditionalGeneration as C
            else:
                from transformers import AutoModelForMultimodalLM as C
            self.model = C.from_pretrained(self.dir, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
            tok = self.proc.tokenizer
            self.yes = sorted({tok.encode(v, add_special_tokens=False)[0] for v in ("Yes", " Yes", "yes", " yes")})
            self.no = sorted({tok.encode(v, add_special_tokens=False)[0] for v in ("No", " No", "no", " no")})

    def _chat(self, images, texts, gen=False):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": t}]}] for im, t in zip(images, texts)]
        kw = dict(add_generation_prompt=True, tokenize=False)
        if self.kind == "qwen35": kw["enable_thinking"] = False
        prompts = [self.proc.apply_chat_template(m, **kw) for m in msgs]
        return self.proc(text=prompts, images=images, padding=True, return_tensors="pt").to("cuda")

    @torch.inference_mode()
    def yesno(self, images, texts):
        inp = self._chat(images, texts)
        logits = self.model(**inp).logits[:, -1].float()
        ly = torch.logsumexp(logits[:, self.yes], 1); ln = torch.logsumexp(logits[:, self.no], 1)
        return torch.sigmoid(ly - ln).cpu().numpy()

    @torch.inference_mode()
    def generate(self, images, texts, max_new=60):
        inp = self._chat(images, texts)
        out = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.inference_mode()
    def hps(self, images, prompts):
        texts = [INSTRUCTION.format(text_prompt=p) + HPS_TAIL for p in prompts]
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}] for t in texts]
        chat = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inp = self.proc(text=chat, images=images, padding=True, return_tensors="pt").to("cuda")
        hs = self.model.model(**inp, return_dict=True).last_hidden_state          # (B, L, H)
        mask = inp["input_ids"] == self.reward_id
        pooled = hs[mask].float().view(hs.shape[0], -1)
        return self.head(pooled)[:, 0].cpu().numpy()                              # mu


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    ap.add_argument("--modes", default="align,artifact,real,aislop"); ap.add_argument("--groups", default="samples,ref_real,ref_flux")
    ap.add_argument("--subset", default=None, help="json of {file: ...} to restrict to (e.g. the labelled set)")
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    M = json.load(open(f"{E}/manifest.json")); G = groups_of(M, a.groups.split(","), a.subset)
    os.makedirs(f"{E}/scores", exist_ok=True); outp = f"{E}/scores/{a.model}.json"
    R = json.load(open(outp)) if os.path.exists(outp) else {}
    t_load = time.time(); jd = Judge(a.model); t_load = time.time() - t_load
    R.setdefault("_timing", {})["load_s"] = t_load; R["_timing"]["vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    modes = a.modes.split(",") if jd.kind != "hpsv3" else ["hps", "hps_mis", "hps_jpeg"]
    import io, random; random.seed(0)
    for g, items in G.items():
        if a.limit: items = items[:a.limit]
        R.setdefault(g, {})
        files = [f for f, _ in items]; prompts = [p for _, p in items]
        mis = prompts[:]; random.shuffle(mis); mis = [m if m != p else prompts[(i + 1) % len(prompts)] for i, (m, p) in enumerate(zip(mis, prompts))]
        for mode in modes:
            todo = [i for i, f in enumerate(files) if mode not in R[g].get(f, {})]
            if not todo: continue
            t0 = time.time(); n = 0
            for k in range(0, len(todo), a.batch):
                idx = todo[k:k + a.batch]; ims = [img448(files[i]) for i in idx]
                if mode.endswith("_jpeg"):
                    ims2 = []
                    for im in ims:
                        b = io.BytesIO(); im.save(b, "JPEG", quality=90); ims2.append(Image.open(b).convert("RGB"))
                    ims = ims2
                if mode in ("hps", "hps_jpeg"): vals = jd.hps(ims, [prompts[i] for i in idx])
                elif mode == "hps_mis": vals = jd.hps(ims, [mis[i] for i in idx])
                elif mode in ("rubric", "describe"):
                    outs = jd.generate(ims, [Q[mode].format(prompt=prompts[i]) for i in idx], max_new=60 if mode == "rubric" else 120)
                    vals = outs
                else:
                    vals = jd.yesno(ims, [Q[mode.replace("_jpeg", "")].format(prompt=prompts[i]) for i in idx])
                    if mode == "align":
                        vals_mis = jd.yesno(ims, [Q[mode].format(prompt=mis[i]) for i in idx])
                for j, i in enumerate(idx):
                    R[g].setdefault(files[i], {})
                    if mode == "rubric":
                        m = re.search(r'\{[^}]*\}', vals[j] or "")
                        try: d = json.loads(m.group(0)) if m else {}
                        except Exception: d = {}
                        for key in ("alignment", "realism", "integrity"):
                            R[g][files[i]][f"rubric_{key}"] = float(d[key]) if key in d and str(d[key]).replace('.', '', 1).isdigit() else None
                        R[g][files[i]]["rubric"] = vals[j]
                    elif mode == "describe": R[g][files[i]]["describe"] = vals[j]
                    else:
                        R[g][files[i]][mode] = float(vals[j])
                        if mode == "align": R[g][files[i]]["align_mis"] = float(vals_mis[j]); R[g][files[i]]["mismatch_prompt"] = mis[i]
                n += len(idx)
                if (k // a.batch) % 10 == 0: json.dump(R, open(outp, "w"))
            dt = time.time() - t0; R["_timing"][f"{g}.{mode}.img_per_s"] = n / max(dt, 1e-6)
            print(f"  {a.model} {g} {mode}: {n} images, {n / dt:.1f} img/s", flush=True)
            json.dump(R, open(outp, "w"))
    R["_timing"]["vram_gb"] = torch.cuda.max_memory_allocated() / 1e9; json.dump(R, open(outp, "w"))
    print(f"done -> {outp}  (load {t_load:.0f}s, peak {R['_timing']['vram_gb']:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
