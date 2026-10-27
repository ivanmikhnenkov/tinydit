"""Judges and reward models used by the stage-2 reward and by the evaluations.

  VLM(name)      Qwen3-VL / Qwen3.5 family: P(yes) of a yes/no question from the first-token logits (VQAScore style),
                 thinking disabled; `generate` for rubric / description text.
  HPSv3()        re-implementation of MizzenAI/HPSv3 (Qwen2-VL-7B + a 3-layer head on the <|Reward|> token); returns mu.
  DinoProbe      logistic probe on DINOv2 CLS features, real photograph vs FLUX render; P(real).
  pixel_stats    mean saturation, sharpness (Laplacian variance), contrast.
Images are (N,3,H,W) uint8; VLM inputs are upsampled to 448x448 (HPSv3's 256*28*28 budget).
"""
from __future__ import annotations
import io, os
import numpy as np, torch
from PIL import Image

J = "out/models/judges"
QUESTIONS = {
 "align":    "Does this image show \"{prompt}\"? Answer yes or no.",
 "artifact": "Look carefully at the animals, people and objects in this image. Are there any anatomical or structural errors, "
             "such as extra or missing limbs, a wrong number of legs, eyes or ears, merged or deformed body parts, duplicated "
             "features, or objects with impossible geometry? Answer yes or no.",
 "real":     "Is this image a real photograph taken with a camera, as opposed to an AI-generated, rendered, painted or illustrated image? Answer yes or no.",
 "aislop":   "Does this image have the typical look of AI-generated pictures: overly smooth or glossy surfaces, oversaturated colours, "
             "a plastic or airbrushed finish, or an unnaturally clean composition? Answer yes or no.",
 "describe": "List every visual error or artifact in this image: wrong anatomy (extra, missing or merged limbs, heads, eyes, ears), "
             "impossible geometry, duplicated or floating parts, unreadable or garbled details. Be specific and brief. If there are none, answer: none.",
}
HPS_INSTRUCTION = ("You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. \n\n"
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
VLM_DIRS = {"qwen3vl8b": ("qwen3vl8b", "qwen3vl"), "qwen35_4b": ("qwen35_4b", "qwen35"), "qwen35_9b": ("qwen35_9b", "qwen35"), "qwen38_27b": ("qwen38_27b", "qwen35")}


def to_pil448(images_u8):
    return [Image.fromarray(np.ascontiguousarray(im.transpose(1, 2, 0))).resize((448, 448), Image.BICUBIC) for im in images_u8]


def pixel_stats(images_u8):
    """-> dict of arrays: sat (mean HSV saturation, 0-1), sharp (Laplacian variance of grey), contrast (grey std, 0-1)"""
    sat, sharp, con = [], [], []
    for im in images_u8:
        pil = Image.fromarray(np.ascontiguousarray(im.transpose(1, 2, 0)))
        hsv = np.asarray(pil.convert("HSV")).astype(np.float32); g = np.asarray(pil.convert("L")).astype(np.float32)
        lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        sat.append(hsv[..., 1].mean() / 255); sharp.append(lap.var()); con.append(g.std() / 255)
    return {"sat": np.array(sat), "sharp": np.array(sharp), "contrast": np.array(con)}


class VLM:
    def __init__(self, name="qwen3vl8b", device="cuda"):
        from transformers import AutoProcessor
        d, kind = VLM_DIRS[name]; self.kind, self.dir, self.dev = kind, os.path.join(J, d), device
        self.proc = AutoProcessor.from_pretrained(self.dir); self.proc.tokenizer.padding_side = "left"
        if kind == "qwen3vl":
            from transformers import Qwen3VLForConditionalGeneration as C
        else:
            from transformers import AutoModelForMultimodalLM as C
        self.model = C.from_pretrained(self.dir, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa").eval()
        tok = self.proc.tokenizer
        self.yes = sorted({tok.encode(v, add_special_tokens=False)[0] for v in ("Yes", " Yes", "yes", " yes")})
        self.no = sorted({tok.encode(v, add_special_tokens=False)[0] for v in ("No", " No", "no", " no")})

    def _inputs(self, pil_images, texts):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": t}]}] for im, t in zip(pil_images, texts)]
        kw = dict(add_generation_prompt=True, tokenize=False)
        if self.kind == "qwen35": kw["enable_thinking"] = False
        prompts = [self.proc.apply_chat_template(m, **kw) for m in msgs]
        return self.proc(text=prompts, images=pil_images, padding=True, return_tensors="pt").to(self.dev)

    @torch.inference_mode()
    def yesno(self, images_u8, question, prompts=None, batch=8):
        """P(yes) per image. `question` is a key of QUESTIONS or a template with {prompt}."""
        q = QUESTIONS.get(question, question); out = []
        pil = to_pil448(images_u8)
        for i in range(0, len(pil), batch):
            texts = [q.format(prompt=(prompts[j] if prompts else "")) for j in range(i, min(i + batch, len(pil)))]
            logits = self.model(**self._inputs(pil[i:i + batch], texts)).logits[:, -1].float()
            ly = torch.logsumexp(logits[:, self.yes], 1); ln = torch.logsumexp(logits[:, self.no], 1)
            out.append(torch.sigmoid(ly - ln).cpu().numpy())
        return np.concatenate(out)

    @torch.inference_mode()
    def generate(self, images_u8, question, prompts=None, max_new=120, batch=8):
        q = QUESTIONS.get(question, question); out = []
        pil = to_pil448(images_u8)
        for i in range(0, len(pil), batch):
            texts = [q.format(prompt=(prompts[j] if prompts else "")) for j in range(i, min(i + batch, len(pil)))]
            inp = self._inputs(pil[i:i + batch], texts)
            gen = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False)
            out += self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return out


class HPSv3:
    """MizzenAI/HPSv3 without its package: the checkpoint is a Qwen2-VL-7B (its config.json says qwen2_5_vl, but the vision
    tower is Qwen2-VL's fc1/fc2 MLP), plus rm_head = Linear(3584,1024)-ReLU-Dropout-Linear(1024,16)-ReLU-Linear(16,2) on the
    hidden state of the <|Reward|> token. Output mu (index 0)."""
    def __init__(self, device="cuda"):
        from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
        from safetensors import safe_open
        base, d = os.path.join(J, "qwen2vl_tok"), os.path.join(J, "hpsv3"); self.dev = device
        self.proc = AutoProcessor.from_pretrained(base, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
        self.proc.tokenizer.add_special_tokens({"additional_special_tokens": ["<|Reward|>"]})
        self.reward_id = self.proc.tokenizer.convert_tokens_to_ids("<|Reward|>")
        cfg = AutoConfig.from_pretrained(base); cfg.vocab_size = len(self.proc.tokenizer)
        if hasattr(cfg, "text_config"): cfg.text_config.vocab_size = len(self.proc.tokenizer)
        if not os.path.exists(os.path.join(d, "model.safetensors")): os.symlink("HPSv3.safetensors", os.path.join(d, "model.safetensors"))
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(d, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        h = cfg.text_config.hidden_size if hasattr(cfg, "text_config") else cfg.hidden_size
        self.head = torch.nn.Sequential(torch.nn.Linear(h, 1024), torch.nn.ReLU(), torch.nn.Dropout(0.05), torch.nn.Linear(1024, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)).to(device).eval()
        with safe_open(os.path.join(d, "HPSv3.safetensors"), "pt") as s:
            self.head.load_state_dict({k[len("rm_head."):]: s.get_tensor(k).float() for k in s.keys() if k.startswith("rm_head.")})

    @torch.inference_mode()
    def score(self, images_u8, prompts, batch=8):
        pil = to_pil448(images_u8); out = []
        for i in range(0, len(pil), batch):
            texts = [HPS_INSTRUCTION.format(text_prompt=p) + HPS_TAIL for p in prompts[i:i + batch]]
            msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}] for t in texts]
            chat = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            inp = self.proc(text=chat, images=pil[i:i + batch], padding=True, return_tensors="pt").to(self.dev)
            hs = self.model.model(**inp, return_dict=True).last_hidden_state
            pooled = hs[inp["input_ids"] == self.reward_id].float().view(hs.shape[0], -1)
            out.append(self.head(pooled)[:, 0].cpu().numpy())
        return np.concatenate(out)


class DinoProbe:
    """P(real photograph) from a logistic probe on DINOv2 CLS features (trained by scripts/rl/train_dino_probe.py)."""
    def __init__(self, path=os.path.join(J, "dino_probe.pt"), device="cuda"):
        st = torch.load(path, map_location="cpu"); self.mu, self.sd, self.w, self.b = st["mu"], st["sd"], st["w"], st["b"]; self.dev = device
    @torch.no_grad()
    def __call__(self, images_u8):
        from .. import metrics
        f = torch.tensor(metrics._dino_feats(images_u8, self.dev)).float()
        return torch.sigmoid(((f - self.mu) / self.sd) @ self.w + self.b).numpy()
