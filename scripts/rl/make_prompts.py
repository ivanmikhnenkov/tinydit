"""Prompt sets for stage 2.
  train.json : 20k COCO train captions (real photos, short) + 10k synthetic
               "subject doing something somewhere" prompts. No FLUX-style captions: they pull toward the render look.
  eval.json  : fixed, never trained on: the 45 DDPO animals ("a photo of a {animal}"), the 40 hand-written prompts of the judge
               study, 40 held-out real captions; plus the 16-image grid prompts and seeds.
    .venv/bin/python scripts/rl/make_prompts.py"""
import json, random, sys
sys.path.insert(0, "src")
from tinydit.rl.rewards import ANIMALS, prompt as animal_prompt
random.seed(11)
ann = json.load(open("out/data/coco/annotations/captions_train2017.json"))["annotations"]   # 590k human captions, real photos
caps = sorted({a["caption"].strip().rstrip(".") for a in ann if 4 <= len(a["caption"].split()) <= 30})
random.shuffle(caps); caps = caps[:20000]
subjects = ["dog", "cat", "horse", "rabbit", "fox", "raccoon", "bear", "deer", "owl", "penguin", "duck", "goat", "cow", "pig", "sheep", "monkey",
            "elephant", "giraffe", "zebra", "lion", "tiger", "squirrel", "hedgehog", "otter", "koala", "kangaroo", "parrot", "eagle", "frog", "turtle",
            "wolf", "llama", "camel", "panda", "red panda", "hamster", "goose", "flamingo", "seal", "walrus"]
actions = ["riding a bicycle", "reading a newspaper", "playing chess", "holding an umbrella", "wearing sunglasses", "wearing a red scarf",
           "sitting at a desk with a laptop", "driving a car", "playing the piano", "eating ice cream", "wearing a raincoat", "carrying a backpack",
           "sitting in an armchair", "drinking from a cup", "wearing a straw hat", "playing football", "sleeping on a sofa", "standing on a surfboard",
           "holding a balloon", "wearing a bow tie", "looking through binoculars", "sitting in a shopping cart", "wearing headphones",
           "painting on a canvas", "playing a guitar", "riding a skateboard", "wearing a birthday hat", "sitting on a suitcase", "wearing boots", "holding a flower"]
places = ["on a beach", "in a library", "in a kitchen", "on a city street", "in a snowy forest", "in a garden", "on a mountain trail",
          "in a train station", "at a café", "in a living room", "on a wooden bridge", "in a meadow", "at a bus stop", "in a supermarket",
          "on a rooftop at sunset", "in a barn", "next to a lake", "in an office", "on a boat", "in a park in autumn", "in a greenhouse", "on a farm"]
objects = ["a red bicycle", "a wooden chair", "a blue umbrella", "a vintage camera", "a stack of books", "a cup of coffee", "a bowl of fruit",
           "a yellow car", "a guitar", "a bouquet of tulips", "a lantern", "a pair of boots", "a birthday cake", "a teapot", "a globe", "a violin"]
synth = set()
art = lambda w: ("an " if w[0] in "aeiou" else "a ") + w
while len(synth) < 8000: synth.add(f"{art(random.choice(subjects))} {random.choice(actions)} {random.choice(places)}")
for o in objects:
    for pl in places: synth.add(f"{o} {pl}")            # all 352 object x place prompts
train = [{"prompt": c, "source": "real_caption"} for c in caps] + [{"prompt": s, "source": "synthetic"} for s in sorted(synth)]
random.shuffle(train)
S = json.load(open("prompts/rl_eval/set_v1.json"))
hand = [p for p in S["prompts"] if p["source"].startswith("hand")]; held = [p for p in S["prompts"] if p["source"] == "heldout_real"]
ev = [{"id": f"animal{i:02d}", "prompt": animal_prompt(a), "source": "animal"} for i, a in enumerate(ANIMALS)] \
   + [{"id": p["id"], "prompt": p["prompt"], "source": p["source"]} for p in hand] \
   + [{"id": p["id"], "prompt": p["prompt"], "source": "heldout_real"} for p in held]
grid = [p["id"] for p in hand if p["id"] in ("usual00", "usual03", "usual09", "usual12", "usual16", "usual19", "odd00", "odd01", "odd02", "odd03", "odd04", "odd06", "odd10", "odd13", "odd16", "odd19")]
json.dump(train, open("prompts/rl_stage2/train.json", "w"))
json.dump({"prompts": ev, "seeds": 2, "grid": grid, "grid_seed": 0}, open("prompts/rl_stage2/eval.json", "w"), indent=1)
from collections import Counter
print("train:", Counter(t["source"] for t in train), "| eval:", Counter(p["source"] for p in ev), "| grid", len(grid))
print("  e.g.", train[0]["prompt"][:80], "|", [t["prompt"] for t in train if t["source"] == "synthetic"][:2])
