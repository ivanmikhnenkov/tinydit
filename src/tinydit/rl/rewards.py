"""Rewards for the RL stage.

The first two are DDPO's (Black et al. 2023) sanity objectives: the JPEG file size of the decoded
image at quality 95, in kB, negated for compressibility. They need no reward model, cannot be
argued with, and pull in opposite directions, so an algorithm that moves both is doing something
real. Prompts are DDPO's 45 common animals.
"""
from __future__ import annotations
import io
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ANIMALS = ["cat", "dog", "horse", "monkey", "rabbit", "zebra", "spider", "bird", "sheep", "deer", "cow", "goat",
           "lion", "tiger", "bear", "raccoon", "fox", "wolf", "lizard", "beetle", "ant", "butterfly", "fish", "shark",
           "whale", "dolphin", "squirrel", "mouse", "rat", "snake", "turtle", "frog", "chicken", "duck", "goose",
           "bee", "pig", "turkey", "fly", "llama", "camel", "bat", "gorilla", "hedgehog", "kangaroo"]
OBJECTIVES = {"compress": -1.0, "incompress": +1.0}      # reward = sign * kB


def prompt(animal: str) -> str:
    return f"a photo of a {animal}"


def _kb(img_u8_chw, quality):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img_u8_chw.transpose(1, 2, 0))).save(buf, format="JPEG", quality=quality)
    return buf.tell() / 1000.0


def jpeg_kb(images_u8, quality: int = 95) -> np.ndarray:
    """images_u8: (N,3,H,W) -> JPEG size in kB per image."""
    with ThreadPoolExecutor(8) as ex:
        return np.array(list(ex.map(lambda im: _kb(im, quality), images_u8)), dtype=np.float32)
