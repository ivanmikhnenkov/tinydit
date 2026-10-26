"""Text-to-image pipeline for tinydit-256: frozen flan-t5-base text encoder, the tinydit transformer,
frozen FLUX.2 autoencoder. Sampling is the training-time sampler: Euler steps on the SD3 timestep shift
(2.8, in the scheduler config) with classifier-free guidance against the empty prompt.

    pipe = DiffusionPipeline.from_pretrained("ivanmikhnenkov/tinydit-256", trust_remote_code=True)
    image = pipe("a lighthouse on a rocky coast at sunset", width=320, height=208).images[0]
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor


class TinyDiTPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(self, tokenizer, text_encoder, transformer, vae, scheduler):
        super().__init__()
        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer,
                              vae=vae, scheduler=scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        patch = self.transformer.config.patch_size if getattr(self, "transformer", None) else 2
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * patch)

    def _embed(self, prompts: list[str], max_sequence_length: int, device):
        """-> T5 states (B,L,D), mask (B,L) bool, L = longest prompt in the batch."""
        tok = self.tokenizer(prompts, padding="longest", max_length=max_sequence_length, truncation=True,
                             return_tensors="pt").to(device)
        emb = self.text_encoder(input_ids=tok.input_ids, attention_mask=tok.attention_mask).last_hidden_state
        return emb, tok.attention_mask.bool()

    @staticmethod
    def _pad(emb, mask, L):
        """Right-pad to length L with masked zeros (padding never reaches cross-attention)."""
        B, l, D = emb.shape
        if l == L:
            return emb, mask
        return (torch.cat([emb, emb.new_zeros(B, L - l, D)], 1),
                torch.cat([mask, mask.new_zeros(B, L - l)], 1))

    def encode_prompt(self, prompt, negative_prompt="", do_classifier_free_guidance=True,
                      num_images_per_prompt=1, device=None, max_sequence_length=128):
        """-> (prompt_embeds, prompt_mask, negative_embeds, negative_mask), all padded to one length.
        The empty negative prompt is the null caption tinydit was trained with (10% caption dropout)."""
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        emb, mask = self._embed(prompt, max_sequence_length, device)
        neg_emb = neg_mask = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = [negative_prompt] * len(prompt) if isinstance(negative_prompt, str) else list(negative_prompt)
            if len(negative_prompt) != len(prompt):
                raise ValueError(f"got {len(negative_prompt)} negative prompts for {len(prompt)} prompts")
            neg_emb, neg_mask = self._embed(negative_prompt, max_sequence_length, device)
            L = max(emb.shape[1], neg_emb.shape[1])
            emb, mask = self._pad(emb, mask, L)
            neg_emb, neg_mask = self._pad(neg_emb, neg_mask, L)
            neg_emb = neg_emb.repeat_interleave(num_images_per_prompt, 0)
            neg_mask = neg_mask.repeat_interleave(num_images_per_prompt, 0)
        emb = emb.repeat_interleave(num_images_per_prompt, 0)
        mask = mask.repeat_interleave(num_images_per_prompt, 0)
        return emb, mask, neg_emb, neg_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] = "",
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 128,
        callback_on_step_end: Callable | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ):
        """Trained at ~256x256 pixels in five shapes: 256x256, 288x224, 224x288, 320x208, 208x320 (W x H).
        Any multiple of 16 works (2D RoPE), quality drops away from that area. 20 steps is the training-time
        sampler, 50 is a little better; the timestep shift matters more than the step count."""
        div = self.vae_scale_factor * self.transformer.config.patch_size
        if height % div or width % div:
            raise ValueError(f"height and width must be multiples of {div}, got {height}x{width}")
        device = self._execution_device
        do_cfg = guidance_scale > 1.0
        emb, mask, neg_emb, neg_mask = self.encode_prompt(prompt, negative_prompt, do_cfg, num_images_per_prompt,
                                                          device, max_sequence_length)
        dtype = self.transformer.dtype
        if do_cfg:
            emb, mask = torch.cat([neg_emb, emb]), torch.cat([neg_mask, mask])
        emb = emb.to(dtype)

        batch = mask.shape[0] // (2 if do_cfg else 1)
        shape = (batch, self.transformer.config.in_channels, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:   # the Euler state stays fp32 whatever the transformer dtype, as in training
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        else:
            latents = latents.to(device=device, dtype=torch.float32)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                x = torch.cat([latents] * 2) if do_cfg else latents
                noise_pred = self.transformer(x.to(dtype), t.expand(x.shape[0]), emb, mask, return_dict=False)[0].float()
                if do_cfg:
                    uncond, cond = noise_pred.chunk(2)
                    noise_pred = uncond + guidance_scale * (cond - uncond)
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            cfg = self.transformer.config
            mean = torch.tensor(cfg.latents_mean, device=device).view(1, -1, 1, 1)
            std = torch.tensor(cfg.latents_std, device=device).view(1, -1, 1, 1)
            z = (latents * std + mean).to(self.vae.dtype)
            image = self.vae.decode(z, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
