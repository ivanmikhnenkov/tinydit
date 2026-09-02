# Interfaces shared between ingest, training and dashboard (run 1)

## Cache layout: `out/cache/<name>/`  (name = `run1`, smoke tests use `smoke`)
```
meta.json               {"ae":"flux2","latent_ch":32,
                         "buckets":{"1_1":[256,256],"4_3":[288,224],"3_4":[224,288],"3_2":[320,208],"2_3":[208,320]},   # [W,H] pixels
                         "sources":["coco","pexels","flux"]}
stats.json              {"mean":[32 floats],"std":[32 floats]}      per-channel latent whitening over all buckets (computed at the end; ingest may refresh it)
<bucket>/lat.npy        float16 npy memmap (capacity, 32, H/8, W/8); valid rows = number of lines in rows.jsonl (extra rows are zeros)
<bucket>/rows.jsonl     one JSON object per latent row, same order as lat.npy:
                          {"src":"coco|pexels|flux", "id":"<source id>", "long":[str,...], "short":[str,...],
                           "crop":0.0-0.36 (fraction of the long side removed), "w":orig_w, "h":orig_h, "val":true|false}
                          `long` = detailed captions (any one may be used), `short` = short captions (any one may be used); both non-empty lists.
<bucket>/val_images.npy uint8 (n_val, 3, H, W) pixels of the rows flagged val, in the order they appear in rows.jsonl (for FID references)
progress.json           ingest bookkeeping (per source), free-form
```
Bucket assignment (real = coco, pexels; synthetic = flux), aspect a = w/h, crop(b) = 1 - min(a, a_b)/max(a, a_b):
- drop if a > 2.0 or a < 0.5
- real: nearest bucket by crop; if crop(1_1) <= 0.30 (i.e. 4:3 / 3:4 frames) route to 1_1 with probability 0.5
- flux (all 1024x1024): 1_1 with p = 0.45, else one of the other four uniformly (crop 22% for 4:3/3:4, 35% for 3:2/2:3)
- resize so the image covers the bucket, centre-crop the residual, LANCZOS. Encode with FLUX.2 AE posterior mean, store fp16.
- caption rule for flux: long = caption_detail unless crop > 0.20, then caption_entity; short = caption_entity. coco: long = GPT-4V caption(s), short = the 5 human captions. pexels: long = i1 Qwen3-VL captions (join by Pexels id), short = cogvlm_caption; if the join misses, long = short.
- val: the first 1000 successfully ingested rows of each source are flagged val (spread over buckets naturally). Never train on val rows.

## Training sampling
Source weights w = {pexels: .60, flux: .25, coco: .15}. Draw a bucket b with P(b) = sum_s w_s P(b|s) (P(b|s) from row counts); within the batch draw each row's source with P(s|b) proportional to w_s P(b|s), then a uniform non-val row of that source in that bucket.
Caption per row: 10% empty, 50% random `long` (T5, max 128 tokens), 40% random `short` (max 48 tokens); long and short encoded as separate sub-batches padded to their own longest.

## metrics.jsonl keys (train.py writes, dashboard.html reads)
Every log interval: step, loss, loss_mse, loss_cos, loss_disp, lr, gnorm, ips, secs, bucket ("1_1"...), tokens
Every val interval: step, val_loss, val_loss_coco, val_loss_pexels, val_loss_flux
Every sample interval: step, sample="samples/<step>.png"
Every eval interval: step, eval="evals/<step>.png", clip_unseen, clip_novel, pickscore_unseen, pickscore_novel, hpsv2_unseen, hpsv2_novel, fid, fd_dino, obj_acc
Prompt sets: UNSEEN = captions of val rows (never trained on), NOVEL = out/prompts_novel.txt (hand-written).
Sample grids show all five bucket shapes.
