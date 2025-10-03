
- `q1.ipynb` — Vision Transformer (ViT) implementation in PyTorch, training on CIFAR-10.
- `q2.ipynb` — Pipeline for text-prompted segmentation using SAM 2 (installation + demo code).
- This README — how to run, a suggested best config, and tips to humanize your write-up.
---
## How to use (step-by-step, Colab)
1. Open Google Colab: https://colab.research.google.com
2. Upload the notebook file (File → Upload notebook) or open from GitHub after you push.
3. **Runtime** → Change runtime type → select **GPU** (e.g., Tesla T4).
4. Run cells from top to bottom. Follow the inline comments.
---
## Q1 — ViT on CIFAR-10 (summary)
- Patchify images, add learnable positional embeddings and a CLS token.
- Basic Transformer encoder with Multi-Head Self-Attention, MLP, residual connections and LayerNorm.
- Training loop with augmentation, optional AdamW + cosine LR scheduler.
- Save best model checkpoint and print test accuracy.
### Suggested best config (starter)
- patch_size: 4 (for CIFAR-10 32x32, try 4 or 8)
- dim: 256
- depth (num encoder blocks): 8
- heads: 8
- mlp_ratio: 4.0
- batch_size: 128
- epochs: 60 (try 100 for better accuracy)
- optimizer: AdamW, lr: 3e-4, weight_decay: 0.05
- scheduler: CosineAnnealingLR or Cosine with Warmup

Short results table (example you should fill after running):
| Model | Patch | Depth | Dim | Best Test Acc (%) |
|-------|-------|-------|-----|-------------------|
| ViT-starter | 4x4 | 8 | 256 | *run and fill* |

---

## Q2 — Text-driven segmentation with SAM 2 (summary)
- The notebook shows how to convert a text prompt to region seeds (using CLIPSeg / GroundingDINO / GLIP approaches).
- Seeds are passed to SAM 2 to get masks, and the final mask is overlaid on the image.
- **Note:** SAM 2 and some detectors require model weights. The cells include `git clone` + install commands and placeholders to download recommended weights. On Colab you'll need to download weights (links provided in notebook comments).

Limitations:
- Accuracy depends on detector + SAM weights.
- Some models need GPU and internet to download weights.
- Fragmented or ambiguous prompts can fail — try short, descriptive prompts.

