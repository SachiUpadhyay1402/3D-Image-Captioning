# 3D Image Captioning

Automatic caption generation for 3D objects using the ModelNet10 dataset. 3D meshes are rendered into 2D images and passed through a BLIP vision-language model to produce natural language descriptions - bridging 3D vision and NLP.

---

## Overview

| Item | Detail |
|------|--------|
| **Dataset** | ModelNet10 (10 object categories) |
| **Core Model** | `Salesforce/blip-image-captioning-large` |
| **Baseline** | BLIP zero-shot captioning |
| **Custom Model** | Transformer with category-aware context |
| **Platform** | Google Colab (GPU) |

---

## Pipeline

```
ModelNet10 (.zip)
      │
      ▼
 3D Mesh Loading (trimesh)
      │
      ▼
 Photorealistic Rendering → 2D Image (640×640)
      │
      ▼
 BLIP Captioning + Category Context Injection
      │
      ▼
 Generated Captions + Evaluation Metrics
```

---

## Categories

The model handles all 10 ModelNet10 classes:

`bathtub` · `bed` · `chair` · `desk` · `dresser` · `monitor` · `night_stand` · `sofa` · `table` · `toilet`

---

## How It Works

1. **Upload** - Upload `ModelNet10.zip` to Colab via `files.upload()`
2. **Render** - Each `.off` 3D mesh is rendered into a 640×640 2D image with photorealistic shading and a 45° rotation
3. **Caption** - The rendered image is passed to BLIP with the object category as context to generate a descriptive sentence
4. **Evaluate** - Captions are scored against reference descriptions using standard NLP metrics

---

## Evaluation Metrics

| Metric | Transformer Model | BLIP Baseline |
|--------|:-----------------:|:-------------:|
| BLEU | 0.5231 | 0.4952 |
| METEOR | 0.4617 | 0.4383 |
| ROUGE-L | 0.6079 | 0.5821 |
| CIDEr | 1.0824 | 0.9743 |
| SPICE | 0.2198 | 0.2014 |

The Transformer model with context injection outperforms the BLIP zero-shot baseline across all metrics.

---

## Dependencies

```bash
pip install trimesh pillow matplotlib torch torchvision \
            transformers opencv-python scipy evaluate \
            nltk rouge_score tabulate
```

---

## Output

- **Per-object captions** printed inline
- **`caption_evaluation_results.csv`** - metric scores table
- **`3D_Captioning_TechniqueComparison.png`** - line graph comparing Transformer vs BLIP baseline

---

## Tech Stack

- **PyTorch** - model inference
- **Hugging Face Transformers** - BLIP model
- **Trimesh** - 3D mesh loading & processing
- **OpenCV / Matplotlib** - rendering pipeline
- **Hugging Face Evaluate** - BLEU, METEOR, ROUGE-L, CIDEr, SPICE
