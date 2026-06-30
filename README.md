# TRINITY: A Multi-Perspective Benchmark for Personal-Style Video Highlight Detection (ECCV 2026)

[![GitHub](https://img.shields.io/badge/GitHub-TRINITY-181717?logo=github)](https://github.com/vanilladucky/TRINITY)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-TRINITY-yellow)](https://huggingface.co/datasets/vanilladucky/TRINITY)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official PyTorch implementation of **TRINITY**, a multi-perspective benchmark that decomposes video highlight saliency into three complementary perspectives — **Event**, **Emotion**, and **Nature** — within a unified temporal framework, together with a shared-backbone, multi-branch architecture that predicts all three perspectives in parallel from a single input sequence.

> Traditional video highlight detection relies on a narrow, event-centric definition of saliency, which often fails to generalize to unconstrained personal videos where highlights are heterogeneous and perspective-dependent. TRINITY addresses this by decomposing highlight saliency into Event, Emotion, and Nature perspectives, and proposes a shared-backbone multi-branch architecture with view-specific experts for parallel multi-perspective prediction.

## 📢 News

- **[2026]** Code and the TRINITY dataset are publicly released.

## Table of Contents

- [🛠️ Installation](#installation)
- [🗂️ Dataset](#dataset)
- [🏗️ Model](#model)
- [🏋️ Training](#training)
- [🧪 Evaluation](#evaluation)
- [📝 Citation](#citation)
- [🙏 Acknowledgements](#acknowledgements)
- [⚖️ License](#license)

## 🛠️ Installation

```bash
git clone https://github.com/vanilladucky/TRINITY.git
cd TRINITY
conda create -n trinity python=3.10 -y
conda activate trinity
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch >= 2.0, and CUDA 11.8+. A single modern GPU (>= 16GB) is sufficient, since the model operates on pre-extracted 512-dim CLIP ViT-B/32 features rather than raw video.

## 🗂️ Dataset

The TRINITY benchmark — pre-extracted CLIP ViT-B/32 features, ground truth, and splits for the Event, Emotion, and Nature perspectives — is hosted on Hugging Face:

**[https://huggingface.co/datasets/vanilladucky/TRINITY](https://huggingface.co/datasets/vanilladucky/TRINITY)**

Each perspective ships as numbered zip shards of per-video `.npz` (and, for Nature ground truth, `.json`) files, plus a single split file:

| Perspective | Features | Ground truth | Split file |
|---|---|---|---|
| Event | `event/event_features_part_{1..145}.zip` — `video_<id>.npz`, key `"data"`, shape `(T, 512)`, float32, 1 fps | `event/event_gt_part_1.zip` — `video_<id>.npz`, key `"data"`, frame-level replay-saliency scores | `mr_hisum_split_original.json` (27,892 / 2,000 / 2,000 train/val/test) |
| Emotion | `emotion/emotion_features_part_{1..11}.zip` — `<clip_id>.npz`, key `"features"`, shape `(T, 512)`, float16 | `emotion/emotion_gt_part_1.zip` — `<clip_id>.npz` with `labels`, `clip_start`, `clip_end`, `clip_duration`, `num_segments`, `segment_duration`, `is_negative` | `emotion_split.json` (16,178 / 1,759 / 1,600) |
| Nature | `nature/nature_feat_part_{1..32}.zip` — `<video_id>.npz`, key `"features"`, shape `(T, 512)`, float16 (includes `_aug` variants), 1 fps | `nature/nature_gt_part_1.zip` — `<video_id>.json`, a list of per-frame aesthetic scores in `[0, 1]` | `nature_split.json` (12,763 / 874 / 1,903) |

> **Note**: For the Event perspective, ground-truth highlight scores follow the replay-based "Most Replayed" supervision introduced by Mr. HiSum; TRINITY adopts its annotations directly rather than re-deriving them.

### Downloading

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='vanilladucky/TRINITY', repo_type='dataset', local_dir='data/trinity_raw')
"
```

Unzip the shards for whichever perspectives you need:

```bash
cd data/trinity_raw
for f in event/event_features_part_*.zip; do unzip -o "$f" -d event_features/; done
unzip -o event/event_gt_part_1.zip -d event_gt/
```

Each feature/ground-truth `.npz` can then be loaded directly with NumPy:

```python
import numpy as np
feat = np.load("event_features/video_1.npz")["data"]  # (T, 512)
```

> **Note**: `MrHiSumDataset` in [`model/full_dataset.py`](model/full_dataset.py) currently reads the Event perspective from a single combined `.h5` file (`features` / `gtscore` keys per video group) rather than the per-video `.npz` shards above. If loading directly from the Hugging Face release, adapt `MrHiSumDataset` to read the `.npz` shards (key `"data"`) or build an `.h5` file from them first. `EmotionDataset` and `NatureDataset` already expect the per-video `.npz` / `.json` layout used by the release.

## 🏗️ Model

```
.
├── train.py                                # Main entry point: joint pretraining / evaluation on TRINITY
├── model/
│   ├── configs.py                          # Argument -> Config object, experiment directory bookkeeping
│   ├── full_dataset.py                     # Event / Emotion / Nature Dataset classes + collator
│   ├── solver.py                           # Checkpoint loading utility
│   ├── solver_combined_5s.py               # Main solver: training loop, evaluation, gradient diagnostics (Fig. 4)
│   ├── solver_combined_aligned_mtl.py      # Solver variant using Aligned-MTL instead of PCGrad (Fig. 4a)
│   ├── pcgrad.py                           # PCGrad (Yu et al., NeurIPS 2020) gradient-surgery optimizer wrapper
│   └── aligned_mtl.py                      # Aligned-MTL (Senushkin et al., CVPR 2023) gradient alignment
└── networks/
    ├── mlp.py                              # MLP baseline
    ├── pgl_sum/                            # PGL-SUM baseline (Koutras & Maragos, ISM 2021)
    ├── vasnet/                             # VASNet baseline (Fajtl et al.)
    └── sl_module/
        ├── transformer.py                  # Single-stream Transformer backbone with RoPE
        ├── transformer_split_heads.py      # Shared + per-perspective split-head Transformer (proposed)
        ├── score_net.py                    # Per-perspective scoring head
        ├── sl_module_combined.py           # Single-backbone wrapper (SL_module baseline)
        └── sl_module_split_heads.py        # Proposed shared-backbone, multi-branch model
```

A shared temporal Transformer backbone captures task-agnostic global context, while per-perspective expert attention heads specialize in Event-, Emotion-, or Nature-driven saliency. RoPE (rotary position embedding) injects relative temporal position directly into self-attention.

The proposed model corresponds to `--model SL_module_split` in `train.py`, which builds `SL_module_split_heads` from [`networks/sl_module/sl_module_split_heads.py`](networks/sl_module/sl_module_split_heads.py). The `MLP`, `PGL_SUM`, `VASNet`, and `SL_module` options are single-perspective baselines used for comparison in the paper's ablations and are not actively maintained beyond that role.

## 🏋️ Training

Joint pretraining on all three TRINITY perspectives:

```bash
python train.py \
    --train true \
    --device cuda:0 \
    --model SL_module_split \
    --epochs 50 \
    --batch_size 16 \
    --lr 5e-5 \
    --l2_reg 1e-4 \
    --dropout_ratio 0.2 \
    --tag trinity_pretrain
```

This jointly optimizes the shared backbone and the three perspective-specific expert branches with round-robin batching (one mini-batch per perspective per step), and reports per-perspective `mAP@50%` / `mAP@15%` on the validation split each epoch. Checkpoints, training history, and gradient-conflict diagnostics (Fig. 4 in the paper) are written to `Summaries/<model>/<tag>/`.

**Useful flags:**
- `--use_CovLoss`: enable the covariance-based orthogonality regularizer between perspective embeddings (only with `--model SL_module_split`).
- `--use_smoothap`: add the SmoothAP ranking loss and a temporal-gradient consistency term on top of BCE.
- `--projection_dim_count`: project input features to a different dimension before the backbone (default: 512, i.e. no projection).
- `--profile_fps`: after training (or loading a checkpoint), measure feature-level inference throughput.

### Gradient-conflict ablation (Aligned-MTL)

[`model/solver_combined_aligned_mtl.py`](model/solver_combined_aligned_mtl.py) is a drop-in alternative to `model/solver_combined_5s.py` that replaces PCGrad with Aligned-MTL for combining the three task gradients, used to produce the "With Aligned-MTL" comparison in Fig. 4a of the paper.

## 🧪 Evaluation

Evaluate an existing checkpoint on all three perspectives:

```bash
python train.py --train false --ckpt_path /path/to/checkpoint.pkl --device cuda
```

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{trinity2026,
  title     = {TRINITY: A Multi-Perspective Benchmark for Personal-Style Video Highlight Detection},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## 🙏 Acknowledgements

The `PGL_SUM` and `VASNet` baseline implementations under `networks/` are adapted from their original authors' public releases. `model/pcgrad.py` and `model/aligned_mtl.py` implement, respectively, PCGrad (Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020) and Aligned-MTL (Senushkin et al., *Independent Component Alignment for Multi-Task Learning*, CVPR 2023).

## ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
