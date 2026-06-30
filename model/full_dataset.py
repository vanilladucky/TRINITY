import json
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# ==========================================
# Dataset roots
# ==========================================
# All paths below can be overridden via environment variables so the code
# does not depend on any particular machine's directory layout. Defaults
# assume a `data/` directory at the repository root, populated by
# downloading https://huggingface.co/datasets/vanilladucky/TRINITY and
# extracting each perspective's feature/ground-truth zip shards into a flat
# directory (see README.md for the exact download + unzip commands).

DATA_ROOT = os.environ.get("TRINITY_DATA_ROOT", "data")

# ==========================================
# 1. Event Dataset Paths
# ==========================================
EVENT_SPLIT_JSON = os.environ.get(
    "TRINITY_EVENT_SPLIT",
    os.path.join(DATA_ROOT, "event", "mr_hisum_split_original.json")
)
EVENT_FEATURE_ROOT = os.environ.get(
    "TRINITY_EVENT_FEATURES",
    os.path.join(DATA_ROOT, "event", "event_features")
)
EVENT_GT_ROOT = os.environ.get(
    "TRINITY_EVENT_GT",
    os.path.join(DATA_ROOT, "event", "event_gt")
)

# ==========================================
# 2. Emotion Dataset Paths
# ==========================================
EMOTION_SPLIT_JSON = os.environ.get(
    "TRINITY_EMOTION_SPLIT",
    os.path.join(DATA_ROOT, "emotion", "emotion_split.json")
)
EMOTION_FEATURE_ROOT = os.environ.get(
    "TRINITY_EMOTION_FEATURES",
    os.path.join(DATA_ROOT, "emotion", "features")
)
EMOTION_GT_ROOT = os.environ.get(
    "TRINITY_EMOTION_GT",
    os.path.join(DATA_ROOT, "emotion", "ground_truth")
)

# ==========================================
# 3. Nature Dataset Paths
# ==========================================
NATURE_SPLIT_JSON = os.environ.get(
    "TRINITY_NATURE_SPLIT",
    os.path.join(DATA_ROOT, "nature", "nature_split.json")
)
NATURE_FEATURE_ROOT = os.environ.get(
    "TRINITY_NATURE_FEATURES",
    os.path.join(DATA_ROOT, "nature", "features")
)
NATURE_GT_ROOT = os.environ.get(
    "TRINITY_NATURE_GT",
    os.path.join(DATA_ROOT, "nature", "ground_truth")
)


# ==========================================
# Data augmentation helpers
# ==========================================

def add_gaussian_noise(features, mean=0.0, std=0.02):
    """Add random Gaussian noise to the feature embeddings."""
    noise = torch.randn_like(features) * std + mean
    return features + noise


def time_masking(features, mask_prob=0.1):
    """Randomly zero out entire time steps (segments)."""
    seq_len = features.shape[0]
    mask = (torch.rand(seq_len) > mask_prob).float()
    mask = mask.unsqueeze(1).to(features.device)
    return features * mask


def channel_dropout(features, drop_prob=0.1):
    """Randomly zero out specific feature channels across all time steps."""
    input_dim = features.shape[1]
    mask = (torch.rand(input_dim) > drop_prob).float()
    mask = mask.unsqueeze(0).to(features.device)
    return features * mask


def semantic_peak_swap(features, gt, margin=2, peak_thresh=0.4, quiet_thresh=0.1):
    """
    Augmentation that locates a highlight peak and swaps it with a quiet
    background region, forcing the model to rely on content rather than
    positional bias within the clip.

    Args:
        features: (T, D) tensor.
        gt: (T,) tensor of ground-truth scores.
        margin: number of frames around the peak to include as context.
        peak_thresh: minimum score for a frame to be considered a peak.
        quiet_thresh: maximum mean score for a region to be considered quiet.
    """
    T = features.shape[0]

    peak_indices = torch.nonzero(gt > peak_thresh).flatten()
    if len(peak_indices) == 0:
        return features, gt  # No peak found in this clip; skip augmentation.

    idx = torch.randint(0, len(peak_indices), (1,)).item()
    peak_center = peak_indices[idx].item()

    p_start = max(0, peak_center - margin)
    p_end = min(T, peak_center + margin)
    window_len = p_end - p_start

    found_target = False
    q_start = 0

    for _ in range(10):
        rand_start = torch.randint(0, T - window_len, (1,)).item()

        if (rand_start < p_end) and (rand_start + window_len > p_start):
            continue

        region_score = gt[rand_start: rand_start + window_len].mean()

        if region_score < quiet_thresh:
            q_start = rand_start
            found_target = True
            break

    if not found_target:
        return features, gt

    q_end = q_start + window_len

    feat_peak = features[p_start:p_end].clone()
    gt_peak = gt[p_start:p_end].clone()

    feat_quiet = features[q_start:q_end].clone()
    gt_quiet = gt[q_start:q_end].clone()

    features[p_start:p_end] = feat_quiet
    gt[p_start:p_end] = gt_quiet

    features[q_start:q_end] = feat_peak
    gt[q_start:q_end] = gt_peak

    return features, gt


def apply_augmentations(features, gt=None):
    if gt is None:
        return features
    features, gt = semantic_peak_swap(features, gt, margin=5)
    return features, gt


# ==========================================
# 1. Event Dataset (Mr. HiSum replay-based annotations)
# ==========================================

class MrHiSumDataset(Dataset):
    """
    Event-perspective dataset, backed by per-video .npz files as released on
    https://huggingface.co/datasets/vanilladucky/TRINITY (event_features_part_*.zip
    and event_gt_part_1.zip). Both the feature and ground-truth .npz files
    store their array under the key "data": features as (T, 512) float32,
    ground truth as (T,) replay-saliency scores in [0, 1].

    Some videos referenced in the split file are not present in the released
    ground-truth shard (a small number of source videos were dropped during
    feature extraction); these are skipped with a warning rather than raising.
    """

    def __init__(self, mode, segment_duration=5, augment=False):
        self.mode = mode
        self.segment_duration = segment_duration
        self.augment = augment

        with open(EVENT_SPLIT_JSON, "r") as f:
            self.data = json.loads(f.read())

        key_name = self.mode + "_keys" if self.mode + "_keys" in self.data else self.mode
        self.raw_keys = list(self.data[key_name])

        self.valid_video_list = []
        missing = 0

        print(f"Scanning {len(self.raw_keys)} keys from {EVENT_FEATURE_ROOT} / {EVENT_GT_ROOT} for {mode}...")

        for vkey in self.raw_keys:
            feat_path = os.path.join(EVENT_FEATURE_ROOT, f"{vkey}.npz")
            gt_path = os.path.join(EVENT_GT_ROOT, f"{vkey}.npz")

            if os.path.exists(feat_path) and os.path.exists(gt_path):
                self.valid_video_list.append({
                    "video_id": vkey,
                    "feat_path": feat_path,
                    "gt_path": gt_path,
                })
            else:
                missing += 1

        print(f"[MrHiSumDataset:{mode}] kept={len(self.valid_video_list)} (missing={missing})")

    def __len__(self):
        return len(self.valid_video_list)

    def _aggregate_features(self, features, segment_duration):
        T, D = features.shape
        n_segments = T // segment_duration
        if n_segments == 0:
            return features.mean(axis=0, keepdims=True)
        truncated = features[:n_segments * segment_duration]
        reshaped = truncated.reshape(n_segments, segment_duration, D)
        return reshaped.mean(axis=1)

    def _aggregate_gtscore(self, gtscore, segment_duration):
        T = len(gtscore)
        n_segments = T // segment_duration
        if n_segments == 0:
            return np.array([gtscore.mean()])
        truncated = gtscore[:n_segments * segment_duration]
        reshaped = truncated.reshape(n_segments, segment_duration)
        aggregated = reshaped.mean(axis=1)

        # Top-15% rule: clips already among the most salient 15% are saturated to 1.0.
        threshold_idx = int(np.ceil(len(aggregated) * 0.85))
        sorted_indices = np.argsort(aggregated)
        top_15_percent_indices = sorted_indices[threshold_idx:]
        aggregated[top_15_percent_indices] = 1.0
        return aggregated

    def __getitem__(self, index):
        record = self.valid_video_list[index]

        with np.load(record['feat_path']) as feat_data:
            raw_feats = np.array(feat_data['data'], dtype=np.float32)
        with np.load(record['gt_path']) as gt_data:
            gts = np.array(gt_data['data'], dtype=np.float32)

        min_len = min(len(raw_feats), len(gts))
        raw_feats = raw_feats[:min_len]
        gts = gts[:min_len]

        feats_5s = self._aggregate_features(raw_feats, self.segment_duration)
        gts_5s = self._aggregate_gtscore(gts, self.segment_duration)

        d = {
            "video_id": record['video_id'],
            "features": torch.tensor(feats_5s, dtype=torch.float32),
            "gtscore": torch.tensor(gts_5s, dtype=torch.float32),
        }

        T = d["features"].shape[0]
        if len(gts_5s) != T:
            min_t = min(len(gts_5s), T)
            d['features'] = d['features'][:min_t]
            d['gtscore'] = d['gtscore'][:min_t]

        if self.mode != "train":
            n_frames = d["features"].shape[0]
            d["n_frames"] = np.array(n_frames)
            d["picks"] = np.arange(n_frames)
            d["gt_summary"] = np.expand_dims(d["gtscore"].numpy(), axis=0)

        return d


# ==========================================
# 2. Emotion Dataset (NPZ features + binary labels)
# ==========================================


import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """Emotion-perspective dataset, backed by per-video .npz files as released

    on https://huggingface.co/datasets/vanilladucky/TRINITY
    (emotion_features_part_*.zip and emotion_gt_part_1.zip). Split-file keys
    are expected to be the exact .npz filenames with sub-folders, e.g.
    "ActivityNet_Captions_raw/v_---9CpRcKoU_0.npz".
    """

    def __init__(self, mode, augment=False):
        self.mode = mode
        self.augment = augment
        self.feature_root = EMOTION_FEATURE_ROOT
        self.gt_root = EMOTION_GT_ROOT
        self.samples = []

        with open(EMOTION_SPLIT_JSON, "r") as f:
            self.data = json.load(f)

        key_name = self.mode + "_keys"
        if key_name in self.data:
            raw_keys = self.data[key_name]
        elif self.mode in self.data:
            raw_keys = self.data[self.mode]
        else:
            raise KeyError(
                f"Cannot find '{key_name}' or '{self.mode}' in {EMOTION_SPLIT_JSON}. "
                f"Available keys: {list(self.data.keys())}"
            )

        missing = 0
        seen = set()

        for key in raw_keys:
            # 补全扩展名，此时 npz_rel 依然带有子文件夹路径，如 "ActivityNet_Captions_raw/v_---9CpRcKoU_0.npz"
            npz_rel = key if key.endswith(".npz") else key + ".npz"

            if npz_rel in seen:
                continue
            seen.add(npz_rel)

            # ====== 核心修改部分：保留子文件夹结构进行路径拼接 ======
            feat_path = os.path.join(self.feature_root, npz_rel)
            gt_path = os.path.join(self.gt_root, npz_rel)

            if os.path.exists(feat_path) and os.path.exists(gt_path):
                # 提取纯粹的文件名作为 video_id（例如：v_---9CpRcKoU_0）
                pure_filename = os.path.basename(npz_rel)
                video_id = os.path.splitext(pure_filename)[0]

                self.samples.append(
                    {
                        "video_id": video_id,
                        "feat_path": feat_path,
                        "gt_path": gt_path,
                    }
                )
            else:
                missing += 1

        print(f"[EmotionDataset:{mode}] split={EMOTION_SPLIT_JSON}")
        print(
            f"[EmotionDataset:{mode}] total videos={len(self.samples)} (missing={missing})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        with np.load(sample["feat_path"]) as f:
            loaded_feats = torch.tensor(f["features"], dtype=torch.float32)

        with np.load(sample["gt_path"]) as g:
            binary_labels = (g["labels"] != 0).astype(np.float32)
            is_negative = (
                bool(g["is_negative"][0]) if "is_negative" in g else False
            )

        min_len = min(loaded_feats.shape[0], len(binary_labels))
        loaded_feats = loaded_feats[:min_len]
        binary_labels = binary_labels[:min_len]

        if self.augment and self.mode == "train":
            loaded_feats, binary_labels_tensor = apply_augmentations(
                loaded_feats, torch.tensor(binary_labels, dtype=torch.float32)
            )
            binary_labels = binary_labels_tensor.numpy()

        d = {
            "video_id": sample["video_id"],
            "features": loaded_feats,
            "gtscore": torch.tensor(binary_labels, dtype=torch.float32),
            "is_negative": is_negative,
        }

        if self.mode != "train":
            n_frames = loaded_feats.shape[0]
            d["n_frames"] = np.array(n_frames)
            d["picks"] = np.arange(n_frames)
            d["gt_summary"] = np.expand_dims(binary_labels, axis=0)

        return d
    
class EmotionDataset_Orig(Dataset):
    """
    Emotion-perspective dataset, backed by per-video .npz files as released
    on https://huggingface.co/datasets/vanilladucky/TRINITY
    (emotion_features_part_*.zip and emotion_gt_part_1.zip). Split-file keys
    are expected to be the exact .npz filenames (with or without the
    extension) as they appear in those archives, e.g. "00000_00000007.npz".
    """

    def __init__(self, mode, augment=False):
        self.mode = mode
        self.augment = augment
        self.feature_root = EMOTION_FEATURE_ROOT
        self.gt_root = EMOTION_GT_ROOT
        self.samples = []

        with open(EMOTION_SPLIT_JSON, 'r') as f:
            self.data = json.load(f)

        key_name = self.mode + '_keys'
        if key_name in self.data:
            raw_keys = self.data[key_name]
        elif self.mode in self.data:
            raw_keys = self.data[self.mode]
        else:
            raise KeyError(
                f"Cannot find '{key_name}' or '{self.mode}' in {EMOTION_SPLIT_JSON}. "
                f"Available keys: {list(self.data.keys())}"
            )

        missing = 0
        seen = set()

        for key in raw_keys:
            npz_rel = key if key.endswith('.npz') else key + '.npz'

            if npz_rel in seen:
                continue
            seen.add(npz_rel)

            feat_path = os.path.join(self.feature_root, npz_rel)
            gt_path = os.path.join(self.gt_root, npz_rel)

            if os.path.exists(feat_path) and os.path.exists(gt_path):
                self.samples.append({
                    'video_id': npz_rel[:-len('.npz')],
                    'feat_path': feat_path,
                    'gt_path': gt_path,
                })
            else:
                missing += 1

        print(f"[EmotionDataset:{mode}] split={EMOTION_SPLIT_JSON}")
        print(f"[EmotionDataset:{mode}] total videos={len(self.samples)} (missing={missing})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        with np.load(sample['feat_path']) as f:
            loaded_feats = torch.tensor(f['features'], dtype=torch.float32)

        with np.load(sample['gt_path']) as g:
            binary_labels = (g['labels'] != 0).astype(np.float32)
            is_negative = bool(g['is_negative'][0]) if 'is_negative' in g else False

        min_len = min(loaded_feats.shape[0], len(binary_labels))
        loaded_feats = loaded_feats[:min_len]
        binary_labels = binary_labels[:min_len]

        if self.augment and self.mode == 'train':
            loaded_feats, binary_labels_tensor = apply_augmentations(
                loaded_feats, torch.tensor(binary_labels, dtype=torch.float32)
            )
            binary_labels = binary_labels_tensor.numpy()

        d = {
            'video_id': sample['video_id'],
            'features': loaded_feats,
            'gtscore': torch.tensor(binary_labels, dtype=torch.float32),
            'is_negative': is_negative,
        }

        if self.mode != 'train':
            n_frames = loaded_feats.shape[0]
            d['n_frames'] = np.array(n_frames)
            d['picks'] = np.arange(n_frames)
            d['gt_summary'] = np.expand_dims(binary_labels, axis=0)

        return d


# ==========================================
# 3. Nature Dataset (NPZ features + continuous aesthetic scores)
# ==========================================
    

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class NatureDataset(Dataset):
    """Nature-perspective dataset produced by the scenic-localization + aesthetic-scoring pipeline."""

    def __init__(self, mode, augment=False):
        self.mode = mode
        assert mode in ['train', 'val', 'test'], "mode should be train, val, or test"
        self.augment = augment

        self.roots = [{'feat': NATURE_FEATURE_ROOT, 'gt': NATURE_GT_ROOT}]
        self.split_file = NATURE_SPLIT_JSON

        with open(self.split_file, 'r', encoding='utf-8') as f:
            self.split_data = json.load(f)

        raw_keys = self.split_data[self.mode + '_keys']
        print(f"[NatureDataset:{mode}] Scanning {len(raw_keys)} candidate videos...")

        self.valid_keys = []
        self.key_to_root = {}

        for video_name in raw_keys:
            # 💡 核心修复：如果 json 中的名字自带 .npz 后缀（如 "xxx.npz"），
            # 剔除后缀拿到纯文件名 "xxx"，防止后面拼接成 "xxx.npz.npz" 和 "xxx.npz.json"
            base_name = video_name[:-4] if video_name.endswith('.npz') else video_name
            
            for root in self.roots:
                f_path = os.path.join(root['feat'], f"{base_name}.npz")
                g_path = os.path.join(root['gt'], f"{base_name}.json")

                # 只有当对应的特征 npz 和标签 json 同时存在时，才视为有效视频
                if os.path.exists(f_path) and os.path.exists(g_path):
                    self.valid_keys.append(video_name)
                    self.key_to_root[video_name] = {'feat': f_path, 'gt': g_path}
                    break

        print(f"[NatureDataset:{mode}] Kept {len(self.valid_keys)} valid videos.")

    def __len__(self):
        return len(self.valid_keys)

    def _aggregate_features(self, features, segment_duration=5):
        T, D = features.shape
        n_segments = T // segment_duration
        if n_segments == 0:
            return features.mean(axis=0, keepdims=True)
        truncated = features[:n_segments * segment_duration]
        return truncated.reshape(n_segments, segment_duration, D).mean(axis=1)

    def _aggregate_gtscore(self, gtscore, segment_duration=5):
        T = len(gtscore)
        n_segments = T // segment_duration
        if n_segments == 0:
            return np.array([gtscore.mean()])
        truncated = gtscore[:n_segments * segment_duration]
        return truncated.reshape(n_segments, segment_duration).mean(axis=1)

    def __getitem__(self, index):
        video_name = self.valid_keys[index]
        paths = self.key_to_root[video_name]

        d = {'video_id': video_name}

        # Features and ground truth are both extracted at 1 fps and need 5s aggregation.
        with np.load(paths['feat']) as feat_data:
            # 安全地将 numpy 数组转换为 PyTorch 连续张量
            raw_features = torch.from_numpy(feat_data['features']).float()

        with open(paths['gt'], 'r', encoding='utf-8') as file:
            # 将 json list 转为 float32 张量
            raw_gtscore = torch.tensor(json.load(file), dtype=torch.float32)

        min_len = min(raw_features.shape[0], raw_gtscore.shape[0])
        raw_features = raw_features[:min_len]
        raw_gtscore = raw_gtscore[:min_len]

        # 进行 5s 聚合处理
        agg_feat = self._aggregate_features(raw_features.numpy())
        agg_gt = self._aggregate_gtscore(raw_gtscore.numpy())
        
        d['features'] = torch.from_numpy(agg_feat).float()
        d['gtscore'] = torch.from_numpy(agg_gt).float()

        if self.augment:
            # 确保全局有定义 apply_augmentations 函数
            d['features'], d['gtscore'] = apply_augmentations(d['features'], d['gtscore'])

        if self.mode != 'train':
            n_frames = d['features'].shape[0]
            d['n_frames'] = np.array(n_frames)
            d['picks'] = np.arange(n_frames)
            d['gt_summary'] = np.expand_dims(d['gtscore'].numpy(), axis=0)

        return d

class NatureDataset_OLD(Dataset):
    """Nature-perspective dataset produced by the scenic-localization + aesthetic-scoring pipeline."""

    def __init__(self, mode, augment=False):
        self.mode = mode
        assert mode in ['train', 'val', 'test'], "mode should be train, val, or test"
        self.augment = augment

        self.roots = [{'feat': NATURE_FEATURE_ROOT, 'gt': NATURE_GT_ROOT}]
        self.split_file = NATURE_SPLIT_JSON

        with open(self.split_file, 'r') as f:
            self.split_data = json.load(f)

        raw_keys = self.split_data[self.mode + '_keys']
        print(f"[NatureDataset:{mode}] Scanning {len(raw_keys)} candidate videos...")

        self.valid_keys = []
        self.key_to_root = {}

        for video_name in raw_keys:
            for root in self.roots:
                f_path = os.path.join(root['feat'], f"{video_name}.npz")
                g_path = os.path.join(root['gt'], f"{video_name}.json")

                if os.path.exists(f_path) and os.path.exists(g_path):
                    self.valid_keys.append(video_name)
                    self.key_to_root[video_name] = {'feat': f_path, 'gt': g_path}
                    break

        print(f"[NatureDataset:{mode}] Kept {len(self.valid_keys)} valid videos.")

    def __len__(self):
        return len(self.valid_keys)

    def _aggregate_features(self, features, segment_duration=5):
        T, D = features.shape
        n_segments = T // segment_duration
        if n_segments == 0:
            return features.mean(axis=0, keepdims=True)
        truncated = features[:n_segments * segment_duration]
        return truncated.reshape(n_segments, segment_duration, D).mean(axis=1)

    def _aggregate_gtscore(self, gtscore, segment_duration=5):
        T = len(gtscore)
        n_segments = T // segment_duration
        if n_segments == 0:
            return np.array([gtscore.mean()])
        truncated = gtscore[:n_segments * segment_duration]
        return truncated.reshape(n_segments, segment_duration).mean(axis=1)

    def __getitem__(self, index):
        video_name = self.valid_keys[index]
        paths = self.key_to_root[video_name]

        d = {'video_id': video_name}

        # Features and ground truth are both extracted at 1 fps and need 5s aggregation.
        with np.load(paths['feat']) as feat_data:
            raw_features = torch.tensor(feat_data['features'], dtype=torch.float32)

        with open(paths['gt'], 'r', encoding='utf-8') as file:
            raw_gtscore = torch.tensor(json.load(file), dtype=torch.float32)

        min_len = min(raw_features.shape[0], raw_gtscore.shape[0])
        raw_features = raw_features[:min_len]
        raw_gtscore = raw_gtscore[:min_len]

        d['features'] = torch.tensor(self._aggregate_features(raw_features.numpy()), dtype=torch.float32)
        d['gtscore'] = torch.tensor(self._aggregate_gtscore(raw_gtscore.numpy()), dtype=torch.float32)

        if self.augment:
            d['features'], d['gtscore'] = apply_augmentations(d['features'], d['gtscore'])

        if self.mode != 'train':
            n_frames = d['features'].shape[0]
            d['n_frames'] = np.array(n_frames)
            d['picks'] = np.arange(n_frames)
            d['gt_summary'] = np.expand_dims(d['gtscore'].numpy(), axis=0)

        return d


# ==========================================
# Batch collation
# ==========================================

def Pad5sColllator(batch):
    """Pad a batch of variable-length 5-second-segment sequences to the same length."""
    B = len(batch)
    lengths = [b["features"].shape[0] for b in batch]
    Tm = max(lengths)
    D = batch[0]["features"].shape[1]

    feats = torch.zeros((B, Tm, D), dtype=torch.float32)
    gts = torch.zeros((B, Tm), dtype=torch.float32)
    mask = torch.zeros((B, Tm), dtype=torch.bool)

    video_id = []

    for i, b in enumerate(batch):
        T = b["features"].shape[0]
        feats[i, :T] = b["features"]
        gts[i, :T] = b["gtscore"]
        mask[i, :T] = True
        video_id.append(b["video_id"])

    return {
        "video_name": video_id,
        "features": feats,
        "gtscore": gts,
        "mask": mask,
    }
