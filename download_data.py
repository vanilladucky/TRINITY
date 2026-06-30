#!/usr/bin/env python3
"""
Download the TRINITY dataset (https://huggingface.co/datasets/vanilladucky/TRINITY)
and extract it into the flat per-perspective directory layout expected by
model/full_dataset.py:

  data/
  ├── event/
  │   ├── mr_hisum_split_original.json
  │   ├── event_features/   # video_<id>.npz  key "data"  shape (T, 512)
  │   └── event_gt/         # video_<id>.npz  key "data"  shape (T,)
  ├── emotion/
  │   ├── emotion_split.json          # downloaded directly from the HF repo
  │   ├── features/                   # <id>.npz  key "features"
  │   └── ground_truth/               # <id>.npz  keys "labels", "is_negative", ...
  └── nature/
      ├── nature_split.json
      ├── features/                   # <id>.npz  key "features"
      └── ground_truth/               # <id>.json  list of per-frame scores

Idempotent: re-running skips any shard whose extraction marker already exists,
so it is safe to call repeatedly when data/ is on a persistent volume.

Usage:
  python download_data.py [--perspectives event,emotion,nature] [--data-root data]
"""

import argparse
import os
import sys
import zipfile

from huggingface_hub import hf_hub_download, list_repo_files

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ID   = "vanilladucky/TRINITY"
REPO_TYPE = "dataset"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perspectives", type=str, default="event,emotion,nature",
                        help="comma-separated subset of {event,emotion,nature} to download")
    parser.add_argument("--data-root", type=str, default="data",
                        help="local directory to populate")
    return parser.parse_args()


def _download_and_extract(remote_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    marker = os.path.join(extract_dir, f".done_{os.path.basename(remote_path)}")
    if os.path.exists(marker):
        return

    local_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=remote_path)

    if remote_path.endswith(".zip"):
        with zipfile.ZipFile(local_path) as zf:
            zf.extractall(extract_dir)
    else:
        dest = os.path.join(extract_dir, os.path.basename(remote_path))
        if not os.path.exists(dest):
            with open(local_path, "rb") as src, open(dest, "wb") as out:
                out.write(src.read())

    open(marker, "w").close()
    print(f"[download_data] extracted {remote_path} -> {extract_dir}")


def download_event(data_root):
    print("[download_data] === Event ===")
    remote_files   = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    feature_shards = sorted(f for f in remote_files if f.startswith("event/event_features_part_"))
    gt_shards      = sorted(f for f in remote_files if f.startswith("event/event_gt_part_"))

    split_dest = os.path.join(data_root, "event")
    _download_and_extract("event/mr_hisum_split_original.json", split_dest)

    feat_dest = os.path.join(data_root, "event", "event_features")
    for shard in feature_shards:
        _download_and_extract(shard, feat_dest)

    gt_dest = os.path.join(data_root, "event", "event_gt")
    for shard in gt_shards:
        _download_and_extract(shard, gt_dest)


def download_emotion(data_root):
    print("[download_data] === Emotion ===")
    remote_files   = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    feature_shards = sorted(f for f in remote_files if f.startswith("emotion/emotion_feature_") and f.endswith(".zip"))
    gt_shards      = sorted(f for f in remote_files if f.startswith("emotion/emotion_gt") and f.endswith(".zip"))

    split_dest = os.path.join(data_root, "emotion")
    _download_and_extract("emotion/emotion_split.json", split_dest)

    feat_dest = os.path.join(data_root, "emotion", "features")
    os.makedirs(feat_dest, exist_ok=True)
    for shard in feature_shards:
        _download_and_extract(shard, feat_dest)

    gt_dest = os.path.join(data_root, "emotion", "ground_truth")
    os.makedirs(gt_dest, exist_ok=True)
    for shard in gt_shards:
        _download_and_extract(shard, gt_dest)


def download_nature(data_root):
    print("[download_data] === Nature ===")
    remote_files   = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    feature_shards = sorted(f for f in remote_files if f.startswith("nature/nature_feat_part_"))
    gt_shards      = sorted(f for f in remote_files if f.startswith("nature/nature_gt_part_"))

    split_dest = os.path.join(data_root, "nature")
    _download_and_extract("nature/nature_split.json", split_dest)

    feat_dest = os.path.join(data_root, "nature", "features")
    os.makedirs(feat_dest, exist_ok=True)
    for shard in feature_shards:
        _download_and_extract(shard, feat_dest)

    gt_dest = os.path.join(data_root, "nature", "ground_truth")
    os.makedirs(gt_dest, exist_ok=True)
    for shard in gt_shards:
        _download_and_extract(shard, gt_dest)


if __name__ == "__main__":
    args = parse_args()
    requested = {p.strip().lower() for p in args.perspectives.split(",") if p.strip()}

    os.makedirs(args.data_root, exist_ok=True)

    if "event"   in requested:
        download_event(args.data_root)
    if "emotion" in requested:
        download_emotion(args.data_root)
    if "nature"  in requested:
        download_nature(args.data_root)

    print("[download_data] done.")
