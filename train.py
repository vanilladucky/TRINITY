# train.py
"""
Main entry point for joint pretraining of the shared-backbone, multi-branch
model on the three TRINITY perspectives (Event, Emotion, Nature).

Usage:
  # Pretrain a new model
  python train.py --train true --device cuda:0 --model SL_module_split \
      --epochs 50 --batch_size 16 --lr 5e-5 --l2_reg 1e-4 --dropout_ratio 0.2 \
      --tag trinity_pretrain

  # Evaluate an existing checkpoint
  python train.py --train false --ckpt_path /path/to/checkpoint.pkl --device cuda
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from model.configs import Config, str2bool
from model.solver import load_ckpt_safely
from model.full_dataset import (
    EmotionDataset,
    MrHiSumDataset,
    NatureDataset,
    Pad5sColllator,
)
from model.solver_combined_5s import Solver100


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='SL_module_split', help='the name of the model')
    parser.add_argument('--epochs', type=int, default=20, help='the number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='the learning rate')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='l2 regularizer')
    parser.add_argument('--dropout_ratio', type=float, default=0.2, help='the dropout ratio')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch size')
    parser.add_argument('--tag', type=str, default='dev', help='a tag for the experiment, used to name the output directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='checkpoint path for inference or weight initialization')
    parser.add_argument('--train', type=str2bool, default='true', help='true to train, false to evaluate only')

    parser.add_argument('--event_weight', type=float, default=1.0)
    parser.add_argument('--emotion_weight', type=float, default=1.0, help='weight allocated to emotion loss')
    parser.add_argument('--nature_weight', type=float, default=1.0)

    parser.add_argument('--augment', action='store_true', help='whether to apply data augmentation during training')
    parser.add_argument('--use_CovLoss', action='store_true', help='whether to use the covariance loss (only with SL_module_split)')
    parser.add_argument('--CovLoss_lambda', type=float, default=1e9)
    parser.add_argument('--device', type=str, default='cuda', help='cuda / cuda:0 / cuda:1 / cpu')
    parser.add_argument('--covloss_type', type=str, default='combined')

    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--lambda_ortho', type=float, default=0)
    parser.add_argument('--projection_dim_count', type=int, default=512)
    parser.add_argument('--use_smoothap', action='store_true', help='whether to use the SmoothAP ranking loss')

    # Feature-level inference speed profiling (used in the rebuttal to report FPS).
    parser.add_argument('--profile_fps', action='store_true', help='measure feature-level FPS on pre-extracted features')
    parser.add_argument('--profile_loader', type=str, default='event', choices=['event', 'emotion', 'nature'],
                         help='which test loader to use for FPS profiling')
    parser.add_argument('--fps_warmup', type=int, default=30, help='number of warmup forward passes before FPS measurement')
    parser.add_argument('--fps_repeat', type=int, default=200, help='number of repeated forward passes for FPS measurement')

    return parser.parse_args()


# ============================================================
# FPS profiling helpers
# ============================================================

def extract_features_from_batch(batch, device):
    """Extract the feature tensor from a dataloader batch (tensor, tuple/list, or dict)."""
    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, (tuple, list)):
        for item in batch:
            if torch.is_tensor(item) and item.dim() >= 2:
                return item.to(device)

    if isinstance(batch, dict):
        for key in ["features", "feature", "x", "feats", "video_features"]:
            if key in batch and torch.is_tensor(batch[key]):
                return batch[key].to(device)

    raise RuntimeError(
        "Cannot extract feature tensor from batch. "
        "Check the Dataset.__getitem__ / Pad5sColllator output format."
    )


def ensure_btd_shape(x):
    """Ensure the feature tensor has shape (B, T, D); (T, D) is promoted to (1, T, D)."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return x


def try_forward_model(model, x, training_type='event'):
    """Call model.forward with the calling convention used across model variants."""
    errors = []

    try:
        return model(x, training_type=training_type)
    except Exception as e:
        errors.append(f"model(x, training_type='{training_type}') failed: {repr(e)}")

    try:
        return model(x, train_type=training_type)
    except Exception as e:
        errors.append(f"model(x, train_type='{training_type}') failed: {repr(e)}")

    try:
        return model(x, training_type)
    except Exception as e:
        errors.append(f"model(x, '{training_type}') failed: {repr(e)}")

    try:
        return model(x)
    except Exception as e:
        errors.append(f"model(x) failed: {repr(e)}")

    try:
        B, T = x.shape[0], x.shape[1]
        mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        return model(x, mask)
    except Exception as e:
        errors.append(f"model(x, mask) failed: {repr(e)}")

    try:
        B, T = x.shape[0], x.shape[1]
        mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        return model(x, mask, training_type=training_type)
    except Exception as e:
        errors.append(f"model(x, mask, training_type='{training_type}') failed: {repr(e)}")

    raise RuntimeError("All forward attempts failed for FPS profiling.\n\n" + "\n".join(errors))


def measure_feature_level_fps(model, dataloader, device, training_type='event', warmup=30, repeat=200):
    """
    Measure feature-level FPS, i.e. model forward passes per second on
    pre-extracted CLIP features. This excludes video decoding, frame
    sampling, and CLIP feature extraction.
    """
    model = model.to(device)
    model.eval()

    batch = next(iter(dataloader))
    x = extract_features_from_batch(batch, device)
    x = ensure_btd_shape(x)

    print("=" * 80)
    print("Feature-level FPS Profiling")
    print("=" * 80)
    print(f"training_type : {training_type}")
    print(f"input shape   : {tuple(x.shape)}")
    print(f"warmup        : {warmup}")
    print(f"repeat        : {repeat}")
    print("-" * 80)

    with torch.no_grad():
        for _ in range(warmup):
            _ = try_forward_model(model, x, training_type=training_type)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeat):
            _ = try_forward_model(model, x, training_type=training_type)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start
    latency_ms_per_clip = total_time / repeat * 1000.0
    fps = repeat / total_time

    print(f"Latency      : {latency_ms_per_clip:.4f} ms/clip")
    print(f"FPS          : {fps:.2f} clips/s")
    print("Note         : feature-level FPS excludes video decoding and CLIP feature extraction.")
    print("=" * 80)

    return fps, latency_ms_per_clip, tuple(x.shape)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    opt = parse_args()
    kwargs = vars(opt)
    config = Config(**kwargs)
    config.device = opt.device

    print("Device being used:", config.device)

    if config.use_CovLoss and config.model != 'SL_module_split':
        raise ValueError("CovLoss can only be used with the SL_module_split model.")

    if config.use_CovLoss:
        print("==== Using CovLoss ====")

    # ============================================================
    # Emotion perspective dataloaders
    # ============================================================
    emotion_train_loader = DataLoader(
        EmotionDataset(mode='train', augment=config.augment),
        batch_size=config.batch_size, shuffle=True, num_workers=4,
        collate_fn=Pad5sColllator, pin_memory=True,
    )
    emotion_val_loader = DataLoader(
        EmotionDataset(mode='val', augment=False),
        batch_size=1, shuffle=False, num_workers=2,
        collate_fn=Pad5sColllator, pin_memory=False,
    )
    emotion_test_loader = DataLoader(
        EmotionDataset(mode='test', augment=False),
        batch_size=1, shuffle=False, num_workers=2,
        collate_fn=Pad5sColllator, pin_memory=False,
    )

    # ============================================================
    # Event perspective dataloaders
    # ============================================================
    event_train_loader = DataLoader(
        MrHiSumDataset(mode='train', augment=config.augment),
        batch_size=config.batch_size, shuffle=True, num_workers=2,
        collate_fn=Pad5sColllator, pin_memory=True,
    )
    event_val_loader = DataLoader(
        MrHiSumDataset(mode='val', augment=False),
        batch_size=1, shuffle=False, num_workers=2,
        collate_fn=Pad5sColllator, pin_memory=False,
    )
    event_test_loader = DataLoader(
        MrHiSumDataset(mode='test', augment=False),
        batch_size=1, shuffle=False, num_workers=2,
        collate_fn=Pad5sColllator, pin_memory=False,
    )

    # ============================================================
    # Nature perspective dataloaders
    # ============================================================
    nature_train_loader = DataLoader(
        NatureDataset(mode='train', augment=config.augment),
        batch_size=config.batch_size, shuffle=True, num_workers=2,
        collate_fn=Pad5sColllator, pin_memory=True,
    )
    nature_val_loader = DataLoader(
        NatureDataset(mode='val', augment=False),
        batch_size=1, shuffle=False, num_workers=2,
        collate_fn=Pad5sColllator, pin_memory=False,
    )
    nature_test_loader = DataLoader(
        NatureDataset(mode='test', augment=False),
        batch_size=1, shuffle=False, num_workers=2,
        collate_fn=Pad5sColllator, pin_memory=False,
    )

    # ============================================================
    # Solver
    # ============================================================
    solver = Solver100(
        config,
        emotion_train_loader, emotion_val_loader, emotion_test_loader,
        event_train_loader, event_val_loader, event_test_loader,
        nature_train_loader, nature_val_loader, nature_test_loader,
        config.beta, config.alpha, config.lambda_ortho,
        config.event_weight, config.emotion_weight, config.nature_weight, config.CovLoss_lambda,
        False, config.projection_dim_count,
    )
    solver.build()

    profile_loader_map = {
        "event": event_test_loader,
        "emotion": emotion_test_loader,
        "nature": nature_test_loader,
    }
    selected_profile_loader = profile_loader_map[opt.profile_loader]

    # ============================================================
    # Train / Test
    # ============================================================
    if config.train:
        (
            best_map50_event_path, best_map50_emotion_path,
            best_map15_event_path, best_map15_emotion_path,
            best_map50_nature_path, best_map15_nature_path
        ) = solver.train(
            use_CovLoss=config.use_CovLoss,
            loss_type=config.covloss_type,
            use_smoothap=config.use_smoothap,
        )

        if opt.profile_fps:
            load_ckpt_safely(solver.model, best_map15_event_path, config.device)
            measure_feature_level_fps(
                model=solver.model, dataloader=selected_profile_loader, device=config.device,
                training_type=opt.profile_loader, warmup=opt.fps_warmup, repeat=opt.fps_repeat,
            )

        solver.test(best_map15_event_path, training_type='event')
        solver.test(best_map15_emotion_path, training_type='emotion')
        solver.test(best_map15_nature_path, training_type='nature')

    else:
        if config.ckpt_path is None:
            print("A trained model checkpoint is required for evaluation. Exiting.")
            exit(1)

        # Load the checkpoint before profiling so the reported FPS matches the evaluated model.
        load_ckpt_safely(solver.model, config.ckpt_path, config.device)

        if opt.profile_fps:
            measure_feature_level_fps(
                model=solver.model, dataloader=selected_profile_loader, device=config.device,
                training_type=opt.profile_loader, warmup=opt.fps_warmup, repeat=opt.fps_repeat,
            )

        solver.test(config.ckpt_path, training_type='emotion')
        solver.test(config.ckpt_path, training_type='event')
        solver.test(config.ckpt_path, training_type='nature')
