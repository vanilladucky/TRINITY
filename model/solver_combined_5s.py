# model/solver_combined_5s.py
"""
Solver for joint Event/Emotion/Nature training with variable-length 5-second
segments. Sequences are kept at their native (padded) length rather than
interpolated to a fixed number of segments.

Includes:
  - Per-step task gradient cosine similarity tracking (Event-Emotion, Event-Nature, Emotion-Nature)
  - Per-step condition number tracking of the stacked task-gradient matrix
  - Automatic saving of diagnostic plots and raw diagnostics JSON (Fig. 4 in the paper)
"""

import json
import math
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from model.pcgrad import PCGrad
from networks.mlp import SimpleMLP
from networks.pgl_sum.pgl_sum import PGL_SUM
from networks.vasnet.vasnet import VASNet
from networks.sl_module.sl_module_combined import SL_module
from networks.sl_module.sl_module_split_heads import SL_module_split_heads




class MaskedTaskOrthogonalityLoss(nn.Module):
    """
    ADVANCED: Orthogonality loss that respects padding masks.

    Use this if you want to exclude padded positions from the computation.
    """

    def __init__(self, method='subspace', reduction='mean'):
        super(MaskedTaskOrthogonalityLoss, self).__init__()
        self.method = method
        self.reduction = reduction

    def forward(self, task_embeddings_and_masks):
        """
        Args:
            task_embeddings_and_masks: List of tuples [(embed1, mask1), (embed2, mask2), ...]
                where embed_i is (B, T, D) and mask_i is (B, T) boolean mask

        Returns:
            loss: Scalar tensor
        """
        if len(task_embeddings_and_masks) < 2:
            raise ValueError("Need at least 2 tasks")

        losses = []
        num_tasks = len(task_embeddings_and_masks)

        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                embed_i, mask_i = task_embeddings_and_masks[i]
                embed_j, mask_j = task_embeddings_and_masks[j]

                if self.method == 'subspace':
                    loss_ij = self._masked_subspace_loss(embed_i, mask_i, embed_j, mask_j)
                else:
                    raise NotImplementedError(f"Method {self.method} not implemented for masked version")

                losses.append(loss_ij)

        if self.reduction == 'mean':
            return torch.stack(losses).mean()
        else:
            return torch.stack(losses).sum()

    def _masked_subspace_loss(self, embed1, mask1, embed2, mask2):
        """
        Compute subspace alignment using only valid (non-padded) positions.
        """
        # Extract valid embeddings
        z1 = embed1[mask1]  # (N1_valid, D)
        z2 = embed2[mask2]  # (N2_valid, D)

        if z1.shape[0] < 2 or z2.shape[0] < 2:
            # Not enough samples to compute covariance
            return torch.tensor(0.0, device=embed1.device)

        # Center
        z1 = z1 - z1.mean(dim=0, keepdim=True)
        z2 = z2 - z2.mean(dim=0, keepdim=True)

        # Compute covariance
        cov1 = (z1.T @ z1) / (z1.shape[0] - 1)
        cov2 = (z2.T @ z2) / (z2.shape[0] - 1)

        # Normalize to correlation
        std1 = torch.sqrt(torch.diag(cov1) + 1e-6)
        std2 = torch.sqrt(torch.diag(cov2) + 1e-6)
        corr1 = cov1 / (std1.unsqueeze(1) * std1.unsqueeze(0))
        corr2 = cov2 / (std2.unsqueeze(1) * std2.unsqueeze(0))

        # Alignment
        alignment = (corr1 * corr2).abs().mean()
        return alignment


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification with sigmoid-activated inputs.

    Drop-in replacement for nn.BCELoss(reduction="none").
    Expects predictions AFTER sigmoid (probabilities in [0, 1]).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "none"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        pred = pred.clamp(eps, 1.0 - eps)

        bce = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)

        p_t = target * pred + (1.0 - target) * (1.0 - pred)
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def _fix_state_dict_keys(state: dict) -> dict:
    """Remove module. prefix and return clean dict."""
    if not isinstance(state, dict):
        return state
    fixed = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        fixed[nk] = v
    return fixed


def load_ckpt_safely(model: nn.Module, ckpt_path: str, device: str, projector: nn.Module = None):
    state = torch.load(ckpt_path, map_location=device)

    # Handle new combined format {"model": ..., "projector": ...}
    if isinstance(state, dict) and "model" in state:
        model_state = state["model"]
        if projector is not None and "projector" in state:
            proj_state = _fix_state_dict_keys(state["projector"])
            projector.load_state_dict(proj_state, strict=True)
            print("[LOAD] projector loaded OK")
        state = model_state

    # unwrap other common wrappers
    if isinstance(state, dict):
        for k in ("state_dict", "net", "network"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a dict-like state_dict. type={type(state)}")

    state = _fix_state_dict_keys(state)

    if "transformer.pos_alpha" in state and hasattr(model, "transformer") and not hasattr(model.transformer, "pos_alpha"):
        state.pop("transformer.pos_alpha", None)

    try:
        model.load_state_dict(state, strict=True)
        print("[LOAD] strict=True OK")
        return
    except RuntimeError as e:
        print("[LOAD] strict=True failed:", str(e).split("\n")[0])

    if "transformer.pos_alpha" in state:
        print("[LOAD] removing key transformer.pos_alpha and retry strict")
        state.pop("transformer.pos_alpha", None)
        try:
            model.load_state_dict(state, strict=True)
            print("[LOAD] strict=True OK after removing pos_alpha")
            return
        except RuntimeError as e:
            print("[LOAD] strict=True still failed:", str(e).split("\n")[0])

    ret = model.load_state_dict(state, strict=False)
    print("[LOAD] strict=False used. ret =", ret)

# ============================================================
# Helpers for evaluation metrics
# ============================================================

def _topk_binary(y: np.ndarray, frac: float) -> np.ndarray:
    """Convert scores to binary labels by taking top-k fraction as positive."""
    y = np.asarray(y).reshape(-1)
    n = len(y)
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argpartition(-y, kth=k - 1)[:k]
    out = np.zeros(n, dtype=np.int32)
    out[idx] = 1
    return out


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average precision safely."""
    y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float32).reshape(-1)
    if len(y_true) == 0:
        return 0.0
    if y_true.min() == y_true.max():
        return 0.0
    return float(average_precision_score(y_true, y_score))

def temporal_gradient_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    L1 loss between temporal derivatives of prediction and ground-truth sequences.
    """
    if pred.size(1) < 2:
        return pred.new_tensor(0.0)

    diff_pred = pred[:, 1:] - pred[:, :-1]
    diff_gt = gt[:, 1:] - gt[:, :-1]

    if mask is not None:
        valid_mask = mask[:, 1:] & mask[:, :-1]
        diff_pred = diff_pred[valid_mask]
        diff_gt = diff_gt[valid_mask]

    if diff_pred.numel() == 0:
        return pred.new_tensor(0.0)

    return nn.L1Loss()(diff_pred, diff_gt)


class TopKPeakSmoothAPLoss(nn.Module):
    """
    Differentiable AP-style ranking loss with top-K anchors and optional peak weighting.
    """

    def __init__(
        self,
        frac: float = 0.15,
        tau: float = 0.05,
        eta: float = 2.0,
        use_peakness: bool = True,
        peakness_scale: float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.frac = frac
        self.tau = tau
        self.eta = eta
        self.use_peakness = use_peakness
        self.peakness_scale = peakness_scale
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            labels = labels.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)

        batch_size = scores.size(0)
        losses: List[torch.Tensor] = []

        for b in range(batch_size):
            m = mask[b]
            s_b = scores[b][m]
            y_b = labels[b][m]

            if s_b.numel() == 0:
                losses.append(scores.new_tensor(0.0))
                continue

            n_valid = s_b.numel()
            k = max(1, int(math.ceil(self.frac * n_valid)))
            _, topk_idx = torch.topk(y_b, k=k, largest=True, sorted=False)

            pos_mask = torch.zeros_like(y_b, dtype=torch.bool)
            pos_mask[topk_idx] = True

            diff = s_b.unsqueeze(1) - s_b.unsqueeze(0)
            p = torch.sigmoid(-diff / self.tau)

            soft_rank = 1.0 + p.sum(dim=1)
            pos_p = p * pos_mask.float().unsqueeze(0)
            soft_pos_rank = pos_mask.float() + pos_p.sum(dim=1)
            precision = soft_pos_rank / (soft_rank + self.eps)

            pos_weights = y_b[pos_mask].pow(self.eta)

            if self.use_peakness and n_valid > 1:
                left = torch.cat([y_b[:1], y_b[:-1]])
                right = torch.cat([y_b[1:], y_b[-1:]])
                local_avg = 0.5 * (left + right)
                peakness = (y_b - local_avg).clamp_min(0.0)
                pos_weights = pos_weights * (1.0 + self.peakness_scale * peakness[pos_mask])

            pos_weights = pos_weights / (pos_weights.sum() + self.eps)
            ap = (precision[pos_mask] * pos_weights).sum()
            losses.append(1.0 - ap)

        output = torch.stack(losses)
        if self.reduction == "mean":
            return output.mean()
        if self.reduction == "sum":
            return output.sum()
        return output


# ============================================================
# Solver for variable-length 5-second segments
# ============================================================

class Solver100(object):
    """
    Solver for combined emotion/event/nature training.
    Works with variable-length sequences from Pad5sCollator.
    No interpolation to fixed length.
    """

    def __init__(
        self,
        config=None,
        emotion_train_loader=None, emotion_val_loader=None, emotion_test_loader=None,
        event_train_loader=None, event_val_loader=None, event_test_loader=None,
        nature_train_loader=None, nature_val_loader=None, nature_test_loader=None,
        beta=0.5, alpha=0.5, lambda_ortho=0.1,
        event_weight=1, emotion_weight=1, nature_weight=1, CovLoss_lambda=1e10,
        use_grad_norm=False, projection_dim_count = 512,
    ):
        self.model, self.optimizer, self.writer, self.scheduler = None, None, None, None

        self.config = config
        self.emotion_train_loader = emotion_train_loader
        self.emotion_val_loader = emotion_val_loader
        self.emotion_test_loader = emotion_test_loader
        self.event_train_loader = event_train_loader
        self.event_val_loader = event_val_loader
        self.event_test_loader = event_test_loader
        self.nature_train_loader = nature_train_loader
        self.nature_val_loader = nature_val_loader
        self.nature_test_loader = nature_test_loader
        self.event_weight = event_weight
        self.emotion_weight = emotion_weight
        self.nature_weight = nature_weight

        self.beta = beta
        self.alpha = alpha
        self.lambda_ortho = lambda_ortho

        self.use_grad_norm = use_grad_norm
        self.global_step = 0

        self.CovLoss_lambda = CovLoss_lambda

        # Binary cross entropy loss for highlight detection
        self.criterion = nn.BCELoss(reduction="none").to(self.config.device)
        self.smoothap = TopKPeakSmoothAPLoss(frac=0.15, tau=0.05).to(self.config.device)

        self.ortho_loss_fn = MaskedTaskOrthogonalityLoss(method='subspace').to(self.config.device)

        self.dimension_count = projection_dim_count

    # ----------------------------
    # Gradient diagnostics helpers
    # ----------------------------

    def _get_shared_params(self):
        """
        Try to pick the shared temporal backbone parameters.

        Covers common naming patterns across variants:
          - self.model.transformer
          - self.model.T_s
          - self.model.shared_backbone / shared_transformer
          - fallback: all model parameters
        """
        m = self.model

        # Most common case: SL_module / SL_module_split_heads expose a `transformer` attribute.
        if hasattr(m, "transformer") and isinstance(getattr(m, "transformer"), nn.Module):
            return [p for p in m.transformer.parameters() if p.requires_grad]

        # Shared backbone naming patterns
        for attr in ["T_s", "shared_transformer", "shared_backbone", "shared_encoder", "shared_temporal"]:
            if hasattr(m, attr) and isinstance(getattr(m, attr), nn.Module):
                return [p for p in getattr(m, attr).parameters() if p.requires_grad]

        # Fallback
        return [p for p in m.parameters() if p.requires_grad]

    def _flat_grad(self, loss, params):
        """
        Flattened gradient vector d(loss)/d(params).
        """
        grads = torch.autograd.grad(
            loss, params,
            retain_graph=True,
            allow_unused=True
        )
        flat = []
        for g in grads:
            if g is None:
                continue
            flat.append(g.contiguous().view(-1))

        if len(flat) == 0:
            return torch.zeros(1, device=self.config.device)

        return torch.cat(flat, dim=0)

    def _cos_sim_safe(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
        """
        Cosine similarity with safety for zero norms.
        """
        if a.numel() != b.numel():
            n = min(a.numel(), b.numel())
            a = a[:n]
            b = b[:n]

        na = torch.norm(a)
        nb = torch.norm(b)
        if na < eps or nb < eps:
            return 0.0
        return float((torch.dot(a, b) / (na * nb + eps)).item())

    def _cond_number_from_task_grads(self, g_list, eps: float = 1e-12) -> float:
        """
        Condition number of stacked gradient matrix G (k x P), k=3 tasks.
        cond(G) = sigma_max / sigma_min.
        """
        lens = [g.numel() for g in g_list]
        n = min(lens)
        if n <= 1:
            return 0.0

        G = torch.stack([g[:n] for g in g_list], dim=0)  # (3, P)

        if not torch.isfinite(G).all():
            return 0.0

        try:
            s = torch.linalg.svdvals(G)
        except Exception:
            try:
                _, s, _ = torch.svd(G)
            except Exception:
                return 0.0

        if not torch.isfinite(s).all():
            return 0.0

        s_max = s.max()
        s_min = s.min()
        return float((s_max / (s_min + eps)).item())

    def generate_grad_diagnostics_plot(
        self,
        sim_ev_em, sim_ev_nt, sim_em_nt,
        cond_list,
        save_dir,
        smooth_weight=0.9
    ):
        """
        Save:
          - gradient cosine similarity plot
          - gradient condition number plot
        """
        if not sim_ev_em:
            print("No gradient diagnostics to plot.")
            return

        os.makedirs(save_dir, exist_ok=True)

        def smooth(vals, w):
            last = vals[0]
            out = []
            for v in vals:
                last = last * w + (1 - w) * v
                out.append(last)
            return out

        steps = list(range(len(sim_ev_em)))

        s1 = smooth(sim_ev_em, smooth_weight)
        s2 = smooth(sim_ev_nt, smooth_weight)
        s3 = smooth(sim_em_nt, smooth_weight)

        # --- Similarity plot ---
        plt.figure(figsize=(10, 5))
        plt.plot(steps, sim_ev_em, alpha=0.15, linewidth=1)
        plt.plot(steps, sim_ev_nt, alpha=0.15, linewidth=1)
        plt.plot(steps, sim_em_nt, alpha=0.15, linewidth=1)

        plt.plot(steps, s1, label="Event vs Emotion", linewidth=2)
        plt.plot(steps, s2, label="Event vs Nature", linewidth=2)
        plt.plot(steps, s3, label="Emotion vs Nature", linewidth=2)

        plt.axhline(0, color="black", linestyle="-", alpha=0.8, linewidth=1)

        plt.xlabel("Training Iterations", fontsize=20)
        plt.ylabel("Gradient Cosine Similarity", fontsize=20)
        plt.title("Task Gradient Similarity over Training Iterations", fontsize=28)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=20)

        plt.tight_layout()

        # Saved as a vector format since dpi only affects rasterized (alpha) elements.
        sim_path = os.path.join(save_dir, "gradient_similarity_steps.pdf")
        plt.savefig(sim_path, format="pdf", dpi=1200)
        plt.close()
        print(f"Saved step-wise gradient similarity plot to: {sim_path}")

        # --- Condition number plot ---
        if cond_list and len(cond_list) == len(steps):
            sc = smooth(cond_list, smooth_weight)

            plt.figure(figsize=(10, 5))
            plt.plot(steps, cond_list, alpha=0.15, linewidth=1)
            plt.plot(steps, sc, linewidth=2, label="Smoothed")

            plt.yscale("log")  # usually spans orders of magnitude
            plt.xlabel("Training Iterations", fontsize=12)
            plt.ylabel("Condition Number (log scale)", fontsize=12)
            plt.title("Condition Number of Task-Gradient Matrix over Training Iterations", fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            cond_path = os.path.join(save_dir, "gradient_condition_steps.png")
            plt.savefig(cond_path, dpi=300)
            plt.close()
            print(f"Saved step-wise gradient condition plot to: {cond_path}")

    # ----------------------------
    # Pairwise gradient similarity (single pair, used outside the main 3-task loop)
    # ----------------------------

    def _compute_gradient_similarity(self, loss_a, loss_b, shared_params):
        """
        Computes the cosine similarity between the gradients of two losses
        with respect to the shared parameters.
        """
        grad_a = torch.autograd.grad(loss_a, shared_params, retain_graph=True, allow_unused=True)
        grad_b = torch.autograd.grad(loss_b, shared_params, retain_graph=True, allow_unused=True)

        vec_a = torch.cat([g.view(-1) for g in grad_a if g is not None])
        vec_b = torch.cat([g.view(-1) for g in grad_b if g is not None])

        if vec_a.numel() == 0 or vec_b.numel() == 0:
            return 0.0

        norm_a = torch.norm(vec_a)
        norm_b = torch.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0

        sim = torch.dot(vec_a, vec_b) / (norm_a * norm_b)
        return sim.item()

    # ============================================================
    # Model build
    # ============================================================

    def build(self):
        """Define your summarization model here."""
        if self.config.model == "MLP":
            self.model = SimpleMLP(1024, [1024], 1).to(self.config.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)

        elif self.config.model == "PGL_SUM":
            self.model = PGL_SUM(
                input_size=1024, output_size=1024,
                num_segments=4, heads=8, fusion="add", pos_enc="absolute"
            ).to(self.config.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)
            self.init_weights(self.model, init_type="xavier")

        elif self.config.model == "VASNet":
            self.model = VASNet(hidden_dim=1024).to(self.config.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
            self.init_weights(self.model, init_type="xavier")

        elif self.config.model == "SL_module":
            # CLIP ViT-B/32 features = 512 dim
            self.model = SL_module(
                input_dim=512, depth=5, heads=8, mlp_dim=3072, dropout_ratio=self.config.dropout_ratio, pos_type = 'rope'
            ).to(self.config.device)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Total Parameters: {total_params:,}")

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.l2_reg
            )

            warmup_epochs = int(self.config.epochs * 0.1)
            if warmup_epochs < 1:
                warmup_epochs = 1
            scheduler_warmup = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=(self.config.epochs - warmup_epochs), eta_min=1e-6
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[scheduler_warmup, scheduler_cosine],
                milestones=[warmup_epochs]
            )

        elif self.config.model == "SL_module_split":
            self.model = SL_module_split_heads(
                input_dim=self.dimension_count,
                depth=5,
                total_heads=8,
                shared_heads=4,
                mlp_dim=3072,
                # mlp_dim = 1024,
                dropout_ratio=self.config.dropout_ratio,
                separate_ffn=False,
                pos_type='rope',  # IMPORTANT
            ).to(self.config.device)
            total_params = sum(p.numel() for p in self.model.parameters())
            if self.dimension_count!=512:
                print("Employing a projector!")
                self.projector = nn.Linear(in_features=512, out_features=self.dimension_count).to(self.config.device)
                print(f"Projector parameter count: {sum(p.numel() for p in self.projector.parameters())}")
                total_params += sum(p.numel() for p in self.projector.parameters())
            else:
                print("Not using any projector")
            print(f"Total Parameters: {total_params:,}")

            if self.dimension_count!=512:
                self.optimizer = torch.optim.AdamW(
                    [
                        {'params': self.model.parameters(), 'lr': self.config.lr},
                        {'params': self.projector.parameters(), 'lr': self.config.lr},
                    ],
                    weight_decay=self.config.l2_reg
                )
            else:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.lr,  
                    weight_decay=self.config.l2_reg
                )
            warmup_epochs = int(self.config.epochs * 0.1)
            if warmup_epochs < 1:
                warmup_epochs = 1
            scheduler_warmup = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            
            scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=(self.config.epochs - warmup_epochs), eta_min=1e-6
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[scheduler_warmup, scheduler_cosine],
                milestones=[warmup_epochs]
            )

            if self.use_grad_norm:
                self.optimizer = PCGrad(self.optimizer)

        else:
            raise ValueError(f"Wrong model: {self.config.model}")

    # ============================================================
    # Data helpers
    # ============================================================

    def _prep_batch(self, data: dict):
        feats = data["features"].to(self.config.device)   # (B, T, D)
        gt = data["gtscore"].to(self.config.device)       # (B, T)
        mask = data.get("mask", None)
        if mask is not None:
            mask = mask.to(self.config.device)            # (B, T)
        return feats, gt, mask

    # ============================================================
    # Covariance regularization losses (used when use_CovLoss=True)
    # ============================================================

    def yellow_loss(self, embed, ep=1e-4):
        mean = torch.mean(embed, dim=(0, 1))
        center = embed.view(-1, embed.shape[-1]) - mean
        var = torch.var(center, dim=0)
        std = torch.sqrt(var + ep)
        loss_per_dim = torch.relu(1 - std)
        loss = torch.sum(loss_per_dim) / embed.shape[-1]
        return loss

    def green_loss(self, embed):
        N_samples = embed.shape[0] * embed.shape[1]
        D = embed.shape[2]

        z = embed.view(-1, D)
        z = z - z.mean(dim=0)
        z_norm = F.normalize(z, p=2, dim=0)
        corr = z_norm.T @ z_norm
        corr_sq = corr.pow(2)
        off_diag_sum = corr_sq.sum() - D
        loss = off_diag_sum / D
        return loss

    # ============================================================
    # Train
    # ============================================================

    def _save_ckpt(self, path: str):
        """Save model (and projector if present) to a single .pkl file."""
        payload = {"model": self.model.state_dict()}
        if hasattr(self, "projector") and self.projector is not None:
            payload["projector"] = self.projector.state_dict()
        torch.save(payload, path)

    def train(self, use_CovLoss=False, loss_type='combined', training_style='round_robin', use_smoothap=False):
        """
        Joint training for emotion, event, and nature detection.

        Args:
            use_CovLoss (bool): Whether to use Covariance Loss.
            loss_type (str): Type of loss to use.
            training_style (str): 'round_robin' (interleaved batches) or 'sequential' (epoch phases).
        """
        best_event_map50, best_event_map15 = -1.0, -1.0
        best_emotion_map50, best_emotion_map15 = -1.0, -1.0
        best_nature_map50, best_nature_map15 = -1.0, -1.0

        global_threshold = 2
        best_event_loss, best_emotion_loss, best_nature_loss = float('inf'), float('inf'), float('inf')
        event_threshold_count, emotion_threshold_count, nature_threshold_count = 0, 0, 0

        os.makedirs(self.config.save_dir_root, exist_ok=True)
        os.makedirs(self.config.best_map50_save_dir, exist_ok=True)
        os.makedirs(self.config.best_map15_save_dir, exist_ok=True)

        history = []

        # Per-step diagnostics (across the whole run)
        step_sim_ev_em = []
        step_sim_ev_nt = []
        step_sim_em_nt = []
        step_cond = []

        if training_style not in ['round_robin', 'sequential']:
            raise ValueError(f"Unknown training_style: {training_style}. Use 'round_robin' or 'sequential'.")

        print(f"==== Starting Joint Training ({training_style.upper()} Strategy) ====")

        if training_style == 'sequential' and (self.use_grad_norm or self.lambda_ortho > 0):
            print("WARNING: PCGrad and Task Orthogonality Loss require simultaneous task gradients.")
            print("These features will be disabled during 'sequential' training.")

        len_event = len(self.event_train_loader)
        len_emotion = len(self.emotion_train_loader)
        len_nature = len(self.nature_train_loader)
        max_steps = max(len_event, len_emotion, len_nature)

        for epoch_i in range(self.config.epochs):
            print(f"[Epoch: {epoch_i}/{self.config.epochs}]")
            self.model.train()

            # choose shared parameters once per epoch (safe if model structure changes)
            shared_params = self._get_shared_params()

            loss_history_event, loss_history_emotion, loss_history_nature = [], [], []
            sim_ev_em_list, sim_ev_nt_list, sim_em_nt_list = [], [], []
            cond_list_epoch = []
            event_yellow, event_green = [], []
            emotion_yellow, emotion_green = [], []
            nature_yellow, nature_green = [], []
            ortho_loss_list = []

            # ==========================================
            # ROUND ROBIN TRAINING STYLE
            # ==========================================
            if training_style == 'round_robin':
                iter_event = iter(self.event_train_loader)
                iter_emotion = iter(self.emotion_train_loader)
                iter_nature = iter(self.nature_train_loader)

                for idx in tqdm(range(max_steps), desc="Joint Train", leave=False):
                    self.optimizer.zero_grad()

                    # Fetch Data
                    try:
                        data_event = next(iter_event)
                    except StopIteration:
                        iter_event = iter(self.event_train_loader)
                        data_event = next(iter_event)

                    try:
                        data_emotion = next(iter_emotion)
                    except StopIteration:
                        iter_emotion = iter(self.emotion_train_loader)
                        data_emotion = next(iter_emotion)

                    try:
                        data_nature = next(iter_nature)
                    except StopIteration:
                        iter_nature = iter(self.nature_train_loader)
                        data_nature = next(iter_nature)

                    # Event Forward
                    feats_ev, gt_ev, mask_ev = self._prep_batch(data_event)
                    if self.dimension_count!=512:
                        feats_ev = self.projector(feats_ev)
                    pred_ev, embed_ev, _, _ = self.model(feats_ev, mask_ev, 'event')
                    per_ev = self.criterion(pred_ev, gt_ev)
                    loss_ev = per_ev[mask_ev].mean()
                    if use_smoothap:
                        additional_loss = temporal_gradient_loss(pred_ev, gt_ev, mask_ev) * 4 + self.smoothap(pred_ev, gt_ev, mask_ev)
                        loss_ev += additional_loss
                    if idx%500==0:
                        print(f"EVENT: BCE Loss: {per_ev[mask_ev].mean().item():.2f}")

                    if use_CovLoss:
                        covloss_ev = self.beta * self.yellow_loss(embed_ev) + self.alpha * self.green_loss(embed_ev)
                        event_yellow.append(self.yellow_loss(embed_ev).cpu().item())
                        event_green.append(self.green_loss(embed_ev).cpu().item())
                        loss_ev += covloss_ev
                    loss_history_event.append(float(loss_ev.item()))

                    # Emotion Forward
                    feats_em, gt_em, mask_em = self._prep_batch(data_emotion)
                    if self.dimension_count!=512:
                        feats_em = self.projector(feats_em)
                    pred_em, embed_em, _, _ = self.model(feats_em, mask_em, 'emotion')
                    per_em = self.criterion(pred_em, gt_em)
                    loss_em = per_em[mask_em].mean()
                    if use_smoothap:
                        additional_loss = temporal_gradient_loss(pred_em, gt_em, mask_em) * 4 + self.smoothap(pred_em, gt_em, mask_em)
                        loss_em += additional_loss
                    if idx%500==0:
                        print(f"EMOTION: BCE Loss: {per_em[mask_em].mean().item():.2f}")

                    if use_CovLoss:
                        covloss_em = self.beta * self.yellow_loss(embed_em) + self.alpha * self.green_loss(embed_em)
                        emotion_yellow.append(self.yellow_loss(embed_em).cpu().item())
                        emotion_green.append(self.green_loss(embed_em).cpu().item())
                        loss_em += covloss_em
                    loss_history_emotion.append(float(loss_em.item()))

                    # Nature Forward
                    feats_nt, gt_nt, mask_nt = self._prep_batch(data_nature)
                    if self.dimension_count!=512:
                        feats_nt = self.projector(feats_nt)
                    pred_nt, embed_nature, _, _ = self.model(feats_nt, mask_nt, 'nature')
                    per_nt = self.criterion(pred_nt, gt_nt)
                    loss_nt = per_nt[mask_nt].mean()
                    if use_smoothap:
                        additional_loss = temporal_gradient_loss(pred_nt, gt_nt, mask_nt) * 4 + self.smoothap(pred_nt, gt_nt, mask_nt)
                        loss_nt += additional_loss
                    if idx%500==0:
                        print(f"NATURE: BCE Loss: {per_nt[mask_nt].mean().item():.2f}")
                    if use_CovLoss:
                        covloss_nt = self.beta * self.yellow_loss(embed_nature) + self.alpha * self.green_loss(embed_nature)
                        nature_yellow.append(self.yellow_loss(embed_nature).cpu().item())
                        nature_green.append(self.green_loss(embed_nature).cpu().item())
                        loss_nt += covloss_nt
                    loss_history_nature.append(float(loss_nt.item()))

                    # -------------------------------
                    # Gradient cosine sim + cond num
                    # -------------------------------
                    # Note: these are computed BEFORE backward() using autograd.grad
                    g_ev = self._flat_grad(loss_ev, shared_params)
                    g_em = self._flat_grad(loss_em, shared_params)
                    g_nt = self._flat_grad(loss_nt, shared_params)

                    sim_ev_em = self._cos_sim_safe(g_ev, g_em)
                    sim_ev_nt = self._cos_sim_safe(g_ev, g_nt)
                    sim_em_nt = self._cos_sim_safe(g_em, g_nt)
                    cond_g = self._cond_number_from_task_grads([g_ev, g_em, g_nt])

                    step_sim_ev_em.append(sim_ev_em)
                    step_sim_ev_nt.append(sim_ev_nt)
                    step_sim_em_nt.append(sim_em_nt)
                    step_cond.append(cond_g)

                    sim_ev_em_list.append(sim_ev_em)
                    sim_ev_nt_list.append(sim_ev_nt)
                    sim_em_nt_list.append(sim_em_nt)
                    cond_list_epoch.append(cond_g)

                    # Joint Backward
                    if use_CovLoss:
                        ortho_loss = self.ortho_loss_fn([
                            (embed_ev, mask_ev),
                            (embed_em, mask_em),
                            (embed_nature, mask_nt)
                        ])
                        ortho_loss_list.append(ortho_loss.cpu().item())
                    else:
                        ortho_loss = 0.0

                    if self.use_grad_norm:
                        total_loss = [loss_ev, loss_em, loss_nt]
                    else:
                        total_loss = loss_ev * self.event_weight + loss_em * self.emotion_weight + loss_nt * self.nature_weight
                        if use_CovLoss:
                            total_loss += self.lambda_ortho * ortho_loss

                    if self.use_grad_norm:
                        self.optimizer.pc_backward(total_loss)
                    else:
                        total_loss.backward()

                    for name, p in self.model.named_parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            print(f"[NaN/Inf grad] step={self.global_step} param={name}")

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.global_step += 1

            # ==========================================
            # SEQUENTIAL TRAINING STYLE
            # ==========================================
            elif training_style == 'sequential':
                # 1. EVENT PHASE
                for data_event in tqdm(self.event_train_loader, desc="Event Phase", leave=False):
                    self.optimizer.zero_grad()
                    feats_ev, gt_ev, mask_ev = self._prep_batch(data_event)
                    pred_ev, embed_ev, _, _ = self.model(feats_ev, mask_ev, 'event')

                    per_ev = self.criterion(pred_ev, gt_ev)
                    loss_ev = per_ev[mask_ev].mean()

                    if use_CovLoss:
                        covloss_ev = self.beta * self.yellow_loss(embed_ev) + self.alpha * self.green_loss(embed_ev)
                        event_yellow.append(self.yellow_loss(embed_ev).cpu().item())
                        event_green.append(self.green_loss(embed_ev).cpu().item())
                        loss_ev += covloss_ev

                    loss_history_event.append(float(loss_ev.item()))
                    total_loss = loss_ev * self.event_weight
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.global_step += 1

                # 2. EMOTION PHASE
                for data_emotion in tqdm(self.emotion_train_loader, desc="Emotion Phase", leave=False):
                    self.optimizer.zero_grad()
                    feats_em, gt_em, mask_em = self._prep_batch(data_emotion)
                    pred_em, embed_em, _, _ = self.model(feats_em, mask_em, 'emotion')

                    per_em = self.criterion(pred_em, gt_em)
                    loss_em = per_em[mask_em].mean()

                    if use_CovLoss:
                        covloss_em = self.beta * self.yellow_loss(embed_em) + self.alpha * self.green_loss(embed_em)
                        emotion_yellow.append(self.yellow_loss(embed_em).cpu().item())
                        emotion_green.append(self.green_loss(embed_em).cpu().item())
                        loss_em += covloss_em

                    loss_history_emotion.append(float(loss_em.item()))
                    total_loss = loss_em * self.emotion_weight
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.global_step += 1

                # 3. NATURE PHASE
                for data_nature in tqdm(self.nature_train_loader, desc="Nature Phase", leave=False):
                    self.optimizer.zero_grad()
                    feats_nt, gt_nt, mask_nt = self._prep_batch(data_nature)
                    pred_nt, embed_nature, _, _ = self.model(feats_nt, mask_nt, 'nature')

                    per_nt = self.criterion(pred_nt, gt_nt)
                    loss_nt = per_nt[mask_nt].mean()

                    if use_CovLoss:
                        covloss_nt = self.beta * self.yellow_loss(embed_nature) + self.alpha * self.green_loss(embed_nature)
                        nature_yellow.append(self.yellow_loss(embed_nature).cpu().item())
                        nature_green.append(self.green_loss(embed_nature).cpu().item())
                        loss_nt += covloss_nt

                    loss_history_nature.append(float(loss_nt.item()))
                    total_loss = loss_nt * self.nature_weight
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.global_step += 1

            # --- END OF EPOCH EVALUATION ---
            if self.scheduler is not None:
                self.scheduler.step()

            train_loss_ev = np.mean(loss_history_event) if loss_history_event else 0.0
            train_loss_em = np.mean(loss_history_emotion) if loss_history_emotion else 0.0
            train_loss_nt = np.mean(loss_history_nature) if loss_history_nature else 0.0

            event_yellow_mean = np.mean(event_yellow) if event_yellow else 0.0
            event_green_mean = np.mean(event_green) if event_green else 0.0
            emotion_yellow_mean = np.mean(emotion_yellow) if emotion_yellow else 0.0
            emotion_green_mean = np.mean(emotion_green) if emotion_green else 0.0
            nature_yellow_mean = np.mean(nature_yellow) if nature_yellow else 0.0
            nature_green_mean = np.mean(nature_green) if nature_green else 0.0

            ortho_loss_mean = np.mean(ortho_loss_list) if ortho_loss_list else 0.0
            cond_mean = float(np.mean(cond_list_epoch)) if cond_list_epoch else 0.0

            # Evaluate Models
            val_map50_ev, val_map15_ev, val_loss_ev = self.evaluate_highlight(
                self.event_val_loader, training_type='event', use_CovLoss=use_CovLoss
            )
            val_map50_em, val_map15_em, val_loss_em = self.evaluate_highlight(
                self.emotion_val_loader, training_type='emotion', use_CovLoss=use_CovLoss
            )
            val_map50_nt, val_map15_nt, val_loss_nt = self.evaluate_highlight(
                self.nature_val_loader, training_type='nature', use_CovLoss=use_CovLoss
            )

            print(f"   Train Loss (Event/Emot/Nature): {train_loss_ev:.4f} / {train_loss_em:.4f} / {train_loss_nt:.4f}")
            print(f"   Val Loss (Event/Emot/Nature): {val_loss_ev:.4f} / {val_loss_em:.4f} / {val_loss_nt:.4f}")
            if training_style == 'round_robin':
                print(f"   Ortho loss: {ortho_loss_mean:.7f}")
                print(f"   Grad cond (epoch mean): {cond_mean:.4e}")
                print(f"   Grad sim (epoch mean) EV-EM/EV-NT/EM-NT: "
                      f"{float(np.mean(sim_ev_em_list)) if sim_ev_em_list else 0.0:.4f} / "
                      f"{float(np.mean(sim_ev_nt_list)) if sim_ev_nt_list else 0.0:.4f} / "
                      f"{float(np.mean(sim_em_nt_list)) if sim_em_nt_list else 0.0:.4f}")
            print(f"   Yellow/Green losses (Event/Emot/Nature): {event_yellow_mean:.6f} / {event_green_mean:.6f} / "
                  f"{emotion_yellow_mean:.6f} / {emotion_green_mean:.6f} / {nature_yellow_mean:.6f} / {nature_green_mean:.6f}")
            print(f"   [EVENT]   MAP50: {val_map50_ev:.4f} | MAP15: {val_map15_ev:.4f}")
            print(f"   [EMOTION] MAP50: {val_map50_em:.4f} | MAP15: {val_map15_em:.4f}")
            print(f"   [NATURE]  MAP50: {val_map50_nt:.4f} | MAP15: {val_map15_nt:.4f}")

            # Save Models
            if val_map50_ev >= best_event_map50:
                best_event_map50 = val_map50_ev
                self._save_ckpt(os.path.join(self.config.best_map50_save_dir, "best_map50_EVENT.pkl"))

            if val_map15_ev >= best_event_map15:
                best_event_map15 = val_map15_ev
                self._save_ckpt(os.path.join(self.config.best_map15_save_dir, "best_map15_EVENT.pkl"))


            if val_map50_em >= best_emotion_map50:
                best_emotion_map50 = val_map50_em
                self._save_ckpt(os.path.join(self.config.best_map50_save_dir, "best_map50_EMOTION.pkl"))

            if val_map15_em >= best_emotion_map15:
                best_emotion_map15 = val_map15_em
                self._save_ckpt(os.path.join(self.config.best_map15_save_dir, "best_map15_EMOTION.pkl"))


            if val_map50_nt >= best_nature_map50:
                best_nature_map50 = val_map50_nt
                self._save_ckpt(os.path.join(self.config.best_map50_save_dir, "best_map50_NATURE.pkl"))

            if val_map15_nt >= best_nature_map15:
                best_nature_map15 = val_map15_nt
                self._save_ckpt(os.path.join(self.config.best_map15_save_dir, "best_map15_NATURE.pkl"))


            # Log history
            history.append({
                "epoch": int(epoch_i),
                "train_loss_event": float(train_loss_ev),
                "train_loss_emotion": float(train_loss_em),
                "train_loss_nature": float(train_loss_nt),
                "val_loss_event": float(val_loss_ev),
                "val_loss_emotion": float(val_loss_em),
                "val_loss_nature": float(val_loss_nt),
                "val_map50_event": float(val_map50_ev),
                "val_map50_emotion": float(val_map50_em),
                "val_map15_event": float(val_map15_ev),
                "val_map15_emotion": float(val_map15_em),
                "val_map50_nature": float(val_map50_nt),
                "val_map15_nature": float(val_map15_nt),
                "grad_sim_event_emotion": float(np.mean(sim_ev_em_list)) if sim_ev_em_list else 0.0,
                "grad_sim_event_nature": float(np.mean(sim_ev_nt_list)) if sim_ev_nt_list else 0.0,
                "grad_sim_emotion_nature": float(np.mean(sim_em_nt_list)) if sim_em_nt_list else 0.0,
                "grad_cond_mean": float(cond_mean),
            })
            with open(os.path.join(self.config.save_dir_root, "history_joint.json"), "w") as f:
                json.dump(history, f, indent=2)

            # Early Stopping
            if event_threshold_count > global_threshold and emotion_threshold_count > global_threshold and nature_threshold_count > global_threshold:
                print("Early stopping!")
                break

            if val_loss_ev < best_event_loss:
                best_event_loss = val_loss_ev
                event_threshold_count = 0
            else:
                event_threshold_count += 1

            if val_loss_em < best_emotion_loss:
                best_emotion_loss = val_loss_em
                emotion_threshold_count = 0
            else:
                emotion_threshold_count += 1

            if val_loss_nt < best_nature_loss:
                best_nature_loss = val_loss_nt
                nature_threshold_count = 0
            else:
                nature_threshold_count += 1

        # -------------------------------------------------------
        # Save diagnostics JSON + plots (only if round_robin used)
        # -------------------------------------------------------
        if training_style == "round_robin" and len(step_sim_ev_em) > 0:
            diag_path = os.path.join(self.config.save_dir_root, "grad_diagnostics_steps.json")
            with open(diag_path, "w") as f:
                json.dump({
                    "sim_ev_em": step_sim_ev_em,
                    "sim_ev_nt": step_sim_ev_nt,
                    "sim_em_nt": step_sim_em_nt,
                    "cond": step_cond
                }, f, indent=2)
            print(f"Saved gradient diagnostics json to: {diag_path}")

            self.generate_grad_diagnostics_plot(
                step_sim_ev_em, step_sim_ev_nt, step_sim_em_nt,
                step_cond,
                save_dir=self.config.save_dir_root
            )

        best_map50_event_path = os.path.join(self.config.best_map50_save_dir, "best_map50_EVENT.pkl")
        best_map50_emotion_path = os.path.join(self.config.best_map50_save_dir, "best_map50_EMOTION.pkl")
        best_map15_event_path = os.path.join(self.config.best_map15_save_dir, "best_map15_EVENT.pkl")
        best_map15_emotion_path = os.path.join(self.config.best_map15_save_dir, "best_map15_EMOTION.pkl")
        best_map50_nature_path = os.path.join(self.config.best_map50_save_dir, "best_map50_NATURE.pkl")
        best_map15_nature_path = os.path.join(self.config.best_map15_save_dir, "best_map15_NATURE.pkl")

        return (
            best_map50_event_path, best_map50_emotion_path,
            best_map15_event_path, best_map15_emotion_path,
            best_map50_nature_path, best_map15_nature_path
        )

    # ============================================================
    # Eval / Test / Infer (unchanged)
    # ============================================================

    @torch.no_grad()
    def evaluate_highlight(self, dataloader=None, training_type='emotion', use_CovLoss=False):
        self.model.eval()

        map50_history = []
        map15_history = []
        loss_history = []
        CovLoss_history = []
        indiv_loss_history = []

        for batch_idx, data in enumerate(tqdm(iter(dataloader), desc=f"eval_map for '{training_type}'", leave=False)):
            feats, gt, mask = self._prep_batch(data)

            if self.dimension_count!=512:
                feats = self.projector(feats)
            pred, component1, _, _ = self.model(feats, mask, training_type)

            if use_CovLoss:
                covloss = self.beta * self.yellow_loss(component1) + self.alpha * self.green_loss(component1)

            per = self.criterion(pred, gt)
            loss = per[mask].mean()
            
            if use_CovLoss:
                CovLoss_history.append(covloss.cpu().item())
                indiv_loss_history.append(loss.item())
                loss_history.append(loss.item() + covloss.cpu().item())
            else:
                loss_history.append(loss.item())

            pred_np = pred.detach().float().cpu().numpy()
            gt_np = gt.detach().float().cpu().numpy()
            mask_np = mask.detach().cpu().numpy() if mask is not None else None

            B = pred_np.shape[0]
            for b in range(B):
                if mask_np is not None:
                    valid = mask_np[b].astype(bool)
                    y_score = pred_np[b][valid]
                    y_gt = gt_np[b][valid]
                else:
                    y_score = pred_np[b]
                    y_gt = gt_np[b]

                if len(y_score) == 0:
                    continue

                y50 = _topk_binary(y_gt, 0.50)
                y15 = _topk_binary(y_gt, 0.15)

                ap50 = _safe_ap(y50, y_score)
                ap15 = _safe_ap(y15, y_score)
                
                map50_history.append(ap50)
                map15_history.append(ap15)

        if len(CovLoss_history) > 0:
            print(f"{training_type} Original Loss: {np.mean(indiv_loss_history)} | CovLoss: {np.mean(CovLoss_history)}")

        final_map50 = float(np.mean(map50_history)) if map50_history else 0.0
        final_map15 = float(np.mean(map15_history)) if map15_history else 0.0

        return final_map50, final_map15, np.mean(loss_history) if loss_history else 0.0

    def test(self, ckpt_path, training_type='emotion'):
        if ckpt_path is not None:
            print("Testing Model: ", ckpt_path)
            print("Device: ", self.config.device)
            load_ckpt_safely(self.model, ckpt_path, self.config.device,
                 projector=self.projector if hasattr(self, "projector") else None)

        if training_type == 'emotion':
            test_loader = self.emotion_test_loader
        elif training_type == 'event':
            test_loader = self.event_test_loader
        else:
            test_loader = self.nature_test_loader

        test_map50, test_map15, _ = self.evaluate_highlight(
            dataloader=test_loader,
            training_type=training_type
        )

        print("------------------------------------------------------")
        print(f"   TEST RESULT ({training_type}) on {ckpt_path}: ")
        print(f"   TEST MAP50 {test_map50:.5f} | MAP15 {test_map15:.5f}")
        print("------------------------------------------------------")

        with open(os.path.join(self.config.save_dir_root, f"results_{training_type}.txt"), "a") as f:
            f.write(f"Testing on Model {ckpt_path}\n")
            f.write(f"Test MAP50   {test_map50}\n")
            f.write(f"Test MAP15   {test_map15}\n\n")

        return test_map50, test_map15

    def infer(self, dataloader, ckpt_path=None, save_dir=None, training_type='emotion'):
        """Run inference on a dataloader."""
        if ckpt_path is not None:
            print("Loading checkpoint:", ckpt_path)
            load_ckpt_safely(self.model, ckpt_path, self.config.device,
                 projector=self.projector if hasattr(self, "projector") else None)
        self.model.eval()
        results = []

        with torch.no_grad():
            for data in dataloader:
                feats, _, mask = self._prep_batch(data)

                score, _ = self.model(feats, mask, training_type)

                score_np = score.detach().cpu().numpy()
                mask_np = mask.detach().cpu().numpy()

                B = score_np.shape[0]
                video_names = data.get("video_name", [None] * B)

                for b in range(B):
                    valid_mask = mask_np[b]
                    valid_score = score_np[b][valid_mask]

                    results.append({
                        "video_id": video_names[b] if isinstance(video_names, list) else video_names,
                        "score": valid_score
                    })

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(results, os.path.join(save_dir, f"inference_results_{training_type}.pt"))

        return results

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """Initialize network weights."""
        for name, param in net.named_parameters():
            if "weight" in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))
                else:
                    raise NotImplementedError(f"initialization method {init_type} is not implemented.")
            elif "bias" in name:
                nn.init.constant_(param, 0.1)