# model/solver_combined_aligned_mtl.py
"""
Variant of solver_combined_5s.py with Aligned-MTL (Senushkin et al., CVPR 2023)
as an optional alternative to PCGrad for combining the three task gradients.
Used to produce the "With Aligned-MTL" comparison in Fig. 4a of the paper.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from model.aligned_mtl import AlignedMTL
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

    This is a drop-in replacement for nn.BCELoss(reduction="none").
    It expects predictions AFTER sigmoid (i.e., probabilities in [0, 1]),
    matching your current usage:
        pred = torch.sigmoid(score)   # in your model forward()
        per  = self.criterion(pred, gt)
        loss = per[mask].mean()

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p_t = p     if y == 1
        p_t = 1 - p if y == 0

    Args:
        alpha (float): Weighting factor for the positive class.
                       Set < 0.5 to down-weight positives, > 0.5 to up-weight.
                       Default: 0.25 (standard from the original paper).
        gamma (float): Focusing parameter. Higher values focus more on hard
                       examples. gamma=0 recovers standard BCE.
                       Default: 2.0 (standard from the original paper).
        reduction (str): 'none' | 'mean' | 'sum'.
                         Use 'none' to match your current masked loss pattern.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "none"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, T) predicted probabilities AFTER sigmoid, in [0, 1]
            target: (B, T) ground-truth labels, in [0, 1]

        Returns:
            loss:   (B, T) per-element focal loss if reduction='none'
        """
        # Clamp for numerical stability (same as BCELoss internally)
        eps = 1e-7
        pred = pred.clamp(eps, 1.0 - eps)

        # Standard BCE component: -[y*log(p) + (1-y)*log(1-p)]
        bce = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)

        # p_t: probability of the correct class
        p_t = target * pred + (1.0 - target) * (1.0 - pred)

        # alpha_t: class-specific weighting
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)

        # Focal modulating factor
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        # Final focal loss
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
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


def load_ckpt_safely(model: nn.Module, ckpt_path: str, device: str):
    """
    Robust checkpoint loading for:
      - flat state_dict
      - nested {"model": state_dict} / {"state_dict": state_dict}
      - nested SL_module custom format {"transformer":..., "score_model":...}

    Also patches known mismatch key: transformer.pos_alpha
    """
    state = torch.load(ckpt_path, map_location=device)

    # unwrap common wrappers
    if isinstance(state, dict):
        for k in ("model", "state_dict", "net", "network"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a dict-like state_dict. type={type(state)}")

    state = _fix_state_dict_keys(state)

    # patch known problematic key if present
    if "transformer.pos_alpha" in state and not hasattr(model.transformer, "pos_alpha"):
        state.pop("transformer.pos_alpha", None)

    # Try strict load first; fallback to non-strict with prints
    try:
        model.load_state_dict(state, strict=True)
        print("[LOAD] strict=True OK")
        return
    except RuntimeError as e:
        print("[LOAD] strict=True failed:", str(e).split("\n")[0])

    # If failing because of pos_alpha mismatch, pop it and try again strict
    if "transformer.pos_alpha" in state:
        print("[LOAD] removing key transformer.pos_alpha and retry strict")
        state.pop("transformer.pos_alpha", None)
        try:
            model.load_state_dict(state, strict=True)
            print("[LOAD] strict=True OK after removing pos_alpha")
            return
        except RuntimeError as e:
            print("[LOAD] strict=True still failed:", str(e).split("\n")[0])

    # Final fallback: non-strict
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
    # AP undefined if only one class present
    if y_true.min() == y_true.max():
        return 0.0
    return float(average_precision_score(y_true, y_score))


# ============================================================
# Solver for variable-length 5-second segments
# ============================================================
class Solver100(object):
    """
    Solver for combined emotion/event training.
    Works with variable-length sequences from Pad5sCollator.
    No interpolation to fixed length.
    """
    
    def __init__(self, config=None, emotion_train_loader=None, emotion_val_loader=None, emotion_test_loader=None, 
                 event_train_loader=None, event_val_loader=None, event_test_loader=None, 
                 nature_train_loader=None, nature_val_loader=None, nature_test_loader=None, 
                 beta = 0.5, alpha = 0.5, lambda_ortho = 0.1,
                 event_weight = 1, emotion_weight=1, nature_weight = 1, CovLoss_lambda = 1e10,
                 use_grad_norm = False, use_aligned_mtl = True):
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
        self.use_aligned_mtl = use_aligned_mtl

        self.global_step = 0
        
        self.CovLoss_lambda = CovLoss_lambda
        
        # Initialize Aligned-MTL if enabled
        if self.use_aligned_mtl:
            self.aligned_mtl = AlignedMTL()

        # Binary cross entropy loss for highlight detection
        self.criterion = nn.BCELoss(reduction="none").to(self.config.device)
        # self.criterion = BinaryFocalLoss(alpha=0.8, gamma=0.5, reduction="none").to(self.config.device)
        
        self.ortho_loss_fn = MaskedTaskOrthogonalityLoss(method='subspace').to(self.config.device)

    def shared_backbone_regularization(self, shared_attn, task_attn=None, *, tv_weight=0.1, kl_weight=0.1, ortho_weight=0.05):
        """
        shared_attn: (layers, batch, heads, seq, seq) or (batch, heads, seq, seq)
        """
        loss = 0.0
        
        if shared_attn.dim() == 5:
            shared_attn = shared_attn.mean(dim=0)
        if task_attn is not None and task_attn.dim() == 5:
            task_attn = task_attn.mean(dim=0)
            
        b, h, t, _ = shared_attn.shape
        
        # 1. KL Divergence to Uniform (Universally Smooth)
        uniform_target = torch.full_like(shared_attn, 1.0 / t)
        log_shared_attn = torch.log(shared_attn + 1e-8)
        
        # --- THE FIX ---
        # Formula: sum( target * (log(target) - log(input)) )
        # Sum over the sequence/keys dimension (dim=-1) to complete the KL math
        # Then take the mean() over Batch, Heads, and Queries so it doesn't inflate!
        kl_per_query = (uniform_target * (torch.log(uniform_target) - log_shared_attn)).sum(dim=-1)
        kl_loss = kl_per_query.mean() 
        
        loss += kl_weight * kl_loss

        # 2. Total Variation (Temporal Smoothness)
        tv_loss = torch.abs(shared_attn[:, :, :, 1:] - shared_attn[:, :, :, :-1]).mean()
        loss += tv_weight * tv_loss
        
        # 3. Orthogonality (Distinctness)
        if task_attn is not None:
            flat_shared = shared_attn.reshape(b, -1)
            flat_task = task_attn.reshape(b, -1)
            sim = F.cosine_similarity(flat_shared, flat_task, dim=-1).mean()
            loss += ortho_weight * sim
        
        return loss
        
    def _compute_gradient_similarity(self, loss_a, loss_b, shared_params):
        """
        Computes the cosine similarity between the gradients of two losses 
        with respect to the shared parameters.
        """
        # Calculate gradients independently. retain_graph=True is essential!
        grad_a = torch.autograd.grad(loss_a, shared_params, retain_graph=True, allow_unused=True)
        grad_b = torch.autograd.grad(loss_b, shared_params, retain_graph=True, allow_unused=True)

        # Flatten and concatenate valid gradients
        vec_a = torch.cat([g.view(-1) for g in grad_a if g is not None])
        vec_b = torch.cat([g.view(-1) for g in grad_b if g is not None])

        # Handle empty tensors or zero vectors to avoid NaN
        if vec_a.numel() == 0 or vec_b.numel() == 0:
            return 0.0
            
        norm_a = torch.norm(vec_a)
        norm_b = torch.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0

        sim = torch.dot(vec_a, vec_b) / (norm_a * norm_b)
        return sim.item()

    def _analyze_gradient_conflicts(self, losses, shared_params):
        """
        Calculates Aligned-MTL metrics: Condition Number and Dominance Rate.
        """
        grads = []
        for loss in losses:
            # Compute gradients for each task separately, retain_graph is crucial here!
            grad = torch.autograd.grad(loss, shared_params, retain_graph=True, allow_unused=True)
            flat_grad = torch.cat([g.view(-1) for g in grad if g is not None])
            
            # Handle empty graph fallback gracefully
            if flat_grad.numel() == 0:
                flat_grad = torch.zeros(1, device=losses[0].device)
                
            grads.append(flat_grad)
        
        # Stack into a G matrix (NumTasks x NumParameters)
        G = torch.stack(grads) # shape: [3, P]
        
        # Compute Singular Values via SVD for stability analysis
        # Since P is huge, we compute SVD on G @ G.T (NumTasks x NumTasks)
        # which has the same non-zero singular values squared.
        gram_matrix = G @ G.T
        S_squared = torch.linalg.eigvalsh(gram_matrix)
        S = torch.sqrt(torch.clamp(S_squared, min=1e-8))
        
        # 1. Condition Number: max_sigma / min_sigma
        # High condition number = High conflict/instability
        cond_number = S.max() / S.min()
        
        # 2. Dominance Rate: How much the top singular value dominates
        dominance = S.max() / S.sum()
        
        return cond_number.item(), dominance.item()

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
            self.model = SL_module(input_dim=512, depth=5, heads=8, mlp_dim=3072, dropout_ratio=self.config.dropout_ratio).to(self.config.device) # The original shared encoder architecture
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
                input_dim=512,
                depth=5, 
                total_heads=8, 
                shared_heads=4,
                mlp_dim=3072, 
                dropout_ratio=self.config.dropout_ratio,
                separate_ffn=False,
                pos_type='rope',
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
            if self.use_grad_norm:
                self.optimizer = PCGrad(self.optimizer)

        else:
            raise ValueError(f"Wrong model: {self.config.model}")

    # -------------------------
    # Prepare batch (no interpolation)
    # -------------------------
    def _prep_batch(self, data: dict):
        """
        Prepare batch from Pad5sCollator output.
        No interpolation - uses variable-length sequences directly.
        
        Args:
            data: Dict with 'features', 'gtscore', 'mask' from Pad5sCollator
        
        Returns:
            feats: (B, T, D) features tensor on device
            gt: (B, T) ground truth tensor on device
            mask: (B, T) boolean mask tensor on device
        """
        feats = data["features"].to(self.config.device)   # (B, T, D)
        gt = data["gtscore"].to(self.config.device)       # (B, T)
        mask = data.get("mask", None)

        # 2. Only move to device if it actually exists
        if mask is not None:
            mask = mask.to(self.config.device)      # (B, T)
        
        return feats, gt, mask
    
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
    
    def generate_gradient_plot(self, sim_ev_em, sim_ev_nt, sim_em_nt, save_dir, smooth_weight=0.9):
        """
        Generates and saves a plot of the gradient cosine similarities 
        tracked during the training run per iteration/step.
        """
        if not sim_ev_em:
            print("No history to plot.")
            return

        # Helper function to smooth out noisy per-iteration data
        def smooth(scalars, weight):
            last = scalars[0]
            smoothed = []
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed

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

    def generate_conflict_report(self, cond_nums, dominance_rates, save_dir, smooth_weight=0.9):
        """
        Generates and saves plots for the Condition Number and Dominance Rate tracking task conflicts.
        """
        if not cond_nums:
            return

        def smooth(scalars, weight):
            last = scalars[0]
            smoothed = []
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed

        steps = list(range(len(cond_nums)))

        # Apply smoothing
        smooth_cond = smooth(cond_nums, smooth_weight)
        smooth_dom = smooth(dominance_rates, smooth_weight)

        # Plot 1: Condition Number (Instability)
        plt.figure(figsize=(10, 5))
        plt.plot(steps, cond_nums, color='red', alpha=0.15, linewidth=1)
        plt.plot(steps, smooth_cond, label='Condition Number', color='red', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Training Iterations', fontsize=12)
        plt.ylabel('Stability (Log Scale)', fontsize=12)
        plt.title('Optimization Stability: Condition Number (Higher = More Conflict)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        
        cond_path = os.path.join(save_dir, "gradient_instability_cond.png")
        plt.savefig(cond_path, dpi=300)
        plt.close()
        print(f"Saved Condition Number plot to: {cond_path}")

        # Plot 2: Dominance Rate
        plt.figure(figsize=(10, 5))
        plt.plot(steps, dominance_rates, color='purple', alpha=0.15, linewidth=1)
        plt.plot(steps, smooth_dom, label='Dominance Rate', color='purple', linewidth=2)
        plt.xlabel('Training Iterations', fontsize=12)
        plt.ylabel('Dominance Rate', fontsize=12)
        plt.title('Task Dominance Rate over Training', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        dom_path = os.path.join(save_dir, "gradient_instability_dom.png")
        plt.savefig(dom_path, dpi=300)
        plt.close()
        print(f"Saved Dominance Rate plot to: {dom_path}")

    def train(self, use_CovLoss = False, loss_type='combined'):
        """Joint training for emotion and event detection."""
        # Initialize trackers for BOTH tasks separately
        best_event_map50 = -1.0
        best_event_map15 = -1.0
        best_emotion_map50 = -1.0
        best_emotion_map15 = -1.0
        best_nature_map50 = -1.0
        best_nature_map15 = -1.0
        
        global_threshold = 2 # How many times validation loss can be stuck at minimum 
        best_event_loss = float('inf')
        best_emotion_loss = float('inf')
        best_nature_loss = float('inf')
        event_threshold_count = 0
        emotion_threshold_count = 0
        nature_threshold_count = 0
        
        # Ensure directories exist
        os.makedirs(self.config.save_dir_root, exist_ok=True)
        os.makedirs(self.config.best_map50_save_dir, exist_ok=True) 
        os.makedirs(self.config.best_map15_save_dir, exist_ok=True)

        history = []

        step_sim_ev_em = []
        step_sim_ev_nt = []
        step_sim_em_nt = []
        
        step_cond_nums = []
        step_dom_rates = []
        
        print("==== Starting Joint Training (Cycling Strategy) ====")
        
        # Determine the length of the epoch based on the LARGER dataset
        len_event = len(self.event_train_loader)
        len_emotion = len(self.emotion_train_loader)
        len_nature = len(self.nature_train_loader)
        max_steps = max(len_event, len_emotion, len_nature)
        
        for epoch_i in range(self.config.epochs):
            print(f"[Epoch: {epoch_i}/{self.config.epochs}]")
            self.model.train()

            loss_history_event = []
            loss_history_emotion = []
            loss_history_nature = []
            
            sim_ev_em_list = []
            sim_ev_nt_list = []
            sim_em_nt_list = []
            
            cond_num_list = []
            dom_rate_list = []
            
            event_yellow, event_green = [], []
            emotion_yellow, emotion_green = [], []
            nature_yellow, nature_green = [], []
            
            ortho_loss_list = []
            
            basic_loss, CovLoss_list = [], []
            
            # Create Iterators at the start of every epoch
            iter_event = iter(self.event_train_loader)
            iter_emotion = iter(self.emotion_train_loader)
            iter_nature = iter(self.nature_train_loader)

            for _ in tqdm(range(max_steps), desc="Joint Train", leave=False):
                # self.optimizer.zero_grad(set_to_none=True)
                self.optimizer.zero_grad()
                
                # --- FETCH EVENT DATA (Cycle if needed) ---
                try:
                    data_event = next(iter_event)
                except StopIteration:
                    iter_event = iter(self.event_train_loader)
                    data_event = next(iter_event)

                # --- FETCH EMOTION DATA (Cycle if needed) ---
                try:
                    data_emotion = next(iter_emotion)
                except StopIteration:
                    iter_emotion = iter(self.emotion_train_loader)
                    data_emotion = next(iter_emotion)
                    
                # --- FETCH NATURE DATA (Cycle if needed) ---
                try:
                    data_nature = next(iter_nature)
                except StopIteration:
                    iter_nature = iter(self.nature_train_loader)
                    data_nature = next(iter_nature)

                # --- FORWARD PASS: EVENT ---
                feats_ev, gt_ev, mask_ev = self._prep_batch(data_event)
                pred_ev, embed_ev, shared_attn, task_attn = self.model(feats_ev, mask_ev, 'event')
                if use_CovLoss: # If using split backbone and using CovLoss
                    covloss = self.beta * self.yellow_loss(embed_ev) + self.alpha * self.green_loss(embed_ev)
                    event_yellow.append(self.yellow_loss(embed_ev).cpu().item())
                    event_green.append(self.green_loss(embed_ev).cpu().item())
                    per_ev = self.criterion(pred_ev, gt_ev) 
                    loss_ev = per_ev[mask_ev].mean()
                    loss_ev += covloss
                else:
                    per_ev = self.criterion(pred_ev, gt_ev)
                    loss_ev = per_ev[mask_ev].mean()

                loss_reg = self.shared_backbone_regularization(
                    shared_attn,
                    task_attn,
                    kl_weight=0.1,       # <--- CHANGE TO THIS
                    tv_weight=0.1,
                    ortho_weight=0.05
                )
                loss_ev+=loss_reg

                loss_history_event.append(float(loss_ev.item()))

                # --- FORWARD PASS: EMOTION ---
                feats_em, gt_em, mask_em = self._prep_batch(data_emotion)
                pred_em, embed_em, shared_attn, task_attn = self.model(feats_em, mask_em, 'emotion')
                if use_CovLoss: # If using split backbone and using CovLoss
                    covloss = self.beta * self.yellow_loss(embed_em) + self.alpha * self.green_loss(embed_em)
                    emotion_yellow.append(self.yellow_loss(embed_em).cpu().item())
                    emotion_green.append(self.green_loss(embed_em).cpu().item())
                    per_em = self.criterion(pred_em, gt_em) 
                    loss_em = per_em[mask_em].mean()
                    loss_em += covloss
                else:
                    per_em = self.criterion(pred_em, gt_em)
                    loss_em = per_em[mask_em].mean()

                loss_reg = self.shared_backbone_regularization(
                    shared_attn,
                    task_attn,
                    kl_weight=0.1,       # <--- CHANGE TO THIS
                    tv_weight=0.1,
                    ortho_weight=0.05
                )
                loss_em+=loss_reg

                loss_history_emotion.append(float(loss_em.item()))
                
                # --- FORWARD PASS: NATURE ---
                feats_nt, gt_nt, mask_nt = self._prep_batch(data_nature)
                pred_nt, embed_nature, shared_attn, task_attn = self.model(feats_nt, mask_nt, 'nature')
                if use_CovLoss: # If using split backbone and using CovLoss
                    covloss = self.beta * self.yellow_loss(embed_nature) + self.alpha * self.green_loss(embed_nature)
                    nature_yellow.append(self.yellow_loss(embed_nature).cpu().item())
                    nature_green.append(self.green_loss(embed_nature).cpu().item())
                    per_nt = self.criterion(pred_nt, gt_nt) 
                    loss_nt = per_nt[mask_nt].mean()
                    loss_nt += covloss
                else:
                    per_nt = self.criterion(pred_nt, gt_nt)
                    loss_nt = per_nt[mask_nt].mean()

                loss_reg = self.shared_backbone_regularization(
                    shared_attn,
                    task_attn,
                    kl_weight=0.1,       # <--- CHANGE TO THIS
                    tv_weight=0.1,
                    ortho_weight=0.05
                )
                loss_nt+=loss_reg

                loss_history_nature.append(float(loss_nt.item()))
                
                shared_params = list(self.model.get_shared_parameters())
                
                # --- Track Cosine Similarities ---
                sim_ev_em = self._compute_gradient_similarity(loss_ev, loss_em, shared_params)
                sim_ev_nt = self._compute_gradient_similarity(loss_ev, loss_nt, shared_params)
                sim_em_nt = self._compute_gradient_similarity(loss_em, loss_nt, shared_params)
                
                sim_ev_em_list.append(sim_ev_em)
                sim_ev_nt_list.append(sim_ev_nt)
                sim_em_nt_list.append(sim_em_nt)

                step_sim_ev_em.append(sim_ev_em)
                step_sim_ev_nt.append(sim_ev_nt)
                step_sim_em_nt.append(sim_em_nt)

                # --- Track Stability (Condition Number & Dominance Rate) ---
                task_losses = [loss_ev, loss_em, loss_nt]
                cond_num, dom_rate = self._analyze_gradient_conflicts(task_losses, shared_params)

                cond_num_list.append(cond_num)
                dom_rate_list.append(dom_rate)

                step_cond_nums.append(cond_num)
                step_dom_rates.append(dom_rate)
                
                # --- JOINT BACKWARD ---
                if use_CovLoss:
                    ortho_loss = self.ortho_loss_fn([
                        (embed_ev, mask_ev),
                        (embed_em, mask_em),
                        (embed_nature, mask_nt)
                    ])
                    ortho_loss_list.append(ortho_loss.cpu().item())
                    
                # --- Loss Calculation and Backward ---
                if self.use_aligned_mtl:
                    # Aligned-MTL: compute optimal task weights
                    task_losses = [loss_ev, loss_em, loss_nt]
                    if use_CovLoss:
                        # Add orthogonality loss to each task loss
                        ortho_per_task = self.lambda_ortho * ortho_loss / 3.0
                        task_losses = [l + ortho_per_task for l in task_losses]
                    
                    task_weights = self.aligned_mtl.backward(self.model, task_losses, self.optimizer)
                    
                elif self.use_grad_norm:
                    # PCGrad
                    total_loss = [loss_ev, loss_em, loss_nt]
                    if use_CovLoss:
                        # Note: PCGrad doesn't handle additional losses well, add to each task
                        ortho_per_task = self.lambda_ortho * ortho_loss / 3.0
                        total_loss = [l + ortho_per_task for l in total_loss]
                    self.optimizer.pc_backward(total_loss)
                else:
                    # Standard weighted sum
                    total_loss = loss_ev * self.event_weight + loss_em * self.emotion_weight + loss_nt * self.nature_weight
                    if use_CovLoss:
                        total_loss += self.lambda_ortho * ortho_loss
                    total_loss.backward()
                    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)   
                self.optimizer.step()

            # --- END OF EPOCH EVALUATION ---
            if self.scheduler is not None:
                self.scheduler.step()

            # Calc average train losses
            train_loss_ev = np.mean(loss_history_event)
            train_loss_em = np.mean(loss_history_emotion)
            train_loss_nt = np.mean(loss_history_nature)
            
            event_yellow_mean = np.mean(event_yellow)
            event_green_mean = np.mean(event_green)
            emotion_yellow_mean = np.mean(emotion_yellow)
            emotion_green_mean = np.mean(emotion_green)
            nature_yellow_mean = np.mean(nature_yellow)
            nature_green_mean = np.mean(nature_green)
            
            ortho_loss_mean = np.mean(ortho_loss_list)

            # --------------------------------------------------------
            # 1. Evaluate Event Head
            # --------------------------------------------------------
            val_map50_ev, val_map15_ev, val_loss_ev = self.evaluate_highlight(
                self.event_val_loader, training_type='event', use_CovLoss=use_CovLoss
            )
            
            # --------------------------------------------------------
            # 2. Evaluate Emotion Head
            # --------------------------------------------------------
            val_map50_em, val_map15_em, val_loss_em = self.evaluate_highlight(
                self.emotion_val_loader, training_type='emotion', use_CovLoss=use_CovLoss
            )
            
            # --------------------------------------------------------
            # 3. Evaluate Nature Head
            # --------------------------------------------------------
            val_map50_nt, val_map15_nt, val_loss_nt = self.evaluate_highlight(
                self.nature_val_loader, training_type='nature', use_CovLoss=use_CovLoss
            )
  
            print(f"   Train Loss (Event/Emot/Nature): {train_loss_ev:.4f} / {train_loss_em:.4f} / {train_loss_nt:.4f}")
            print(f"   Val Loss (Event/Emot/Nature): {val_loss_ev:.4f} / {val_loss_em:.4f} / {val_loss_nt:.4f}")
            print(f"   Orth loss: {ortho_loss_mean:.7f}")
            print(f"   Yellow/Green losses (Event/Emot/Nature): {event_yellow_mean:.6f} / {event_green_mean:.6f} / {emotion_yellow_mean:.6f} / {emotion_green_mean:.6f} / {nature_yellow_mean:.6f} / {nature_green_mean:.6f}")
            print(f"   [EVENT]   MAP50: {val_map50_ev:.4f} | MAP15: {val_map15_ev:.4f}")
            print(f"   [EMOTION] MAP50: {val_map50_em:.4f} | MAP15: {val_map15_em:.4f}")
            print(f"   [NATURE] MAP50: {val_map50_nt:.4f} | MAP15: {val_map15_nt:.4f}")

            # --- SAVE CHECKPOINTS (Separated by Task) ---
            
            # Save Best EVENT Model
            if val_map50_ev >= best_event_map50:
                best_event_map50 = val_map50_ev
                torch.save(self.model.state_dict(), os.path.join(self.config.best_map50_save_dir, "best_map50_EVENT.pkl"))
            
            if val_map15_ev >= best_event_map15:
                best_event_map15 = val_map15_ev
                torch.save(self.model.state_dict(), os.path.join(self.config.best_map15_save_dir, "best_map15_EVENT.pkl"))

            # Save Best EMOTION Model
            if val_map50_em >= best_emotion_map50:
                best_emotion_map50 = val_map50_em
                torch.save(self.model.state_dict(), os.path.join(self.config.best_map50_save_dir, "best_map50_EMOTION.pkl"))

            if val_map15_em >= best_emotion_map15:
                best_emotion_map15 = val_map15_em
                torch.save(self.model.state_dict(), os.path.join(self.config.best_map15_save_dir, "best_map15_EMOTION.pkl"))
                
            # Save Best NATURE Model
            if val_map50_nt >= best_nature_map50:
                best_nature_map50 = val_map50_nt
                torch.save(self.model.state_dict(), os.path.join(self.config.best_map50_save_dir, "best_map50_NATURE.pkl"))

            if val_map15_nt >= best_nature_map15:
                best_nature_map15 = val_map15_nt
                torch.save(self.model.state_dict(), os.path.join(self.config.best_map15_save_dir, "best_map15_NATURE.pkl"))

            # Log history
            history.append({
                "epoch": int(epoch_i),
                "train_loss_event": float(train_loss_ev),
                "train_loss_emotion": float(train_loss_em),
                "val_loss_event":float(val_loss_ev),
                "val_loss_emotion":float(val_loss_em),
                "val_map50_event": float(val_map50_ev),
                "val_map50_emotion": float(val_map50_em),
                "val_map15_event": float(val_map15_ev),
                "val_map15_emotion": float(val_map15_em),
                "val_map50_nature": float(val_map50_nt),
                "val_map15_nature": float(val_map15_nt),
                "grad_sim_event_emotion": float(np.mean(sim_ev_em_list)),
                "grad_sim_event_nature": float(np.mean(sim_ev_nt_list)),
                "grad_sim_emotion_nature": float(np.mean(sim_em_nt_list)),
                "grad_condition_number": float(np.mean(cond_num_list)),
                "grad_dominance_rate": float(np.mean(dom_rate_list))
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
                event_threshold_count+=1
            
            if val_loss_em < best_emotion_loss:
                best_emotion_loss = val_loss_em
                emotion_threshold_count = 0 
            else:
                emotion_threshold_count+=1
                
            if val_loss_nt < best_nature_loss:
                best_nature_loss = val_loss_nt
                nature_threshold_count = 0 
            else:
                nature_threshold_count+=1
                
        # Generate final plots representing the full training run
        self.generate_gradient_plot(step_sim_ev_em, step_sim_ev_nt, step_sim_em_nt, self.config.save_dir_root)
        self.generate_conflict_report(step_cond_nums, step_dom_rates, self.config.save_dir_root)

        # Return paths to best checkpoints
        best_map50_event_path = os.path.join(self.config.best_map50_save_dir, "best_map50_EVENT.pkl")
        best_map50_emotion_path = os.path.join(self.config.best_map50_save_dir, "best_map50_EMOTION.pkl")
        best_map15_event_path = os.path.join(self.config.best_map15_save_dir, "best_map15_EVENT.pkl")
        best_map15_emotion_path = os.path.join(self.config.best_map15_save_dir, "best_map15_EMOTION.pkl")
        best_map50_nature_path = os.path.join(self.config.best_map50_save_dir, "best_map50_NATURE.pkl")
        best_map15_nature_path = os.path.join(self.config.best_map15_save_dir, "best_map15_NATURE.pkl")
        
        return best_map50_event_path, best_map50_emotion_path, best_map15_event_path, best_map15_emotion_path, best_map50_nature_path, best_map15_nature_path 

    @torch.no_grad()
    def evaluate_highlight(self, dataloader=None, training_type='emotion', use_CovLoss=False):
        """
        Evaluate highlight detection performance.

        Args:
            dataloader: DataLoader for evaluation
            training_type: Task type ('emotion', 'event', or 'nature')
            use_CovLoss: Whether to compute CovLoss

        Returns:
            final_map50, final_map15, mean_loss
        """
        self.model.eval()

        map50_history = []
        map15_history = []
        loss_history = []
        CovLoss_history = []
        indiv_loss_history = []

        for batch_idx, data in enumerate(tqdm(iter(dataloader), desc=f"eval_map for '{training_type}'", leave=False)):
            feats, gt, mask = self._prep_batch(data)

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

            # Evaluate per sample
            pred_np = pred.detach().float().cpu().numpy()
            gt_np = gt.detach().float().cpu().numpy()

            B = pred_np.shape[0]
            for b in range(B):
                y_score = pred_np[b]
                y_gt = gt_np[b]

                if len(y_score) == 0:
                    continue

                # Create binary labels from top-k
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
        """
        Test function.

        Args:
            ckpt_path: Path to checkpoint
            training_type: Task type

        Returns:
            test_map50, test_map15
        """
        if ckpt_path is not None:
            print("Testing Model: ", ckpt_path)
            print("Device: ", self.config.device)
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.config.device), strict=True)

        # Choose appropriate test loader
        if training_type == 'emotion':
            test_loader = self.emotion_test_loader
        elif training_type == 'event':
            test_loader = self.event_test_loader
        else:  # Nature
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
            load_ckpt_safely(self.model, ckpt_path, self.config.device)

        self.model.eval()
        results = []

        with torch.no_grad():
            for data in dataloader:
                feats, _, mask = self._prep_batch(data)

                score, _ = self.model(feats, mask, training_type)
                
                # Get scores for each sample in batch
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