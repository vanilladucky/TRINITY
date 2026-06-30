"""
Aligned-MTL: Independent Component Alignment for Multi-Task Learning

Based on the paper:
"Independent Component Alignment for Multi-Task Learning" (CVPR 2023)
https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html

This implementation is adapted to work with PyTorch optimizers directly,
without requiring the LibMTL framework.
"""

import torch
import torch.nn as nn


class AlignedMTL:
    """
    Aligned-MTL gradient alignment for multi-task learning.
    
    This method computes optimal task weights by aligning task gradients
    using eigenvalue decomposition of the gradient Gram matrix.
    
    Usage:
        aligned_mtl = AlignedMTL()
        
        # In training loop:
        optimizer.zero_grad()
        
        # Compute per-task losses
        loss_task1 = ...
        loss_task2 = ...
        loss_task3 = ...
        
        # Apply Aligned-MTL
        task_weights = aligned_mtl.backward(
            model=model,
            losses=[loss_task1, loss_task2, loss_task3],
            optimizer=optimizer
        )
        
        optimizer.step()
    """
    
    def __init__(self):
        pass
    
    def backward(self, model, losses, optimizer):
        """
        Compute Aligned-MTL weights and apply weighted gradients.
        
        Args:
            model: The neural network model
            losses: List of per-task loss tensors [loss_task1, loss_task2, ...]
            optimizer: PyTorch optimizer
            
        Returns:
            task_weights: numpy array of shape (num_tasks,) with computed weights
        """
        num_tasks = len(losses)
        
        # Step 1: Compute per-task gradients
        grads = []
        for i, loss in enumerate(losses):
            # Zero out any existing gradients
            optimizer.zero_grad()
            
            # Compute gradients for this task
            # retain_graph=True because we need to compute gradients for other tasks
            loss.backward(retain_graph=(i < num_tasks - 1))
            
            # Collect gradients from all parameters into a single vector
            grad_vec = self._get_gradient_vector(model)
            grads.append(grad_vec)
        
        # Stack gradients: (num_tasks, num_params)
        grads = torch.stack(grads)
        
        # Step 2: Compute Gram matrix M = G @ G^T where G is the gradient matrix
        M = torch.matmul(grads, grads.t())  # (num_tasks, num_tasks)
        
        # Step 3: Eigenvalue decomposition
        # torch.symeig is deprecated, use torch.linalg.eigh instead
        lmbda, V = torch.linalg.eigh(M)  # lmbda: (num_tasks,), V: (num_tasks, num_tasks)
        
        # Step 4: Determine numerical rank
        tol = (
            torch.max(lmbda)
            * max(M.shape[-2:])
            * torch.finfo(lmbda.dtype).eps
        )
        rank = (lmbda > tol).sum().item()
        
        # Step 5: Sort eigenvalues in descending order and truncate to rank
        order = torch.argsort(lmbda, descending=True)
        lmbda = lmbda[order][:rank]
        V = V[:, order][:, :rank]
        
        # Step 6: Compute alignment matrix B
        sigma = torch.diag(1 / lmbda.sqrt())  # (rank, rank)
        B = lmbda[-1].sqrt() * ((V @ sigma) @ V.t())  # (num_tasks, num_tasks)
        
        # Step 7: Compute task weights alpha
        alpha = B.sum(dim=0)  # (num_tasks,)
        
        # Step 8: Apply weighted gradients
        optimizer.zero_grad()
        weighted_grads = torch.zeros_like(grads[0])
        for i in range(num_tasks):
            weighted_grads += alpha[i] * grads[i]
        
        # Set the weighted gradients back to model parameters
        self._set_gradient_vector(model, weighted_grads)
        
        return alpha.detach().cpu().numpy()
    
    def _get_gradient_vector(self, model):
        """
        Extract gradients from all model parameters and concatenate into a single vector.
        
        Args:
            model: PyTorch model
            
        Returns:
            grad_vec: 1D tensor of all gradients
        """
        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.view(-1))
            else:
                # If no gradient, append zeros
                grad_list.append(torch.zeros_like(param).view(-1))
        
        return torch.cat(grad_list)
    
    def _set_gradient_vector(self, model, grad_vec):
        """
        Set gradients of all model parameters from a single concatenated vector.
        
        Args:
            model: PyTorch model
            grad_vec: 1D tensor of all gradients
        """
        pointer = 0
        for param in model.parameters():
            num_param = param.numel()
            param.grad = grad_vec[pointer:pointer + num_param].view_as(param).clone()
            pointer += num_param


class AlignedMTLOptimizer:
    """
    Wrapper optimizer that applies Aligned-MTL gradient alignment.
    
    This is a convenience wrapper that makes Aligned-MTL easier to use
    by wrapping a standard PyTorch optimizer.
    
    Usage:
        base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer = AlignedMTLOptimizer(base_optimizer, model)
        
        # In training loop:
        optimizer.zero_grad()
        
        loss_task1 = ...
        loss_task2 = ...
        loss_task3 = ...
        
        task_weights = optimizer.backward([loss_task1, loss_task2, loss_task3])
        optimizer.step()
    """
    
    def __init__(self, base_optimizer, model):
        """
        Args:
            base_optimizer: PyTorch optimizer (e.g., Adam, SGD)
            model: The neural network model
        """
        self.base_optimizer = base_optimizer
        self.model = model
        self.aligned_mtl = AlignedMTL()
    
    def zero_grad(self):
        """Zero gradients."""
        self.base_optimizer.zero_grad()
    
    def backward(self, losses):
        """
        Compute Aligned-MTL weights and apply weighted gradients.
        
        Args:
            losses: List of per-task loss tensors
            
        Returns:
            task_weights: numpy array of computed weights
        """
        return self.aligned_mtl.backward(self.model, losses, self.base_optimizer)
    
    def step(self):
        """Perform optimization step."""
        self.base_optimizer.step()
    
    def state_dict(self):
        """Return optimizer state dict."""
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.base_optimizer.load_state_dict(state_dict)
    
    @property
    def param_groups(self):
        """Access parameter groups."""
        return self.base_optimizer.param_groups
