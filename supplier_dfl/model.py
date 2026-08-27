from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from .planning import SupplierSelectionMILP


class CostPredictor(nn.Module):
    def __init__(self, feature_dim: int, center: np.ndarray, scale: np.ndarray):
        super().__init__()
        output_dim = len(center)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 10),
            nn.Tanh(),
            nn.Linear(10, output_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.center + self.scale * self.network(features)


class _SignedIdentityDecisionLoss(Function):
    @staticmethod
    def forward(ctx, predicted_costs, true_costs, planner):
        pred_np = predicted_costs.detach().cpu().numpy()
        decisions, _ = planner.solve_many(pred_np)
        w = torch.as_tensor(decisions, dtype=predicted_costs.dtype, device=predicted_costs.device)
        ctx.save_for_backward(true_costs)
        return (true_costs * w).sum(dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        (true_costs,) = ctx.saved_tensors
        # Minimization: signed-identity surrogate J ≈ -I.
        grad = -true_costs * grad_output[:, None]
        return grad, None, None


class SignedIdentityLoss(nn.Module):
    def __init__(self, planner: SupplierSelectionMILP):
        super().__init__()
        self.planner = planner

    def forward(self, predicted_costs: torch.Tensor, true_costs: torch.Tensor) -> torch.Tensor:
        return _SignedIdentityDecisionLoss.apply(predicted_costs, true_costs, self.planner).mean()


class _PerturbResolveDecisionLoss(Function):
    @staticmethod
    def forward(ctx, predicted_costs, true_costs, planner, perturbation):
        pred_np = predicted_costs.detach().cpu().numpy()
        decisions, _ = planner.solve_many(pred_np)
        w = torch.as_tensor(decisions, dtype=predicted_costs.dtype, device=predicted_costs.device)
        ctx.save_for_backward(predicted_costs.detach(), true_costs.detach(), w)
        ctx.planner = planner
        ctx.perturbation = float(perturbation)
        return (true_costs * w).sum(dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        predicted, true_costs, w = ctx.saved_tensors
        lam = ctx.perturbation

        # The task gradient dL/dw is the true objective vector. Re-solve after
        # perturbing predicted costs in that direction.
        q = predicted.cpu().numpy() + lam * true_costs.cpu().numpy()
        perturbed, _ = ctx.planner.solve_many(q)
        wq = torch.as_tensor(perturbed, dtype=predicted.dtype, device=predicted.device)

        # Interpolation-style black-box gradient for minimization.
        grad_estimate = (wq - w) / lam
        return grad_estimate * grad_output[:, None], None, None, None


class PerturbAndResolveLoss(nn.Module):
    def __init__(self, planner: SupplierSelectionMILP, perturbation: float = 0.05):
        super().__init__()
        if perturbation <= 0:
            raise ValueError("perturbation must be positive")
        self.planner = planner
        self.perturbation = float(perturbation)

    def forward(self, predicted_costs: torch.Tensor, true_costs: torch.Tensor) -> torch.Tensor:
        return _PerturbResolveDecisionLoss.apply(
            predicted_costs,
            true_costs,
            self.planner,
            self.perturbation,
        ).mean()
