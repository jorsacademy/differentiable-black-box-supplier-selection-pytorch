from __future__ import annotations
import argparse
import numpy as np
import torch
from supplier_dfl import (
    PerturbAndResolveLoss,
    SupplierSelectionMILP,
    generate_contextual_sourcing_data,
    train_and_compare,
)


def self_test():
    planner = SupplierSelectionMILP()
    data = generate_contextual_sourcing_data(3, seed=7)
    d = planner.solve(data.objective_costs[0])
    oracle = planner.brute_force_active_set_oracle(data.objective_costs[0])
    assert abs(d.objective - oracle.objective) <= 1e-6
    assert d.max_violation <= 1e-7

    pred = torch.tensor(data.objective_costs[:2], dtype=torch.float32, requires_grad=True)
    true = torch.tensor(data.objective_costs[1:3], dtype=torch.float32)
    layer = PerturbAndResolveLoss(planner, perturbation=0.05)
    loss = layer(pred, true)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    print("Differentiable black-box supplier-selection self-test: OK")


def print_result(result):
    print("=" * 108)
    print("DIFFERENTIABLE BLACK-BOX SUPPLIER SELECTION — MSE vs SIGNED IDENTITY vs PERTURB-AND-RESOLVE")
    print("=" * 108)
    for m in (result.mse, result.signed_identity, result.perturb_resolve):
        print(
            f"{m.name:<20} RMSE={m.cost_rmse:8.3f} mean_regret={m.mean_regret:9.3f} "
            f"p90={m.p90_regret:9.3f} relative={m.mean_relative_regret_pct:7.3f}% "
            f"train_solves={m.training_solver_calls:5d} time={m.finetune_seconds:6.2f}s"
        )
    a = result.paired_pr_minus_mse
    b = result.paired_pr_minus_signed
    print()
    print(f"perturb-resolve - MSE regret    : {a[0]:.3f} [95% CI {a[1]:.3f}, {a[2]:.3f}]")
    print(f"perturb-resolve - signed regret : {b[0]:.3f} [95% CI {b[1]:.3f}, {b[2]:.3f}]")
    print("Negative paired differences favor perturb-and-resolve.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=160)
    parser.add_argument("--validation-samples", type=int, default=48)
    parser.add_argument("--test-samples", type=int, default=80)
    parser.add_argument("--warmstart-epochs", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--perturbation", type=float, default=0.05)
    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        print_result(train_and_compare(
            seed=args.seed,
            train_samples=args.train_samples,
            validation_samples=args.validation_samples,
            test_samples=args.test_samples,
            warmstart_epochs=args.warmstart_epochs,
            finetune_epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            perturbation=args.perturbation,
        ))
