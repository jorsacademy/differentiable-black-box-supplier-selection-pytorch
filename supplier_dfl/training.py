from __future__ import annotations
import copy, math, time
from dataclasses import dataclass
import numpy as np
import torch
from scipy.stats import t as student_t
from .data import generate_contextual_sourcing_data
from .model import CostPredictor, SignedIdentityLoss, PerturbAndResolveLoss
from .planning import SupplierSelectionMILP


@dataclass(frozen=True)
class MethodMetrics:
    name: str
    cost_rmse: float
    mean_regret: float
    median_regret: float
    p90_regret: float
    mean_relative_regret_pct: float
    max_feasibility_violation: float
    training_solver_calls: int
    finetune_seconds: float


@dataclass(frozen=True)
class ExperimentResult:
    mse: MethodMetrics
    signed_identity: MethodMetrics
    perturb_resolve: MethodMetrics
    paired_pr_minus_mse: tuple[float, float, float]
    paired_pr_minus_signed: tuple[float, float, float]


def _batches(n, batch_size, rng):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield idx[start:start+batch_size]


def _paired_ci(d):
    d = np.asarray(d, dtype=float)
    mean = float(d.mean())
    half = float(student_t.ppf(0.975, len(d)-1) * d.std(ddof=1) / math.sqrt(len(d)))
    return mean, mean-half, mean+half


def evaluate_model(name, model, features, true_costs, true_optima, planner, solver_calls=0, seconds=0.0):
    model.eval()
    with torch.no_grad():
        predicted = model(torch.as_tensor(features, dtype=torch.float32)).cpu().numpy()
    rmse = float(np.sqrt(np.mean((predicted - true_costs)**2)))
    regrets, relative = [], []
    max_v = 0.0
    for cp, ct, optimum in zip(predicted, true_costs, true_optima):
        d = planner.solve(cp)
        realized = float(ct @ d.vector)
        regret = max(realized - float(optimum), 0.0)
        regrets.append(regret)
        relative.append(100.0 * regret / max(abs(float(optimum)), 1e-9))
        max_v = max(max_v, d.max_violation)
    r = np.asarray(regrets)
    return MethodMetrics(
        name=name,
        cost_rmse=rmse,
        mean_regret=float(r.mean()),
        median_regret=float(np.median(r)),
        p90_regret=float(np.quantile(r, 0.9)),
        mean_relative_regret_pct=float(np.mean(relative)),
        max_feasibility_violation=max_v,
        training_solver_calls=int(solver_calls),
        finetune_seconds=float(seconds),
    ), r


def _mse_epoch(model, optimizer, x, c, batch_size, rng):
    model.train()
    for idx in _batches(len(x), batch_size, rng):
        pred = model(torch.as_tensor(x[idx], dtype=torch.float32))
        true = torch.as_tensor(c[idx], dtype=torch.float32)
        loss = ((pred - true)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def _decision_epoch(model, optimizer, loss_module, x, c, batch_size, rng):
    model.train()
    for idx in _batches(len(x), batch_size, rng):
        pred = model(torch.as_tensor(x[idx], dtype=torch.float32))
        true = torch.as_tensor(c[idx], dtype=torch.float32)
        loss = loss_module(pred, true)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def train_and_compare(
    *,
    seed=42,
    train_samples=160,
    validation_samples=48,
    test_samples=80,
    warmstart_epochs=6,
    finetune_epochs=6,
    batch_size=16,
    perturbation=0.05,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    planner = SupplierSelectionMILP()

    data = generate_contextual_sourcing_data(
        train_samples + validation_samples + test_samples,
        seed=seed,
    )
    i1, i2 = train_samples, train_samples + validation_samples
    xtr, xv, xt = data.features[:i1], data.features[i1:i2], data.features[i2:]
    ctr, cv, ct = data.objective_costs[:i1], data.objective_costs[i1:i2], data.objective_costs[i2:]
    _, zv = planner.solve_many(cv)
    _, zt = planner.solve_many(ct)

    center, scale = ctr.mean(0), ctr.std(0) + 1e-3
    common = CostPredictor(xtr.shape[1], center, scale)
    opt = torch.optim.Adam(common.parameters(), lr=2e-3)
    for _ in range(warmstart_epochs):
        _mse_epoch(common, opt, xtr, ctr, batch_size, rng)

    mse = copy.deepcopy(common)
    signed = copy.deepcopy(common)
    perturb = copy.deepcopy(common)

    mse_opt = torch.optim.Adam(mse.parameters(), lr=8e-4)
    signed_opt = torch.optim.Adam(signed.parameters(), lr=1e-4)
    perturb_opt = torch.optim.Adam(perturb.parameters(), lr=8e-4)
    signed_loss = SignedIdentityLoss(planner)
    perturb_loss = PerturbAndResolveLoss(planner, perturbation=perturbation)

    best_mse, best_signed, best_perturb = map(lambda m: copy.deepcopy(m.state_dict()), (mse, signed, perturb))
    best_mse_rmse = best_signed_regret = best_perturb_regret = float("inf")
    call_counts = {"mse": 0, "signed": 0, "perturb": 0}
    times = {"mse": 0.0, "signed": 0.0, "perturb": 0.0}

    for epoch in range(1, finetune_epochs + 1):
        before = planner.solve_calls
        t0 = time.perf_counter()
        _mse_epoch(mse, mse_opt, xtr, ctr, batch_size, rng)
        times["mse"] += time.perf_counter() - t0
        call_counts["mse"] += planner.solve_calls - before

        before = planner.solve_calls
        t0 = time.perf_counter()
        _decision_epoch(signed, signed_opt, signed_loss, xtr, ctr, batch_size, rng)
        times["signed"] += time.perf_counter() - t0
        call_counts["signed"] += planner.solve_calls - before

        before = planner.solve_calls
        t0 = time.perf_counter()
        _decision_epoch(perturb, perturb_opt, perturb_loss, xtr, ctr, batch_size, rng)
        times["perturb"] += time.perf_counter() - t0
        call_counts["perturb"] += planner.solve_calls - before

        mse_val, _ = evaluate_model("mse", mse, xv, cv, zv, planner)
        signed_val, _ = evaluate_model("signed_identity", signed, xv, cv, zv, planner)
        perturb_val, _ = evaluate_model("perturb_resolve", perturb, xv, cv, zv, planner)

        if mse_val.cost_rmse < best_mse_rmse:
            best_mse_rmse = mse_val.cost_rmse
            best_mse = copy.deepcopy(mse.state_dict())
        if signed_val.mean_regret < best_signed_regret:
            best_signed_regret = signed_val.mean_regret
            best_signed = copy.deepcopy(signed.state_dict())
        if perturb_val.mean_regret < best_perturb_regret:
            best_perturb_regret = perturb_val.mean_regret
            best_perturb = copy.deepcopy(perturb.state_dict())

        print(
            f"epoch={epoch:02d} mse_regret={mse_val.mean_regret:.2f} "
            f"signed_regret={signed_val.mean_regret:.2f} "
            f"perturb_regret={perturb_val.mean_regret:.2f}"
        )

    mse.load_state_dict(best_mse)
    signed.load_state_dict(best_signed)
    perturb.load_state_dict(best_perturb)

    mse_m, mse_r = evaluate_model("mse", mse, xt, ct, zt, planner, call_counts["mse"], times["mse"])
    signed_m, signed_r = evaluate_model("signed_identity", signed, xt, ct, zt, planner, call_counts["signed"], times["signed"])
    perturb_m, perturb_r = evaluate_model("perturb_resolve", perturb, xt, ct, zt, planner, call_counts["perturb"], times["perturb"])

    return ExperimentResult(
        mse=mse_m,
        signed_identity=signed_m,
        perturb_resolve=perturb_m,
        paired_pr_minus_mse=_paired_ci(perturb_r - mse_r),
        paired_pr_minus_signed=_paired_ci(perturb_r - signed_r),
    )
