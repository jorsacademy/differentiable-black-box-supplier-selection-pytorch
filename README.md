# Differentiable Black-Box Supplier Selection in PyTorch

A from-scratch ML × OR project for contextual supplier selection. A neural network predicts the objective coefficients of a mixed-integer sourcing model, then discrete supplier-selection decisions are trained through with two black-box gradient estimators.

This repository does **not** copy PyEPO source code, package structure, APIs, datasets, or notebooks. PyEPO is cited only as related work. The optimization model, data generator, autograd layers, training loop, oracles, tests, and benchmarking code are independently implemented for this supplier-selection case study.

## Decision pipeline

```text
market / supplier context
        ↓
PyTorch cost predictor
        ↓
predicted sourcing coefficients
        ↓
exact supplier-selection MILP
        ↓
allocation + supplier activation
        ↓
realized true sourcing objective
        ↓
decision regret
```

The project compares three predictors with the same neural architecture and the same MSE warm start:

```text
standard MSE
signed-identity black-box gradient
perturb-and-resolve black-box gradient
```

## Supplier-selection MILP

There are four suppliers and three products.

Decision variables:

```text
x[s,p] >= 0     allocation from supplier s to product p
y[s] in {0,1}   whether supplier s is activated
```

The model enforces:

- exact product-demand satisfaction;
- total supplier capacities;
- supplier-product allocation ceilings;
- minimum confirmed throughput for an activated supplier;
- linking constraints between `x` and `y`;
- diversification: exactly two or three suppliers may be active.

The feasible region is fixed across observations. The contextual predictor estimates all 16 objective coefficients:

```text
12 allocation costs
4 supplier activation costs
```

All downstream mixed-integer problems are solved with `scipy.optimize.milp`, using the HiGHS backend.

## Independent exact oracle

The main solver is not trusted by itself.

For regression testing, an independent oracle enumerates all feasible two- and three-supplier active sets. For each binary activation pattern it solves the remaining continuous allocation LP with `scipy.optimize.linprog`.

The MILP objective must match the best active-set LP objective to numerical tolerance.

This gives an independent exactness check for the declared four-supplier benchmark.

## Contextual data

Each observation contains ten synthetic exogenous features. Objective coefficients contain:

- supplier-specific shocks;
- product-specific nonlinear responses;
- `tanh` and sinusoidal effects;
- feature interactions;
- contextual supplier fixed-charge variation;
- moderate irreducible noise.

The data are synthetic by design. The repository is a transparent differentiable-optimization benchmark, not a calibrated procurement model.

## Method 1 — MSE

The regression baseline minimizes

```text
mean((c_hat - c)^2)
```

and never calls the MILP during fine-tuning.

This is the cheapest training method in solver calls.

## Method 2 — signed identity

The discrete optimizer has no ordinary derivative. The signed-identity estimator replaces the minimization solver Jacobian by a negative identity approximation:

```text
dw*/dc_hat ≈ -I
```

If the downstream task loss is the realized true objective,

```text
L = c_true^T w*(c_hat)
```

the surrogate backward signal becomes

```text
dL/dc_hat ≈ -c_true
```

The forward pass still solves the exact supplier-selection MILP under the predicted costs.

This estimator needs one MILP solve per training observation per epoch.

## Method 3 — perturb and resolve

The stronger black-box estimator explicitly probes how the discrete optimum changes.

Forward:

```text
w  = argmin_w c_hat^T w
```

The task gradient with respect to the chosen decision is the true cost vector:

```text
dL/dw = c_true
```

Perturb predicted costs:

```text
q = c_hat + lambda * c_true
```

and solve again:

```text
w_q = argmin_w q^T w
```

The backward approximation is

```text
dL/dc_hat ≈ (w_q - w) / lambda
```

Default benchmark:

```text
lambda = 0.03
```

The implementation is a custom `torch.autograd.Function`; the regression suite independently reconstructs the second MILP solve and checks the resulting gradient tensor.

Perturb-and-resolve requires two MILP solves per training observation per epoch.

## Decision regret

Prediction error is not the primary KPI.

For true objective `c` and the decision selected under predicted objective `c_hat`:

```text
regret(c_hat, c)
=
c^T w*(c_hat) - c^T w*(c)
```

Lower is better.

Reported metrics include:

- coefficient RMSE;
- mean / median / 90th-percentile regret;
- mean relative regret;
- maximum feasibility violation;
- fine-tuning solver-call count;
- fine-tuning wall time;
- paired Student-t confidence intervals.

## Development benchmark

Seed-42 development configuration:

```text
training samples      120
validation samples     40
test samples           60
MSE warm-start epochs  12
fine-tuning epochs      8
batch size             20
perturbation lambda    0.03
```

Observed results:

```text
MSE
  coefficient RMSE       13.133
  mean regret           678.769
  p90 regret           1427.733
  mean relative regret   10.506%
  training MILP solves        0

signed identity
  coefficient RMSE       13.847
  mean regret           743.000
  p90 regret           1519.750
  mean relative regret   11.513%
  training MILP solves      960

perturb and resolve
  coefficient RMSE       13.783
  mean regret           650.824
  p90 regret           1342.139
  mean relative regret   10.120%
  training MILP solves     1920
```

Paired regret differences:

```text
perturb-resolve - MSE
-27.946
95% CI [-66.794, 10.902]

perturb-resolve - signed identity
-92.176
95% CI [-153.301, -31.051]
```

Negative values favor perturb-and-resolve.

The first confidence interval includes zero, so this run does **not** establish statistically conclusive superiority over MSE. The second interval is below zero for this fixed synthetic development experiment, supporting a difference relative to the signed-identity estimator.

No claim is made that either estimator universally dominates ordinary prediction training.

## Why solver-call counts matter

Decision-focused learning can improve task alignment while being substantially more expensive.

For `N` training observations and `E` fine-tuning epochs:

```text
MSE                 0 solver calls during fine-tuning
signed identity     N * E solver calls
perturb-resolve     2 * N * E solver calls
```

The repository reports this computational trade-off instead of comparing regret alone.

## Regression suite

The tests cover:

- MILP vs independent active-set enumeration oracle;
- post-solve feasibility and diversification;
- deterministic data generation;
- predictor shape and gradient flow;
- exact signed-identity gradient formula;
- perturb-and-resolve gradient vs manually reconstructed second solve;
- zero self-regret under true costs;
- end-to-end three-method training smoke test.

## Run

Install:

```bash
pip install -r requirements.txt
```

Self-test:

```bash
python supplier_selection_blackbox.py --self-test
```

Tests:

```bash
python -m unittest discover -s tests -v
```

Development-sized experiment:

```bash
python supplier_selection_blackbox.py \
  --seed 42 \
  --train-samples 120 \
  --validation-samples 40 \
  --test-samples 60 \
  --warmstart-epochs 12 \
  --finetune-epochs 8 \
  --batch-size 20 \
  --perturbation 0.03
```

## Validated GitHub Actions run

GitHub Actions run `33101948992` completed successfully on Ubuntu 24.04 / CPython 3.12.14 with CPU PyTorch 2.13.0, NumPy 2.5.2, and SciPy 1.18.1. The supplier-selection self-test and all **8 regression tests** passed before the end-to-end differentiable black-box smoke experiment.

The CI smoke configuration used:

```text
training samples       48
validation samples     16
test samples           20
MSE warm-start epochs   3
fine-tuning epochs      2
batch size              12
perturbation lambda    0.03
```

Observed GitHub-runner results:

```text
MSE
  coefficient RMSE       18.055
  mean regret           1110.763
  p90 regret            1782.437
  mean relative regret    17.056%
  training MILP solves         0

signed identity
  coefficient RMSE       18.140
  mean regret           1096.338
  p90 regret            1782.437
  mean relative regret    16.829%
  training MILP solves        96

perturb and resolve
  coefficient RMSE       18.143
  mean regret           1079.026
  p90 regret            1782.437
  mean relative regret    16.612%
  training MILP solves       192
```

Paired smoke differences:

```text
perturb-resolve - MSE
-31.737
95% CI [-77.664, 14.190]

perturb-resolve - signed identity
-17.312
95% CI [-53.545, 18.922]
```

Both confidence intervals include zero. The smoke run validates the solver/autograd/training mechanics and the expected solver-call accounting; it is **not** used as a statistical superiority claim.

Run: https://github.com/jorsacademy/differentiable-black-box-supplier-selection-pytorch/actions/runs/33101948992

## Exactness and scope

The supplier-selection MILP is solved to HiGHS optimality subject to numerical tolerances. For the declared four-supplier benchmark, an independent active-set enumeration oracle cross-checks the optimum.

This exactness applies only to the downstream optimization problem.

It does **not** imply that:

- the neural predictor is globally optimal;
- the black-box gradient is an exact derivative of the discrete argmin map;
- perturb-and-resolve must outperform MSE on other datasets;
- synthetic regret differences represent procurement savings;
- the model captures all real sourcing constraints.

Production procurement would require calibrated supplier prices, lead-time distributions, quality/yield, contractual commitments, multi-period dynamics, uncertainty, risk constraints, and operational validation.

## Related work

The project is conceptually related to differentiable black-box combinatorial optimization and decision-focused learning, including:

- Vlastelica et al., **Differentiation of Blackbox Combinatorial Solvers**, ICLR 2020.
- Sahoo et al., **Backpropagation through Combinatorial Algorithms: Identity with Projection Works**, ICLR 2023.
- Tang & Khalil, **PyEPO: a PyTorch-based End-to-End Predict-then-Optimize Library for Linear and Integer Programming**, Mathematical Programming Computation, 2024.

PyEPO provides broad reusable implementations of multiple predict-then-optimize methods. This repository intentionally does not reproduce that framework; it isolates one sourcing MILP and exposes the gradient estimators, solver calls, independent oracle, and downstream regret directly.
