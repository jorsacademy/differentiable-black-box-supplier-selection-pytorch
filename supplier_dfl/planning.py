from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class SupplierSelectionSpec:
    demand: np.ndarray
    supplier_capacity: np.ndarray
    minimum_throughput: np.ndarray
    product_capacity: np.ndarray

    def __post_init__(self):
        demand = np.asarray(self.demand, dtype=float)
        cap = np.asarray(self.supplier_capacity, dtype=float)
        minload = np.asarray(self.minimum_throughput, dtype=float)
        pcap = np.asarray(self.product_capacity, dtype=float)
        if demand.ndim != 1 or cap.ndim != 1 or minload.ndim != 1:
            raise ValueError("demand/capacity vectors must be one-dimensional")
        if pcap.shape != (len(cap), len(demand)):
            raise ValueError("product_capacity must be [supplier, product]")
        if np.any(demand <= 0) or np.any(cap <= 0) or np.any(minload < 0) or np.any(pcap <= 0):
            raise ValueError("all physical parameters must be positive/nonnegative")
        if np.any(minload > cap):
            raise ValueError("minimum throughput cannot exceed supplier capacity")


@dataclass(frozen=True)
class SourcingDecision:
    vector: np.ndarray
    objective: float
    max_violation: float

    @property
    def allocations(self):
        return self.vector[:-4].reshape(4, 3)

    @property
    def active(self):
        return np.rint(self.vector[-4:]).astype(int)


def default_spec() -> SupplierSelectionSpec:
    return SupplierSelectionSpec(
        demand=np.array([82.0, 71.0, 63.0]),
        supplier_capacity=np.array([112.0, 98.0, 88.0, 76.0]),
        minimum_throughput=np.array([28.0, 24.0, 20.0, 16.0]),
        product_capacity=np.array([
            [62.0, 55.0, 48.0],
            [54.0, 58.0, 50.0],
            [47.0, 50.0, 56.0],
            [45.0, 44.0, 46.0],
        ]),
    )


class SupplierSelectionMILP:
    """Small exact sourcing MILP with a fixed feasible region."""

    def __init__(self, spec: SupplierSelectionSpec | None = None):
        self.spec = spec or default_spec()
        self.n_suppliers = len(self.spec.supplier_capacity)
        self.n_products = len(self.spec.demand)
        if (self.n_suppliers, self.n_products) != (4, 3):
            raise ValueError("benchmark implementation expects 4 suppliers and 3 products")
        self.n_alloc = self.n_suppliers * self.n_products
        self.n_vars = self.n_alloc + self.n_suppliers
        self.solve_calls = 0

        rows, lower, upper = [], [], []

        # Exact demand satisfaction.
        for p in range(self.n_products):
            row = np.zeros(self.n_vars)
            for s in range(self.n_suppliers):
                row[self._x(s, p)] = 1.0
            rows.append(row); lower.append(self.spec.demand[p]); upper.append(self.spec.demand[p])

        # Supplier total capacity and minimum confirmed throughput when active.
        for s in range(self.n_suppliers):
            row = np.zeros(self.n_vars)
            for p in range(self.n_products):
                row[self._x(s, p)] = 1.0
            row[self._y(s)] = -self.spec.supplier_capacity[s]
            rows.append(row); lower.append(-np.inf); upper.append(0.0)

            row = np.zeros(self.n_vars)
            for p in range(self.n_products):
                row[self._x(s, p)] = -1.0
            row[self._y(s)] = self.spec.minimum_throughput[s]
            rows.append(row); lower.append(-np.inf); upper.append(0.0)

        # Supplier-product allocation ceilings.
        for s in range(self.n_suppliers):
            for p in range(self.n_products):
                row = np.zeros(self.n_vars)
                row[self._x(s, p)] = 1.0
                row[self._y(s)] = -self.spec.product_capacity[s, p]
                rows.append(row); lower.append(-np.inf); upper.append(0.0)

        # Diversification: activate 2 or 3 suppliers.
        row = np.zeros(self.n_vars)
        row[-self.n_suppliers:] = 1.0
        rows.append(row); lower.append(2.0); upper.append(3.0)

        self.A = np.vstack(rows)
        self.lb = np.asarray(lower, dtype=float)
        self.ub = np.asarray(upper, dtype=float)
        self.constraint = LinearConstraint(self.A, self.lb, self.ub)
        self.bounds = Bounds(np.zeros(self.n_vars), np.r_[np.full(self.n_alloc, np.inf), np.ones(self.n_suppliers)])
        self.integrality = np.r_[np.zeros(self.n_alloc, dtype=int), np.ones(self.n_suppliers, dtype=int)]

    def _x(self, supplier: int, product: int) -> int:
        return supplier * self.n_products + product

    def _y(self, supplier: int) -> int:
        return self.n_alloc + supplier

    def solve(self, objective: np.ndarray) -> SourcingDecision:
        c = np.asarray(objective, dtype=float)
        if c.shape != (self.n_vars,):
            raise ValueError(f"objective must have shape ({self.n_vars},)")
        self.solve_calls += 1
        result = milp(
            c=c,
            integrality=self.integrality,
            bounds=self.bounds,
            constraints=self.constraint,
            options={"time_limit": 20.0},
        )
        if result.x is None or result.status != 0:
            raise RuntimeError(f"HiGHS MILP failed: status={result.status}, message={result.message}")
        vector = np.asarray(result.x, dtype=float)
        return SourcingDecision(
            vector=vector,
            objective=float(c @ vector),
            max_violation=self.max_constraint_violation(vector),
        )

    def solve_many(self, objectives: np.ndarray):
        objectives = np.asarray(objectives, dtype=float)
        decisions, values = [], []
        for c in objectives:
            d = self.solve(c)
            decisions.append(d.vector)
            values.append(d.objective)
        return np.asarray(decisions), np.asarray(values)

    def max_constraint_violation(self, vector: np.ndarray) -> float:
        z = np.asarray(vector, dtype=float)
        lhs = self.A @ z
        violations = [
            np.max(np.maximum(self.lb - lhs, 0.0)),
            np.max(np.maximum(lhs - self.ub, 0.0)),
            np.max(np.maximum(-z, 0.0)),
            np.max(np.maximum(z[-self.n_suppliers:] - 1.0, 0.0)),
            np.max(np.abs(z[-self.n_suppliers:] - np.rint(z[-self.n_suppliers:]))),
        ]
        return float(max(violations))

    def brute_force_active_set_oracle(self, objective: np.ndarray) -> SourcingDecision:
        """Independent oracle: enumerate all 2/3-supplier active sets and solve the continuous LP."""
        from itertools import combinations
        from scipy.optimize import linprog

        c = np.asarray(objective, dtype=float)
        best = None
        for k in (2, 3):
            for active in combinations(range(self.n_suppliers), k):
                active = set(active)
                # Allocation-only LP with active-set capacity constraints.
                A_eq = np.zeros((self.n_products, self.n_alloc))
                for p in range(self.n_products):
                    for s in range(self.n_suppliers):
                        A_eq[p, self._x(s, p)] = 1.0
                b_eq = self.spec.demand.copy()

                A_ub, b_ub = [], []
                bounds = []
                for s in range(self.n_suppliers):
                    for p in range(self.n_products):
                        upper = self.spec.product_capacity[s, p] if s in active else 0.0
                        bounds.append((0.0, float(upper)))
                for s in range(self.n_suppliers):
                    row = np.zeros(self.n_alloc)
                    row[s*self.n_products:(s+1)*self.n_products] = 1.0
                    A_ub.append(row); b_ub.append(self.spec.supplier_capacity[s])
                    row2 = -row
                    A_ub.append(row2); b_ub.append(-self.spec.minimum_throughput[s] if s in active else 0.0)

                res = linprog(
                    c=c[:self.n_alloc],
                    A_ub=np.asarray(A_ub),
                    b_ub=np.asarray(b_ub),
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=bounds,
                    method="highs",
                )
                if not res.success:
                    continue
                y = np.array([1.0 if s in active else 0.0 for s in range(self.n_suppliers)])
                vector = np.r_[res.x, y]
                obj = float(c @ vector)
                if best is None or obj < best.objective - 1e-9:
                    best = SourcingDecision(vector, obj, self.max_constraint_violation(vector))
        if best is None:
            raise RuntimeError("active-set oracle found no feasible solution")
        return best
