import math
import unittest
import numpy as np
import torch

from supplier_dfl.data import generate_contextual_sourcing_data
from supplier_dfl.model import CostPredictor, PerturbAndResolveLoss, SignedIdentityLoss
from supplier_dfl.planning import SupplierSelectionMILP
from supplier_dfl.training import train_and_compare


class SupplierSelectionDFLTests(unittest.TestCase):
    def test_milp_matches_independent_active_set_oracle(self):
        planner = SupplierSelectionMILP()
        data = generate_contextual_sourcing_data(1, seed=11)
        milp = planner.solve(data.objective_costs[0])
        oracle = planner.brute_force_active_set_oracle(data.objective_costs[0])
        self.assertAlmostEqual(milp.objective, oracle.objective, places=6)

    def test_postsolve_feasibility(self):
        planner = SupplierSelectionMILP()
        data = generate_contextual_sourcing_data(5, seed=12)
        for c in data.objective_costs:
            d = planner.solve(c)
            self.assertLessEqual(d.max_violation, 1e-7)
            self.assertIn(int(d.active.sum()), (2, 3))

    def test_data_generation_reproducible(self):
        a = generate_contextual_sourcing_data(10, seed=13)
        b = generate_contextual_sourcing_data(10, seed=13)
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_equal(a.objective_costs, b.objective_costs)

    def test_predictor_shape_and_gradient(self):
        data = generate_contextual_sourcing_data(12, seed=14)
        model = CostPredictor(10, data.objective_costs.mean(0), data.objective_costs.std(0)+1e-3)
        x = torch.tensor(data.features)
        out = model(x)
        self.assertEqual(tuple(out.shape), (12, 16))
        out.mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_signed_identity_gradient_is_negative_true_cost(self):
        planner = SupplierSelectionMILP()
        data = generate_contextual_sourcing_data(2, seed=15)
        pred = torch.tensor(data.objective_costs, requires_grad=True)
        true = torch.tensor(data.objective_costs + 1.0)
        loss = SignedIdentityLoss(planner)(pred, true)
        loss.backward()
        expected = -true / len(true)
        torch.testing.assert_close(pred.grad, expected)

    def test_perturb_resolve_gradient_matches_manual_second_solve(self):
        planner = SupplierSelectionMILP()
        data = generate_contextual_sourcing_data(2, seed=16)
        pred_np = data.objective_costs.copy()
        true_np = generate_contextual_sourcing_data(2, seed=17).objective_costs
        pred = torch.tensor(pred_np, requires_grad=True)
        true = torch.tensor(true_np)
        lam = 0.05
        loss = PerturbAndResolveLoss(planner, perturbation=lam)(pred, true)
        loss.backward()

        w, _ = planner.solve_many(pred_np)
        wq, _ = planner.solve_many(pred_np + lam * true_np)
        expected = torch.tensor((wq - w) / lam / len(pred_np), dtype=pred.dtype)
        torch.testing.assert_close(pred.grad, expected, atol=2e-5, rtol=2e-5)

    def test_true_cost_decision_has_zero_regret(self):
        planner = SupplierSelectionMILP()
        data = generate_contextual_sourcing_data(3, seed=18)
        for c in data.objective_costs:
            d = planner.solve(c)
            regret = float(c @ d.vector) - d.objective
            self.assertLessEqual(abs(regret), 1e-7)

    def test_short_end_to_end_training_smoke(self):
        result = train_and_compare(
            seed=19,
            train_samples=32,
            validation_samples=12,
            test_samples=12,
            warmstart_epochs=1,
            finetune_epochs=1,
            batch_size=8,
        )
        for m in (result.mse, result.signed_identity, result.perturb_resolve):
            self.assertTrue(math.isfinite(m.mean_regret))
            self.assertLessEqual(m.max_feasibility_violation, 1e-6)
        self.assertGreater(result.perturb_resolve.training_solver_calls, result.signed_identity.training_solver_calls)


if __name__ == "__main__":
    unittest.main()
