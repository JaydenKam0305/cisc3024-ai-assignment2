"""Correctness checks for the probability model and evaluation pipeline."""
import unittest
import numpy as np
import torch
from experiment import Classifier, metrics

class Checks(unittest.TestCase):
    def test_kl_identical_gaussians(self):
        m = Classifier(True)
        with torch.no_grad():
            m.head.weight.zero_(); m.head.bias.zero_(); m.log_var.zero_()
        self.assertAlmostEqual(m.kl().item(), 0., places=6)

    def test_logit_moments_against_sampled_weights(self):
        torch.manual_seed(12)
        m = Classifier(True)
        x = torch.rand(3, 64)
        h = m.features(x).detach()
        a = torch.cat([h, torch.ones(3, 1)], dim=1)
        mu, var = m.moments(x)
        mean = torch.cat([m.head.weight, m.head.bias[:, None]], dim=1).detach()
        weights = mean + m.log_var.detach().exp().sqrt() * torch.randn(50000, 10, 33)
        logits = torch.einsum('scd,nd->snc', weights, a)
        torch.testing.assert_close(logits.mean(0), mu.detach(), atol=.003, rtol=.02)
        torch.testing.assert_close(logits.var(0), var.detach(), atol=.0001, rtol=.04)

    def test_jensen_bound_and_gradients(self):
        torch.manual_seed(13)
        m = Classifier(True)
        x, y = torch.rand(8, 64), torch.arange(8)
        mu, var = m.moments(x)
        deterministic = (torch.logsumexp(mu + .5 * var, -1) - mu[torch.arange(8), y]).mean()
        z = mu[None] + torch.randn(20000, 8, 10) * var.sqrt()[None]
        mc_nll = (torch.logsumexp(z, -1) - z[:, torch.arange(8), y]).mean()
        self.assertGreaterEqual(deterministic.item() + 0.001, mc_nll.item())
        loss = m.objective(x, y, 100)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_probabilities_and_perfect_predictions(self):
        for b in [False, True]:
            p = Classifier(b).probabilities(torch.rand(4,64))
            self.assertTrue(np.allclose(p.sum(1), 1, atol=1e-6))
        result = metrics(np.eye(10), np.arange(10))
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['nll'], 0)
        self.assertEqual(result['brier'], 0)
        self.assertEqual(result['ece'], 0)

if __name__ == '__main__':
    unittest.main()
