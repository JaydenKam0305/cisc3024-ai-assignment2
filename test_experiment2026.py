"""Independent algebraic and differential checks for the 2026 implementation."""
import unittest
import torch
from experiment import Classifier
from experiment2026 import features, hidden_jacobian, correction, posterior, predictive

class RichBLLChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.h=torch.randn(19,5,dtype=torch.float64)
        self.j=torch.randn(19,11,dtype=torch.float64)
        self.g=correction(self.h,self.j)

    def test_projection_and_positive_correction(self):
        c=torch.linalg.solve(self.h.T@self.h+torch.eye(5,dtype=torch.float64),self.h.T@self.j)
        self.assertLess(torch.linalg.norm(self.g-(torch.eye(5)+c@c.T)).item(),1e-12)
        self.assertGreater(torch.linalg.eigvalsh(self.g-torch.eye(5)).min().item(),-1e-10)

    def test_kernel_and_transformed_feature_equivalence(self):
        x=torch.randn(5,dtype=torch.float64); n=len(self.h); sig=.1
        p=posterior(self.h,self.g)
        k=self.h@self.g@self.h.T; kx=self.h@self.g@x
        kernel=x@self.g@x-kx@torch.linalg.solve(k+sig*torch.eye(n,dtype=torch.float64),kx)
        l=torch.linalg.cholesky(self.g); psi=self.h@l; psix=x@l
        transformed=psix@torch.linalg.solve(torch.eye(5,dtype=torch.float64)+psi.T@psi/sig,psix)
        self.assertAlmostEqual((x@p@x).item(),kernel.item(),places=9)
        self.assertAlmostEqual(transformed.item(),kernel.item(),places=9)

    def test_full_rich_covariance_dominates_bll(self):
        diff=posterior(self.h,self.g)-posterior(self.h,torch.eye(5,dtype=torch.float64))
        self.assertGreater(torch.linalg.eigvalsh(diff).min().item(),-1e-10)

    def test_subsample_population_scaling(self):
        # Repeating each row four times is equivalent to an N/k=4 Gram scaling.
        a=posterior(self.h,self.g,population_size=4*len(self.h))
        b=posterior(self.h.repeat(4,1),self.g)
        self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=1e-10))

    def test_jacobian_directional_finite_difference(self):
        m=Classifier(False).double(); x=torch.randn(3,64,dtype=torch.float64)
        j=hidden_jacobian(m,x,2); params=list(m.features.parameters())
        direction=torch.randn(sum(p.numel() for p in params),dtype=torch.float64); direction/=direction.norm()
        eps=1e-5; offset=0
        with torch.no_grad():
            for p in params:
                q=direction[offset:offset+p.numel()].reshape_as(p); p.add_(eps*q); offset+=p.numel()
            plus=m.moments(x)[0][:,2].clone(); offset=0
            for p in params:
                q=direction[offset:offset+p.numel()].reshape_as(p); p.add_(-2*eps*q); offset+=p.numel()
            minus=m.moments(x)[0][:,2].clone()
        self.assertTrue(torch.allclose(j@direction,(plus-minus)/(2*eps),atol=1e-8,rtol=1e-5))

    def test_predictive_integral_normalization(self):
        mu=torch.randn(7,10,dtype=torch.float64); var=torch.rand_like(mu)
        p=predictive(mu,var,samples=256)
        self.assertTrue((p>=0).all())
        self.assertTrue(torch.allclose(torch.tensor(p.sum(1)),torch.ones(7,dtype=torch.float64),atol=1e-12))

if __name__=='__main__':unittest.main()
