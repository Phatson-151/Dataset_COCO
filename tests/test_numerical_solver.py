"""Independent numerical checks of the FFT operators and convex subproblems."""
import unittest
import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve2d
from paper_scsc.solver import (conv, adj, kernel_fft, solve_z, solve_d,
                               init_dict, make_mask, sigmoid_loss_grad)


class NumericalSolverTests(unittest.TestCase):
    def test_linear_convolution_on_valid_support(self):
        rng = np.random.default_rng(9)
        d = rng.normal(size=(2, 5, 5)).astype('float32')
        z = rng.normal(size=(1, 2, 17, 19)).astype('float32')
        got = conv(kernel_fft(d, (17, 19)), z)[0]
        ref = sum(convolve2d(z[0, k], d[k], mode='full')[:17, :19]
                  for k in range(2))
        valid = make_mask((17, 19), 5).astype(bool)
        np.testing.assert_allclose(got[valid], ref[valid], atol=2e-5)

    def test_adjoint_identity(self):
        rng = np.random.default_rng(10)
        d = init_dict(np.zeros((1, 16, 16)), 3, 5)
        z = rng.normal(size=(2, 3, 16, 16)).astype('float32')
        x = rng.normal(size=(2, 16, 16)).astype('float32')
        df = kernel_fft(d, (16, 16))
        self.assertAlmostEqual(float(np.sum(conv(df, z)*x)),
                               float(np.sum(z*adj(df, x))), places=4)

    def test_logistic_gradient_finite_difference(self):
        rng = np.random.default_rng(8)
        z = rng.normal(size=(1, 2, 5, 5))
        w = rng.normal(size=(4, 2)); b = rng.normal(size=4)
        y = rng.choice([-1, 0, 1], size=(1, 4, 5, 5)).astype('int8')
        _, gw, gb = sigmoid_loss_grad(z, w, b, y, .3)
        eps = 1e-4
        for k in range(w.size+b.size):
            wp=w.copy();wm=w.copy();bp=b.copy();bm=b.copy()
            if k<w.size:wp.flat[k]+=eps;wm.flat[k]-=eps
            else:bp[k-w.size]+=eps;bm[k-w.size]-=eps
            fd=(sigmoid_loss_grad(z,wp,bp,y,.3)[0]-sigmoid_loss_grad(z,wm,bm,y,.3)[0])/(2*eps)
            self.assertAlmostEqual(fd, np.r_[gw.ravel(),gb][k], places=2)

    def test_supervised_scalar_against_independent_optimizer(self):
        # f=1 gives independent scalar convex problems, with a closed l1 term.
        d=np.ones((1,1,1),'float32');x=np.full((1,4,4),.4,'float32')
        w=np.array([[1.3]],'float32');b=np.array([-.1],'float32')
        y=np.ones((1,1,4,4),'int8')
        z,diag=solve_z(d,x,w,b,y,.7,.03,maxiter=2000,tol=1e-5)
        ref=minimize(lambda t:.5*(t[0]-.4)**2+.03*abs(t[0])+
                     .7*np.logaddexp(0,-(1.3*t[0]-.1)),[.4],method='BFGS')
        self.assertTrue(diag['converged'])
        np.testing.assert_allclose(z,ref.x[0],atol=1e-3)
        zu,_=solve_z(d,x,w*0,b,None,0,.03,maxiter=2000,tol=1e-5)
        self.assertGreater(float(np.mean(z-zu)),.1)

    def test_masked_pixels_cannot_influence_codes(self):
        d=init_dict(np.zeros((1,16,16)),2,3)
        rng=np.random.default_rng(1);x=rng.normal(size=(1,16,16)).astype('float32')*.1
        mask=(rng.random(x.shape)>.5).astype('float32')
        altered=x.copy();altered[mask==0]=999
        args=(np.zeros((4,2),'float32'),np.zeros(4,'float32'),None,0,.02)
        a,_=solve_z(d,x,*args,M_override=mask,maxiter=200)
        c,_=solve_z(d,altered,*args,M_override=mask,maxiter=200)
        np.testing.assert_allclose(a,c,atol=1e-6)

    def test_global_dictionary_update_decreases_data_term(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(3,16,16)).astype('float32')*.1
        d=init_dict(x,2,3)
        z,_=solve_z(d,x,np.zeros((4,2),'float32'),np.zeros(4,'float32'),None,0,.01,maxiter=1000)
        new,diag=solve_d(x,z,d,maxiter=2000,adaptive=True)
        m=make_mask((16,16),3)
        cost=lambda a:np.sum(((conv(kernel_fft(a,(16,16)),z)-x)*m)**2)
        self.assertTrue(diag['converged'])
        self.assertLessEqual(cost(new),cost(d)+1e-5)
        self.assertLessEqual(np.linalg.norm(new,axis=(1,2)).max(),1.00001)
