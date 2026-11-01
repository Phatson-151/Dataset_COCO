"""Independent numerical checks of the FFT operators and convex subproblems."""
import unittest
import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve2d
from paper_scsc.solver import (conv, adj, kernel_fft, solve_z, solve_d,
                               init_dict, make_mask, sigmoid_loss_grad,logistic_prox)


class NumericalSolverTests(unittest.TestCase):
    def test_logistic_prox_newton_cycle_regression(self):
        # Undamped Newton oscillates near +/-8; the exact root is zero.
        a=np.array([-8.,8.]);y=np.array([1.,-1.])
        np.testing.assert_allclose(logistic_prox(a,y,0.,1.,1/16),0.,atol=1e-9)

    def test_logistic_prox_against_bracketed_scalar_oracle(self):
        from scipy.optimize import brentq
        from scipy.special import expit
        for gamma in [.05,1.,10.,100.]:
            for rho in [1.,1/16,1/64]:
                a=np.linspace(-30,30,31);y=np.where(np.arange(31)%2,1.,-1.);b=np.linspace(-3,3,31)
                actual=logistic_prox(a,y,b,gamma,rho)
                expected=[brentq(lambda v:rho*(v-aa)-gamma*yy*expit(-yy*(v+bb)),
                    min(aa,aa+gamma*yy/rho)-1,max(aa,aa+gamma*yy/rho)+1,xtol=1e-12)
                    for aa,yy,bb in zip(a,y,b)]
                np.testing.assert_allclose(actual,expected,atol=1e-8)

    def test_ill_conditioned_high_K_dictionary_is_finite_and_decreases_loss(self):
        z=np.ones((2,12,8,8),'float32')*1000
        z*=np.linspace(1,1.01,12,dtype='float32')[None,:,None,None]
        x=np.full((2,8,8),.4,'float32');d=np.zeros((12,3,3),'float32')
        new,diag=solve_d(x,z,d,maxiter=2000,adaptive=True)
        self.assertTrue(np.isfinite(new).all())
        m=make_mask((8,8),3)
        self.assertLess(float(np.sum(m*(conv(kernel_fft(new,(8,8)),z)-x)**2)),float(np.sum(m*x*x))*.2)

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

    def test_high_K_supervised_objective_against_epigraph_optimizer(self):
        # Independent smooth bounded formulation z=u-v, u,v>=0 exercises the
        # low-rank multi-head Woodbury path, not just the one-filter case.
        from scipy.special import expit
        rng=np.random.default_rng(61);K=12
        d=rng.uniform(-1,1,K);w=rng.normal(size=(4,K))*.3
        b=np.array([.1,-.2,.05,-.1]);y=np.array([1,-1,1,-1]);beta=.05;gamma=.3
        def oracle(t):
            z=t[:K]-t[K:];logits=w@z+b
            value=.5*(d@z-.4)**2+beta*t.sum()+gamma*np.logaddexp(0,-y*logits).sum()
            grad=d*(d@z-.4)+gamma*w.T@(-y*expit(-y*logits))
            return value,np.r_[grad+beta,-grad+beta]
        ref=minimize(oracle,np.zeros(2*K),jac=True,method='L-BFGS-B',
            bounds=[(0,None)]*(2*K),options={'maxiter':3000,'ftol':1e-13,'gtol':1e-9})
        self.assertTrue(ref.success)
        labels=np.broadcast_to(y[None,:,None,None],(1,4,4,4)).astype('int8')
        z,diag=solve_z(d[:,None,None].astype('float32'),np.full((1,4,4),.4,'float32'),
            w.astype('float32'),b.astype('float32'),labels,gamma,beta,maxiter=3000,tol=1e-4,adaptive=True)
        zz=z[0,:,0,0]
        got=.5*(d@zz-.4)**2+beta*abs(zz).sum()+gamma*np.logaddexp(0,-y*(w@zz+b)).sum()
        self.assertTrue(diag['converged'])
        self.assertLess(abs(got-ref.fun),.003)

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
