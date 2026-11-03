"""Masked FISTA and differentiable label-free unrolled coding.

Convolution follows solver.py's causal top-left convention. Step sizes use the
Fourier operator norm (a bound after masking). Returned convergence is the
proximal-gradient mapping criterion, never ADMM residuals in disguise.
"""
import time
import numpy as np
from scipy.fft import rfft2, irfft2
from .solver import make_mask, kernel_fft, conv, adj, soft, proj


def code(D, X, beta=.015, maxiter=160, tol=.003, mask=None, z0=None):
    started=time.perf_counter();shape=X.shape[-2:]
    F=kernel_fft(D,shape);M=make_mask(shape,D.shape[-1])[None] if mask is None else mask
    L=max(float(np.max(np.sum(abs(F)**2,axis=0))),1e-5)
    z=np.zeros((len(X),len(D),*shape),np.float32) if z0 is None else z0.copy()
    y=z.copy();t=1.;trace=[];success=False
    for it in range(maxiter):
        grad=adj(F,M*(conv(F,y)-X))
        new=soft(y-grad/L,beta/L)
        if (it+1)%10==0 or it==0 or it==maxiter-1:
            rr=M*(conv(F,new)-X);obj=float(.5*np.sum(rr**2)+beta*np.sum(abs(new)))
            # Proximal-gradient mapping at the returned feasible point.
            gn=adj(F,rr);prox=soft(new-gn/L,beta/L)
            residual=float(np.linalg.norm(new-prox));limit=tol*max(float(np.linalg.norm(new)),1.)
            trace.append(dict(iter=it+1,objective=obj,mapping_norm=residual,mapping_tolerance=limit,L=L,
                primal_relative=residual/limit,dual_relative_proxy=residual/limit))
            if residual<=limit:success=True;z=new;break
        tn=float((1+np.sqrt(1+4*t*t))/2)
        yn=new+((t-1)/tn)*(new-z)
        if np.sum((y-new)*(new-z))>0:yn=new.copy();tn=1.
        z,y,t=new,yn,tn
    return z,dict(solver='FISTA',iterations=it+1,converged=success,residuals=trace,
        maxiter=maxiter,tolerance=tol,seconds=time.perf_counter()-started)


def dictionary(D,X,Z,maxiter=80,tol=.003):
    shape=X.shape[-2:];f=D.shape[-1];M=make_mask(shape,f)[None];ZF=rfft2(Z)
    # Frobenius bound on each frequency's spectral norm; safe for any rank.
    L=max(float(np.max(np.sum(abs(ZF)**2,axis=(0,1)))),1e-5)
    d=D.copy();y=d.copy();t=1.;trace=[];success=False
    def gradient(v):
        err=M*(conv(kernel_fft(v,shape),Z)-X)
        full=irfft2(np.sum(ZF.conj()*rfft2(err)[:,None],axis=0),s=shape).real
        return full[:,:f,:f].astype(np.float32)
    def project(v):return v/np.maximum(1,np.linalg.norm(v,axis=(1,2),keepdims=True))
    initial=float(.5*np.sum((M*(conv(kernel_fft(d,shape),Z)-X))**2))
    for it in range(maxiter):
        new=project(y-gradient(y)/L)
        if it==0 or (it+1)%10==0 or it==maxiter-1:
            residual=float(np.linalg.norm(new-project(new-gradient(new)/L)))
            limit=tol*max(float(np.linalg.norm(new)),1.)
            cost=float(.5*np.sum((M*(conv(kernel_fft(new,shape),Z)-X))**2))
            trace.append(dict(iter=it+1,objective=cost,mapping_norm=residual,mapping_tolerance=limit,L=L,
                primal_relative=residual/limit,dual_relative_proxy=residual/limit))
            if residual<=limit:success=True;d=new;break
        tn=float((1+np.sqrt(1+4*t*t))/2);yn=new+(t-1)/tn*(new-d)
        if np.sum((y-new)*(new-d))>0:yn=new.copy();tn=1.
        d,y,t=new,yn,tn
    cost=float(.5*np.sum((M*(conv(kernel_fft(d,shape),Z)-X))**2));rejected=cost>initial+1e-5
    return (D.copy() if rejected else d),dict(solver='projected_FISTA',iterations=it+1,
        converged=success and not rejected,residuals=trace,initial_feasible_objective=initial,
        candidate_feasible_objective=cost,descent_guard_rejected=rejected)


def torch_code(D,X,beta,steps=24,z0=None):
    """Labels are never encoder inputs; autograd differentiates the unrolling."""
    import torch
    import torch.nn.functional as functional
    h,w=X.shape[-2:];f=D.shape[-1]
    F=torch.fft.rfft2(functional.pad(D,(0,w-f,0,h-f)))
    L=torch.clamp(torch.max(torch.sum(abs(F)**2,dim=0)).detach(),min=1e-5)
    M=torch.ones_like(X);M[:,:f-1,:]=0;M[:,:,:f-1]=0
    z=torch.zeros((len(X),len(D),h,w),dtype=X.dtype) if z0 is None else z0.detach().clone();y=z;t=1.
    for _ in range(steps):
        r=M*(torch.fft.irfft2(torch.sum(F[None]*torch.fft.rfft2(y),dim=1),s=(h,w))-X)
        grad=torch.fft.irfft2(F.conj()[None]*torch.fft.rfft2(r)[:,None],s=(h,w))
        v=y-grad/L;new=torch.sign(v)*torch.relu(abs(v)-beta/L)
        tn=float((1+np.sqrt(1+4*t*t))/2);y=new+(t-1)/tn*(new-z);z=new;t=tn
    return z


def features(Z,rgb):
    """Native-grid signed codes + local energy, shared across both arms."""
    from scipy.ndimage import uniform_filter
    energy=abs(Z)
    return np.concatenate([Z*10,energy*10,
        uniform_filter(energy,size=(1,1,5,5),mode='reflect')*10,
        uniform_filter(energy,size=(1,1,15,15),mode='reflect')*10,
        rgb,uniform_filter(rgb,size=(1,1,9,9),mode='reflect')],axis=1).astype(np.float32)


def torch_features(Z,rgb):
    import torch
    import torch.nn.functional as F
    def pool(x,k):return F.avg_pool2d(F.pad(x,(k//2,)*4,mode='reflect'),k,stride=1)
    return torch.cat([Z*10,abs(Z)*10,pool(abs(Z),5)*10,pool(abs(Z),15)*10,rgb,pool(rgb,9)],dim=1)
