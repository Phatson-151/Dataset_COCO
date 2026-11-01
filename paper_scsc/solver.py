"""Paper Eq (5), Eq (6/7), Algorithm 1/2; FFT-domain *scaled* ADMM.

Uses circular convolutions with a reconstruction boundary mask. Both Z and D
subproblems are split ADMM; classifier via L-BFGS (the paper says GD, this is
an explicitly disclosed solver substitution, for faster convex convergence).
Full supervised sparse codes during training: matches paper, unlike task-driven.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.fft import rfft2,irfft2
from scipy.optimize import minimize
from scipy.special import expit

DT=np.float32

def four(a):return rfft2(a,axes=(-2,-1),workers=1)
def back(a,sh):return irfft2(a,s=sh,axes=(-2,-1),workers=1).astype(DT)
def kernel_fft(D,hw):
    h,w=hw;f=D.shape[-1]
    padded=np.zeros((len(D),h,w),DT)
    padded[:,:f,:f]=D
    return four(padded)
def conv(Df,Z):return back(np.sum(Df[None]*four(Z),axis=1),Z.shape[-2:])
def adj(Df,x):return back(np.conj(Df)[None]*four(x)[:,None],x.shape[-2:])
def soft(a,t):return np.sign(a)*np.maximum(np.abs(a)-t,0.)

def logistic_prox(a,y,b,gamma,rho):
    """Unique logistic proximal root with safeguarded Newton, in float64.

    Undamped Newton can cycle when gamma/rho is large. Monotonicity gives
    the exact bracket between a and a+gamma*y/rho; every fallback bisects it.
    """
    a=np.asarray(a,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    other=a+gamma*y/rho;lo=np.minimum(a,other);hi=np.maximum(a,other)
    v=a.copy()
    for _ in range(120):
        s=expit(-y*(v+b));g=rho*(v-a)-gamma*y*s
        done=np.abs(g)<1e-10
        if np.all(done):break
        lo=np.where(g<0,v,lo);hi=np.where(g>0,v,hi)
        candidate=v-g/(rho+gamma*s*(1-s))
        trial=rho*(candidate-a)-gamma*y*expit(-y*(candidate+b))
        # Merely lying inside the bracket is insufficient: Newton can bounce
        # extremely close to opposite ends without shrinking it meaningfully.
        accept=(candidate>lo)&(candidate<hi)&(np.abs(trial)<=.5*np.abs(g))
        v=np.where(done,v,np.where(accept,candidate,(lo+hi)*.5))
    residual=rho*(v-a)-gamma*y*expit(-y*(v+b))
    if not np.isfinite(residual).all() or np.max(np.abs(residual),initial=0.)>1e-8:
        raise FloatingPointError('Logistic proximal stationarity failed')
    return v
def proj(D,f):
    # Compact support + Euclidean unit BALL (paper), not unit sphere.
    out=np.zeros_like(D);out[:,:f,:f]=D[:,:f,:f]
    norm=np.sqrt((out*out).sum((1,2),keepdims=True))
    return out/np.maximum(1,norm)
def sigmoid_loss_grad(Z,W,b,labels,alpha):
    feat=np.moveaxis(Z,1,-1)
    coords=np.where(labels!=0)
    n,c,y,x=coords
    yy=labels[coords].astype(DT)
    if len(yy)==0:raise ValueError('No supervision points')
    X=feat[n,y,x,:]
    logits=np.sum(X*W[c],axis=1)+b[c]
    L=np.logaddexp(0,-yy*logits).sum()+alpha*(np.square(W).sum()+np.square(b).sum())
    gradlog=-yy*expit(-yy*logits)
    gw=np.zeros_like(W);gb=np.zeros_like(b)
    np.add.at(gw,c,gradlog[:,None]*X)
    np.add.at(gb,c,gradlog)
    return float(L),gw+2*alpha*W,gb+2*alpha*b

def fit_cls(Z,labels,W,b,alpha=1., maxiter=100):
    K=Z.shape[1];C=W.shape[0]
    def fg(v):
        a=v[:C*K].reshape(C,K);bb=v[C*K:]
        L,g,bg=sigmoid_loss_grad(Z,a,bb,labels,alpha)
        return L,np.r_[g.ravel(),bg].astype('float64')
    opt=minimize(fg,np.r_[W.ravel(),b].astype('float64'),jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'gtol':1e-5,'ftol':1e-10})
    return opt.x[:C*K].reshape(C,K).astype(DT),opt.x[C*K:].astype(DT),{'iterations':int(opt.nit),'success':bool(opt.success),'loss':float(opt.fun),'message':str(opt.message)}

def make_mask(hw,f):
    h,w=hw
    M=np.ones((h,w),DT)
    # Kernels are anchored at index (0, 0), not FFT-shifted. A causal f x f
    # convolution wraps only in its first f-1 rows/columns. Exclude exactly
    # those samples; a symmetric f//2 mask does not exclude all wrapped data.
    edge=f-1
    M[:edge,:]=0;M[:,:edge]=0
    return M

def solve_z(D,x,W,b,labels,gamma,beta,rho=1.,maxiter=150,tol=0.003,zinit=None, M_override=None,adaptive=False):
    """ADMM split u0=D*z, u1=z, u2=W*z; all exact proximal operators."""
    N,H,WW=x.shape;K=len(D);C=len(W);shape=(H,WW)
    # Eq. (3) needs only [D; I]. Zero-weight logistic constraints are redundant
    # and otherwise add a large number of zero arrays to inference.
    if gamma==0:
        W=W[:0];b=b[:0];C=0
    Df=kernel_fft(D,shape)
    # Eq. (10): one K x K inverse and rank-one Woodbury corrections.
    # Avoid an H*W*K*K inverse tensor (especially for K=49 or 100).
    B=np.linalg.inv(np.eye(K,dtype=DT)+W.T@W)
    Bd=np.einsum('kl,lhw->khw',B,Df.conj(),optimize=True)
    denom=1+np.sum(Df*Bd,axis=0)
    E=W.T@np.linalg.inv(np.eye(C,dtype=DT)+W@W.T) if C and K>8 else None
    M=make_mask(shape,D.shape[-1]) if M_override is None else M_override;M0=M[None] if M.ndim==2 else M
    z=np.zeros((N,K,H,WW),DT) if zinit is None else zinit.copy()
    u0=conv(Df,z);u1=z.copy();u2=np.einsum('ck,nkhw->nchw',W,z,optimize=True)
    lam0=np.zeros_like(u0);lam1=np.zeros_like(z);lam2=np.zeros_like(u2)
    trace=[]; converged=False
    for it in range(maxiter):
        # FFT solve of block linear system, using all channels and all examples.
        R0=four(u0-lam0);R1=four(u1-lam1)
        R2=four(u2-lam2)
        rhs=np.conj(Df)[None]*R0[:,None]+R1+np.einsum('ck,nchw->nkhw',W,R2,optimize=True)
        if E is not None:
            projected=np.einsum('ck,nkhw->nchw',W,rhs,optimize=True)
            base=rhs-np.einsum('kc,nchw->nkhw',E,projected,optimize=True)
        else:
            base=np.einsum('kl,nlhw->nkhw',B,rhs,optimize=True) if C else rhs
        zf=base-Bd[None]*(np.sum(Df[None]*base,axis=1)/denom)[...,None,:,:]
        z=back(zf,shape)
        # zf is already available: do not FFT all N*K spatial maps again.
        q0=back(np.sum(Df[None]*zf,axis=1),shape)
        q2=np.einsum('ck,nkhw->nchw',W,z,optimize=True)
        old0=u0;old1=u1;old2=u2
        a0=q0+lam0
        u0=(M0*x+rho*a0)/(M0+rho)
        u1=soft(z+lam1,beta/rho)
        u2=q2+lam2
        if gamma>0 and labels is not None:
            # Prox of gamma*log(1+exp(-y*(v+b))). Newton updates with a
            # stationarity check; 8 iterations can be insufficient for high gamma.
            # Hessian >= rho > 0 guarantees a unique proximal minimizer.
            mask=labels!=0; yy=labels[mask].astype(DT)
            a=u2[mask];bb=np.broadcast_to(b[None,:,None,None],u2.shape)[mask]
            u2[mask]=logistic_prox(a,yy,bb,gamma,rho)
        lam0+=q0-u0;lam1+=z-u1;lam2+=q2-u2
        if it==0 or (it+1)%10==0 or it==maxiter-1:
            r2=np.sum((q0-u0)**2)+np.sum((z-u1)**2)+np.sum((q2-u2)**2)
            ds=adj(Df,u0-old0)+(u1-old1)+np.einsum('ck,nchw->nkhw',W,u2-old2,optimize=True)
            s2=rho*rho*np.sum(ds*ds)
            norm=np.sqrt(np.sum(q0*q0)+np.sum(z*z)+np.sum(q2*q2))
            dualnorm=np.sqrt(np.sum(u0*u0)+np.sum(u1*u1)+np.sum(u2*u2))
            ldual=adj(Df,lam0)+lam1+np.einsum('ck,nchw->nkhw',W,lam2,optimize=True)
            eps_pri=np.sqrt(q0.size+z.size+q2.size)*1e-4+tol*max(norm,dualnorm)
            eps_dual=np.sqrt(z.size)*1e-4+tol*rho*np.linalg.norm(ldual)
            pr=float(np.sqrt(r2));dr=float(np.sqrt(s2))
            if not np.isfinite(pr+dr):raise FloatingPointError('Nonfinite sparse ADMM residual')
            trace.append({'iter':it+1,'primal_residual':pr,'dual_residual':dr,
                          'rho':float(rho),
                          'coding_objective':float(.5*np.sum(M0*(q0-x)**2)+beta*np.abs(z).sum()+
                              (gamma*np.logaddexp(0,-labels*(q2+b[None,:,None,None]))[labels!=0].sum() if gamma>0 and labels is not None else 0)),
                          'primal_tolerance':float(eps_pri),'dual_tolerance':float(eps_dual),
                          'primal_relative':float(pr/max(eps_pri,1e-12)),
                          'dual_relative_proxy':float(dr/max(eps_dual,1e-12))})
            if it>=9 and pr<=eps_pri and dr<=eps_dual:
                converged=True;break
            if adaptive and (it+1)%20==0:
                ratio=(pr/max(eps_pri,1e-12))/max(dr/max(eps_dual,1e-12),1e-12)
                newrho=min(rho*2,64.) if ratio>10 else max(rho/2,1/64) if ratio<.1 else rho
                lam0*=rho/newrho;lam1*=rho/newrho;lam2*=rho/newrho;rho=newrho
    return z,{'iterations':it+1,'converged':converged,'residuals':trace,'maxiter':maxiter,'tolerance':tol}

def solve_d(x,Z,D0,rho=1.,maxiter=150,tol=0.003,M_override=None,adaptive=False):
    """Paper boundary-masked D ADMM: split v0=Z*d, v1=d in unit ball.

    FFT update (Z*Z+I)d = Z*(v0-lambda0)+(v1-lambda1).
    The reconstruction mask is inside the v0 proximity operation.
    """
    N,K,H,W=Z.shape;f=D0.shape[-1];sh=(H,W)
    # Supervised codes can make Gram blocks ill-conditioned. Float32 Woodbury
    # subtraction can lose the identity term and overflow even with finite Z.
    Zf=four(Z).astype(np.complex128);Xf=four(x)
    zmat=Zf.reshape(N,K,-1).transpose(2,0,1)
    # Stable spectral solve of (I + Z* Z). Normal-equation Gram inverses and
    # Woodbury Z* inv(I+ZZ*) factors can catastrophically cancel for coherent
    # supervised codes, even in float64. SVD preserves the identity/nullspace.
    _,singular,Vh=np.linalg.svd(zmat,full_matrices=False)
    rank=min(N,K)
    Vh=Vh.reshape(H,W//2+1,rank,K)
    inverse_values=(1/(1+singular**2)).reshape(H,W//2+1,rank)
    v1=proj(np.pad(D0,((0,0),(0,H-f),(0,W-f))),f)
    lam1=np.zeros_like(v1)
    v0=back(np.sum(four(v1)[None]*Zf,axis=1),sh)
    lam0=np.zeros_like(v0)
    M=make_mask(sh,f)[None] if M_override is None else M_override
    initial_reconstruction=back(np.sum(four(v1)[None]*Zf,axis=1),sh)
    initial_cost=float(.5*np.sum(M*(initial_reconstruction-x)**2))
    trace=[];converged=False
    for i in range(maxiter):
        rhs=np.einsum('nkhw,nhw->khw',Zf.conj(),four(v0-lam0),optimize=True)+four(v1-lam1)
        vr=np.einsum('hwrk,khw->rhw',Vh,rhs,optimize=True)
        scaled=vr*np.moveaxis(inverse_values,-1,0)
        if N<K:
            dhat=rhs+np.einsum('hwrk,rhw->khw',Vh.conj(),scaled-vr,optimize=True)
        else:
            dhat=np.einsum('hwrk,rhw->khw',Vh.conj(),scaled,optimize=True)
        d=back(dhat,sh)
        q0=back(np.sum(dhat[None]*Zf,axis=1),sh)
        old0=v0;old1=v1
        v0=(M*x+rho*(q0+lam0))/(M+rho)
        v1=proj(d+lam1,f)
        lam0+=(q0-v0);lam1+=(d-v1)
        if i==0 or (i+1)%10==0 or i==maxiter-1:
            pr=np.sqrt(np.sum((q0-v0)**2)+np.sum((d-v1)**2))
            ds=rho*(back(np.einsum('nkhw,nhw->khw',Zf.conj(),four(v0-old0),optimize=True),sh)+(v1-old1))
            du=np.linalg.norm(ds)
            n_pr=np.sqrt(np.sum(q0*q0)+np.sum(d*d))
            n_aux=np.sqrt(np.sum(v0*v0)+np.sum(v1*v1))
            ldual=back(np.einsum('nkhw,nhw->khw',Zf.conj(),four(lam0),optimize=True),sh)+lam1
            eps_pri=np.sqrt(q0.size+d.size)*1e-4+tol*max(n_pr,n_aux)
            eps_dual=np.sqrt(d.size)*1e-4+tol*rho*np.linalg.norm(ldual)
            feasible_reconstruction=back(np.sum(four(v1)[None]*Zf,axis=1),sh)
            feasible_cost=float(.5*np.sum(M*(feasible_reconstruction-x)**2))
            if not np.isfinite(float(pr)+float(du)+feasible_cost):
                raise FloatingPointError('Nonfinite dictionary ADMM state')
            trace.append({'iter':i+1,'primal_residual':float(pr),'dual_residual':float(du),
                          'rho':float(rho),'reconstruction_objective':float(.5*np.sum(M*(q0-x)**2)),
                          'feasible_dictionary_objective':feasible_cost,
                          'primal_tolerance':float(eps_pri),'dual_tolerance':float(eps_dual),
                          'primal_relative':float(pr/max(eps_pri,1e-12)),
                          'dual_relative_proxy':float(du/max(eps_dual,1e-12))})
            # Return the feasible compact dictionary, not the unconstrained d.
            # With large Z, a small d-v1 residual can amplify substantially in
            # convolution; also require descent of the actually returned point.
            if i>=9 and pr<=eps_pri and du<=eps_dual and feasible_cost<=initial_cost+1e-7*max(1.,initial_cost):
                converged=True;break
            if adaptive and (i+1)%20==0:
                ratio=(pr/max(eps_pri,1e-12))/max(du/max(eps_dual,1e-12),1e-12)
                newrho=min(rho*2,64.) if ratio>10 else max(rho/2,1/64) if ratio<.1 else rho
                lam0*=rho/newrho;lam1*=rho/newrho;rho=newrho
    rejected=feasible_cost>initial_cost+1e-7*max(1.,initial_cost)
    result=D0.copy() if rejected else v1[:,:f,:f].copy()
    return result,{'iterations':i+1,'converged':converged and not rejected,'residuals':trace,
                  'maxiter':maxiter,'tolerance':tol,'initial_feasible_objective':initial_cost,
                  'candidate_feasible_objective':feasible_cost,'descent_guard_rejected':rejected}

def objective(x,Z,D,W,b,labels,beta,gamma,alpha):
    R=conv(kernel_fft(D,x.shape[-2:]),Z)-x
    M=make_mask(x.shape[-2:],D.shape[-1])
    re=.5*np.square(R*M).sum();sp=beta*np.abs(Z).sum()
    if gamma>0 and labels is not None:
        cl,_,_=sigmoid_loss_grad(Z,W,b,labels,alpha)
    else:cl=0
    return {'reconstruction':float(re),'sparsity':float(sp),'classification':float(cl),'total':float(re+sp+gamma*cl)}

def init_dict(x,K,f,seed=42):
    rng=np.random.default_rng(seed)
    D=np.stack([x[rng.integers(len(x)),rng.integers(x.shape[-2]-f+1):][:f,:f] for _ in range(0)]) if False else rng.normal(size=(K,f,f)).astype(DT)
    D-=D.mean((1,2),keepdims=True)
    D/=np.maximum(np.linalg.norm(D,axis=(1,2),keepdims=True),1e-8)
    return D

def train(items,gamma=0.,K=8,f=9,beta=.015,alpha=1.,seed=42,outer=3,admm_max=150,tol=.003,per_class=80,cls_max=100,D0=None,snapshot_hook=None,bootstrap_classifier=True):
    x=np.stack([p['x'] for p in items]).astype(DT)
    D=init_dict(x,K,f,seed) if D0 is None else D0.copy()
    labels=__import__('paper_scsc.data',fromlist=['sample_anchors']).sample_anchors(items,per_class,seed,filter_size=f)
    W=np.zeros((3,K),DT);b=np.zeros(3,DT);Z=np.zeros((len(x),K,*x.shape[-2:]),DT)
    rows=[];ts=time.time()
    # Pre-fit a valid classifier on LABEL-FREE codes before the first outer.
    # This is only an initialization of theta in Algorithm 2 (not a changed
    # objective). Without it the first supervised coding step has W=0.
    bootstrap={'enabled':bool(bootstrap_classifier and gamma>0)}
    if bootstrap_classifier and gamma>0:
        z0,d0=solve_z(D,x,np.zeros_like(W),b,None,0,beta,maxiter=admm_max,tol=tol)
        W,b,c0=fit_cls(z0,labels,W,b,alpha,maxiter=cls_max)
        bootstrap.update({'coding':d0,'classifier':c0})
        del z0
    for o in range(outer):
        Wcode=W if gamma>0 else np.zeros_like(W)
        Z,diag_z=solve_z(D,x,Wcode,b,labels,gamma,beta,maxiter=admm_max,tol=tol,zinit=Z)
        D,diag_d=solve_d(x,Z,D,maxiter=admm_max,tol=tol)
        W,b,diag_c=fit_cls(Z,labels,W,b,alpha,maxiter=cls_max)
        obj=objective(x,Z,D,W,b,labels,beta,gamma,alpha)
        rows.append({'outer':o+1,'loss':obj,'z_admm':diag_z,'d_admm':diag_d,'classifier':diag_c,'elapsed_s':round(time.time()-ts,2),
         'filter_norms':np.linalg.norm(D.reshape(K,-1),axis=1).tolist()})
        if snapshot_hook is not None: snapshot_hook(o+1,D.copy(),W.copy(),b.copy())
    return {'D':D,'W':W,'b':b,'train_log':rows,'seconds':time.time()-ts,'labels_count':int(np.count_nonzero(labels)),'bootstrap':bootstrap}
