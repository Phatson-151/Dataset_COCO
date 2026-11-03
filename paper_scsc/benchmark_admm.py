"""ADMM benchmark reference with configurable absolute stopping tolerance.
Production solver stays unchanged; stricter tolerance makes objectives comparable.
"""
from .solver import *

def solve_z_reference(D,x,W,b,labels,gamma,beta,rho=1.,maxiter=150,tol=0.003,zinit=None, M_override=None,adaptive=False,chunk_size=1,absolute_tolerance=1e-7):
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
    q0=u0.copy();q2=u2.copy()
    chunk_size=N if chunk_size is None else max(1,int(chunk_size))
    for it in range(maxiter):
        checking=it==0 or (it+1)%10==0 or it==maxiter-1
        if checking:old0=u0.copy();old1=u1.copy();old2=u2.copy()
        for start in range(0,N,chunk_size):
            sl=slice(start,min(start+chunk_size,N))
            R0=four(u0[sl]-lam0[sl]);R1=four(u1[sl]-lam1[sl]);R2=four(u2[sl]-lam2[sl])
            rhs=np.conj(Df)[None]*R0[:,None]+R1+np.einsum('ck,nchw->nkhw',W,R2,optimize=True)
            if E is not None:
                projected=np.einsum('ck,nkhw->nchw',W,rhs,optimize=True)
                base=rhs-np.einsum('kc,nchw->nkhw',E,projected,optimize=True)
            else:base=np.einsum('kl,nlhw->nkhw',B,rhs,optimize=True) if C else rhs
            zf=base-Bd[None]*(np.sum(Df[None]*base,axis=1)/denom)[...,None,:,:]
            zz=back(zf,shape)
            p0=back(np.sum(Df[None]*zf,axis=1),shape)
            p2=np.einsum('ck,nkhw->nchw',W,zz,optimize=True)
            mask0=M0 if M0.shape[0]==1 else M0[sl]
            v0=(mask0*x[sl]+rho*(p0+lam0[sl]))/(mask0+rho)
            v1=soft(zz+lam1[sl],beta/rho);v2=p2+lam2[sl]
            if gamma>0 and labels is not None:
                mask=labels[sl]!=0;yy=labels[sl][mask].astype(DT)
                bb=np.broadcast_to(b[None,:,None,None],v2.shape)[mask]
                v2[mask]=logistic_prox(v2[mask],yy,bb,gamma,rho)
            z[sl]=zz;q0[sl]=p0;q2[sl]=p2
            u0[sl]=v0;u1[sl]=v1;u2[sl]=v2
            lam0[sl]+=p0-v0;lam1[sl]+=zz-v1;lam2[sl]+=p2-v2
        if it==0 or (it+1)%10==0 or it==maxiter-1:
            r2=np.sum((q0-u0)**2)+np.sum((z-u1)**2)+np.sum((q2-u2)**2)
            ds=adj(Df,u0-old0)+(u1-old1)+np.einsum('ck,nchw->nkhw',W,u2-old2,optimize=True)
            s2=rho*rho*np.sum(ds*ds)
            norm=np.sqrt(np.sum(q0*q0)+np.sum(z*z)+np.sum(q2*q2))
            dualnorm=np.sqrt(np.sum(u0*u0)+np.sum(u1*u1)+np.sum(u2*u2))
            ldual=adj(Df,lam0)+lam1+np.einsum('ck,nchw->nkhw',W,lam2,optimize=True)
            eps_pri=np.sqrt(q0.size+z.size+q2.size)*absolute_tolerance+tol*max(norm,dualnorm)
            eps_dual=np.sqrt(z.size)*absolute_tolerance+tol*rho*np.linalg.norm(ldual)
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
    return z,{'iterations':it+1,'converged':converged,'residuals':trace,'maxiter':maxiter,'tolerance':tol,'compute_chunk_size':chunk_size}
