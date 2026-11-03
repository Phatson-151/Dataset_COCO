"""V2 task-driven CSC/SCSC, honest native-grid evaluation, restartable artifacts."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,hashlib,resource
from pathlib import Path
import numpy as np
from scipy.special import softmax,expit,logsumexp
from scipy.ndimage import gaussian_filter,uniform_filter
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from skimage.metrics import structural_similarity
from native_final import native_image,choose_tiles,tile_signal,coverage_positions,jwrite,NAMES
from .data import ROOT,manifest
from .solver import init_dict,make_mask,solve_z
from . import v2_solver as vs
from .experiments import metrics,from_cm,psnr,observed_components,original_gray
OUT=ROOT/'final output'
BASE=dict(seed=42,tile=64,stride=44,tiles_per_image=12,filter_size=11,points=128,
    beta=.015,alpha=1.,outer=30,max_admm=160,eval_admm=160,tol=.003,adaptive=False,
    solver='fista',loss='softmax',feature='signed_energy_rgb',task_steps=24,task_lr=.003)

def config(K=4,gamma=.2,N=10,**kw):return dict(BASE,K=K,gamma=gamma,N=N,**kw)
def tag(cfg):
    base=f'K{cfg["K"]}_g{cfg["gamma"]:g}_N{cfg["N"]}'
    if cfg['seed']!=42:base+=f'_seed{cfg["seed"]}'
    if cfg['solver']!='fista' or cfg['loss']!='softmax':base+=f'_{cfg["solver"]}_{cfg["loss"]}'
    if cfg.get('variant'):base+='_'+cfg['variant']
    return base

def directory(s):p=OUT/s;p.mkdir(parents=True,exist_ok=True);return p

def bank_for(cfg):
    signals=[];targets=[];rgbs=[];records=[];profiles=[]
    for entry in manifest()['train'][:cfg['N']]:
        im=native_image(entry);H,W=im['shape'];T=cfg['tile'];rng=np.random.default_rng(cfg['seed']+im['id'])
        count=cfg['tiles_per_image'];coords=choose_tiles(im,T,count//2,cfg['seed'])
        coords += [(int(rng.integers(H-T+1)),int(rng.integers(W-T+1))) for _ in range(count-len(coords))]
        for y,x in coords:
            signals.append(im['x'][y:y+T,x:x+T]);rgbs.append(im['rgb'][y:y+T,x:x+T].transpose(2,0,1)/255.)
            target=im['gt'][:,y:y+T,x:x+T].argmax(0).astype(np.int64)
            ignore=im['ignore'][y:y+T,x:x+T]|(make_mask((T,T),cfg['filter_size'])==0)
            target[ignore]=-1;targets.append(target)
            records.append(dict(image_id=im['id'],origin_yx=[y,x],gt_counts=np.bincount(target[target>=0],minlength=4).tolist()))
        profiles.append(dict(image_id=im['id'],native_shape=im['shape'],training_tiles=count))
    return np.stack(signals).astype('float32'),np.stack(rgbs).astype('float32'),np.stack(targets),dict(images=profiles,tiles=records)

def encode(D,X,cfg,z0=None,mask=None):
    if cfg['solver']=='admm':
        return solve_z(D,X,np.zeros((4,len(D)),np.float32),np.zeros(4,np.float32),None,0,cfg['beta'],
            maxiter=cfg['eval_admm'],tol=cfg['tol'],adaptive=True,zinit=z0,M_override=mask)
    return vs.code(D,X,cfg['beta'],cfg['eval_admm'],cfg['tol'],mask,z0)

def feature_array(Z,rgb,cfg):
    if cfg.get('feature')=='signed':return Z*10
    return vs.features(Z,rgb)

def fit_head(Z,rgb,Y,cfg):
    F=feature_array(Z,rgb,cfg);rng=np.random.default_rng(cfg['seed']);xx=[];yy=[]
    for f,y in zip(F,Y):
        valid=np.flatnonzero(y.ravel()>=0);ix=rng.choice(valid,min(len(valid),1024),replace=False)
        xx.append(f.reshape(len(f),-1)[:,ix].T);yy.append(y.ravel()[ix])
    x=np.concatenate(xx).astype(np.float64);y=np.concatenate(yy);mean=x.mean(0);scale=np.maximum(x.std(0),.02)
    x=(x-mean)/scale
    if cfg['loss']=='ova':
        from sklearn.multiclass import OneVsRestClassifier
        clf=OneVsRestClassifier(LogisticRegression(C=1/cfg['alpha'],max_iter=400,solver='lbfgs')).fit(x,y)
        W=np.stack([c.coef_[0] for c in clf.estimators_]);b=np.array([c.intercept_[0] for c in clf.estimators_]);nit=max(int(c.n_iter_[0]) for c in clf.estimators_)
    else:
        clf=LogisticRegression(C=1/cfg['alpha'],max_iter=400,solver='lbfgs').fit(x,y)
        W=clf.coef_;b=clf.intercept_;nit=int(clf.n_iter_[0])
    return W.astype('float32'),b.astype('float32'),mean.astype('float32'),scale.astype('float32'),dict(success=nit<400,iterations=nit,sampled_pixels=len(y))

def train(cfg):
    import torch
    torch.set_num_threads(1);torch.manual_seed(cfg['seed'])
    key=tag(cfg);path=directory('checkpoints')/key;path.mkdir(exist_ok=True)
    if (path/'completed.json').exists():
        assert json.loads((path/'config.json').read_text())==cfg
        return json.loads((path/'completed.json').read_text())
    start=time.time();X,rgb,Y,record=bank_for(cfg);jwrite(path/'config.json',cfg)
    jwrite(path/'software.json',dict(source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['paper_scsc/v2.py','paper_scsc/v2_solver.py','paper_scsc/solver.py']},torch=torch.__version__,numpy=np.__version__,protocol='v2'))
    record['protocol']='v2 label-free task encoder, 30 outer, warmstart differentiable FISTA, 12 tiles';record['label_counts_per_class']=np.bincount(Y[Y>=0],minlength=4);record['tile_signals_sha256']=hashlib.sha256(X.tobytes()).hexdigest()
    record['tile_labels_sha256']=hashlib.sha256(Y.tobytes()).hexdigest();jwrite(path/'training_bank.json',record)
    D=init_dict(X,cfg['K'],cfg['filter_size'],cfg['seed']);Z,zd=encode(D,X,cfg)
    W,b,mean,scale,cd=fit_head(Z,rgb,Y,cfg)
    td=torch.tensor(D,requires_grad=True);tw=torch.tensor(W,requires_grad=True);tb=torch.tensor(b,requires_grad=True)
    opt=torch.optim.Adam([td,tw,tb],lr=cfg['task_lr']);rng=np.random.default_rng(cfg['seed']);history=[]
    # Balanced natural/ROI crop bank; moderate inverse-sqrt weighting only in task loss.
    counts=np.bincount(Y[Y>=0],minlength=4).astype(float);weights=np.sqrt(counts.sum()/np.maximum(counts,1));weights/=weights.mean()
    tm=torch.tensor(mean);ts=torch.tensor(scale);cw=torch.tensor(weights,dtype=torch.float32)
    for outer in range(cfg['outer']):
        previous=D.copy();Z,zd=encode(D,X,cfg,Z)
        if cfg['solver']=='admm':
            from .solver import solve_d
            D,dd=solve_d(X,Z,D,maxiter=160,tol=cfg['tol'],adaptive=True)
        else:D,dd=vs.dictionary(D,X,Z,maxiter=60,tol=cfg['tol'])
        with torch.no_grad():td.copy_(torch.from_numpy(D))
        taskloss=0.;ce=0.;rcloss=0.
        if cfg['gamma']>0 and outer>=4:
            ix=rng.choice(len(X),min(8,len(X)),replace=False);tx=torch.tensor(X[ix]);tr=torch.tensor(rgb[ix]);ty=torch.tensor(Y[ix])
            opt.zero_grad();tz=vs.torch_code(td,tx,cfg['beta'],cfg['task_steps'],torch.tensor(Z[ix]))
            tf=tz*10 if cfg.get('feature')=='signed' else vs.torch_features(tz,tr)
            logits=torch.einsum('cf,nfhw->nchw',tw,(tf-tm[None,:,None,None])/ts[None,:,None,None])+tb[None,:,None,None]
            if cfg['loss']=='ova':
                valid=ty>=0;target=torch.nn.functional.one_hot(torch.clamp(ty,min=0),4).permute(0,3,1,2).float()
                ce=torch.nn.functional.binary_cross_entropy_with_logits(logits.permute(0,2,3,1)[valid],target.permute(0,2,3,1)[valid])
            else:ce=torch.nn.functional.cross_entropy(logits,ty,weight=cw,ignore_index=-1)
            import torch.nn.functional as functional
            f=td.shape[-1];F=torch.fft.rfft2(functional.pad(td,(0,cfg['tile']-f,0,cfg['tile']-f)))
            recon=torch.fft.irfft2(torch.sum(F[None]*torch.fft.rfft2(tz),dim=1),s=tx.shape[-2:])
            valid=ty>=0;rcloss=torch.mean((recon[valid]-tx[valid])**2)*100
            # The task term acts THROUGH label-free inference, never labels in Z.
            taskloss=rcloss+cfg['gamma']*min((outer-3)/10,1.)*ce+1e-5*torch.sum(tw**2)
            taskloss.backward();torch.nn.utils.clip_grad_norm_([td,tw,tb],5.);opt.step()
            with torch.no_grad():td.div_(torch.clamp(torch.linalg.vector_norm(td,dim=(1,2),keepdim=True),min=1))
            D=td.detach().numpy().copy();ce=float(ce.detach());rcloss=float(rcloss.detach());taskloss=float(taskloss.detach())
        F=vs.kernel_fft(D,X.shape[-2:]);R=vs.conv(F,Z)-X;M=make_mask(X.shape[-2:],cfg['filter_size'])
        components=dict(reconstruction=float(.5*np.sum((M*R)**2)),sparsity=float(cfg['beta']*abs(Z).sum()),classification=ce)
        components['total']=components['reconstruction']+components['sparsity']+cfg['gamma']*ce
        row=dict(outer=outer+1,z_admm=zd,d_admm=dd,classifier=cd,after_classifier=components,
            dictionary_relative_change=float(np.linalg.norm(D-previous)/np.linalg.norm(previous)),
            task_objective=taskloss,task_reconstruction_scaled=rcloss,elapsed_s=time.time()-start)
        history.append(row)
        np.savez_compressed(path/f'outer_{outer+1:02}.npz',D=D,W=tw.detach().numpy(),b=tb.detach().numpy(),mean=mean,scale=scale)
        jwrite(path/'history.json',history)
        print(key,'outer',outer+1,'task',round(taskloss,5),'t',round(time.time()-start,1),flush=True)
    Z,ed=encode(D,X,cfg);We,be,me,se,ec=fit_head(Z,rgb,Y,cfg)
    np.savez_compressed(path/'model.npz',D=D,W=We,b=be,mean=me,scale=se,W_training=tw.detach().numpy(),b_training=tb.detach().numpy())
    summary=dict(id=key,config=cfg,status='completed',seconds=time.time()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        z_converged=sum(r['z_admm']['converged'] for r in history),d_converged=sum(r['d_admm']['converged'] for r in history),total_subproblems=2*cfg['outer'],
        fresh_encoding=ed,fresh_classifier=ec,bootstrap_encoding=history[0]['z_admm'],bootstrap_classifier=cd)
    jwrite(path/'completed.json',summary);return summary

def infer(im,model,cfg,signal=None,observed=None):
    signal=im['x'] if signal is None else signal;H,W=signal.shape;T=cfg['tile'];halo=cfg['filter_size']-1;core=T-halo
    xp=np.pad(signal,((halo,core),(halo,core)),mode='reflect');rp=np.pad(im['rgb'].transpose(2,0,1)/255.,((0,0),(halo,core),(halo,core)),mode='reflect')
    mp=np.pad(observed,((halo,core),(halo,core)),mode='reflect') if observed is not None else None
    coords=[(y,x) for y in coverage_positions(H,core,cfg['stride']) for x in coverage_positions(W,core,cfg['stride'])]
    sums=np.zeros((4,H,W),np.float32);recs=np.zeros((H,W),np.float32);wt=np.zeros((H,W),np.float32);diagnostics=[]
    taper=np.outer(np.hanning(core+2)[1:-1],np.hanning(core+2)[1:-1]).astype('float32')
    for i in range(0,len(coords),16):
        block=coords[i:i+16];X=np.stack([xp[y:y+T,x:x+T] for y,x in block]);rgb=np.stack([rp[:,y:y+T,x:x+T] for y,x in block]).astype('float32')
        mask=None if mp is None else np.stack([mp[y:y+T,x:x+T]*make_mask((T,T),cfg['filter_size']) for y,x in block])
        Z,diag=encode(model['D'],X,cfg,mask=mask);diagnostics.append(dict(batch_size=len(block),**diag))
        F=feature_array(Z,rgb,cfg);F=(F-model['mean'][None,:,None,None])/model['scale'][None,:,None,None]
        scores=np.einsum('cf,nfhw->nchw',model['W'],F,optimize=True)+model['b'][None,:,None,None]
        rec=vs.conv(vs.kernel_fft(model['D'],(T,T)),Z)
        for j,(y,x) in enumerate(block):
            h=min(core,H-y);w=min(core,W-x);t=taper[:h,:w];sl=(slice(halo,halo+h),slice(halo,halo+w))
            sums[:,y:y+h,x:x+w]+=scores[j,:,sl[0],sl[1]]*t;recs[y:y+h,x:x+w]+=rec[j,sl[0],sl[1]]*t;wt[y:y+h,x:x+w]+=t
    assert wt.min()>0
    return sums/wt[None],recs/wt,dict(tile_count=len(coords),batch_count=len(diagnostics),converged_batches=sum(d['converged'] for d in diagnostics),min_weight=float(wt.min()),batches=diagnostics)


def calibrate(cfg):
    key=tag(cfg);p=directory('calibration')/f'{key}.json'
    if p.exists():return json.loads(p.read_text())
    model=np.load(OUT/'checkpoints'/key/'model.npz');scores=[];labels=[]
    for e in manifest()['val']:
        im=native_image(e);s,rec,diag=infer(im,model,cfg);valid=~im['ignore'];y=im['gt'].argmax(0)[valid]
        np.savez_compressed(directory('raw_val')/f'{key}_{im["id"]}.npz',raw_logits=s,reconstruction=rec+im['low'])
        jwrite(directory('raw_val')/f'{key}_{im["id"]}.json',diag)
        # Deterministic spatial sampling solely to fit low-dimensional calibration.
        scores.append(s[:,valid][:,::13].T);labels.append(y[::13])
    S=np.concatenate(scores).astype(float);Y=np.concatenate(labels)
    def fun(v):
        A=S/np.exp(v[0])+np.r_[v[1:],0][None];P=softmax(A,axis=1)
        return float(np.mean(logsumexp(A,axis=1)-A[np.arange(len(Y)),Y])+.002*np.sum(v**2))
    fit=minimize(fun,np.zeros(4),method='L-BFGS-B',bounds=[(-2,2)]+[(-3,3)]*3)
    result=dict(temperature=float(np.exp(fit.x[0])),bias=[*fit.x[1:].tolist(),0.],selection_split='VAL',success=bool(fit.success),objective=float(fit.fun),rule='Shared temperature and class-bias CE calibration; identical procedure both arms')
    jwrite(p,result);return result


def evaluate(cfg,split='val'):
    key=tag(cfg);path=directory('metrics')/f'{key}_{split}.json'
    if path.exists():return json.loads(path.read_text())
    start=time.time();cal=calibrate(cfg);model=np.load(OUT/'checkpoints'/key/'model.npz');rows=[];ys=[];ss=[]
    for e in manifest()[split]:
        im=native_image(e)
        if split=='val':
            raw=np.load(OUT/'raw_val'/f'{key}_{im["id"]}.npz');s=raw['raw_logits'];rec=raw['reconstruction'];diag=json.loads((OUT/'raw_val'/f'{key}_{im["id"]}.json').read_text())
        else:s,rec,diag=infer(im,model,cfg);rec+=im['low']
        adjusted=s/cal['temperature']+np.array(cal['bias'],np.float32)[:,None,None]
        probs=softmax(adjusted,axis=0) if cfg['loss']=='softmax' else expit(adjusted)
        # AP and PR use the actual class probability. Log probabilities preserve rankings.
        scores=np.log(np.maximum(probs,1e-12));row=metrics(im,scores,rec,diag);rows.append(row)
        valid=~im['ignore'];ys.append(im['gt'].argmax(0)[valid]);ss.append(scores[:,valid])
        pred=probs.argmax(0).astype('uint8')
        np.savez_compressed(directory('score_maps')/f'{key}_{split}_{im["id"]}.npz',logits=scores,raw_logits=s,
            probabilities=probs,prediction_internal=pred,prediction=np.array([1,2,3,0],np.uint8)[pred],reconstruction=rec,internal_class_order=np.array(NAMES))
    total=from_cm(np.sum([np.array(r['confusion_gt_rows_pred_cols']) for r in rows],axis=0));Y=np.concatenate(ys);S=np.concatenate(ss,axis=1)
    total['pooled_pixel_AP']={n:float(average_precision_score(Y==c,S[c])) for c,n in enumerate(NAMES)}
    total['macro_AP_foreground']=float(np.mean([total['pooled_pixel_AP'][n] for n in NAMES[:3]]))
    total['macro_AP_all_classes']=float(np.mean(list(total['pooled_pixel_AP'].values())))
    for out,field in [('PSNR_db_mean_image','PSNR_db'),('SSIM_mean_image','SSIM'),('relative_reconstruction_error_mean_image','relative_reconstruction_error')]:total[out]=float(np.mean([r[field] for r in rows]))
    total['class_PSNR_db_mean_image']={n:float(np.mean([r['class_PSNR_db'][n] for r in rows if r['class_PSNR_db'][n] is not None])) for n in NAMES}
    result=dict(id=key,config=cfg,split=split,aggregate=total,per_image=rows,seconds=time.time()-start)
    jwrite(path,result);print('EVAL',key,split,'AP',total['macro_AP_foreground'],'mIoU',total['all_class_mIoU'],'FPR',total['background_FPR'],flush=True);return result


def inpaint(cfg):
    key=tag(cfg);path=directory('metrics')/f'{key}_inpainting.json'
    if path.exists():return json.loads(path.read_text())
    model=np.load(OUT/'checkpoints'/key/'model.npz');rows=[]
    for e in manifest()['test_sealed']:
        im=native_image(e,with_targets=False);original=original_gray(im);rng=np.random.default_rng(42+im['id']);observed=np.ones(original.size,bool)
        observed[rng.choice(original.size,original.size//2,replace=False)]=False;observed=observed.reshape(original.shape)
        signal,low,masked=observed_components(original,observed)
        # No hidden RGB enters even unused classifier features in inpainting.
        inp=dict(im,rgb=np.repeat(masked[:,:,None],3,axis=2)*255)
        _,high,diag=infer(inp,model,cfg,signal,observed.astype('float32'));rec=high+low
        rows.append(dict(image_id=im['id'],PSNR_all_db=psnr(original,rec),PSNR_missing_db=psnr(original,rec,~observed),SSIM=float(structural_similarity(original,rec,data_range=1.)),relative_error=float(np.linalg.norm(original-rec)/np.linalg.norm(original)),missing_fraction=float((~observed).mean()),encode=diag))
        np.savez_compressed(directory('inpainting')/f'{key}_{im["id"]}.npz',observed=observed,masked=masked,reconstruction=rec)
    result=dict(id=key,per_image=rows,split='previously_seen_test',out_of_domain='unavailable');jwrite(path,result);return result
