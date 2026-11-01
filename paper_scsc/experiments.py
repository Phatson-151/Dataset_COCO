"""Reproducible native-pixel experiments, separate from legacy release scripts.

Algorithm 2 is solved jointly over a fixed bank of intact native-pixel tiles.
The D step uses ALL bank tiles, never an EMA of minibatch dictionaries.
This remains a sampled-tile approximation to full-frame dictionary learning.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
import marshal
import platform
import resource
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit
from sklearn.metrics import average_precision_score, confusion_matrix
from skimage.metrics import structural_similarity
from native_final import (native_image, create_train_bank, coverage_positions,
                          jwrite, NAMES)
from paper_scsc.data import ROOT, manifest
from paper_scsc.solver import (init_dict, solve_z, solve_d, fit_cls, objective,
                               conv, kernel_fft, make_mask)

OUT = ROOT / 'final output'
SEED = 42
BASE = dict(seed=SEED, tile=64, stride=44, tiles_per_image=4,
            filter_size=11, points=24, beta=.015, alpha=1., outer=10,
            max_admm=1000, eval_admm=1000, tol=.003, adaptive=True)


def tag(k, gamma, n):
    return f'K{k}_g{gamma:g}_N{n}'


def config(k=4, gamma=.2, n=10):
    return dict(BASE, K=k, gamma=gamma, N=n)


def directory(name):
    p=OUT/name;p.mkdir(parents=True,exist_ok=True);return p


def source_revision():
    """Version metadata also works when the release ZIP has no .git directory."""
    try:
        root=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=ROOT,
            text=True,capture_output=True)
    except FileNotFoundError:root=None
    if root is not None and root.returncode==0 and Path(root.stdout.strip()).resolve()==ROOT.resolve():
        return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    release=ROOT/'RELEASE_MANIFEST.json'
    if release.exists():return json.loads(release.read_text()).get('source_commit','unversioned_source_archive')
    return 'unversioned_source_archive'


def bank_for(cfg):
    bank,profile=create_train_bank(manifest()['train'][:cfg['N']],cfg['tile'],
        cfg['tiles_per_image'],cfg['filter_size'],cfg['points'],cfg['seed'])
    return np.stack([p['x'] for p in bank]), np.stack([p['labels'] for p in bank]), bank,profile


def train(cfg):
    lock=directory('checkpoints')/f'{tag(cfg["K"],cfg["gamma"],cfg["N"])}.lock'
    with lock.open('w') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        return _train(cfg)


def _train(cfg):
    key=tag(cfg['K'],cfg['gamma'],cfg['N']);path=directory('checkpoints')/key
    path.mkdir(exist_ok=True)
    if (path/'config.json').exists() and json.loads((path/'config.json').read_text())!=cfg:
        raise ValueError(f'Configuration collision: {key}')
    if (path/'completed.json').exists():
        existing=json.loads((path/'config.json').read_text())
        if existing!=cfg:raise ValueError(f'Configuration collision: {key}')
        return json.loads((path/'completed.json').read_text())
    jwrite(path/'config.json',cfg)
    prior_history=json.loads((path/'history.json').read_text()) if (path/'history.json').exists() else []
    software_file=path/(f'software_resume_{len(prior_history):02}.json' if prior_history else 'software.json')
    jwrite(software_file,dict(**IMPORTED_SOFTWARE,
        note='Safeguarded logistic prox, stable SVD D systems, cached FFT work; global ADMM criteria unchanged.'))
    start=time.time();X,Y,bank,profile=bank_for(cfg)
    bank_record={'images':profile,'tiles':[
        {k:v for k,v in p.items() if k not in ('x','labels')} for p in bank],
        'label_counts_per_class':np.count_nonzero(Y,axis=(0,2,3)),
        'tile_signals_sha256':hashlib.sha256(X.tobytes()).hexdigest(),
        'tile_labels_sha256':hashlib.sha256(Y.tobytes()).hexdigest()}
    if prior_history and (path/'training_bank.json').exists():
        previous=json.loads((path/'training_bank.json').read_text())
        for name in ['tile_signals_sha256','tile_labels_sha256']:
            if previous[name]!=bank_record[name]:raise ValueError('Cannot resume on a changed native TRAIN bank')
    jwrite(path/'training_bank.json',bank_record)
    opts=dict(maxiter=cfg['max_admm'],tol=cfg['tol'],adaptive=cfg['adaptive'])
    history=prior_history
    event={}
    if history:
        if len(history)>cfg['outer']:raise ValueError('Checkpoint outer count exceeds configuration')
        saved=np.load(path/f'outer_{len(history):02}.npz')
        D,W,b=[saved[k] for k in ['D','W','b']]
        Z=np.zeros((len(X),cfg['K'],cfg['tile'],cfg['tile']),'float32')
        boot=json.loads((path/'bootstrap.json').read_text())
        bootstrap_z,bootstrap_c=boot['encoding'],boot['classifier']
        event=json.loads((path/'resume_event.json').read_text()) if (path/'resume_event.json').exists() else {}
        elapsed=max(history[-1]['elapsed_s'],event.get('prior_process_elapsed_s',0))
        start-=elapsed
        jwrite(path/f'resume_{len(history):02}.json',dict(completed_outer_preserved=len(history),
            prior_training_wall_s=elapsed,sparse_code_initialization='zeros; current D and theta restored; inner ADMM solves the same convex objective',
            software_metadata=software_file.name))
        print(f'{key} resume after outer {len(history)}; D/theta restored, Z reinitialized',flush=True)
    else:
        D=init_dict(X,cfg['K'],cfg['filter_size'],cfg['seed'])
        W=np.zeros((4,cfg['K']),'float32');b=np.zeros(4,'float32')
        Z,bootstrap_z=solve_z(D,X,W,b,None,0,cfg['beta'],**opts)
        W,b,bootstrap_c=fit_cls(Z,Y,W,b,cfg['alpha'],maxiter=400)
        jwrite(path/'bootstrap.json',dict(encoding=bootstrap_z,classifier=bootstrap_c))
        print(f'{key} bootstrap Z={bootstrap_z["converged"]}({bootstrap_z["iterations"]}) '
              f't={time.time()-start:.1f}s',flush=True)
    for outer in range(len(history),cfg['outer']):
        before=objective(X,Z,D,W,b,Y,cfg['beta'],cfg['gamma'],cfg['alpha'])
        Z,zd=solve_z(D,X,W if cfg['gamma'] else W*0,b,
            Y if cfg['gamma'] else None,cfg['gamma'],cfg['beta'],zinit=Z,**opts)
        after_z=objective(X,Z,D,W,b,Y,cfg['beta'],cfg['gamma'],cfg['alpha'])
        old=D.copy();D,dd=solve_d(X,Z,D,**opts)
        after_d=objective(X,Z,D,W,b,Y,cfg['beta'],cfg['gamma'],cfg['alpha'])
        # CRITICAL: theta update on the same supervised Z, not label-free recoding.
        W,b,cd=fit_cls(Z,Y,W,b,cfg['alpha'],maxiter=400)
        after_c=objective(X,Z,D,W,b,Y,cfg['beta'],cfg['gamma'],cfg['alpha'])
        row=dict(outer=outer+1,z_admm=zd,d_admm=dd,classifier=cd,
            before=before,after_z=after_z,after_d=after_d,after_classifier=after_c,
            dictionary_relative_change=float(np.linalg.norm(D-old)/np.linalg.norm(old)),
            filter_norms=np.linalg.norm(D,axis=(1,2)),elapsed_s=time.time()-start,
            peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        history.append(row)
        np.savez_compressed(path/f'outer_{outer+1:02}.npz',D=D,W=W,b=b)
        jwrite(path/'history.json',history)
        print(f'{key} outer {outer+1}/{cfg["outer"]} Z={zd["converged"]}({zd["iterations"]}) '
              f'D={dd["converged"]}({dd["iterations"]}) obj={after_c["total"]:.4f} '
              f't={time.time()-start:.1f}s',flush=True)
    # Paper section 5.2: fresh evaluation classifier for BOTH dictionaries.
    Ze,ed=solve_z(D,X,W*0,b*0,None,0,cfg['beta'],**opts)
    We,be,ec=fit_cls(Ze,Y,W*0,b*0,cfg['alpha'],maxiter=600)
    np.savez_compressed(path/'model.npz',D=D,W=We,b=be,W_training=W,b_training=b)
    summary=dict(id=key,config=cfg,status='completed',seconds=time.time()-start,
        peak_rss_MiB=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,event.get('prior_peak_rss_MiB',0)),
        timing_scope='Training-process wall time; resumed jobs include recorded segment/restart/wait overhead, not a controlled speed benchmark',
        z_converged=sum(r['z_admm']['converged'] for r in history),
        d_converged=sum(r['d_admm']['converged'] for r in history),
        total_subproblems=2*cfg['outer'],bootstrap_encoding=bootstrap_z,
        bootstrap_classifier=bootstrap_c,fresh_encoding=ed,fresh_classifier=ec)
    jwrite(path/'completed.json',summary)
    return summary


def infer(signal,D,W,b,cfg,observed=None):
    """Inference accepts signal and optional observed mask ONLY; never GT.

    Reflect padding supplies context at image edges. Only the valid causal
    convolution interior of each tile is overlap blended (no circular seams).
    All original pixels receive positive weight. No downsampling anywhere.
    """
    H,WW=signal.shape;T=cfg['tile'];halo=D.shape[-1]-1;core=T-halo
    stride=min(cfg['stride'],core)
    padded=np.pad(signal,((halo,core),(halo,core)),mode='reflect')
    obs=np.pad(observed,((halo,core),(halo,core)),mode='reflect') if observed is not None else None
    coords=[(y,x) for y in coverage_positions(H,core,stride) for x in coverage_positions(WW,core,stride)]
    sums=np.zeros((4,H,WW),'float32');rec=np.zeros((H,WW),'float32');wt=np.zeros_like(rec)
    taper=np.outer(np.hanning(core+2)[1:-1],np.hanning(core+2)[1:-1]).astype('float32')
    diagnostics=[]
    blocks=[coords[i:i+16] for i in range(0,len(coords),16)]
    workers=min(len(blocks),int(cfg.get('eval_workers',os.environ.get('SCSC_EVAL_WORKERS','2'))))
    def code_block(block):
        X=np.stack([padded[y:y+T,x:x+T] for y,x in block])
        mask=make_mask((T,T),D.shape[-1])
        if obs is not None:mask=np.stack([obs[y:y+T,x:x+T]*mask for y,x in block])
        Z,diag=solve_z(D,X,W*0,b*0,None,0,cfg['beta'],maxiter=cfg['eval_admm'],
            tol=cfg['tol'],M_override=mask,adaptive=cfg['adaptive'])
        scores=np.einsum('ck,nkhw->nchw',W,Z,optimize=True)+b[None,:,None,None]
        rr=conv(kernel_fft(D,(T,T)),Z)
        return block,scores,rr,diag
    # Batches are independent; blend in their original order for reproducible
    # arithmetic. Parallelism changes scheduling, not predictions or stopping.
    with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
        for block,scores,rr,diag in pool.map(code_block,blocks):
            diagnostics.append(dict(batch_size=len(block),**diag))
            for j,(y,x) in enumerate(block):
                h=min(core,H-y);w=min(core,WW-x);t=taper[:h,:w]
                sums[:,y:y+h,x:x+w]+=scores[j,:,halo:halo+h,halo:halo+w]*t
                rec[y:y+h,x:x+w]+=rr[j,halo:halo+h,halo:halo+w]*t
                wt[y:y+h,x:x+w]+=t
    assert wt.min()>0 and np.isfinite(rec).all()
    return sums/wt[None],rec/wt,dict(tile_count=len(coords),
        batch_count=len(diagnostics),converged_batches=sum(d['converged'] for d in diagnostics),
        min_weight=float(wt.min()),encoding_workers=workers,batches=diagnostics)


def psnr(a,b,mask=None):
    e=(a-b)**2
    if mask is not None:e=e[mask]
    return float(-10*np.log10(max(float(np.mean(e)),1e-12))) if e.size else None


def original_gray(im):
    return cv2.cvtColor(im['rgb'],cv2.COLOR_RGB2GRAY).astype('float32')/255


def observed_components(gray,observed):
    """Inpainting input decomposition, with no access to hidden pixel values."""
    masked=np.where(observed,gray,0.)
    low=gaussian_filter(masked,2)/np.maximum(gaussian_filter(observed.astype('float32'),2),1e-8)
    high=np.where(observed,gray-low,0)
    return high,low,masked


def metrics(im,scores,reconstruction,diag):
    valid=~im['ignore'];truth=np.argmax(im['gt'],axis=0);pred=scores.argmax(axis=0)
    y=truth[valid];p=pred[valid];cm=confusion_matrix(y,p,labels=range(4))
    result=from_cm(cm);result['image_id']=im['id'];result['native_shape']=im['shape']
    result['ap']={name:float(average_precision_score(y==c,scores[c][valid]))
        if 0<np.sum(y==c)<len(y) else None for c,name in enumerate(NAMES)}
    result['macro_ap_foreground']=float(np.mean([v for n,v in result['ap'].items() if n!='Background' and v is not None]))
    original=im['x']+im['low']
    result['PSNR_db']=psnr(original,reconstruction)
    result['class_PSNR_db']={name:psnr(original,reconstruction,valid&(truth==c)) for c,name in enumerate(NAMES)}
    result['SSIM']=float(structural_similarity(original,reconstruction,data_range=1.))
    result['relative_reconstruction_error']=float(np.linalg.norm(original-reconstruction)/np.linalg.norm(original))
    result['encode']=diag
    return result


def from_cm(cm):
    per={}
    for c,name in enumerate(NAMES):
        tp=int(cm[c,c]);sup=int(cm[c].sum());num=int(cm[:,c].sum())
        precision=tp/max(num,1);recall=tp/max(sup,1)
        per[name]=dict(precision=precision,recall=recall,f1=2*tp/max(sup+num,1),
            iou=tp/(sup+num-tp) if sup+num-tp else None,tp=tp,fp=num-tp,fn=sup-tp,support=sup,predicted=num)
    return dict(confusion_gt_rows_pred_cols=cm.tolist(),
        normalized_confusion=(cm/np.maximum(cm.sum(1,keepdims=True),1)).tolist(),per_class=per,
        foreground_mIoU=float(np.mean([per[n]['iou'] for n in NAMES[:3] if per[n]['iou'] is not None])),
        all_class_mIoU=float(np.mean([per[n]['iou'] for n in NAMES if per[n]['iou'] is not None])),
        pixel_accuracy=float(np.trace(cm)/max(cm.sum(),1)),
        class_confusion_rate=float(1-np.trace(cm)/max(cm.sum(),1)),
        foreground_class_confusion_rate=float((cm[:3,:3].sum()-np.trace(cm[:3,:3]))/max(cm[:3].sum(),1)),
        background_FPR=float(cm[3,:3].sum()/max(cm[3].sum(),1)))


def evaluate(cfg,split='val'):
    lock=directory('metrics')/f'{tag(cfg["K"],cfg["gamma"],cfg["N"])}_{split}.lock'
    with lock.open('w') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        return _evaluate(cfg,split)


def _evaluate(cfg,split='val'):
    key=tag(cfg['K'],cfg['gamma'],cfg['N']);path=directory('metrics')/f'{key}_{split}.json'
    if path.exists():return json.loads(path.read_text())
    started=time.time()
    model=np.load(OUT/'checkpoints'/key/'model.npz');D,W,b=[model[k] for k in ('D','W','b')]
    rows=[];pooled_y=[];pooled_scores=[]
    for entry in manifest()[split]:
        im=native_image(entry)
        scores,recon,diag=infer(im['x'],D,W,b,cfg);recon+=im['low']
        # No smoothing, threshold tuning, GT filtering or per-arm offsets.
        row=metrics(im,scores,recon,diag);rows.append(row)
        valid=~im['ignore'];pooled_y.append(im['gt'].argmax(0)[valid]);pooled_scores.append(scores[:,valid])
        destination=directory('score_maps')/f'{key}_{split}_{im["id"]}.npz'
        np.savez_compressed(destination,logits=scores,probabilities=expit(scores),
            prediction_internal=scores.argmax(0).astype('uint8'),
            prediction=np.array([1,2,3,0],dtype='uint8')[scores.argmax(0)],
            reconstruction=recon,internal_class_order=np.array(NAMES))
        print(f'{key} {split} {im["id"]} AP={row["macro_ap_foreground"]:.4f} '
              f'mIoU={row["foreground_mIoU"]:.4f} PSNR={row["PSNR_db"]:.2f} '
              f'encoding={diag["converged_batches"]}/{diag["batch_count"]}',flush=True)
    total=from_cm(np.sum([np.array(r['confusion_gt_rows_pred_cols']) for r in rows],axis=0))
    y=np.concatenate(pooled_y);s=np.concatenate(pooled_scores,axis=1)
    total['pooled_pixel_AP']={name:float(average_precision_score(y==c,s[c])) if np.any(y==c) else None for c,name in enumerate(NAMES)}
    total['macro_AP_foreground']=float(np.mean([total['pooled_pixel_AP'][n] for n in NAMES[:3] if total['pooled_pixel_AP'][n] is not None]))
    total['PSNR_db_mean_image']=float(np.mean([r['PSNR_db'] for r in rows]))
    total['class_PSNR_db_mean_image']={n:float(np.mean([r['class_PSNR_db'][n] for r in rows if r['class_PSNR_db'][n] is not None])) for n in NAMES}
    total['SSIM_mean_image']=float(np.mean([r['SSIM'] for r in rows]))
    total['relative_reconstruction_error_mean_image']=float(np.mean([r['relative_reconstruction_error'] for r in rows]))
    result=dict(id=key,config=cfg,split=split,aggregate=total,per_image=rows,
        seconds=time.time()-started,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    jwrite(path,result);return result


def inpaint(cfg):
    started=time.time()
    key=tag(cfg['K'],cfg['gamma'],cfg['N']);model=np.load(OUT/'checkpoints'/key/'model.npz')
    D,W,b=[model[k] for k in ('D','W','b')];rows=[]
    for e in manifest()['test_sealed']:
        im=native_image(e,with_targets=False);original=original_gray(im);rng=np.random.default_rng(SEED+im['id'])
        observed=np.ones(original.size,bool);observed[rng.choice(original.size,original.size//2,replace=False)]=False
        observed=observed.reshape(original.shape)
        # Lowpass uses ONLY observed pixels. Never reuse complete-image lowpass.
        signal,low,masked=observed_components(original,observed)
        _,high,diag=infer(signal,D,W,b,cfg,observed.astype('float32'))
        reconstruction=high+low
        rows.append(dict(image_id=im['id'],PSNR_all_db=psnr(original,reconstruction),
            PSNR_missing_db=psnr(original,reconstruction,~observed),
            SSIM=float(structural_similarity(original,reconstruction,data_range=1.)),
            relative_error=float(np.linalg.norm(original-reconstruction)/np.linalg.norm(original)),
            missing_fraction=float((~observed).mean()),
            observed_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),encode=diag))
        np.savez_compressed(directory('inpainting')/f'{key}_{im["id"]}.npz',
            observed=observed,masked=masked,reconstruction=reconstruction)
        print(f'inpainting {key} {im["id"]} {rows[-1]["PSNR_missing_db"]:.2f} dB',flush=True)
    jwrite(directory('metrics')/f'{key}_inpainting.json',dict(id=key,per_image=rows,
        split='previously_seen_test',out_of_domain='not run: no independent other-category dataset',
        seconds=time.time()-started,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','eval','inpaint'])
    p.add_argument('--K',type=int,default=4);p.add_argument('--gamma',type=float,default=.2)
    p.add_argument('--N',type=int,default=10);p.add_argument('--split',default='val')
    a=p.parse_args();cfg=config(a.K,a.gamma,a.N)
    if a.action=='inpaint' or a.split=='test_sealed' or (a.action=='eval' and a.K>=100):cfg['eval_workers']=4
    if a.action=='train':train(cfg)
    elif a.action=='eval':evaluate(cfg,a.split)
    else:inpaint(cfg)


IMPORTED_SOFTWARE=dict(source_commit=source_revision(),
    file_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['paper_scsc/solver.py','paper_scsc/experiments.py','paper_scsc/data.py','native_final.py']},
    loaded_function_bytecode_sha256={f.__name__:hashlib.sha256(marshal.dumps(f.__code__)).hexdigest()
        for f in [solve_z,solve_d,fit_cls,_train]},
    fingerprint_scope='Captured at module import, before waiting on any model lock')

if __name__=='__main__':main()
