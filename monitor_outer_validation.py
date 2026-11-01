"""Supplementary outer-iteration monitoring, without full-VAL training-head claims.

One geometry-fixed native 64x64 centre crop per VAL image, same pixels for every
checkpoint and arm. These use the TRAINING heads, not the fresh final classifiers.
Full native VAL results and hyperparameter selection remain in metrics/*_val.json.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/.cache')
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from native_final import native_image,tile_signal,jwrite,NAMES
from paper_scsc.data import manifest
from paper_scsc.solver import solve_z,make_mask
from paper_scsc.experiments import OUT,tag,from_cm


def main():
    selection=json.loads((OUT/'experiment_configs/selection.json').read_text())
    rows=[];details=[];coordinates=[]
    for cfg in selection['paired_models']:
        key=tag(cfg['K'],cfg['gamma'],cfg['N']);T=cfg['tile'];X=[];truth=[]
        for e in manifest()['val']:
            im=native_image(e);H,W=im['shape'];y=(H-T)//2;x=(W-T)//2
            signal,gt,ignore,h,w=tile_signal(im,y,x,T,True)
            valid=(make_mask((T,T),cfg['filter_size'])>0)&~ignore
            X.append(signal);truth.append((gt.argmax(0),valid))
            if cfg['gamma']==0:coordinates.append(dict(image_id=im['id'],origin_yx=[y,x],tile=T,valid_pixels=int(valid.sum()),class_support=[int((gt[c]&valid).sum()) for c in range(4)]))
        X=np.stack(X);yt=np.concatenate([g[v] for g,v in truth])
        for outer in range(1,cfg['outer']+1):
            a=np.load(OUT/'checkpoints'/key/f'outer_{outer:02}.npz')
            D,W,b=[a[k] for k in ['D','W','b']]
            z,diag=solve_z(D,X,W*0,b*0,None,0,cfg['beta'],maxiter=cfg['eval_admm'],tol=cfg['tol'],adaptive=cfg['adaptive'])
            scores=np.einsum('ck,nkhw->nchw',W,z,optimize=True)+b[None,:,None,None]
            s=np.concatenate([scores[i][:,valid] for i,(_,valid) in enumerate(truth)],axis=1)
            pred=s.argmax(0);cm=np.bincount(yt*4+pred,minlength=16).reshape(4,4)
            m=from_cm(cm);aps={name:float(average_precision_score(yt==c,s[c])) if 0<np.sum(yt==c)<len(yt) else None for c,name in enumerate(NAMES)}
            fg=[aps[n] for n in NAMES[:3] if aps[n] is not None]
            row=dict(arm='CSC' if cfg['gamma']==0 else 'SCSC',id=key,outer=outer,
                foreground_AP_present_classes=float(np.mean(fg)) if fg else None,
                foreground_mIoU=m['foreground_mIoU'],pixel_accuracy=m['pixel_accuracy'],
                encode_converged=diag['converged'],encode_iterations=diag['iterations'],
                **{f'{n}_AP':aps[n] for n in NAMES})
            rows.append(row);details.append(dict(**row,encode=diag,confusion=cm.tolist()))
            print(key,'outer-monitor',outer,row['foreground_AP_present_classes'],m['foreground_mIoU'],diag['converged'],flush=True)
    folder=OUT/'tables';folder.mkdir(exist_ok=True)
    with (folder/'outer_validation_training_head_sampled.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    jwrite(OUT/'metrics/outer_validation_training_head_sampled.json',dict(
        purpose='Diagnostic only: fixed native centre crops and training heads; not full-VAL fresh-classifier metrics, not model selection',
        coordinates=coordinates,per_checkpoint=details))
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for arm in ['CSC','SCSC']:
        r=[v for v in rows if v['arm']==arm]
        for ax,field in zip(axs,['foreground_AP_present_classes','foreground_mIoU']):
            ax.plot([v['outer'] for v in r],[v[field] for v in r],'o-',label=arm);ax.legend();ax.set_xlabel('Outer iteration');ax.set_ylabel(field);ax.grid(alpha=.25)
    fig.suptitle('VAL diagnostic: fixed native centre crops, TRAIN heads (not fresh final classifiers)')
    fig.tight_layout();folder=OUT/'figures/convergence';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'selected_outer_validation_sampled.png',dpi=180,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
