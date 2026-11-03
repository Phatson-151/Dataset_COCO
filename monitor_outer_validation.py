"""Native centre-crop VAL monitoring with label-free codes and training heads."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
import json,csv,numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import average_precision_score
from native_final import native_image,NAMES,jwrite
from paper_scsc.data import manifest
from paper_scsc.v2 import OUT,tag,infer,from_cm

def main():
 selection=json.loads((OUT/'experiment_configs/selection.json').read_text());rows=[];details=[];coords=[]
 for cfg in selection['paired_models']:
  key=tag(cfg);T=cfg['tile'];images=[]
  for e in manifest()['val']:
   full=native_image(e);h,w=full['shape'];y=(h-T)//2;x=(w-T)//2
   im={k:(v[:,y:y+T,x:x+T] if k=='gt' else v[y:y+T,x:x+T] if isinstance(v,np.ndarray) and v.shape[:2]==(h,w) else v) for k,v in full.items()};im['shape']=(T,T);images.append(im)
   if cfg['gamma']==0:coords.append(dict(image_id=im['id'],origin_yx=[y,x],tile=T))
  for outer in range(1,cfg['outer']+1):
   a=np.load(OUT/'checkpoints'/key/f'outer_{outer:02}.npz');ys=[];ss=[];enc=[]
   for im in images:
    s,_,d=infer(im,a,cfg);valid=~im['ignore'];ys.append(im['gt'].argmax(0)[valid]);ss.append(softmax(s,axis=0)[:,valid]);enc.append(d)
   y=np.concatenate(ys);s=np.concatenate(ss,axis=1);cm=np.bincount(y*4+s.argmax(0),minlength=16).reshape(4,4);m=from_cm(cm)
   aps={n:float(average_precision_score(y==c,s[c])) if 0<np.sum(y==c)<len(y) else None for c,n in enumerate(NAMES)}
   row=dict(arm='CSC' if cfg['gamma']==0 else 'SCSC',id=key,outer=outer,foreground_AP_present_classes=float(np.mean([aps[n] for n in NAMES[:3] if aps[n] is not None])),foreground_mIoU=m['foreground_mIoU'],all_class_mIoU=m['all_class_mIoU'],pixel_accuracy=m['pixel_accuracy'])
   rows.append(row);details.append(dict(**row,encode=enc,confusion=cm.tolist()))
   print(key,'monitor',outer,flush=True)
 folder=OUT/'tables';folder.mkdir(exist_ok=True)
 with (folder/'outer_validation_training_head_sampled.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 jwrite(OUT/'metrics/outer_validation_training_head_sampled.json',dict(purpose='Diagnostic native centre crops, training heads, label-free coding; not full-VAL fresh-head model selection',coordinates=coords,per_checkpoint=details))
 fig,axs=plt.subplots(1,2,figsize=(12,4))
 for arm in ['CSC','SCSC']:
  r=[v for v in rows if v['arm']==arm]
  for ax,field in zip(axs,['foreground_AP_present_classes','all_class_mIoU']):ax.plot([v['outer'] for v in r],[v[field] for v in r],label=arm);ax.set_xlabel('Outer iteration');ax.set_ylabel(field);ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Centre-crop VAL diagnostic, training heads; not final evaluation');fig.tight_layout();p=OUT/'figures/convergence';p.mkdir(parents=True,exist_ok=True);fig.savefig(p/'selected_outer_validation_sampled.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
