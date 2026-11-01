"""Transparent, paired spatial probability refinement; VAL selects all knobs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix
from native_final import native_image,jwrite
from paper_scsc.data import manifest
from paper_scsc.experiments import from_cm
from rapid_context import OUT,read,metrics,sha

def refine(p,sigma):
    q=gaussian_filter(p,sigma=(0,sigma,sigma),mode='reflect') if sigma else p.copy()
    return q/np.maximum(q.sum(0,keepdims=True),1e-12)

def load_p(arm,e,split,gamma):
    if arm=='CSC':path=OUT/f'rgb_sparse_context_CSC_{split}_{e["image_id"]}.npz'
    elif arm=='RGB_only':path=OUT/f'rgb_context_only_CSC_{split}_{e["image_id"]}.npz'
    elif split=='test':path=OUT/f'gamma_reselection/selected_test_{e["image_id"]}.npz'
    elif gamma==2:path=OUT/f'rgb_sparse_context_SCSC_val_{e["image_id"]}.npz'
    else:path=OUT/f'gamma_reselection/g{gamma:g}_val_{e["image_id"]}.npz'
    return np.load(path)['probabilities']

def calibrate(rows):
    options=[]
    for bias in [0,.5,1,1.5,2]:
        cm=np.zeros((4,4),dtype='int64')
        for e,p in rows:
            im=native_image(e);v=~im['ignore'];d=np.log(np.maximum(p[:,v],1e-12));d[3]+=bias
            cm+=confusion_matrix(im['gt'].argmax(0)[v],d.argmax(0),labels=range(4))
        options.append(dict(bias=bias,**from_cm(cm)))
    return max(options,key=lambda r:r['all_class_mIoU']),options

def main():
    start=time.time();folder=OUT/'spatial_refinement';folder.mkdir(exist_ok=True)
    jwrite(folder/'protocol.json',dict(sigmas=[0,3,7,15,31],criterion='Common sigma maximizing mean CSC/SCSC VAL all-class mIoU; bias calibrated separately on VAL with same grid',
        note='Explicit postprocessing, not raw activation; previously-seen TEST exploratory evaluation, not independent holdout',source_sha256=sha(__file__)))
    while not (OUT/'gamma_reselection/results.json').exists():time.sleep(1)
    gamma=read(OUT/'gamma_reselection/results.json')['selected']['gamma'];grid=[]
    for sigma in [0,3,7,15,31]:
        record=dict(sigma=sigma,arms={})
        for arm in ['CSC','SCSC']:
            rows=[(e,refine(load_p(arm,e,'val',gamma),sigma)) for e in manifest()['val']]
            best,options=calibrate(rows);record['arms'][arm]=dict(best=best,bias_grid=options)
        record['mean_VAL_mIoU']=float(np.mean([r['best']['all_class_mIoU'] for r in record['arms'].values()]))
        grid.append(record);jwrite(folder/'validation_grid.json',grid)
        print('SPATIAL VAL',sigma,record['mean_VAL_mIoU'],flush=True)
    best=max(grid,key=lambda r:r['mean_VAL_mIoU']);sigma=best['sigma']
    jwrite(folder/'selection.json',dict(gamma=gamma,sigma=sigma,selected_before_TEST=True,record=best))
    results={}
    for arm in ['CSC','SCSC','RGB_only']:
        valrows=[(e,refine(load_p(arm,e,'val',gamma),sigma)) for e in manifest()['val']]
        cal,_=calibrate(valrows);bias=cal['bias']
        val=metrics(valrows,bias)
        for e,p in valrows:np.savez_compressed(folder/f'{arm}_val_{e["image_id"]}.npz',probabilities=p)
        rows=[]
        for e in manifest()['test_sealed']:
            p=refine(load_p(arm,e,'test',gamma),sigma);rows.append((e,p))
            d=np.log(np.maximum(p,1e-12));d[3]+=bias;pred=d.argmax(0)
            np.savez_compressed(folder/f'{arm}_test_{e["image_id"]}.npz',probabilities=p,prediction_internal=pred.astype('uint8'),prediction_public=np.array([1,2,3,0],dtype='uint8')[pred])
        results[arm]=dict(VAL=val,TEST_PREVIOUSLY_SEEN=metrics(rows,bias),bias=bias)
        print('SPATIAL TEST',arm,results[arm]['TEST_PREVIOUSLY_SEEN']['foreground_AP'],results[arm]['TEST_PREVIOUSLY_SEEN']['all_class_mIoU'],flush=True)
    jwrite(folder/'results.json',dict(gamma=gamma,sigma=sigma,arms=results,elapsed_s=time.time()-start))

if __name__=='__main__':main()
