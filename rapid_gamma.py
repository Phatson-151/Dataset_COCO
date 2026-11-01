"""Predeclared VAL-only gamma reselection for the new contextual head."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','4')
import time,pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import rapid_context as rc
from native_final import native_image,jwrite
from paper_scsc.data import manifest
from paper_scsc.experiments import from_cm

def main():
    start=time.time();folder=rc.OUT/'gamma_reselection';folder.mkdir(exist_ok=True)
    protocol=dict(gammas=[.2,2,10,100],family='rgb_sparse_context',
        criterion='Maximize SCSC VAL all-class mIoU, independent of CSC score or TEST',
        fixed_head='same 120 iterations,15 leaves,seed and training points as CSC',
        declared_before_reading_new_TEST=True,source_sha256=rc.sha(__file__))
    jwrite(folder/'protocol.json',protocol);records=[];models={}
    for gamma in [.2,10,100]:
        arm=f'SCSC_g{gamma:g}';rc.ARMS[arm]=f'K4_g{gamma:g}_N10';Xs=[];ys=[]
        for e in manifest()['train']:
            im,idx,y=rc.samples(e);Z=rc.codes(e,arm)
            X=rc.features(im['rgb'],Z,'rgb_sparse_context');Xs.append(X[idx]);ys.append(y)
        model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,learning_rate=.1,l2_regularization=2,early_stopping=False,min_samples_leaf=80,random_state=rc.SEED)
        model.fit(np.concatenate(Xs),np.concatenate(ys));(folder/f'g{gamma:g}.pkl').write_bytes(pickle.dumps(model));models[gamma]=model
        rows=[]
        for e in manifest()['val']:
            p=rc.predict(model,e,arm,'rgb_sparse_context');rows.append((e,p));np.savez_compressed(folder/f'g{gamma:g}_val_{e["image_id"]}.npz',probabilities=p)
        base=rc.metrics(rows);options=[]
        for bias in [0,.5,1,1.5,2]:
            cm=np.zeros((4,4),dtype='int64')
            for e,p in rows:
                im=native_image(e);v=~im['ignore'];d=np.log(np.maximum(p[:,v],1e-12));d[3]+=bias
                cm+=confusion_matrix(im['gt'].argmax(0)[v],d.argmax(0),labels=range(4))
            options.append(dict(bias=bias,**from_cm(cm)))
        best=max(options,key=lambda r:r['all_class_mIoU']);base.update({k:v for k,v in best.items() if k!='bias'});base['background_logit_bias']=best['bias']
        records.append(dict(gamma=gamma,arm=arm,VAL=base,calibration_grid=options));jwrite(folder/'validation.json',records)
        print('GAMMA VAL',gamma,'AP',base['foreground_AP'],'mIoU',base['all_class_mIoU'],'elapsed',time.time()-start,flush=True)
    # The base run's gamma=2 is one declared candidate. Wait for its VAL only.
    while not (rc.OUT/'validation.json').exists():time.sleep(1)
    existing=rc.read(rc.OUT/'validation.json')
    matches=[r for r in existing if r['name']=='rgb_sparse_context_SCSC']
    if not matches:raise RuntimeError('Base gamma2 contextual VAL result not ready')
    records.append(dict(gamma=2,arm='SCSC',VAL=matches[0]['VAL']))
    selected=max(records,key=lambda r:r['VAL']['all_class_mIoU'])
    jwrite(folder/'selection.json',dict(selected_gamma=selected['gamma'],criterion=protocol['criterion'],all_VAL=records,selected_before_final_TEST=True))
    gamma=selected['gamma'];arm=selected['arm']
    model=models[gamma] if gamma!=2 else pickle.loads((rc.OUT/'rgb_sparse_context_SCSC.pkl').read_bytes())
    rows=[]
    for e in manifest()['test_sealed']:
        p=rc.predict(model,e,arm,'rgb_sparse_context');rows.append((e,p));np.savez_compressed(folder/f'selected_test_{e["image_id"]}.npz',probabilities=p)
    selected['TEST_PREVIOUSLY_SEEN']=rc.metrics(rows,selected['VAL']['background_logit_bias'])
    jwrite(folder/'results.json',dict(selected=selected,all_VAL=records,elapsed_s=time.time()-start))
    print('GAMMA COMPLETE',gamma,selected['TEST_PREVIOUSLY_SEEN'],time.time()-start,flush=True)

if __name__=='__main__':main()
