"""Time-limited, paired contextual classifier experiment. Never writes old results.

This is a frozen-dictionary SCSC extension, not a paper reproduction or newly
trained dictionary. Both arms receive identical features, labels and budgets.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib,json,time,pickle
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import uniform_filter,gaussian_filter
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,confusion_matrix
from native_final import native_image,jwrite,NAMES
from paper_scsc.data import ROOT,manifest
from paper_scsc.experiments import infer,from_cm

OUT=ROOT/'rapid_context_output'
ARMS={'CSC':'K4_g0_N10','SCSC':'K4_g2_N10'}
FAMILIES=['sparse_context','rgb_sparse_context','rgb_context_only']
SEED=20261101

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load_model(arm):
    p=ROOT/'final output/checkpoints'/ARMS[arm]
    return np.load(p/'model.npz'),read(p/'config.json')

def codes(entry,arm):
    folder=OUT/'codes';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{arm}_{entry["image_id"]}.npz'
    if path.exists():return np.load(path)['Z']
    a,cfg=load_model(arm)
    # Linear overlap blending commutes with the original linear classifier.
    # Its full-rank W permits recovery of the saved blended Z without recoding.
    split=next(s for s in ['train','val','test_sealed'] if entry in manifest()[s])
    previous=ROOT/'final output/score_maps'/f'{ARMS[arm]}_{split}_{entry["image_id"]}.npz'
    if previous.exists():
        saved=np.load(previous);logits=saved['logits'];shape=logits.shape
        Z=np.linalg.solve(a['W'].astype('float64'),(logits-a['b'][:,None,None]).reshape(4,-1)).reshape(shape).astype('float32')
        error=float(np.max(np.abs(np.einsum('ck,khw->chw',a['W'],Z)+a['b'][:,None,None]-logits)))
        assert error<1e-6
        diag=dict(source='recovered linear blended codes from saved logits',linear_roundtrip_max_error=error,
            source_sha256=sha(previous),condition_number=float(np.linalg.cond(a['W'])))
    else:
        im=native_image(entry,with_targets=False);cfg['eval_workers']=4
        Z,_,diag=infer(im['x'],a['D'],np.eye(4,dtype='float32'),np.zeros(4,'float32'),cfg)
        diag['source']='fresh label-free full-image coding from frozen dictionary'
    np.savez_compressed(path,Z=Z)
    jwrite(path.with_suffix('.json'),diag)
    print('codes',arm,entry['image_id'],diag['source'],flush=True)
    return Z

def features(rgb,Z,family):
    """Only RGB and label-free codes enter features; no mask/GT arguments."""
    blocks=[]
    if family!='rgb_context_only':
        blocks=[Z,np.abs(Z)]
        for size in [5,17,49,97]:blocks.append(uniform_filter(np.abs(Z),size=(1,size,size),mode='reflect'))
    if family!='sparse_context':
        color=rgb.astype('float32')/255
        blocks.extend([color.transpose(2,0,1),cv2.cvtColor(color,cv2.COLOR_RGB2Lab).transpose(2,0,1)/100])
        for size in [9,33,97]:blocks.append(uniform_filter(color,size=(size,size,1),mode='reflect').transpose(2,0,1))
        gray=cv2.cvtColor(color,cv2.COLOR_RGB2GRAY)
        for size in [5,25]:
            mean=uniform_filter(gray,size)
            blocks.append(np.sqrt(np.maximum(uniform_filter(gray**2,size)-mean**2,0))[None])
    return np.ascontiguousarray(np.concatenate(blocks).reshape(-1,rgb.shape[0]*rgb.shape[1]).T,dtype='float32')

def samples(entry):
    im=native_image(entry);truth=im['gt'].argmax(0).ravel();valid=(~im['ignore']).ravel()
    rng=np.random.default_rng(SEED+im['id']);indices=[]
    for c in range(4):
        idx=np.flatnonzero(valid&(truth==c));n=min(1200,len(idx))
        if n:indices.extend(rng.choice(idx,n,replace=False))
    indices=np.array(indices,dtype='int64');return im,indices,truth[indices]

def metrics(rows,bias=0):
    cm=np.zeros((4,4),dtype='int64');ys=[];ss=[]
    for e,proba in rows:
        im=native_image(e);valid=~im['ignore'];y=im['gt'].argmax(0)[valid]
        prob=proba[:,valid];decision=np.log(np.maximum(prob,1e-12));decision[3]+=bias
        cm+=confusion_matrix(y,decision.argmax(0),labels=range(4));ys.append(y);ss.append(prob)
    result=from_cm(cm);y=np.concatenate(ys);s=np.concatenate(ss,axis=1)
    result['AP']={name:float(average_precision_score(y==c,s[c])) for c,name in enumerate(NAMES)}
    result['foreground_AP']=float(np.mean([result['AP'][n] for n in NAMES[:3]]))
    result['background_logit_bias']=bias
    return result

def predict(model,entry,arm,family):
    im=native_image(entry,with_targets=False)
    Z=None if family=='rgb_context_only' else codes(entry,arm)
    X=features(im['rgb'],Z,family)
    return model.predict_proba(X).T.reshape(4,*im['shape']).astype('float32')

def main():
    OUT.mkdir(exist_ok=True);start=time.time()
    protocol=dict(seed=SEED,train_images=10,train_pixels_per_class_per_image=1200,
      candidates=FAMILIES,max_iter=120,max_leaf_nodes=15,learning_rate=.1,l2_regularization=2,
      early_stopping=False,min_samples_leaf=80,
      selection='Common family maximizing mean CSC/SCSC VAL all-class mIoU; never optimize the gap',
      calibration='Each arm/family selects BG logit bias from [0,.5,1,1.5,2] on VAL all-class mIoU',
      limitations=['Frozen existing dictionaries, not new dictionary training','Previously-seen TEST, tiny data, one seed',
       'Modified method: nonlinear readout, dense TRAIN sampling, multi-scale context; extra RGB features in one family',
       'Model comparison can fail; do not suppress negative outcomes'],
      source_sha256=sha(__file__),dictionary_sha256={arm:sha(ROOT/'final output/checkpoints'/k/'model.npz') for arm,k in ARMS.items()})
    jwrite(OUT/'protocol.json',protocol)
    train_cache={arm:[] for arm in ARMS};labels=[];sample_record=[]
    # Cache only sampled TRAIN rows, not the enormous full-image feature matrix.
    for e in manifest()['train']:
        im,idx,y=samples(e);labels.append(y);sample_record.append(dict(image_id=im['id'],indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),counts=np.bincount(y,minlength=4)))
        for arm in ARMS:
            Z=codes(e,arm);X=features(im['rgb'],Z,'rgb_sparse_context')
            train_cache[arm].append(X[idx]);del X
    y=np.concatenate(labels);jwrite(OUT/'training_samples.json',sample_record)
    train_cache={arm:np.concatenate(rows) for arm,rows in train_cache.items()}
    records=[];models={}
    for family in FAMILIES:
        for arm in (['CSC'] if family=='rgb_context_only' else ARMS):
            name=f'{family}_{arm}';path=OUT/f'{name}.pkl'
            X=train_cache[arm]
            cols=slice(0,24) if family=='sparse_context' else slice(24,None) if family=='rgb_context_only' else slice(None)
            if path.exists():model=pickle.loads(path.read_bytes())
            else:
                model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,learning_rate=.1,l2_regularization=2,early_stopping=False,min_samples_leaf=80,random_state=SEED)
                model.fit(X[:,cols],y);path.write_bytes(pickle.dumps(model))
            models[name]=model
            rows=[]
            for e in manifest()['val']:
                p=predict(model,e,arm,family);rows.append((e,p))
                np.savez_compressed(OUT/f'{name}_val_{e["image_id"]}.npz',probabilities=p)
            # AP ranking does not depend on the decision-only BG offset.
            base=metrics(rows,0);options=[]
            for bias in [0,.5,1,1.5,2]:
                cm=np.zeros((4,4),dtype='int64')
                for e,p in rows:
                    im=native_image(e);v=~im['ignore'];d=np.log(np.maximum(p[:,v],1e-12));d[3]+=bias
                    cm+=confusion_matrix(im['gt'].argmax(0)[v],d.argmax(0),labels=range(4))
                options.append(dict(bias=bias,**from_cm(cm)))
            best=max(options,key=lambda r:r['all_class_mIoU'])
            base.update({k:v for k,v in best.items() if k!='bias'});base['background_logit_bias']=best['bias']
            record=dict(name=name,family=family,arm=arm,VAL=base,calibration_grid=options)
            records.append(record);jwrite(OUT/'validation.json',records)
            print('VAL',name,'AP',base['foreground_AP'],'mIoU',base['all_class_mIoU'],'bias',best['bias'],'elapsed',time.time()-start,flush=True)
    paired=[f for f in FAMILIES if f!='rgb_context_only']
    selected=max(paired,key=lambda f:np.mean([r['VAL']['all_class_mIoU'] for r in records if r['family']==f]))
    jwrite(OUT/'selection.json',dict(family=selected,criterion=protocol['selection'],selected_before_TEST=True))
    for record in records:
        if record['family'] not in [selected,'rgb_context_only']:continue
        name=record['name'];model=models[name];rows=[]
        for e in manifest()['test_sealed']:
            p=predict(model,e,record['arm'],record['family']);rows.append((e,p))
            np.savez_compressed(OUT/f'{name}_test_{e["image_id"]}.npz',probabilities=p)
        record['TEST_PREVIOUSLY_SEEN']=metrics(rows,record['VAL']['background_logit_bias'])
        jwrite(OUT/'results.json',dict(selected_family=selected,records=records,elapsed_s=time.time()-start))
        print('TEST',name,record['TEST_PREVIOUSLY_SEEN'],flush=True)
    print('COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':main()
