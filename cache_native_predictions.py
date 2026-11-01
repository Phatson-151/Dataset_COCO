"""Only cache raw full-original-grid predictions; no threshold/GT in inference."""
import argparse, json, numpy as np
from native_final import ROOT,manifest,native_image,native_predict_cached
p=argparse.ArgumentParser();p.add_argument('--K',type=int,required=True);p.add_argument('--split',choices=['val','test_sealed'],default='val')
p.add_argument('--arm',choices=['CSC','SCSC','both'],default='both')
a=p.parse_args();out=ROOT/'final output';config=json.loads((out/'protocol.json').read_text())['args'];args=argparse.Namespace(**config)
for arm in (('CSC','SCSC') if a.arm=='both' else [a.arm]):
 weights=np.load(out/'models'/f'K{a.K}_{arm}_fresh.npz')
 model={'K':a.K,'arm':arm,'D':weights['D'],'W':weights['W'],'b':weights['b']}
 for e in manifest()[a.split]:
  im=native_image(e)
  scores,recon,diag=native_predict_cached(model,im,a.split,args,out)
  print(f'Cached original {a.split} K{a.K} {arm} {im["id"]} shape={im["shape"]} tiles={diag["tiles"]} converged={diag["converged"]}/{diag["tiles"]}',flush=True)
