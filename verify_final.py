"""Independent v2 artifact checks; stored predictions are re-read, not retrained."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import softmax,expit
from sklearn.metrics import average_precision_score
from skimage.metrics import structural_similarity
from PIL import Image
from native_final import jwrite,native_image,NAMES
from paper_scsc.data import ROOT,manifest
from paper_scsc.v2 import OUT,tag,psnr,from_cm,original_gray

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)
def main():
 # A second invocation after report generation checks all input digests and the
 # new report; expensive AP recomputation is reused only for identical inputs.
 input_files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.log','.lock','.md'] and p.name not in ['verification.json','SHA256SUMS.json','GOAL_STATUS.json'] and 'raw_val' not in p.parts]
 input_files += [p for p in (ROOT/'data').rglob('*') if p.is_file()]
 input_files += [ROOT/name for name in ['native_final.py','coco_scsc_manifest.json','paper_scsc/data.py','paper_scsc/solver.py','paper_scsc/v2.py','paper_scsc/v2_solver.py','verify_final.py']]
 input_files += list((ROOT/'source_snapshots').glob('*.py'))
 fingerprint=hashlib.sha256()
 for p in sorted(set(input_files)):
  fingerprint.update(str(p.relative_to(ROOT)).encode());fingerprint.update(bytes.fromhex(digest(p)))
 signature=fingerprint.hexdigest();prior=OUT/'verification.json'
 if prior.exists():
  cached=json.loads(prior.read_text())
  if cached.get('verification_input_signature')==signature:
   if (OUT/'GOAL_STATUS.json').exists():
    from verify_report_v2 import main as check_report
    check_report()
   print('All numerical/source/PNG input digests unchanged; reusing previous complete artifact checks')
   return
 selection=json.loads((OUT/'experiment_configs/selection.json').read_text());models=0;nonconverged=[];classfail=[]
 for p in sorted((OUT/'checkpoints').glob('*/completed.json')):
  c=json.loads((p.parent/'config.json').read_text());h=json.loads((p.parent/'history.json').read_text());assert c['K']<=49
  assert len(h)==c['outer']==30;assert len(list(p.parent.glob('outer_*.npz')))==30
  sw=json.loads((p.parent/'software.json').read_text())
  for name,sha in sw['source_hashes'].items():
   archived=ROOT/'source_snapshots'/f'{Path(name).stem}_{sha}.py'
   assert digest(ROOT/name)==sha or (archived.exists() and digest(archived)==sha),('Training source unavailable',p.parent.name,name)
  for checkpoint in p.parent.glob('outer_*.npz'):
   with np.load(checkpoint) as a:
    assert a['D'].shape==(c['K'],c['filter_size'],c['filter_size']);assert all(np.isfinite(a[k]).all() for k in a.files)
    assert np.linalg.norm(a['D'],axis=(1,2)).max()<=1.00001
  with np.load(p.parent/'model.npz') as a:
   features=c['K'] if c['feature']=='signed' else 4*c['K']+6
   assert a['W'].shape==(4,features);assert a['mean'].shape==(features,)
   assert np.all(a['scale']>0)
  completed=json.loads(p.read_text())
  if not completed['fresh_classifier']['success']:classfail.append(p.parent.name)
  for row in h:
   for step in ['z_admm','d_admm']:
    d=row[step];last=d['residuals'][-1]
    assert all(np.isfinite(v) for v in last.values() if isinstance(v,(int,float)))
    if d['converged']:
     assert last['primal_relative']<=1 and last['dual_relative_proxy']<=1
    else:nonconverged.append(dict(id=p.parent.name,outer=row['outer'],step=step,solver=d.get('solver','ADMM')))
  models+=1
 images={int(e['image_id']):native_image(e) for s in ['train','val','test_sealed'] for e in manifest()[s]}
 predcount=0;inferencefails=[]
 for p in sorted((OUT/'metrics').glob('*.json')):
  r=json.loads(p.read_text())
  if r.get('split') not in ['val','test_sealed'] or 'config' not in r:continue
  assert len(r['per_image'])==len(manifest()[r['split']])
  c=r['config'];key=r['id'];cm=np.zeros((4,4),np.int64);ys=[];ss=[];ps=[];ssims=[]
  cal=json.loads((OUT/'calibration'/f'{key}.json').read_text());assert cal['selection_split']=='VAL'
  for row,e in zip(r['per_image'],manifest()[r['split']]):
   assert row['image_id']==e['image_id'];im=images[e['image_id']];valid=~im['ignore'];truth=im['gt'].argmax(0)
   with np.load(OUT/'score_maps'/f'{key}_{r["split"]}_{im["id"]}.npz') as a:
    assert a['probabilities'].shape==(4,*im['shape']);assert all(np.isfinite(a[f]).all() for f in a.files if a[f].dtype.kind in 'fc')
    adjusted=a['raw_logits']/cal['temperature']+np.array(cal['bias'],np.float32)[:,None,None]
    expected=softmax(adjusted,axis=0) if c['loss']=='softmax' else expit(adjusted)
    close(a['probabilities'],expected);close(np.exp(a['logits']),np.maximum(expected,1e-12))
    if c['loss']=='softmax':close(a['probabilities'].sum(0),1)
    pred=a['probabilities'].argmax(0);np.testing.assert_array_equal(a['prediction_internal'],pred);np.testing.assert_array_equal(a['prediction'],np.array([1,2,3,0],np.uint8)[pred])
    yy=truth[valid];pp=pred[valid];local=np.bincount(yy*4+pp,minlength=16).reshape(4,4);close(local,row['confusion_gt_rows_pred_cols']);cm+=local
    ys.append(yy);ss.append(a['logits'][:,valid]);original=original_gray(im)
    ps.append(psnr(original,a['reconstruction']));close(ps[-1],row['PSNR_db'])
    ssims.append(float(structural_similarity(original,a['reconstruction'],data_range=1.)));close(ssims[-1],row['SSIM'])
    for ch,name in enumerate(NAMES):
     ap=float(average_precision_score(yy==ch,a['logits'][ch][valid])) if 0<np.sum(yy==ch)<len(yy) else None
     if ap is not None:close(ap,row['ap'][name])
   if row['encode']['converged_batches']!=row['encode']['batch_count']:inferencefails.append(dict(id=key,image_id=im['id'],split=r['split']))
   predcount+=1
  total=from_cm(cm);a=r['aggregate'];close(total['confusion_gt_rows_pred_cols'],a['confusion_gt_rows_pred_cols'])
  for name in ['foreground_mIoU','all_class_mIoU','pixel_accuracy','background_FPR']:close(total[name],a[name])
  y=np.concatenate(ys);s=np.concatenate(ss,axis=1)
  for ch,name in enumerate(NAMES):close(average_precision_score(y==ch,s[ch]),a['pooled_pixel_AP'][name])
  close(np.mean(ps),a['PSNR_db_mean_image']);close(np.mean(ssims),a['SSIM_mean_image'])
 inpaintings=0
 for e in manifest()['test_sealed']:
  im=images[e['image_id']];original=original_gray(im);masks=[]
  for cfg in selection['paired_models']:
   key=tag(cfg)
   with np.load(OUT/'inpainting'/f'{key}_{im["id"]}.npz') as a:
    obs=a['observed'];assert np.count_nonzero(~obs)==obs.size//2;masks.append(obs)
    expected=original.copy();expected[~obs]=0;close(a['masked'],expected)
    r=json.loads((OUT/'metrics'/f'{key}_inpainting.json').read_text());row=next(v for v in r['per_image'] if v['image_id']==im['id'])
    close(psnr(original,a['reconstruction']),row['PSNR_all_db']);close(psnr(original,a['reconstruction'],~obs),row['PSNR_missing_db'])
   inpaintings+=1
  np.testing.assert_array_equal(*masks)
 pngs=list((OUT/'figures').rglob('*.png'))
 for p in pngs:
  with Image.open(p) as im:im.verify()
  if 'figure4_multiclass' in p.parts:
   with Image.open(p) as im:assert np.asarray(im)[0,0,:3].max()==0,('Figure4 canvas is not black',p)
 selected_convergence=[dict(id=tag(c),outer=c['outer'],**{k:json.loads((OUT/'checkpoints'/tag(c)/'completed.json').read_text())[k] for k in ['z_converged','d_converged']}) for c in selection['paired_models']]
 result=dict(protocol='V2',verification_input_signature=signature,selected_model_convergence=selected_convergence,artifact_consistency_passed=True,models_checked=models,original_images_verified=len(images),
  stored_native_predictions_rechecked=predcount,paired_inpaintings_verified=inpaintings,PNG_files_verified=len(pngs),
  nonconverged_primary_subproblems=nonconverged,unsuccessful_classifier_fits=classfail,nonconverged_inference_images=inferencefails,
  K100_excluded=True,all_training_subproblems_converged=not nonconverged,independent_holdout_available=False,training_seed_count=3,
  fully_verified_unseen_superiority=False,note='Consistency of saved predictions, not fresh inference replay or proof of superiority. FISTA mapping tests differ from ADMM residual criteria.')
 jwrite(OUT/'verification.json',result)
 hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json' and not any(v in p.parts for v in ['raw_val'])}
 jwrite(OUT/'SHA256SUMS.json',hashes)
 print(json.dumps({k:(len(v) if isinstance(v,list) else v) for k,v in result.items()},indent=2))
if __name__=='__main__':main()
