"""Write ablations, seed variability and label-free TRAIN transfer diagnostics."""
import json,csv,numpy as np
from scipy.special import softmax,expit
from sklearn.metrics import average_precision_score
from native_final import jwrite,NAMES
from paper_scsc import v2 as ex

def result(c,split='val'):return json.loads((ex.OUT/'metrics'/f'{ex.tag(c)}_{split}.json').read_text())['aggregate']
def csvwrite(name,rows):
 with (ex.directory('tables')/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 sel=json.loads((ex.OUT/'experiment_configs/selection.json').read_text());rows=[]
 for group in ['recipe_search','joint_refinement','initial_K_configs','budgeted_ADMM_pilots','ablations','controls','seed_configs']:
  for c in sel.get(group,[]):
   a=result(c);rows.append(dict(group=group,id=ex.tag(c),arm='CSC' if c['gamma']==0 else 'SCSC',K=c['K'],gamma=c['gamma'],seed=c['seed'],solver=c['solver'],loss=c['loss'],feature=c['feature'],tiles_per_image=c['tiles_per_image'],beta=c['beta'],alpha=c['alpha'],filter_size=c['filter_size'],VAL_AP=a['macro_AP_foreground'],VAL_all_mIoU=a['all_class_mIoU'],VAL_foreground_mIoU=a['foreground_mIoU'],VAL_Background_FPR=a['background_FPR'],PSNR=a['PSNR_db_mean_image']))
 csvwrite('ablations_and_controls.csv',rows)
 seeds=[]
 for c in sel['seed_configs']:
  a=result(c,'test_sealed');seeds.append(dict(seed=c['seed'],arm='CSC' if c['gamma']==0 else 'SCSC',id=ex.tag(c),foreground_AP=a['macro_AP_foreground'],foreground_mIoU=a['foreground_mIoU'],all_class_mIoU=a['all_class_mIoU'],Background_FPR=a['background_FPR'],PSNR=a['PSNR_db_mean_image']))
 csvwrite('three_seed_TEST_metrics.csv',seeds);summary={}
 for field in ['foreground_AP','foreground_mIoU','all_class_mIoU','Background_FPR','PSNR']:
  pairs=[]
  for seed in [42,43,44]:
   a=next(r for r in seeds if r['seed']==seed and r['arm']=='CSC');b=next(r for r in seeds if r['seed']==seed and r['arm']=='SCSC');pairs.append(b[field]-a[field])
  summary[field]=dict(SCSC_minus_CSC_per_seed=pairs,mean=float(np.mean(pairs)),sample_std=float(np.std(pairs,ddof=1)))
 jwrite(ex.directory('metrics')/'three_seed_summary.json',dict(paired=summary,caveat='3 training seeds, fixed 5 previously-seen TEST images; not population significance or pristine holdout'))
 transfers=[]
 for c in sel['paired_models']:
  X,rgb,Y,_=ex.bank_for(c);m=np.load(ex.OUT/'checkpoints'/ex.tag(c)/'model.npz');Z,d=ex.encode(m['D'],X,c)
  F=ex.feature_array(Z,rgb,c);S=np.einsum('cf,nfhw->nchw',m['W'],(F-m['mean'][None,:,None,None])/m['scale'][None,:,None,None])+m['b'][None,:,None,None]
  P=softmax(S,axis=1);rng=np.random.default_rng(73);aps={}
  for ch,name in enumerate(NAMES):
   pos=np.flatnonzero(Y.ravel()==ch);neg=np.flatnonzero((Y.ravel()!=ch)&(Y.ravel()>=0));n=min(len(pos),len(neg),4096)
   ix=np.r_[rng.choice(pos,n,False),rng.choice(neg,n,False)];scores=P[:,ch].ravel()[ix]
   aps[name]=float(average_precision_score(np.r_[np.ones(n),np.zeros(n)],scores))
  # Compare warmstarted unrolled and production coding WITHOUT any labels.
  import torch
  from paper_scsc.v2_solver import torch_code
  with torch.no_grad():short=torch_code(torch.tensor(m['D']),torch.tensor(X[:8]),c['beta'],c['task_steps'],torch.tensor(Z[:8])).numpy()
  gap=float(np.linalg.norm(short-Z[:8])/max(np.linalg.norm(Z[:8]),1e-12))
  transfers.append(dict(id=ex.tag(c),balanced_TRAIN_anchor_AP=aps,production_encoding_converged=d['converged'],warmstarted_unrolled_relative_code_gap=gap))
 jwrite(ex.directory('metrics')/'label_free_train_transfer.json',dict(protocol='Balanced TRAIN coordinates, AP baseline=.5. Codes never receive labels; fresh TRAIN head; diagnostic only, not held-out performance.',rows=transfers))
 print(json.dumps(summary,indent=2));print(json.dumps(transfers,indent=2))
if __name__=='__main__':main()
