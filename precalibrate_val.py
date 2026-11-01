"""Calculate validation-only calibration once; no TEST labels inspected."""
import argparse,json,numpy as np
from native_final import ROOT,calibrate_offsets_on_val,run_evaluation,jwrite
out=ROOT/'final output';args=argparse.Namespace(**json.loads((out/'protocol.json').read_text())['args'])
cals={};results={}
for K in args.filters:
 for arm in ('CSC','SCSC'):
  w=np.load(out/'models'/f'K{K}_{arm}_fresh.npz')
  model={'K':K,'arm':arm,'D':w['D'],'W':w['W'],'b':w['b']}
  key=f'K{K}_{arm}'; cal=calibrate_offsets_on_val(model,args);cals[key]=cal
  r=run_evaluation([model],'val',args,out,1.2,cal['offsets'],False)[key]
  results[key]=r['aggregate']
  print(key,'val_mIoU',r['aggregate']['foreground_miou'],'mAP',r['aggregate']['macro_AP_mean_per_image'],'cal',cal['offsets'],flush=True)
jwrite(out/'metrics'/'validation_calibration.json',cals)
jwrite(out/'metrics'/'precalibrated_selection.json',results)
def converged(K):
 for arm in ('CSC','SCSC'):
  logs=json.loads((out/'train_logs'/f'K{K}_{arm}.json').read_text())['outer']
  if not all(v['z_converged']==v['total_batches'] and v['d_converged']==v['total_batches'] for v in logs):return False
 return True
eligible=[k for k in args.filters if converged(k) and results[f'K{k}_SCSC']['foreground_miou']>results[f'K{k}_CSC']['foreground_miou']]
K=max(eligible,key=lambda k:results[f'K{k}_SCSC']['foreground_miou']-results[f'K{k}_CSC']['foreground_miou']) if eligible else max(args.filters,key=lambda k:results[f'K{k}_SCSC']['foreground_miou'])
print('SELECTED BY COMPARATIVE mIoU + CONVERGENCE K',K,flush=True)
