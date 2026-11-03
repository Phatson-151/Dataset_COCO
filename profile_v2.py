"""Resource feasibility and numerical diagnostics, kept separate from training."""
import time,resource,json
import numpy as np
from native_final import native_image,jwrite
from paper_scsc.data import manifest
from paper_scsc.solver import init_dict,make_mask
from paper_scsc.v2_solver import code
from paper_scsc.v2 import OUT,directory

def main():
 im=native_image(manifest()['train'][0],with_targets=False);X=im['x'][None];rows=[]
 for K in [4,16,49]:
  D=init_dict(X,K,11);start=time.perf_counter();z,d=code(D,X,.015,maxiter=10,tol=.003)
  rows.append(dict(K=K,image_id=im['id'],native_shape=im['shape'],iterations=d['iterations'],converged=d['converged'],seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,finite=bool(np.isfinite(z).all()),status='budgeted single-image coding feasibility only; no dictionary training, no scientific score'))
 jwrite(directory('experiment_configs')/'experiment_status.json',{p.parent.name:json.loads(p.read_text())['status'] for p in (OUT/'checkpoints').glob('*/completed.json')})
 env=OUT/'experiment_configs/environment.json'
 if env.exists():
  environment=json.loads(env.read_text());environment['ADMM_reference_parallel_jobs']=4;jwrite(env,environment)
 jwrite(directory('experiment_configs')/'full_frame_feasibility.json',dict(rows=rows,full_frame_training=False,conclusion='Single-image coding supported, full multi-image optimization not performed; main results retain sampled native tiles.'))
 jwrite(directory('experiment_configs')/'profiling_summary.json',dict(full_frame_coding_pilot=rows,training_wall_times=[dict(id=p.parent.name,seconds=json.loads(p.read_text())['seconds'],peak_rss_MiB=json.loads(p.read_text())['peak_rss_MiB']) for p in (OUT/'checkpoints').glob('*/completed.json')],note='Shared 4-CPU host; no controlled full-training speedup claim. Fixed-problem accuracy-matched benchmark is separate.'))
if __name__=='__main__':main()
