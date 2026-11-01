"""Run one K candidate with identical native_final.py defaults; for parallel CPU orchestration."""
import argparse
from pathlib import Path
from native_final import ROOT,create_train_bank,manifest,train_model,jwrite
p=argparse.ArgumentParser();p.add_argument('--filters',type=int,nargs='+',default=[8]);p.add_argument('--outer',type=int,default=10);p.add_argument('--tile',type=int,default=64);p.add_argument('--stride',type=int,default=48)
p.add_argument('--tiles-per-image',type=int,default=4);p.add_argument('--batch',type=int,default=4);p.add_argument('--filter-size',type=int,default=11)
p.add_argument('--beta',type=float,default=.015);p.add_argument('--gamma',type=float,default=.05);p.add_argument('--alpha',type=float,default=.25)
p.add_argument('--dict-rate',type=float,default=.25);p.add_argument('--points',type=int,default=24);p.add_argument('--max-admm',type=int,default=500)
p.add_argument('--eval-admm',type=int,default=220);p.add_argument('--tol',type=float,default=.003);p.add_argument('--seed',type=int,default=42)
args=p.parse_args();out=ROOT/'final output';(out/'models').mkdir(parents=True,exist_ok=True);(out/'train_logs').mkdir(exist_ok=True)
bank,profile=create_train_bank(manifest()['train'],args.tile,args.tiles_per_image,args.filter_size,args.points,args.seed)
for K in args.filters:
 for arm in ('CSC','SCSC'):
  obj=train_model(bank,K,arm,args,out)
  jwrite(out/'models'/f'K{K}_{arm}_metadata.json',{k:v for k,v in obj.items() if k not in ('D','W','b')})
