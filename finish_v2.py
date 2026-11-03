"""Render, independently validate and report completed v2 experiments."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference-recovery',action='store_true');p.add_argument('--package',action='store_true');args=p.parse_args()
 steps=[]
 if args.reference_recovery:steps.append(('admm_reference_recovery',['run_admm_reference_controls.py']))
 steps += [('additional_analysis',['analyze_v2.py']),('outer_monitor',['monitor_outer_validation.py']),('profile',['profile_v2.py']),('figures_tables',['generate_paper_outputs.py']),('verification_before_report',['verify_final.py']),('thai_report',['write_final_report.py']),('verification_final',['verify_final.py']),('validation_tests',['-m','unittest','discover','-s','tests','-v'])]

 folder=ROOT/'final output/logs';folder.mkdir(parents=True,exist_ok=True)
 for name,cmd in steps:
  print('START',name,flush=True)
  with (folder/f'{name}.log').open('w') as f:subprocess.run([sys.executable,*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('PASS',name,flush=True)
 subprocess.run([sys.executable,'hash_inventory_v2.py'],cwd=ROOT,check=True)
 if args.package:subprocess.run([sys.executable,'package_final.py'],cwd=ROOT,check=True)
 print('V2 FINALIZATION COMPLETE',flush=True)
if __name__=='__main__':main()
