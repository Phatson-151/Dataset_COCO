"""Seal final canonical artifacts after all logs have closed; verify every digest."""
import hashlib,json
from pathlib import Path
from native_final import jwrite
from paper_scsc.v2 import ROOT,OUT

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def main():
 files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json' and p.suffix!='.lock' and 'raw_val' not in p.parts]
 inventory={str(p.relative_to(ROOT)):digest(p) for p in files};jwrite(OUT/'SHA256SUMS.json',inventory)
 for name,expected in inventory.items():assert digest(ROOT/name)==expected,('Artifact changed while sealing',name)
 print(json.dumps(dict(canonical_files_hashed=len(inventory),all_digests_verified=True,exclusions=['self-referential SHA256SUMS.json','redundant raw_val cache','ephemeral *.lock'])))
if __name__=='__main__':main()
