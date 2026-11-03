"""Package all v2 code, inputs and canonical artifacts without caches/credentials."""
import hashlib,json,zipfile
from pathlib import Path
from native_final import jwrite
ROOT=Path(__file__).resolve().parent

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 sources=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.txt','.json']]
 sources += [p for folder in ['paper_scsc','tests','source_snapshots'] for p in (ROOT/folder).rglob('*.py')]
 sources=[p for p in sources if p.name!='RELEASE_MANIFEST.json']
 manifest=dict(protocol='V2',source_base_commit='80a44dd6001e542b1637e511d70fe23a7cbdaca1',publication_branch='research/scsc-coco-v2',source_file_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},note='Exact publication commit is the Git commit containing this manifest; content hashes avoid self-referential commit metadata.')
 jwrite(ROOT/'RELEASE_MANIFEST.json',manifest)
 files=[*sources,ROOT/'RELEASE_MANIFEST.json']
 for folder in ['data','references','final output']:
  files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix!='.lock' and 'raw_val' not in p.parts and '__pycache__' not in p.parts]
 destination=ROOT/'SCSC_COCO_V2_COMPLETE.zip'
 with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
  for p in sorted(set(files)):z.write(p,p.relative_to(ROOT))
 with zipfile.ZipFile(destination) as z:assert z.testzip() is None
 print(json.dumps(dict(archive=str(destination),bytes=destination.stat().st_size,sha256=digest(destination),files=len(set(files)))))
if __name__=='__main__':main()
