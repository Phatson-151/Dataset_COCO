"""Create and CRC-check the requested ZIP, excluding legacy outputs/caches."""
import hashlib
import json
import zipfile
from pathlib import Path
from paper_scsc.data import ROOT
from paper_scsc.experiments import OUT


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    verified=json.loads((OUT/'verification.json').read_text())
    assert verified['artifact_consistency_passed']
    assert (OUT/'FINAL_REPORT_TH.md').exists()
    archive=ROOT.parent/'SCSC_COCO_COMPLETE_PAPER_EXPERIMENTS_FINAL.zip'
    top='SCSC_COCO_PAPER_EXPERIMENTS_FINAL'
    entries={}
    source=['requirements.txt','PAPER_AUDIT.md','run_paper_experiments.py',
        'generate_paper_outputs.py','verify_final.py','write_final_report.py',
        'package_final.py','native_final.py','coco_scsc_manifest.json',
        'DATA_ATTRIBUTION.md','SCSC_COCO_MASTER_PROMPT_CODEX_CLOUD_REVISED.txt']
    for name in source:entries[name]=ROOT/name
    entries['README.md']=ROOT/'README_NEW_EXPERIMENTS.md'
    for path in (ROOT/'paper_scsc').glob('*.py'):entries[str(path.relative_to(ROOT))]=path
    for name in ['test_native_pipeline.py','test_numerical_solver.py','test_new_experiment_protocol.py']:
        entries[f'tests/{name}']=ROOT/'tests'/name
    for folder in [ROOT/'references',ROOT/'data',OUT]:
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix=='.lock' or 'pilot' in path.parts or '__pycache__' in path.parts:continue
            entries[str(path.relative_to(ROOT))]=path
    manifest={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for name,p in entries.items()}
    with zipfile.ZipFile(archive,'w',allowZip64=True) as z:
        for name,path in sorted(entries.items()):
            compress=zipfile.ZIP_STORED if path.suffix.lower() in ['.npz','.png','.jpg','.jpeg','.pdf'] else zipfile.ZIP_DEFLATED
            z.write(path,f'{top}/{name}',compress_type=compress,compresslevel=6 if compress==zipfile.ZIP_DEFLATED else None)
        z.writestr(f'{top}/RELEASE_MANIFEST.json',json.dumps(dict(files=manifest,
            artifact_consistency_passed=True,fully_verified_final=False,
            caution='Previously-seen TEST, sampled native tiles, one seed; read FINAL_REPORT_TH.md'),indent=2),compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip();assert bad is None,bad
        assert len(z.namelist())==len(entries)+1
    digest=sha(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n')
    print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest,
        entries=len(entries)+1,CRC_check='passed'),indent=2))


if __name__=='__main__':main()
