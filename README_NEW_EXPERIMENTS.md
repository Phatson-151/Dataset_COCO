# SCSC COCO — new paper experiments

Entry points for the **new** experiments are `run_paper_experiments.py`,
`paper_scsc/experiments.py`, `generate_paper_outputs.py`, and `verify_final.py`.
`PAPER_AUDIT.md` compares the implementation with both supplied research PDFs.
The historical README and scripts describe an older run and are not evidence
for the new results. All newly generated results are under `final output/`.

## Environment and execution

Python 3.12; CPU implementation, no GPU required. Existing cloud virtualenv:
`/workspace/.venvs/dataset-coco`. For independent reproduction:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-experiment-lock.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
python -m unittest discover -s tests -v
python run_paper_experiments.py
python generate_paper_outputs.py
python verify_final.py
python write_final_report.py
python verify_final.py
python package_final.py
```

The sweep runner reuses **only completed checkpoints from its own new run**.
To reproduce from scratch preserve/move an existing `final output` directory
before running; never mix outputs from different source versions or protocols.
Configuration collisions are rejected; preserve the dataset and the references.
Parallel independent experiments use CPU subprocesses, each with one BLAS thread.

## Method and data

- 18 original COCO JPEGs, official COCO segmentation rasterization, checksums,
  disjoint fixed image IDs: TRAIN 10, VAL 3, previously-seen TEST 5.
- Native-pixel training bank: four 64x64 intact crops per image, no resizing.
  Algorithm 2 alternates supervised Z, global **bank-wide** D, and theta on the
  same Z. This is **sampled-tile training**, not full-image optimization.
- Four one-vs-all logistic heads; grayscale dictionaries; unit-ball constraints.
  FFT ADMM with Woodbury linear solves; actual primal/dual stopping tests.
- At evaluation both models receive a freshly fitted TRAIN classifier using
  label-free codes, as in paper section 5.2. Every original image pixel is
  predicted by overlapping, valid-interior native tiles with reflected borders.
- Fixed argmax logits; no GT-assisted prediction, smoothing or offset tuning.
- Public prediction IDs: Background=0, Person=1, Car=2, Motorcycle=3.
  Internal logit/OVA sigmoid channels: Person, Car, Motorcycle, Background.
  `probabilities` are individual OVA sigmoid scores, not a normalized softmax.
- Gamma selected on VAL at K=4, then K selected on VAL with that gamma.
  Criterion: pooled foreground macro AP. N uses nested TRAIN subsets at fixed
  K=4 and VAL-selected gamma, fixed VAL, and final N=10. The N curve is not a
  K-by-N grid. TEST never selects models or thresholds in this new run.
- TEST images were used in earlier work. They are **not an independent holdout**.
  No unseen-data or general statistical superiority claim is justified here.
- Inpainting removes exactly floor(50% of pixels), paired seeded masks; lowpass
  interpolation and masked sparse coding use observed pixels only.

## Output guide

| Folder/file | Evidence |
|---|---|
| `experiment_configs/` | Protocol, split IDs, image audit, versions, status and VAL-only selection |
| `checkpoints/<model>/` | Config, native crop locations, ten outer checkpoints, full ADMM histories, fresh and training classifiers |
| `score_maps/*.npz` | Native logits, OVA probabilities, both prediction ID conventions and reconstructions |
| `figures/figure3_dictionaries/` | All learned filters, paired K and dictionary evolution |
| `figures/figure4_multiclass/` | Every TEST image, 4 classes × 4 panels, raw scores, binary and multiclass predictions |
| `figures/figure5_ap_psnr/` | Gamma/K/N curves, aggregate and class-specific, combined and individual panels |
| `figures/figure6_class_psnr/` | Person/Car/Motorcycle GT-region reconstruction PSNR |
| `tables/` | Tables 1–2, full metrics, numeric curve data, runtime and convergence |
| `inpainting/` | Paired observed masks and reconstructions |
| `REPORT_METRICS.json` | New selected-model VAL and previously-seen TEST results |
| `FINAL_REPORT_TH.md` | Findings, failure analysis and limitations in Thai |
| `verification.json`, `SHA256SUMS.json` | Independent consistency checks and file hashes |

Models with iteration-limit failures must remain visible. “Completed” means the
experiment executed, not that every subproblem converged or SCSC beat CSC.
Read the convergence table and Thai report before drawing conclusions.

The final ZIP excludes virtualenvs, caches, Git metadata, unused legacy scripts,
and pilot-only checkpoints. It includes only new results plus the required code,
original 18-image dataset, annotations and the two reference PDFs.
