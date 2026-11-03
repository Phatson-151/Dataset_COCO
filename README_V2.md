# CSC/SCSC COCO v2

Read `final output/FINAL_REPORT_TH.md`, `GOAL_STATUS.json` and `verification.json`
for measured wins, losses and limitations. This is a task-driven extension of
Affara et al., not an exact reproduction of their binary Algorithm 2.

## Reproduce

Python 3.12, CPU, approximately 4 CPU cores and 16 GB RAM recommended.
Use the existing isolated checkout; do not create a worktree unless requested.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v2-lock.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export XDG_CACHE_HOME=/workspace/.cache MPLCONFIGDIR=/workspace/.cache/matplotlib
python -m unittest discover -s tests -v
python run_v2_experiments.py
python benchmark_v2.py
python analyze_v2.py
python monitor_outer_validation.py
python profile_v2.py
python generate_paper_outputs.py
python verify_final.py
python write_final_report.py
python verify_final.py
python -m unittest discover -s tests -v
python hash_inventory_v2.py
python package_final.py
```

Run from the repository root. Existing completed jobs are reused only when their
configuration matches. Independent reproduction from scratch requires preserving
and moving `final output` first; never mix v1 and v2 outputs. Training jobs write
source SHA256 hashes. Verification rejects a source mismatch unless an exact SHA256-matching source snapshot is preserved in source_snapshots/. The only versioned change during this run adds an optional ADMM D-step ceiling; FISTA computation is unchanged. An interrupted job
without a completed marker is restarted from its initialization; it is not claimed
to have resumed its optimizer state. Preserve its logs when diagnosing failures.

The run is staged: recipe search (beta/alpha/filter size), gamma sweep, paired K
sweep, a top-two-K joint gamma refinement followed by a fixed-gamma K curve, nested N curve, 2x2 solver/loss ablation, 3 seeds and feature/sampling
controls. K = 4,8,12,16,32,49 only. Gamma=100 is a classification weight and is
not K=100. TEST is evaluated after recipe/gamma/K selection is frozen.

## What changed

* Masked FISTA coding, projected FISTA reconstruction dictionary updates, a
  differentiable warmstarted 24-step FISTA encoder for the SCSC task gradient.
  Adam performs the extra task-driven dictionary/head update; no labels enter
  the encoder. This is truncated differentiation, not an exact implicit gradient; the spectral step size is detached during backpropagation and gradient tests freeze that same step.
* Thirty outer iterations, four reconstruction-only warmup iterations, gradual
  supervision ramp. Twelve 64x64 native crops/image, half ROI and half uniform.
* Both arms use signed/absolute/5px/15px pooled sparse features and RGB/9px RGB
  context. The reconstruction dictionary remains grayscale, not an RGB dictionary.
* Fresh multinomial linear softmax head; VAL CE temperature/bias calibration for
  each arm with identical fitting rules. VAL is tuning data. OVA controls retained.
* Primary selection maximizes all-class VAL mIoU, preferring Background IoU and
  recall >=0.1. AP is a tiebreaker. Seed42 is the primary display; seeds43/44 are
  reported without selecting a winning seed. All negative results remain visible.

Task mean loss and generative sum losses have different units; gamma cannot be
compared numerically to v1. The precise definitions are in `paper_scsc/v2.py`.
Do not describe FISTA mapping tests as ADMM residuals. The historical row keys
`z_admm`/`d_admm` are retained for plotting compatibility only; every FISTA
trace carries its true solver name and gradient-mapping stopping quantities.

## Outputs

All new results replace the old `final output` in this branch. The v1 branch
retains the original results. K=100 models and v1 failed-attempt artifacts are
not included as v2 evidence.

* Figures 3–6, Tables 1–2, dense class probabilities, binary/multiclass maps,
  all per-class metrics, raw/normalized confusion, PR and IoU charts,
  reconstruction and 50%-missing inpainting, convergence/loss/runtime/memory,
  native training coverage, per-outer native-centre-crop diagnostics.
* Figure 4 includes class-colored probability maps on black at shared [0,1]
  scale and `colored_points_black_*.png`: p>=0.35, 9x9 local maxima, same rule
  for both arms. This supplement never filters predictions with GT and never
  changes dense evaluation metrics. All five TEST images are shown.
* `tables/ablations_and_controls.csv`, `three_seed_TEST_metrics.csv`,
  `metrics/solver_benchmark.json`, `three_seed_summary.json`, and
  `label_free_train_transfer.json` explain what each change achieved.
* Full configs, crop coordinates, source hashes, 30 checkpoints per model,
  final models, raw/calibrated native scores and provenance SHA256 inventory.
  `raw_val/` is a redundant local cache and is excluded from Git; the canonical
  `score_maps` already include raw logits, calibrated probabilities and reconstructions.

## Limits

The only available dataset is 18 supplied COCO images: TRAIN 10, VAL 3,
previously-seen TEST 5. Additional COCO access returned proxy CONNECT 403.
N>10, a new independent holdout, RGB dictionary training and other-dataset
inpainting are not claimed. A single-image budgeted full-frame coding pilot
is not full-frame dictionary training. Artifact verification rereads saved arrays
and recomputes their metrics; it is not independent retraining or proof of
population-level superiority. `verify_final_v1.py` is retained solely as the
historical v1 validator, not the v2 entry point.

ADMM reference ablations use a D-step ceiling of 1000 at unchanged tolerances. Earlier ceiling160 pilots remain separately labeled; they are not treated as converged controls.
