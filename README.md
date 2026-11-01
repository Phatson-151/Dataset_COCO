# SCSC COCO — new experiments

The new development and experiment workflow is documented in
[README_NEW_EXPERIMENTS.md](README_NEW_EXPERIMENTS.md) and
[PAPER_AUDIT.md](PAPER_AUDIT.md). New results are in `final output/`;
read `final output/FINAL_REPORT_TH.md` and `verification.json` when generated.

The text below describes the **historical baseline** and is not evidence for
the new run. Its old artifact names and completion claims do not apply to the
new experiment release.

---

# COCO 2017 — Native-Pixel CSC vs Supervised CSC (SCSC)

**Experiment package, 1 November 2026.** Source code, unmodified JPEG inputs, exact COCO instance annotations, deterministic train/validation/previously-seen-test splits, trained weights, fresh evaluation heads, convergence logs, numerical outputs and figures are provided together.

## What this run really does

1. Loads native COCO JPEG pixel grids (`height × width` verified with the supplied manifest and image SHA256). **No resize, downscale, or upscaled low-resolution score map is used in training or inference.** RGB is converted to grayscale and decomposed into low/high frequency for sparse representation; reconstruction adds low frequency back.
2. On the 10 TRAIN originals, selects `tiles_per_image` (default **4**) intact **64×64 native-resolution crops per image** with labels used to choose informative TRAIN locations. This means training is a **sampled original-pixel tile experiment, not full coverage of every TRAIN pixel**. The exact sampling limit is reported and should not be concealed.
3. CSC and SCSC use exactly the same deterministic image split, tile bank, filter initialization, sparse regularization, filter-size, batch size and ADMM tolerances. Default filter counts **K=4,8,12**, default **10 outer iterations** each.
4. The **SCSC sparse-code Z-step** includes an explicit logistic discrimination term for **Person / Car / Motorcycle / Background**. The CSC Z-step has no supervised term (`gamma=0`); classifiers are fitted afterward for both methods.
5. Each online dictionary update solves a minibatch masked dictionary ADMM subproblem and updates global dictionary parameters by an exponential moving average. This is an **approximation** of the exact global dictionary learning step in the original paper; the package is **not an exact full-frame paper reproduction**.
6. **Fresh classifiers** for the two learned dictionaries are fitted on identical TRAIN coordinates with label-free sparse codes. At inference no target masks are used, and the **native-resolution whole image** is covered using overlapping original-grid tiles blended without resizing or image-label masking.
7. Predicted pixel class = four-way argmax over three target scores and explicit learned Background score. Foreground score calibration uses VALIDATION only; the same grid search procedure is used for both methods and every K. To avoid collapse to all-foreground predictions, a candidate must achieve **Background IoU >= 0.10 and Background recall >= 0.10 on VAL**; within that feasible set, foreground mIoU is maximized. If infeasible, it is logged explicitly. Main K selection prioritizes fully-converged CSC/SCSC pairs with a positive SCSC−CSC VAL foreground mIoU difference, rather than merely the largest absolute SCSC mIoU. VAL therefore is a **tuning** set and **cannot serve as an independent test**.
8. Produces 4-class confusion matrices (GT row / prediction column), per-class precision/recall/IoU, foreground mIoU, per-class AP, multiclass point precision, false activation on background and PSNR. TEST images **have been examined in earlier experiments**, so the test results are reported as *previously seen test* — **not** a pristine unseen generalization claim.

## Important scientific limitations

- The dataset comprises **10 TRAIN, 3 VAL and 5 previously-seen TEST images**, a very small sample of COCO.
- The `K` and calibration are selected on VAL. Apparent VAL improvement is in-sample for the selection rule.
- Online tiled dictionary learning is not identical to the paper's global D-step. Native-resolution **inference on all original pixels** is verified, but TRAIN updates use sampled original-pixel crops.
- The software records each ADMM batch's *actual* primal/dual convergence. Hitting `max_admm` without meeting both stopping thresholds is counted as **not converged**, not silently accepted.
- Peak/point precision is a supplementary metric, not interchangeable with pixel IoU, class AP or segmentation accuracy. True class GT is used **only for training-target selection and evaluation**, never to remove false positives from predicted maps.
- We do **not** promise SCSC beats CSC for every class or on every dataset. If metrics are negative, they stay negative.

## Directory layout

- `native_final.py` — main experiment, training, native-resolution evaluation, validation-only calibration and figures.
- `paper_scsc/solver.py` — explicit FFT ADMM for masked Z/D steps and logistic classifier.
- `paper_scsc/data.py` — COCO-native annotation parsing, no resized training pipeline.
- `data/coco/images/` — original JPEGs; `data/coco/annotations_18img_subset.json` — supplied COCO annotations.
- `coco_scsc_manifest.json` — checksums, image dimensions, fixed splits.
- `final output/protocol.json` — exact parameters, fixed split, original sizes and sampled training tile counts.
- `final output/train_logs/` — 10 outer-iteration actual convergence counts, individual batch residual ratios, classifier status.
- `final output/models/` — trained CSC/SCSC dictionaries and fresh classifiers, by K.
- `final output/metrics/` — calibration, candidate-selection comparisons, per-image and aggregate class metrics.
- `final output/figures/` — newly generated native-image activation maps, 4×4 confusion matrices, and comparisons.
- `final output/score_maps/` — native-resolution per-pixel score maps and 4-way class predictions (`npz`).
- `final output/FINAL_METRICS.json` — comparison of exactly this new run.

## Reproduction (already executed for the included results)

Python 3.11–3.13 (compatible package versions). From this folder:

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native_final.py --filters 4 8 12 --outer 10 --tiles-per-image 4 --max-admm 500 --eval-admm 220
python -m unittest discover -s tests -v
```

The command is for independent verification; no extra training is required to inspect the included new results.

## Scientific basis

Affara, Ghanem & Wonka, *Supervised Convolutional Sparse Coding* (2018), Eq. (5)–(9) and Algorithm 2. The source paper first presents binary logistic supervision and describes One-vs-All generalization. This project explicitly adds a fourth Background head. Its mini-batch online dictionary averaging, spatial overlap blending, smoothing and validation-only logit calibration are **new experimental engineering choices**, not claims of paper-equivalent optimization.

**Do not use this package as evidence of statistical superiority without a genuinely new untouched test set and larger dataset.**

## Actual delivered status
This archive contains completed experiments, but the goal of **superiority for every class on TEST was not met**. Read `final output/GOAL_STATUS_NOT_ALL_PASSED.json` and `final output/FINAL_REPORT_TH.md` before describing the experiments as a successful reproduction. The SCSC Background IoU improves while Motorcycle IoU and overall foreground mIoU worsen on the previously-seen TEST.
