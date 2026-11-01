# Paper-to-code audit for the new experiments

Both PDFs in `references/` were read as extracted text (the extraction tool was
`pdftotext -layout`). Their SHA256 digests are recorded in the experiment metadata.
The SCSC reference is Affara, Ghanem and Wonka, arXiv:1804.02678v1; `ADMM.pdf`
is Wohlberg's *SPORCO: A Python package for standard and convolutional sparse
representations*, SciPy 2017. The latter is a software/method reference, not
a claim that SPORCO itself implements supervised CSC.

| Reference | Existing implementation problem | New implementation / validation |
|---|---|---|
| SCSC Eq. 1–3, masked reconstruction + l1 + unit ball | Symmetric `f//2` boundary mask with a non-centred FFT kernel retained wrapped pixels | `make_mask` excludes the first `f-1` rows/columns, matching causal top-left kernel support; compared against direct linear convolution |
| SCSC Eq. 5–8, sum logistic loss | Supervision exists, but the legacy training loop refits theta on label-free re-encodings every outer iteration | `experiments.train` fits theta on the **same supervised Z** after the global D update; only the final evaluation classifier is fitted on label-free codes |
| SCSC Algorithm 2, shared D update | `native_final.train_model` averages minibatch dictionaries via EMA | New runner uses one D subproblem over the complete fixed TRAIN tile bank; no EMA |
| SCSC Eq. 9–10, Fourier solve / Woodbury | Dense per-frequency K-by-K inverse tensor in Z | One inverse of I+W'W and rank-one frequency corrections; numerical scalar-reference test and adjoint test |
| SCSC Eq. 8, logistic proximal | Newton prox, without independent numerical regression test | Independent BFGS comparison of scalar supervised convex objective, including nonzero gamma |
| SCSC Eq. 7, classifier | L-BFGS instead of paper gradient descent | Explicitly retained solver substitution for the same convex objective; logs success, message and iterations; gradient finite-difference test |
| SPORCO pp. 2–3, ADMM residuals | Some prior runs hit iteration limits | Fixed atol=1e-4, rtol=.003; record raw primal/dual, their tolerances, objective and actual success; adaptive rho uses normalized residual balance and rescales scaled duals; ceiling 1000, never count ceiling as convergence |
| SPORCO p. 3, low-frequency handling | Image decomposed by FFT Tikhonov filter | Retained for ordinary reconstruction; sparse high-frequency component + stored low-frequency component, all grayscale on native pixel grids |
| COCO annotation semantics | Rounded polygons via OpenCV fillPoly differ from official masks | Official pycocotools rasterizer for polygons, compressed and uncompressed RLE; target-class crowd/overlap pixels ignored |
| SCSC p. 10, positive/negative sampling | Balanced sampled anchors, native tiles | Same deterministic positive/negative anchors for both arms; 24 of each sign/head/tile when available; four native 64x64 tiles per TRAIN image |
| SCSC Fig. 3 | No new comparable paired figures | All filters of each paired K, symmetric fixed grayscale limits; no filters selected by appearance |
| SCSC p. 11 / Fig. 4 | Legacy rendering emphasizes sparse points | Fresh classifier, dense raw OVA probabilities, binary argmax classes and multiclass maps; every TEST image, all 4 classes, fixed shared display range; native NPZ predictions accompany figures |
| SCSC Fig. 5 / Table 1 | Only K sweep | Staged gamma sweep, paired K sweep with gamma selected on VAL, nested N sweep with fixed VAL/TEST. AP pooled across valid pixels; class reconstruction PSNR uses GT masks for evaluation only |
| SCSC Fig. 6 | Not implemented | Reconstruction PSNR inside each Person/Car/Motorcycle GT region, not repeated whole-image PSNR |
| SCSC Table 2 / section 5.4 | Not implemented | Exact floor(50% pixels) missing mask, seeded per image and shared across arms. Lowpass computed from observed pixels only; masked highpass coding; report all-pixel and missing-pixel PSNR, SSIM, relative error |

## Declared adaptations and limitations

- Four OVA heads including explicit Background adapt the paper's binary setup
  using its proposed one-vs-all extension. Internal score order is Person, Car,
  Motorcycle, Background; saved final class IDs are Background=0, Person=1,
  Car=2, Motorcycle=3. OVA sigmoid values are **not** softmax probabilities and
  need not sum to 1. The fixed decision rule is argmax of the four logits.
- Sparse training and inference use **grayscale converted from native RGB**,
  not a multi-channel RGB dictionary. Original RGB is retained for visualization.
- TRAIN contains only sampled native pixels, not all pixels of each image.
  The global D step is exact as a convex subproblem **of the tile bank** up to
  its reported residuals, not exact full-frame dictionary learning. Full-frame
  pilot timings are documented separately. Nonconvex outer optimization has no
  asserted global-optimality guarantee and is fixed at 10 iterations.
- Sparse codes warm-start from the previous outer iterate and dictionary ADMM
  starts from the current feasible D. Scaled duals are reset between subproblems,
  rather than retained across all outer iterations; this affects efficiency and
  tolerance-limited iterates, not the convex subproblem's objective. Within an
  ADMM solve, adaptive-rho changes rescale duals to preserve unscaled multipliers.
- D-step acceptance additionally checks reconstruction descent of the actually
  returned feasible compact dictionary. A small full-domain d-v1 residual can
  amplify through large sparse codes; residual success alone may not protect the
  feasible dictionary's data objective. Rejected candidates remain failures and
  return the previous feasible D; the original and candidate costs are logged.
- Inference covers every native-resolution pixel with reflect-padded,
  overlapping tiles and uses only valid convolution interiors; no resizing,
  GT masking, smoothing, logit-offset tuning or TEST threshold selection.
  Padding supplies reflected image context at borders; tiled codes differ from
  global full-image sparse codes.
- beta=.015 (instead of paper .5) is in highpass grayscale [0,1] units;
  alpha=1 is as in the paper. Losses are sums, not means. gamma is swept over
  0,.05,.1,.15,.2,.5,1,2,5,10,100. This is a controlled adaptation, not a
  reproduction of the paper's numerical scores on other datasets.
- Target-class crowds and overlap between target classes are ignored for
  classification. Other-category objects and unannotated regions are Background
  by the task's definition; this does not establish perfect background annotation.
- Only 18 COCO images exist here: TRAIN 10, VAL 3, previously-seen TEST 5.
  The attempt to access `https://images.cocodataset.org/zips/val2017.zip`
  returned proxy CONNECT HTTP 403. No independent new holdout or N>=20 data
  were obtained. The test is not pristine and cannot establish unseen-data
  superiority. No cross-dataset / out-of-domain inpainting claim is made.
- One training seed (42); paired image-level bootstrap intervals describe
  variability on these five previously-seen images, not training-seed variability
  or evidence of population-level statistical significance.
- N sweep is a learning curve, not model selection: N=10 is used for final
  training. Gamma and K are selected by maximum pooled VAL foreground AP,
  not by seeking a positive SCSC-minus-CSC difference. All paired baselines
  are retained whether they win or lose. Failed and nonconverged runs remain
  visible in status records.
- The inpainting lowpass is normalized Gaussian interpolation from observed
  pixels (same for both arms), not the unavailable full-image lowpass. This is
  an explicitly shared preprocessing adaptation. No hidden original pixels
  enter reconstruction inference. Missing pixels are not restored from GT.

## Evidence and entry points

`python run_paper_experiments.py` runs the staged experiments. Every model has
configuration, tile locations and checksums, initialization seed, ten checkpoints,
full ADMM traces, classifier diagnostics, wall time and peak RSS. `model.npz`
contains both training and freshly fitted evaluation classifiers. Figures and
tables must be generated exclusively from these new outputs, never legacy JSON.
