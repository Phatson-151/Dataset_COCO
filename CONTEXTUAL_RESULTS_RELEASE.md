# SCSC COCO contextual results — v1.0.0

Branch: `research/scsc-coco-contextual-v1`  
Tag: `scsc-coco-contextual-v1.0.0`  
Parent research release: `80a44dd`

This release contains the complete new contextual experiment: source scripts,
trained multiclass heads, frozen dictionaries inherited from the parent release,
raw and spatially refined probability maps, predictions, sparse-code caches,
encoding diagnostics, logs, figures, configuration, validation grids, sampling
records, and Thai reports. All files in `rapid_context_output/` are included.

Start with [the Thai report](rapid_context_output/REPORT_TH.md),
[the class comparison figure](rapid_context_output/figures/final_metric_comparison.png),
and [the verification record](rapid_context_output/verification.json).

| Matched modified workflow | Foreground pooled AP | All-class pooled mIoU |
|---|---:|---:|
| CSC + contextual classifier + refinement | 0.248708 | 0.234874 |
| SCSC + contextual classifier + refinement | 0.265934 | 0.275133 |

These are frozen-dictionary extensions with newly trained nonlinear classifiers.
VAL selects supervision gamma 0.2 and common Gaussian refinement sigma 31.
Both arms receive the same contextual feature, classifier, and refinement
procedure. Raw probabilities before refinement are retained separately.

The five TEST images were previously seen. SCSC wins per-image mIoU on 3/5
images, and the paired bootstrap confidence interval crosses zero. Predictions
remain coarse, particularly for Person. This release does not establish unseen
generalization or statistically significant superiority.

The original experiment reports and `verification.json` preserve their creation-time
statement that the work was unpublished. The user subsequently authorized this
new branch and tag; that historical statement is not the publication status of
this release. Experimental evidence and its original checksums remain unchanged.

`rapid_context_output/SHA256SUMS.json` verifies the curated original experiment
packet. `CONTEXTUAL_RELEASE_MANIFEST.json` additionally inventories every new
output file, including caches and logs, plus the new experiment scripts and this
release documentation. It excludes itself to avoid a circular checksum.

Download the branch or tag as a GitHub ZIP for a complete repository snapshot.
The separate curated ZIP created in the cloud is an alternative package, not a
Git-tracked file; its SHA256 is recorded in the release manifest. GitHub's generated
source ZIP is a different archive and has a different checksum.

Reproduction commands are in [the experiment README](rapid_context_output/README_TH.md).
The earlier paper experiment is preserved under `final output/`. This release
does not modify `main`, the previous branch, or its experiment evidence.
