# Output coverage against research/scsc-coco-final and the master prompt

| Evaluation | v2 artifact / implementation |
|---|---|
| Original image SHA256, dimensions, split integrity, official masks | experiment_configs/dataset_audit.json; native and protocol tests |
| Paper/ADMM audit | PAPER_AUDIT.md (historical audit clearly marked), README_V2.md, FINAL_REPORT_TH.md (new adaptations) |
| Gamma sweep incl. zero baseline | experiment_configs/selection.json; metrics/*_val.json; Table 1 |
| K sweep paired CSC/SCSC | K=4,8,12,16,32,49; fixed selected gamma; initial sweep and joint refinement retained |
| K=100 | Excluded at the user's request; no new K100 model/evaluation |
| N sweep | Nested N=1,5,10 at K=4, fixed VAL and TEST; no duplicated images |
| N>10 / independent new TEST / other-dataset inpainting | Not run: only 18 inputs and additional COCO access CONNECT 403 |
| Beta/alpha/filter-size controls | Staged .005/.015/.03, .1/1/10, 11/21; full recipe metrics retained |
| Feature/sampling controls | Signed-only vs energy/RGB context; 4 vs 12 crops/image; paired arms |
| Solver/loss ablations | FISTA/ADMM x softmax/OVA x CSC/SCSC, K=4; VAL results |
| Matched-objective solver timing | metrics/solver_benchmark.json: K4/16/49, three repeats, strict ADMM reference tolerance |
| Multiple seeds | 42/43/44, paired models; three_seed_TEST_metrics.csv and three_seed_summary.json |
| Training/inference code transfer | metrics/label_free_train_transfer.json; no labels in either encoder |
| Figure 3 | All learned filters for each paired K; evolution at 1/15/30; all 30 checkpoints/model saved |
| Figure 4, four classes x four panels | Dense raw class probabilities, binary and multiclass maps for every TEST image |
| Figure 4 clearer display | Black panels, fixed class colors/[0,1] probability scale; paired point overlays and fixed-scale probability differences |
| Point diagnostics | display_peaks_*.json: selection is GT-free; point precision uses GT only afterward for evaluation |
| Figure 5(a-f) | AP/PSNR vs gamma/K/N: macro and all classes, composite and individual high-resolution PNGs, source CSV |
| Figure 6(a-c) | True class-region PSNR vs gamma for Person/Car/Motorcycle |
| Table 1 | VAL gamma AP/PSNR, transposed table, selected TEST pair |
| Table 2 | Paired 50%-missing inpainting; full/missing PSNR, SSIM, relative error, plots and arrays |
| Complete multiclass metrics | Per-class AP/precision/recall/F1/IoU/support/TP/FP/FN; foreground/all mIoU, accuracy, Background FPR |
| Confusion/PR/IoU figures | Raw/normalized 4x4 confusion, PR curves, IoU comparison and CSVs |
| Reconstruction | Native grayscale highpass+lowpass, PSNR/SSIM/relative error, visual comparisons |
| Loss/convergence/runtime/memory | All histories, solver-specific stopping quantities, wall times and peak RSS; failures remain failures |
| Outer validation diagnostic | Fixed native centre crops, every outer checkpoint, training heads, separately labeled from final fresh heads |
| Coverage and dictionary diversity | native_training_coverage.csv and dictionary_diversity.csv |
| Bootstrap | Paired image bootstrap for primary seed; distinct from three training seeds |
| Full-frame feasibility/profiling | Single-image native coding budget pilot K4/16/49; not full-frame multi-image dictionary training |
| Historical broken-prox / K100 overflow experiments | Not rerun as v2 science: replaced by solver numerical regression tests and FISTA benchmark; preserved on v1 branch |
| Consistency/provenance | verify_final.py: source hashes, models, native arrays, metrics, paired masks, PNG integrity; SHA256SUMS.json |
| Thai final report and goal status | FINAL_REPORT_TH.md, PROJECT_STATUS_TH.md, GOAL_STATUS.json, REPORT_METRICS.json |
| Reproduction and archive | README_V2.md, requirements-v2-lock.txt, package_final.py; canonical files published individually in Git |

All artifact paths in this table without a repository prefix are under `final output/`.
Artifact consistency is not a claim of independent unseen superiority. Scientific
negative results and iteration-limit failures are retained and reported.
