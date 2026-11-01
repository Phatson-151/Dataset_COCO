# Data attribution and provenance

The 18 image files in this package are **original COCO 2017 JPEGs** bundled in the source project archive. The `coco_scsc_manifest.json` contains per-file SHA256 hashes and original pixel dimensions, checked by the experiment on load. The `annotations_18img_subset.json` contains COCO-derived instance annotations. No images are synthesized and no annotations are invented to improve results.

Dataset reference: Lin et al. (2014), **Microsoft COCO: Common Objects in Context**, ECCV. Image copyright and licensing belong to the corresponding rights holders; the COCO dataset does not imply that all underlying photographs have the same license. Redistribution for coursework should follow the applicable original-image licenses and institution requirements.

The image annotations are processed for only three target categories (Person, Car, Motorcycle); remaining non-target pixels are treated as Background for this experiment, except ignored ambiguous/crowd target-category pixels. This does not claim to segment all 80 COCO categories.
