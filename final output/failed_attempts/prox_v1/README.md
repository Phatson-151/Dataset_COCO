# Superseded implementation attempts — not final results

These were newly run experiments in this task, but are scientifically invalid as final paper results because the undamped logistic Newton prox can cycle. K100 SCSC also overflowed in float32 D Gram/Woodbury arithmetic. Preserved as failure evidence. All supervised models are retrained with safeguarded scalar roots and double-precision D systems. Gamma-zero baseline does not use the logistic prox and was retained only when finite, with actual residuals and feasible objective descent.
