These runs used a RANDOM 80/10/10 split of all USPTO-MIT reactions with a clean mapping (n_train 374,846, n_test 46,963),
because chem_experiment.splits failed to recognise the USPTO-MIT file (it looked for "val" reactions among the first
100,000 entries only). Their numbers (81.85 % and 85.69 % top-1) are NOT on the official test set, and the checkpoints
have seen most of the official test reactions during training. Do not report them. Fixed on 2026-09-20.
