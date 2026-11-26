"""Hypothesis runners.

    runners/_common.py              shared inference + analysis machinery
    runners/h1_history_length.py    H1 - does the gap depend on history length
    runners/h2_query_type.py        H2 - does the gap depend on query type
    runners/h3_backbone_strength.py H3 - does the gap shrink with capability

Each runner has two stages. `--stage inference` runs one backbone and writes
one shard per (backbone, L); `--stage analyse` reads every shard and produces
the screen, the declared-family test, the secondary ablation and the
sensitivity grid. Nothing in the analysis stage touches a model, so it is
free to re-run.

See PREREGISTRATION.md §6 for what each hypothesis tests and why the
confirmatory test for H1 and H3 is an item-level difference-in-differences
rather than a rank test over the sweep axis.
"""
