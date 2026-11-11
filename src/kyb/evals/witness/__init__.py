"""The witness backend eval (report-only backend comparison).

It decides the witness default by data: the same recorded reader claims and layouts, and a small
hand-labelled crop set, re-witnessed by each backend. Only the witness differs. The production
comparator (`kyb.witness`) judges every response, so the eval measures what a backend would do
in the pipeline, not in isolation.

- `labelled`: the labelled crop set (correct-value rate, false accepts on the negatives);
- `fullrun`: a recorded run re-verified through `verify_document` (verified rate, agreement);
- `faults`: fault-injected witnesses that must make the eval fail (the eval's own check);
- `backends`: the API adapter and the dev/eval-only `agy` adapter, behind kyb's witness port;
- `record`: record/replay, so `make verify` runs offline.
"""
