"""`make verify` (EVALS.md, spec): the eight suites, the agent-path gates and the report-only
benchmarks, written to one JSON scorecard.

Everything runs offline by replay. A suite whose producer has not landed yet (L2 facts and
screening, L4 decide, L5 copilot) is `pending`, never passing: each pending case names the lane
and the hook it waits on, so wiring a lane in is a one-line change in that suite.
"""
