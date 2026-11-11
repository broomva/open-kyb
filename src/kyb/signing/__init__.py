"""Officer sign-off (spec): the review queue, `sign_decision`, and the officers who may sign.

Every decision is a recommendation until an officer signs it. A signature is a separate ledger
record that references the `decision_id`; the decision itself is never edited.
"""
