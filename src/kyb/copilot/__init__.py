"""The copilot (agent-driven-review §5): answers only from the record.

`ask` hands the agent a verified evidence bundle; `record_answer` checks each structured
statement against the record and renders the text itself. The agent supplies no prose for a
factual statement, so there is nothing to fabricate and nothing to paraphrase.
"""
