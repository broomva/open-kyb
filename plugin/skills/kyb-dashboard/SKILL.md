---
name: kyb-dashboard
description: Oversee open-kyb cases and workspace activity, answer case-bound questions, archive/restore cases, or request a bounded evidence retry from the inline/browser dashboard. Native chat confirmation stays separate from UI intent and officer signing.
---

# open-kyb dashboard

First check the connected server's actual tools: `open_dashboard`,
`dashboard_snapshot`, `dashboard_intent`, `prepare_dashboard_action`, `dashboard_handoff`,
`archive_case` and `restore_case`.
Installing this skill alone does not upgrade the MCP server. If tools are missing,
report a skill/server version mismatch and request a matching server build/reload;
do not invent a UI resource, simulate a successful connection or change shared
host registration on your own. An unreleased checkout needs `KYB_DEV_REPO` when
installing/launching the existing development server path.

Use `open_dashboard(surface="auto")`, optionally with `case` and `run`. If it refuses
with "no kyb workspace resolves", this project has no reviews yet: say so and offer to
review a packet with kyb-review, which creates the workspace. Opening is
read-only. Prefer the returned inline resource only when negotiated; otherwise show
the returned browser link. Never claim a returned URL has opened a browser. A local
URL is private to the machine running the server; it is not a hosted phone URL.

One connected workspace holds many cases. Cases are not separate filesystem
workspaces. The case list includes completed/signed recommendations as well as
unfinished reviews. Read the selected case and review attempt explicitly; never
apply an action to the last case mentioned in the chat by accident.

For a question or review request from the UI, call `dashboard_intent` with its exact
`workspace_id`, `case`, `run`, `snapshot_id`, `case_state_hash`, `kind` and question. It validates the
context without calling a provider or starting work. A workspace mismatch refuses:
use the correct connection rather than changing the server's workspace silently.

- `question`: call `ask(case, question, run, workspace_id)` and `record_answer` with
  that same context. Follow kyb-review's evidence/answer rules and show the accepted
  rendered answer verbatim. Reply in chat; the panel is not a second conversation.
- `continue_review`: inspect `review_status`, explain the next step and ask the
  officer's native confirmation before starting/continuing review work. Use the
  existing kyb-review workflow; preserve context and reader isolation.
- `review_recommendation`: explain the recorded recommendation through the cited
  answer flow. This request never authorizes a signature.

Archive/restore requests are case-wide queue membership changes, including all attempts.
They never stop work, delete evidence, change a recommendation or consume officer signing assent.
For `archive_case` / `restore_case` intent:

1. Revalidate `dashboard_intent` with the exact shown context and `case_state_hash`.
2. Keep only one native question pending. Ask **Archive case / Cancel** or
   **Restore case / Cancel**, naming the case and workspace. For archive, disclose
   removal from the active queue, preservation of all evidence/history and that
   work already in progress may continue. Restore starts no work.
3. On explicit native assent, revalidate the intent again, then call the corresponding
   `archive_case` / `restore_case` tool with `case`, `workspace_id` and
   `expected_event_hash=case_state_hash`. No answer, Cancel or a UI message is not assent.
   If queue membership changed, stop and request fresh confirmation; do not replace the pin.
4. Report the returned membership and refresh the case snapshot. On uncertain delivery,
   read the case lifecycle first: an already-applied target needs no second mutation.
   Never repeat an old archive answer after a later restore.

The CLI equivalents are `kyb -w WORKSPACE dashboard archive CASE --workspace-id ID`
and `dashboard restore CASE --workspace-id ID --expected-event-hash HASH`. A later
archive after restoration must also pass `--expected-event-hash HASH`. Shared
registration is unchanged. The browser exposes intent/copy only, no archive mutation endpoint.
`review_queue` and `kyb queue` omit archived cases by default; `include_archived=true`
or `--include-archived` explicitly includes their unsigned recommendations.

The dashboard's `prepare_dashboard_action` records only the existing evidence-loop
plan. It starts no run and makes no provider call. It is not authorization. An app's
`ui/message`, copied request, model statement or button press is also not assent.

For a staged request:

Keep only one native confirmation pending. If another UI intent arrives, resolve
or cancel the outstanding native question before presenting the next one. A
rendered app does not prove the host supports chat messages or native questions.

1. Call `dashboard_handoff(case, plan_id, workspace_id)` yourself. The request id is
   a workspace-bound identity; it is not the core plan id. Read the canonical plan, source
   run, fingerprint, attempt/time budget and allocation receipt. Refuse a stale
   or expired request; if already allocated, inspect the recorded run instead of dispatching.
2. Present **one native AskUserQuestion/request_user_input** with the exact case,
   source run, plan id, route (originating harness), maximum attempts and elapsed
   seconds, full-packet reread disclosure and mandatory stop rules. Choices:
   **Start bounded retry** / **Cancel**. No answer or Cancel starts nothing.
   If the host has no native question capability, stop with the prepared summary;
   do not invent an approval endpoint, accept a browser boolean or infer assent.
3. On an explicit native answer, call `dashboard_handoff` again. If context changed,
   validity is false or an allocation now exists, do not dispatch the old answer.
   Otherwise call existing `start_loop_repair(case, plan_id, workspace_id)`. It revalidates
   evidence, assurance, policy, lineage, elapsed time and budget under its lock.
4. Show each fresh run. Continue the existing `kyb-review` workflow with fresh
   isolated readers and witnesses. No model/account/provider fallback. Do not read
   raw documents into the parent or pass gap reasons/expected answers to readers.
   Follow recorded targets from `read_plan` only. Preserve the review date.
5. Use `plan_loop(case, run=<fresh>, previous_plan=<prior plan>, workspace_id=...)` for subsequent
   attempts. Preserve the original budget; never reset it or hide an attempt.
   Stop on ready/manual-review/no-progress/budget-exhausted/blocked, any refusal or
   unresolved candidate. A repair failure is reported, not retried indefinitely.
6. Refresh the dashboard with the explicit new run when the user selects it; never
   silently replace a historical selection. On uncertain transport failure, read
   `dashboard_handoff`/allocation receipts before any retry.

This confirmation covers only the displayed bounded repair. It never signs a
recommendation. Preserve `kyb-review`'s separate officer AskUserQuestion and signing
permission flow. Readiness, code recommendation and signature are separate states.
Do not present citation acceptance as accuracy, unknown usage as zero, a rate
estimate as billed cost, or run-level diagnostics as release acceptance.
