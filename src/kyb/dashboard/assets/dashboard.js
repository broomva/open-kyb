/* @ts-check */
/** @typedef {import('./contracts').DashboardSnapshot} Snapshot */
/** @typedef {import('./contracts').CaseDetail} CaseDetail */
/** @typedef {import('./contracts').RepairHandoff} RepairHandoff */
/** @typedef {import('./contracts').ChatIntent} ChatIntent */
/** @typedef {import('./contracts').HostCapabilities} HostCapabilities */
/** @typedef {import('./contracts').OpenSummary} OpenSummary */
(() => {
    'use strict';
    /** @returns {HTMLElement} */
    function el(id) { const node = document.getElementById(id); if (!node)
        throw new Error('Missing interface element: ' + id); return node; }
    /** @returns {HTMLInputElement} */
    function input(id) { const node = el(id); if (!(node instanceof HTMLInputElement))
        throw new Error('Missing input'); return node; }
    /** @returns {HTMLTextAreaElement} */
    function area(id) { const node = el(id); if (!(node instanceof HTMLTextAreaElement))
        throw new Error('Missing text field'); return node; }
    const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const button = (text, action, style = 'secondary', disabled = false) => `<button class="${style}" data-action="${action}" ${disabled ? 'disabled' : ''}>${esc(text)}</button>`;
    const row = (label, value, tone = '') => `<div class="row"><span>${esc(label)}</span><span class="${tone}">${esc(value)}</span></div>`;
    // An unknown is set as quiet text, never as a display numeral, so it cannot read as a measurement.
    const stat = (value, label) => `<div class="stat"><strong${typeof value === 'number' ? '' : ' class="unknown"'}>${esc(value ?? 'Not measured')}</strong><span>${esc(label)}</span></div>`;
    const meta = value => `<span class="meta">${esc(value)}</span>`;
    const asRecord = value => { if (value === null || typeof value !== 'object' || Array.isArray(value))
        throw new Error('The server returned an unreadable response. Refresh the dashboard.'); return /** @type {Record<string,unknown>} */ (value); };
    /** @returns {Snapshot} */
    function parseSnapshot(value) { const d = asRecord(value), w = asRecord(d.workspace_context); if (d.version !== 'kyb-dashboard-v2' || !Array.isArray(d.cases) || typeof w.id !== 'string' || typeof w.label !== 'string')
        throw new Error('Dashboard and server versions do not match. Reopen it with the matching MCP server.'); return /** @type {Snapshot} */ (value); }
    const previewNode = document.getElementById('preview-data');
    /** @type {Snapshot[]|null} */ const previews = previewNode ? JSON.parse(previewNode.textContent || '{}').snapshots.map(parseSnapshot) : null;
    const embedded = !previews && window.parent !== window;
    const query = previews ? new URLSearchParams() : new URLSearchParams(location.search);
    /** @type {Snapshot|null} */ let state = null;
    /** @type {RepairHandoff|ChatIntent|null} */ let pending = null;
    /** @type {HostCapabilities} */ let host = {};
    let selected = query.get('case'), pinned = query.get('run'), view = selected ? 'overview' : 'queue';
    let filter = 'all', search = '', questionDraft = '', ready = false, themeChosen = false;
    /** @type {string|null} */ let browserUrl = null;
    let opened = false;
    // After a compact open: records come through the host's tool proxy; without one, offer the
    // browser alternative rather than an empty panel
    function loadOpened() { if (state)
        return; if (host.serverTools) {
        refresh();
        return;
    } const browser = el('browser'); browser.hidden = !(browserUrl && host.openLinks); el('connection').innerHTML = '<strong>Tool connection unavailable</strong>This host cannot load records into the panel. ' + (browserUrl && host.openLinks ? 'Use the browser alternative.' : 'Open the browser link your agent shared.'); }
    let work = 0, generation = 0, refreshId = 0, recordError = false, delivery = 'not_sent', nextId = 0;
    /** @type {Map<number,{resolve:(value:unknown)=>void,reject:(error:Error)=>void,timer:number}>} */ const requests = new Map();
    const canMessage = () => Boolean(embedded && ready && host.serverTools && host.message?.text);
    const busy = () => work > 0;
    function note(text, error = false) { el('notice').textContent = text; el('notice').className = error ? 'error' : ''; }
    function notify(method, params) { window.parent.postMessage({ jsonrpc: '2.0', method, params }, '*'); }
    /** @returns {Promise<unknown>} */
    function rpc(method, params) { const id = ++nextId; return new Promise((resolve, reject) => { const timer = window.setTimeout(() => { requests.delete(id); reject(new Error('Delivery could not be confirmed. Check your chat before sending again.')); }, 15000); requests.set(id, { resolve, reject, timer }); window.parent.postMessage({ jsonrpc: '2.0', id, method, params }, '*'); }); }
    function unpack(value) { const result = asRecord(value); if (result.isError)
        throw new Error('The request was refused. Refresh the case and review its state.'); if (result.structuredContent)
        return result.structuredContent; const content = /** @type {Array<{type:string,text?:string}>} */ (result.content || []); const text = content.find(c => c.type === 'text')?.text; if (!text)
        throw new Error('No record was returned. Refresh the dashboard.'); return JSON.parse(text); }
    /** @returns {Promise<unknown>} */
    async function call(name, args = {}) {
        if (previews) {
            if (name !== 'dashboard_snapshot')
                throw new Error('This sample is read-only. Open the connected dashboard to send a real request.');
            const c = asRecord(args);
            const match = previews.find(s => c.case ? s.selected?.case === c.case && (!c.run || s.selected.run === c.run) : !s.selected)
                || (c.case && previews[0].cases.some(row => row.case === c.case && !row.runs.length) ? previews[0] : null);
            if (!match)
                throw new Error('That attempt is not included in this sample.');
            return structuredClone(match);
        }
        if (!['dashboard_snapshot', 'prepare_dashboard_action', 'dashboard_handoff', 'dashboard_intent'].includes(name))
            throw new Error('This action is not available in the dashboard.');
        if (embedded) {
            if (!ready || !host.serverTools)
                throw new Error('The MCP connection is unavailable. Ask your agent to reopen the dashboard.');
            return unpack(await rpc('tools/call', { name, arguments: args }));
        }
        const endpoint = { dashboard_snapshot: 'snapshot', prepare_dashboard_action: 'prepare', dashboard_intent: 'intent' }[name];
        if (!endpoint)
            throw new Error('The agent must revalidate this request in your chat.');
        const url = new URL(endpoint, location.href);
        url.search = '';
        const reading = endpoint === 'snapshot';
        if (reading)
            Object.entries(args).forEach(([key, value]) => { if (value !== null && value !== undefined)
                url.searchParams.set(key, String(value)); });
        const response = await fetch(url, { method: reading ? 'GET' : 'POST', cache: 'no-store', ...(reading ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(args) }) });
        const data = await response.json();
        if (!response.ok)
            throw new Error(data.error || 'Unable to load the record. Refresh and try again.');
        return data;
    }
    function syncURL() { if (embedded || previews)
        return; const u = new URL(location.href); for (const [key, value] of [['case', selected], ['run', pinned]]) {
        if (value)
            u.searchParams.set(key, value);
        else
            u.searchParams.delete(key);
    } history.replaceState(null, '', u); }
    function receive(value) { const data = parseSnapshot(value), extra = asRecord(value); if (typeof extra.browser_url === 'string')
        browserUrl = extra.browser_url; if (state && state.workspace_context.id !== data.workspace_context.id) {
        pending = null;
        selected = null;
        pinned = null;
        view = 'queue';
        note('The workspace changed. Review the new context before requesting work.');
    } state = data; if (data.selected) {
        selected = data.selected.case;
        pinned = data.selected.run;
    } el('workspace').textContent = data.workspace_context.label; syncURL(); render(); }
    async function refresh(showNotice = false) { const seq = ++refreshId; generation++; work++; try {
        const previous = state?.selected?.snapshot_id;
        const result = await call('dashboard_snapshot', { case: selected, run: pinned, workspace_id: state?.workspace_context.id });
        if (seq !== refreshId)
            return;
        recordError = false;
        receive(result);
        if (pending && previous !== state?.selected?.snapshot_id) {
            pending = null;
            delivery = 'not_sent';
            note('The evidence changed. Review this attempt before requesting work again.');
        }
        else if (showNotice)
            note('Records refreshed. Your selected attempt is unchanged.');
    }
    catch (error) {
        if (seq === refreshId) {
            recordError = true;
            note(error.message, true);
            if (!state)
                el('main').innerHTML = '<h1>Unable to load cases</h1><p>Return to the case list to choose an available record.</p>' + button('Open case list', 'reset');
        }
    }
    finally {
        work--;
        render();
    } }
    function navigate(next) { view = next; render(); window.scrollTo({ top: 0, behavior: 'instant' }); el('main').focus({ preventScroll: true }); }
    function selectCase(caseId, run = null) { generation++; selected = caseId; pinned = run; pending = null; questionDraft = ''; delivery = 'not_sent'; view = 'overview'; syncURL(); note(''); refresh(); render(); window.scrollTo({ top: 0, behavior: 'instant' }); }
    function selectedDetail() { return state?.selected?.case === selected && (!pinned || state.selected.run === pinned) ? state.selected : null; }
    const recommendationLabel = status => ({ APPROVED: 'Approved', REJECTED: 'Rejected', MANUAL_REVIEW: 'Manual review' })[status] || status;
    const readinessLabel = value => ({ ready: 'No blocking gaps identified', repair: 'Missing evidence can be retried', manual_review: 'Officer review needed', no_progress: 'Stopped: no new evidence', budget_exhausted: 'Retry limit reached', blocked: 'Evidence review blocked', unavailable: 'Evidence check not ready' })[value] || 'Not checked';
    const filters = [['all', 'Active'], ['needs_work', 'Needs work'], ['review_complete', 'Awaiting officer'], ['signed', 'Signed off'], ['rejected', 'Rejected recommendation'], ['archived', 'Archived']];
    /** @param {import('./contracts').CaseSummary} c @param {string} f */
    function inFilter(c, f) {
        const membership = f === 'archived' ? c.lifecycle.state === 'archived' : c.lifecycle.state !== 'archived';
        return membership && (['all', 'archived'].includes(f) || (f === 'rejected' ? c.recommendation?.status === 'REJECTED' : f === 'needs_work' ? ['needs_work', 'not_started', 'unavailable'].includes(c.process.state) : c.process.state === f));
    }
    const chevron = '<svg class="chev" viewBox="0 0 16 16" aria-hidden="true"><path d="m6 3.5 4.5 4.5L6 12.5"/></svg>';
    function caseCards(compact = false) {
        if (!state) return '';
        const rows = state.cases.filter(c => {
            if (compact) return c.lifecycle.state !== 'archived' || c.case === selected;
            const matchesSearch = !search || c.case.toLowerCase().includes(search.toLowerCase()) || c.display_name.toLowerCase().includes(search.toLowerCase());
            return inFilter(c, filter) && matchesSearch;
        });
        if (!rows.length) return '<p class="empty">' + (filter === 'archived' ? 'No archived cases match these filters.' : 'No active cases match these filters.') + '</p>' + button('Clear filters', 'clear-filters');
        return rows.map(c => {
            const r = c.recommendation, tone = r ? { APPROVED: 'good', REJECTED: 'bad', MANUAL_REVIEW: 'warn' }[r.status] : '';
            const lifecycle = c.lifecycle.state !== 'active' ? `<span class="badge">${c.lifecycle.state === 'archived' ? 'Archived' : 'Archive state unavailable'}</span>` : '';
            return `<button class="case-row" data-case="${esc(c.case)}" aria-pressed="${selected === c.case}" data-static><span><span class="case-name">${esc(c.display_name)}</span><span class="case-state"><i class="dot" data-state="${esc(c.process.state)}" aria-hidden="true"></i>${esc(c.process.label)}${c.latest_run ? ` <span class="meta">· ${esc(c.latest_run)}</span>` : ''}${lifecycle}</span></span>${r ? `<span class="case-rec"><small>Recommendation</small><span class="badge ${tone}">${esc(recommendationLabel(r.status))}</span></span>` : '<span class="case-rec"></span>'}${chevron}</button>`;
        }).join('');
    }
    function queueView() { if (!state)
        return ''; const counts = Object.fromEntries(filters.map(([id]) => [id, state.cases.filter(c => inFilter(c, id)).length])); return `<p class="kicker">Caseload</p><h1>Cases</h1><p class="lede">Choose a case to review its evidence, ask a question or request the next step.</p>${!state.cases.length ? '<section class="card"><h2>Add your first case in chat</h2><p>Ask your agent to review a packet with <code>kyb-review</code>. It will add the documents to this workspace. Return here and refresh to see the case.</p></section>' : `<div class="find"><label for="case-search" class="sr-only">Find a case</label><input id="case-search" type="search" value="${esc(search)}" placeholder="Find a case by name or reference"></div><div class="filters seg" role="group" aria-label="Filter cases">${filters.map(([id, label]) => `<button data-filter="${id}" aria-pressed="${filter === id}" data-static>${label}<span class="count">${counts[id]}</span></button>`).join('')}</div><div id="case-list">${caseCards()}</div>`}<details><summary>How the workspace works</summary><p>One workspace holds many cases. Each review attempt keeps its own evidence and history. Questions and work requests return to your officer chat; the agent uses the same workspace through MCP or <code>kyb -w &lt;workspace&gt;</code>.</p></details>`; }
    function caseHeader(title) { if (!state)
        return ''; const c = state.cases.find(c => c.case === selected), d = selectedDetail(); return `${button('← All cases', 'queue', 'back')}<p class="kicker">${esc(c?.display_name || selected)}</p><h1>${title}</h1>${c?.lifecycle.state === 'archived' ? '<p class="warn">Archived from the active queue. Evidence and history are preserved. Work already in progress may continue.</p>' : c?.lifecycle.state === 'unavailable' ? '<p class="error">Archive state is unavailable. Refresh the records before changing queue membership.</p>' : ''}${c?.runs.length ? `<div class="toolbar"><label>Review attempt<select id="run">${c.runs.map(r => `<option value="${esc(r)}" ${r === d?.run ? 'selected' : ''}>${esc(r)}${r === c.latest_run ? ' · latest' : ''}</option>`).join('')}</select></label><span class="badge">${d?.historical ? 'Earlier attempt' : 'Latest recorded'}</span></div>` : ''}${d?.historical ? '<p class="warn">You are viewing an earlier attempt. New work must start from the latest attempt.</p>' : ''}<div class="tabs seg" role="group" aria-label="Case sections">${[['overview', 'Overview'], ['evidence', 'Evidence'], ['result', 'Outcome'], ['evals', 'Checks']].map(([id, label]) => `<button data-action="${id}" aria-pressed="${view === id}" data-static>${label}</button>`).join('')}</div>`; }
    function noCase() { return '<h1>Choose a case</h1><p>Open a case to inspect its evidence and review attempts.</p>' + button('Open case list', 'queue'); }
    /** @param {CaseDetail|null} d */
    function retryReason(d) { if (!d?.loop)
        return 'Finish document review and verification before requesting an evidence retry.'; if (d.historical)
        return 'Select the latest review attempt before requesting work.'; if (d.loop.candidates.some(c => c.state !== 'supported'))
        return 'Unresolved identity matches need officer review before another retry.'; if (d.readiness !== 'repair')
        return readinessLabel(d.readiness) + '.'; return ''; }
    function overviewView() { if (!state)
        return ''; const d = selectedDetail(), c = state.cases.find(c => c.case === selected), rec = d?.recommendations.at(-1) || c?.recommendation; const retry = d && !retryReason(d); return caseHeader('Case review') + `<section class="card glass"><div class="status-grid"><div><small>Review progress</small><strong>${esc(d?.process.label || c?.process.label || 'Not started')}</strong></div><div><small>KYB recommendation</small><strong>${esc(rec ? recommendationLabel(rec.status) : 'Not issued')}</strong></div><div><small>Officer sign-off</small><strong>${esc(rec ? (rec.signature_state === 'signed' ? 'Signed' : 'Awaiting signature') : 'Not requested')}</strong></div></div></section><section class="next-step"><h2>${retry ? 'Retry missing evidence' : rec ? 'Review the recommendation' : 'Continue the case review'}</h2><p>${retry ? 'Some evidence is missing or failed source checks. Set retry limits, then ask your agent to review those gaps.' : rec ? 'Read the recorded recommendation and its supporting evidence. Only an officer can sign it.' : 'The agent prepares documents, reads them, verifies claims and applies policy. It will ask before continuing work.'}</p>${retry ? button('Review retry limits', 'action') : button(rec ? 'Review recommendation' : 'Continue review in chat', rec ? 'result' : 'continue', 'primary', Boolean(previews) || (embedded && (!ready || !host.serverTools)) || busy())}${previews && !retry ? '<p class="helper">This is a sample. Work requests require a connected dashboard.</p>' : ''}</section><h2>Ask about this case</h2><p>The agent answers in your chat using verified sources from this case and attempt.</p><label for="question">Question<textarea id="question" maxlength="2000" placeholder="For example: which evidence still needs review?">${esc(questionDraft)}</textarea></label>${button(canMessage() ? 'Ask in chat' : 'Create question for chat', 'question', 'primary', !d || Boolean(previews) || (embedded && (!ready || !host.serverTools)) || busy())}${!d ? '<p class="helper">Review the documents before asking evidence questions.</p>' : previews ? '<p class="helper">Questions are disabled in this sample. No MCP connection is attached.</p>' : ''}${button('Inspect evidence', 'evidence')}${button('View case activity', 'activity')}${membershipControls()}`; }
    function membershipControls() {
        const c = state?.cases.find(c => c.case === selected);
        if (!c) return '';
        const archived = c.lifecycle.state === 'archived';
        const disabled = Boolean(previews) || c.lifecycle.state === 'unavailable' || (embedded && (!ready || !host.serverTools)) || recordError || busy();
        const label = archived ? 'Restore case' : 'Archive case';
        return `<details><summary>Case queue</summary><p>${archived ? 'Return this case to the active queue without starting work.' : 'Remove this case from the active queue. Keep all evidence, attempts, recommendations, signatures and history. Work already in progress may continue.'}</p>${button(canMessage() ? label + ' in chat' : 'Create ' + (archived ? 'restore' : 'archive') + ' request', archived ? 'restore' : 'archive', 'secondary', disabled)}<p class="helper">${previews ? 'This sample cannot change queue membership.' : 'Your agent asks you to confirm in chat before changing queue membership.'}</p></details>`;
    }
    function evidenceView() { const d = selectedDetail(); if (!d)
        return caseHeader('Evidence') + '<p>No evidence report is recorded yet. Continue the review in chat first.</p>' + button('Open case overview', 'overview'); const gaps = d.loop?.gaps || []; const descriptions = { failed_claim: 'A claim failed its source check. A fresh attempt can look for stronger evidence.', uncovered: 'A material field is not covered by verified evidence.', jurisdiction: 'The record does not establish a country of incorporation.', nondisclosure: 'Controller information was explicitly withheld. Re-reading cannot resolve this.', conflict: 'The documents disagree. The competing values remain unresolved.', identity: 'A possible identity match needs review before it can change the facts.', assurance: 'Reader independence or verification assurance needs attention.' }; return caseHeader('Evidence') + `<p>See what is supported, what is missing and where each verified quote came from.</p><section class="next-step"><h2>${esc(readinessLabel(d.readiness))}</h2><p>${esc(retryReason(d) || 'A retry creates a separate review attempt. It does not change the existing evidence.')}</p>${button('Review retry limits', 'action', 'primary', Boolean(retryReason(d)))}</section><h2>Evidence gaps</h2>${gaps.length ? gaps.map(g => `<details><summary><span class="badge ${g.repairable ? 'warn' : ''}">${g.repairable ? 'Retry possible' : 'Officer review'}</span>${esc(g.field.replaceAll('_', ' '))}</summary><p>${esc(descriptions[g.kind] || 'This recorded finding needs review.')}</p><p class="meta">Document references: ${esc(g.doc_ids.join(', ') || 'not specified')}</p><details><summary>Recorded finding</summary><p>${esc(g.reason)}</p>${meta(g.gap_id)}</details></details>`).join('') : '<p>No blocking gaps were identified in the available report. This does not approve the applicant.</p>'}${d.loop?.candidates.length ? `<section class="card"><h2>Possible identity matches</h2><p>These suggestions have not changed the verified facts.</p>${d.loop.candidates.map(c => row(c.left + ' ↔ ' + c.right, c.state)).join('')}</section>` : ''}<h2>Source quotes</h2>${d.evidence.length ? d.evidence.map(e => `<details><summary>${esc(e.document_name)} · page ${e.page}<span class="badge">${e.verified ? 'Verified quote' : 'Unverified context'}</span></summary><blockquote>${esc(e.excerpt)}</blockquote><p class="meta">${esc(e.field || 'Context only')} · ${esc(e.evidence_id)}<br>Page coordinates: ${esc(e.bbox.join(', '))}</p></details>`).join('') : '<p>No verified source quotes are available for this attempt yet.</p>'}${button('Back to case overview', 'overview')}`; }
    function actionView() { const d = selectedDetail(); if (!d?.loop)
        return caseHeader('Retry missing evidence') + '<p>' + esc(retryReason(d)) + '</p>' + button('Open case overview', 'overview'); const inherited = Boolean(d.loop.previous_plan), budget = d.retry_budget || {}; return caseHeader('Retry missing evidence') + `<p>Ask your agent to revisit missing or failed evidence. Choose the limits before requesting work.</p><section class="card"><h2>Retry limits</h2><p>The agent may need to reread the whole packet. It must stop when these limits are reached, no new evidence is found, or a check refuses the action.</p><label>Maximum new attempts<input id="attempts" type="number" min="1" step="1" value="${esc(budget.max_attempts ?? 2)}" ${inherited ? 'readonly' : ''} aria-describedby="attempt-help"></label><p class="helper" id="attempt-help">Each attempt keeps its own evidence and history.${inherited ? ' These limits were inherited and cannot be increased.' : ''}</p><label>Time limit in seconds<input id="seconds" type="number" min="1" step="1" value="${esc(budget.max_elapsed_seconds ?? 900)}" ${inherited ? 'readonly' : ''} aria-describedby="time-help"></label><p class="helper" id="time-help">900 seconds is 15 minutes. Time runs from the saved plan, including time awaiting confirmation.</p><p class="field-error" id="retry-error" role="alert"></p>${row('Execution', 'Your current chat agent')}${row('Cost', 'Not available; no billing receipt')}${row('Signing', 'Separate officer approval')}${button(busy() ? 'Creating request…' : canMessage() ? 'Request retry in chat' : 'Create retry request', 'prepare', 'primary', Boolean(previews) || (embedded && (!ready || !host.serverTools)) || recordError || Boolean(retryReason(d)) || busy())}<p class="helper">${previews ? 'This sample cannot create requests. Open the connected dashboard to use this action.' : canMessage() ? 'Your agent will ask you to confirm in chat before starting.' : 'This browser cannot send to your chat. After creating the request, copy it into the same officer conversation.'}</p></section>${button('Back to evidence', 'evidence')}`; }
    function requestView() { if (!pending)
        return noCase(); const current = pending, retryBudget = 'budget' in current ? current.budget : null, retry = Boolean(retryBudget), kind = 'kind' in current ? current.kind : 'retry', title = retry ? 'Evidence retry request' : kind === 'question' ? 'Case question' : kind === 'archive_case' ? 'Archive request' : kind === 'restore_case' ? 'Restore request' : 'Review request'; return caseHeader(title) + `<section class="card"><h2>${delivery === 'sent' ? 'Sent to your chat' : delivery === 'unknown' ? 'Check your chat' : 'Ready to copy to chat'}</h2><p>${delivery === 'sent' ? (retry ? 'Your agent must ask for confirmation before starting the retry.' : ['archive_case', 'restore_case'].includes(kind) ? 'Your agent must ask you to confirm in chat before changing queue membership. Refresh records after it completes.' : 'The agent will respond in your conversation. This panel does not stream chat replies.') : delivery === 'unknown' ? 'Delivery was not confirmed. Check your conversation before resending this request.' : 'Copy this request into the conversation where your open-kyb agent is connected to this workspace.'}</p>${row('Case', pending.case)}${['archive_case', 'restore_case'].includes(kind) ? row('Scope', 'Whole case · all attempts') : row('Review attempt', pending.run || 'Not started')}${retry ? row('Maximum new attempts', retryBudget.max_attempts) + row('Time limit', retryBudget.max_elapsed_seconds + ' seconds') + row('Request expires', new Date('expires_at' in current ? current.expires_at : '').toLocaleTimeString()) : ''}${button('Copy request', 'copy', 'primary')}${canMessage() ? button('Send request to chat', 'send', 'secondary', delivery !== 'not_sent' || busy()) : ''}<details><summary>Request text and references</summary><textarea id="handoff" readonly aria-label="Request text">${esc(pending.message)}</textarea></details><p class="helper">${retry ? 'Saving or copying a request starts no run. The saved plan remains in the audit history. If already sent, cancel any pending confirmation in chat.' : ['archive_case', 'restore_case'].includes(kind) ? 'Only queue membership changes. Evidence and history stay intact; work in progress and recommendations/signatures are unchanged.' : 'A question does not authorize a new review or an officer signature.'}</p></section>${button('Back to case overview', 'overview')}`; }
    function timelineView(caseOnly = false) { if (!state)
        return ''; const t = state.timeline, events = t.events.filter(e => !caseOnly || e.case === selected); return (caseOnly ? caseHeader('Case activity') : '<p class="kicker">History</p><h1>Activity</h1>') + `<p>Saved events across ${caseOnly ? 'this case' : 'the workspace'}. An allocated attempt is not proof that an agent has finished it.</p>${!t.chain_verified ? `<p class="error">${esc(t.error || 'The activity chain could not be verified.')}</p>` : events.length ? `<ol class="timeline">${events.map(e => `<li><strong>${esc(e.label)}</strong><div class="scope">Case ${esc(e.case)}${e.run ? ' · attempt ' + esc(e.run) : ''}</div><time datetime="${esc(e.at)}">${esc(new Date(e.at).toLocaleString())}</time>${!caseOnly ? `<button data-case="${esc(e.case)}" data-run="${esc(e.run || '')}" data-static>Open case</button>` : ''}</li>`).join('')}</ol>` : '<p>No recorded case activity yet. New case and review events will appear after the agent records them.</p>'}${t.truncated ? '<p class="helper">Showing the latest 200 case events. Use the workspace audit log for the full history.</p>' : ''}`; }
    function resultView() { const d = selectedDetail(); if (!d)
        return caseHeader('Outcome') + '<p>No recommendation is available until the review is complete.</p>' + button('Open case overview', 'overview'); const metrics = d.loop?.metrics || {}; return caseHeader('Outcome') + `<p>The recorded KYB recommendation and evidence changes for this attempt.</p>${d.recommendations.length ? d.recommendations.map(r => `<section class="card"><h2>${esc(recommendationLabel(r.status))}</h2>${row('Officer sign-off', r.signature_state === 'signed' ? 'Signed' : 'Awaiting officer signature', r.signature_state === 'signed' ? '' : 'warn')}${row('Risk score', r.risk_score)}<p>Review completion does not replace officer sign-off. Ask the agent to explain the recorded recommendation with its sources.</p>${button('Explain in chat', 'recommendation', 'primary', Boolean(previews) || (embedded && (!ready || !host.serverTools)) || busy())}<details><summary>Decision and policy references</summary><p class="meta">Decision ${esc(r.decision_id)}<br>Policy ${esc(r.rules_version)}</p></details></section>`).join('') : '<section class="card"><h2>Recommendation not issued</h2><p>The review still needs work before code can issue a recommendation.</p>' + button('Open case overview', 'overview') + '</section>'}<h2>Evidence changes</h2><div class="stats">${stat(metrics.verified_assertions, 'Verified supported inputs')}${stat(metrics.unresolved_gaps, 'Remaining evidence gaps')}${stat(metrics.verified_assertion_delta ?? 'Not compared', 'Change from the previous attempt')}</div><p class="helper">Comparison is available only when the recorded attempt has a valid predecessor.</p><details><summary>Usage and cost</summary>${row('Whole-run tokens', 'Not measured')}${row('Whole-run provider calls', 'Not measured')}${row('Billed cost', 'Not available')}${row('Cost per new supported input', 'Not available')}<p class="helper">Missing measurements are not zero. Document receipts alone do not establish whole-run usage or billed cost.</p></details>`; }
    function checksView() { const d = selectedDetail(); if (!d)
        return caseHeader('Review checks') + '<p>Checks appear after document verification has been recorded.</p>'; return caseHeader('Review checks') + `<p>Checks for this case and attempt. Passing a source check is not an accuracy evaluation or release approval.</p><section class="card"><h2>Evidence checks</h2>${row('Evidence readiness', readinessLabel(d.readiness))}${row('Claims refused by verification', d.loop?.metrics.verification_refusals ?? 'Not available')}${row('Independent extraction accuracy', 'Not evaluated')}${row('Model drift', 'Not evaluated')}</section><h2>Reader and verifier records</h2>${d.agents.length ? d.agents.map(a => `<section class="card"><h3>${esc(a.document)}</h3>${row('Accepted source claims', a.verified + ' of ' + a.claims)}${row('Material fields matched', a.coverage_matched + ' of ' + a.coverage_total)}${row('Reader isolation', a.isolated ? 'Established' : 'Not established', a.isolated ? '' : 'warn')}${row('Verification assurance', a.assurance)}<p class="helper">These counts describe checks against sources. They do not measure extraction accuracy.</p><details><summary>Agent and check versions</summary><p class="meta">Reader ${esc(a.reader_model || 'not recorded')}<br>Verifier ${esc(a.witness_model || 'not recorded')}<br>${esc([a.comparator, a.binding, a.coverage_version].filter(Boolean).join(' / '))}</p></details></section>`).join('') : '<p>No reader or verification records are available yet.</p>'}<details><summary>Fact and diagnostic references</summary><p class="meta">${esc(d.loop?.version || 'Not available')}<br>${esc(d.loop?.fact_set_hash || 'No fact set recorded')}</p></details>`; }
    function render() { if (!state)
        return; const browser = el('browser'); browser.hidden = !(embedded && browserUrl && host.openLinks); el('connection').innerHTML = previews ? '<strong>Sample workspace · read-only preview</strong>Explore cases and screens. This sample has no MCP connection and cannot create requests.' : embedded ? (ready && host.serverTools ? '<strong>Connected through MCP</strong>Questions and work requests return to your chat. The agent must confirm work before starting.' : '<strong>Records displayed · tool connection unavailable</strong>Ask your agent to reopen the dashboard or use the browser alternative. Work requests are disabled until tools connect.') : '<strong>Connected browser · copy requests to chat</strong>This panel reads and saves workspace records. Copy questions and retry requests into the officer’s chat to run them.'; el('updated').textContent = 'Checked ' + new Date(state.observed_at).toLocaleTimeString(); el('queue').hidden = view === 'queue' || view === 'timeline';
        document.querySelector('.workspace').setAttribute('data-page', view);
        el('queue').innerHTML = '<p class="label">Cases in this workspace</p><div id="case-rail">' + caseCards(true) + '</div>'; const views = { queue: queueView, timeline: () => timelineView(false), overview: overviewView, evidence: evidenceView, action: actionView, chat: requestView, activity: () => timelineView(true), result: resultView, evals: checksView }; let html = (view === 'queue' || view === 'timeline') ? views[view]() : !selected ? noCase() : !state.cases.some(c => c.case === selected) || (busy() && !selectedDetail() && state.cases.some(c => c.case === selected && c.runs.length)) ? '<h1>Loading case</h1><p>Reading the selected attempt. No work has started.</p>' : views[view]?.() || overviewView(); const errors = [...state.errors, ...(selectedDetail()?.errors || [])]; if (errors.length)
        html += '<details><summary>Unavailable records</summary>' + errors.map(e => `<p>${esc(e)}</p>`).join('') + '</details>'; el('main').innerHTML = '<div class="screen">' + html + '</div>'; document.querySelectorAll('nav button').forEach(b => { const node = /** @type {HTMLButtonElement} */ (b); node.setAttribute('aria-current', node.dataset.view === (view === 'queue' ? 'queue' : view === 'timeline' ? 'timeline' : 'overview') ? 'page' : 'false'); }); window.kybMaterials?.refresh(); }
    function scope() { if (!state || !selected)
        throw new Error('Choose a case first.'); const d = selectedDetail(); return { case: selected, run: d?.run || pinned, workspace_id: state.workspace_context.id, snapshot_id: d?.snapshot_id || null, case_state_hash: state.cases.find(c => c.case === selected)?.lifecycle.event_hash || null }; }
    async function sendPending() { if (!pending || !canMessage() || delivery !== 'not_sent')
        return; const seq = generation, id = pending.request_id; work++; render(); try {
        let checked;
        if ('plan' in pending) {
            checked = /** @type {RepairHandoff} */ (await call('dashboard_handoff', { case: pending.case, plan_id: pending.plan_id, workspace_id: pending.workspace_id }));
            if (!checked.valid || checked.allocation)
                throw new Error('This retry is out of date or already allocated. Refresh the case before continuing.');
        }
        else {
            checked = /** @type {ChatIntent} */ (await call('dashboard_intent', { case: pending.case, run: pending.run, workspace_id: pending.workspace_id, snapshot_id: pending.snapshot_id, kind: pending.kind, question: pending.question, case_state_hash: pending.case_state_hash }));
        }
        if (seq !== generation || pending?.request_id !== id)
            throw new Error('The selected case changed. No request was sent.');
        delivery = 'unknown';
        await rpc('ui/message', { role: 'user', content: { type: 'text', text: checked.message } });
        delivery = 'sent';
        note('Request sent. Continue in your chat.');
    }
    catch (error) {
        note(error.message, true);
    }
    finally {
        work--;
        render();
    } }
    async function createIntent(kind) { if (previews || busy())
        return; const args = scope(); if (recordError)
        throw new Error('Refresh the case before sending a request.'); if (kind === 'question') {
        questionDraft = area('question').value.trim();
        if (!questionDraft) {
            note('Enter a question about this case.', true);
            area('question').focus();
            return;
        }
    } const seq = generation; work++; render(); try {
        const result = /** @type {ChatIntent} */ (await call('dashboard_intent', { ...args, kind, question: questionDraft }));
        if (seq !== generation)
            return;
        pending = result;
        delivery = 'not_sent';
        navigate('chat');
    }
    catch (error) {
        note(error.message, true);
    }
    finally {
        work--;
        render();
    } if (canMessage() && pending)
        await sendPending(); }
    async function act(action) {
        if (action === 'reset') {
            selected = null;
            pinned = null;
            pending = null;
            view = 'queue';
            syncURL();
            note('');
            await refresh();
            return;
        }
        if (['queue', 'timeline', 'overview', 'evidence', 'action', 'activity', 'result', 'evals'].includes(action)) {
            generation++;
            navigate(action);
            return;
        }
        if (action === 'clear-filters') {
            filter = 'all';
            search = '';
            render();
            return;
        }
        if (action === 'copy' && pending) {
            if ('expires_at' in pending && Date.now() >= Date.parse(pending.expires_at)) { note('This request expired. Review the retry limits and create a new request.', true); return; }
            try {
                await navigator.clipboard.writeText(pending.message);
                note('Request copied. Paste it into the officer conversation for this workspace.');
            }
            catch {
                const text = area('handoff');
                text.closest('details').open = true;
                text.focus();
                text.select();
                note('Select and copy the request text.');
            }
            return;
        }
        if (action === 'send') {
            await sendPending();
            return;
        }
        if (action === 'question' || action === 'continue' || action === 'recommendation' || action === 'archive' || action === 'restore') {
            await createIntent({ question: 'question', continue: 'continue_review', recommendation: 'review_recommendation', archive: 'archive_case', restore: 'restore_case' }[action]);
            return;
        }
        if (action === 'prepare') {
            if (previews || busy() || recordError)
                return;
            const attempts = input('attempts'), seconds = input('seconds');
            if (!attempts.reportValidity() || !seconds.reportValidity()) {
                el('retry-error').textContent = 'Choose a positive whole number for each retry limit.';
                return;
            }
            const d = selectedDetail();
            if (!d || retryReason(d))
                return;
            const {case_state_hash: _membership, ...context} = scope();
            const args = { ...context, run: d.run, snapshot_id: d.snapshot_id, max_attempts: Number(attempts.value), max_elapsed_seconds: Number(seconds.value) }, seq = generation;
            work++;
            render();
            try {
                const result = /** @type {RepairHandoff} */ (await call('prepare_dashboard_action', args));
                if (seq !== generation)
                    return;
                pending = result;
                delivery = 'not_sent';
                note('Retry request saved. No review has started.');
                navigate('chat');
            }
            catch (error) {
                note(error.message, true);
            }
            finally {
                work--;
                render();
            }
            if (canMessage() && pending)
                await sendPending();
        }
    }
    document.addEventListener('click', event => { if (!(event.target instanceof Element))
        return; const target = event.target.closest('button,a'); if (!(target instanceof HTMLElement))
        return; if (target instanceof HTMLButtonElement && target.disabled)
        return; if (target.dataset.case) {
        selectCase(target.dataset.case, target.dataset.run || null);
        return;
    } if (target.dataset.filter) {
        filter = target.dataset.filter;
        render();
        return;
    } if (target.dataset.view) {
        generation++;
        navigate(target.dataset.view);
        return;
    } if (target.dataset.action) {
        act(target.dataset.action).catch(error => note(error.message, true));
        return;
    } if (target.classList.contains('brand')) {
        event.preventDefault();
        generation++;
        navigate('queue');
    } });
    document.addEventListener('input', event => { if (event.target instanceof HTMLInputElement && event.target.id === 'case-search') {
        search = event.target.value;
        el('case-list').innerHTML = caseCards();
    } if (event.target instanceof HTMLTextAreaElement && event.target.id === 'question')
        questionDraft = event.target.value; });
    document.addEventListener('change', event => { if (event.target instanceof HTMLSelectElement && event.target.id === 'run')
        selectCase(selected, event.target.value); });
    el('refresh').onclick = () => refresh(true);
    el('browser').onclick = async () => { if (!browserUrl || !host.openLinks)
        return; const url = new URL(browserUrl); url.search = ''; if (selected)
        url.searchParams.set('case', selected); if (pinned)
        url.searchParams.set('run', pinned); try {
        await rpc('ui/open-link', { url: url.href });
    }
    catch (error) {
        note(error.message, true);
    } };
    function theme(value) { document.documentElement.dataset.theme = value; el('theme').setAttribute('aria-label', 'Switch to ' + (value === 'dark' ? 'light' : 'midnight') + ' mode'); window.kybMaterials?.invalidate(); }
    el('theme').onclick = () => { themeChosen = true; theme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'); };
    theme(matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    requestAnimationFrame(() => document.documentElement.classList.add('controls-ready'));
    window.addEventListener('message', event => { if (!embedded || event.source !== window.parent)
        return; const msg = asRecord(event.data); if (msg.jsonrpc !== '2.0')
        return; // A host request or notification can reuse a numeric id; only a result/error envelope answers ours.
        const response = typeof msg.method !== 'string' && ('result' in msg || 'error' in msg), id = typeof msg.id === 'number' ? msg.id : null, req = response && id !== null ? requests.get(id) : null; if (response && !req)
        return; if (req) {
        clearTimeout(req.timer);
        requests.delete(id);
        if (msg.error)
            req.reject(new Error(String(asRecord(msg.error).message || 'The host refused this request.')));
        else
            req.resolve(msg.result);
        return;
    } try {
        if (msg.method === 'ui/notifications/tool-result') {
            const data = unpack(msg.params), version = asRecord(data).version;
            if (version === 'kyb-dashboard-v2' && !state)
                receive(data);
            // open_dashboard returns a compact result; read the pinned snapshot through the host
            if (version === 'kyb-dashboard-open-v1' && !state) {
                const open = /** @type {OpenSummary} */ (data);
                selected = open.case ?? null;
                pinned = open.run ?? null;
                if (typeof open.browser_url === 'string')
                    browserUrl = open.browser_url;
                opened = true;
                if (ready)
                    loadOpened();
            }
        }
        if (msg.method === 'ui/notifications/host-context-changed' && !themeChosen) {
            const params = asRecord(msg.params);
            if (params.theme === 'dark' || params.theme === 'light')
                theme(params.theme);
        }
        if (msg.method === 'ping' || msg.method === 'ui/resource-teardown')
            window.parent.postMessage({ jsonrpc: '2.0', id: msg.id, result: {} }, '*');
    }
    catch (error) {
        note(error.message, true);
    } });
    if (embedded) {
        rpc('ui/initialize', { appInfo: { name: 'open-kyb case workspace', version: '2.0.0' }, appCapabilities: { availableDisplayModes: ['inline', 'fullscreen'] }, protocolVersion: '2025-06-18' }).then(value => { const result = asRecord(value); host = /** @type {HostCapabilities} */ (result.hostCapabilities || {}); ready = true; const ctx = asRecord(result.hostContext || {}); if (!themeChosen && (ctx.theme === 'dark' || ctx.theme === 'light'))
            theme(ctx.theme); notify('ui/notifications/initialized', {}); render(); if (opened)
            loadOpened(); }).catch(error => note('Unable to connect to the host. ' + error.message, true));
        new ResizeObserver(() => { if (ready)
            notify('ui/notifications/size-changed', { height: Math.ceil(document.body.getBoundingClientRect().height) }); }).observe(el('app'));
    }
    else {
        refresh();
    }
})();
