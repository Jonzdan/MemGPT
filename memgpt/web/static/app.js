const API = '';

// ── View switching ──────────────────────────────────────────────
function switchView(name, el) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('view-' + name).classList.add('active');
  el.classList.add('active');

  if (name === 'overview') loadStats();
  if (name === 'sessions') loadSessions();
}

// ── Helpers ─────────────────────────────────────────────────────
function opClass(op = '') {
  if (op.includes('core')) return 'core';
  if (op.includes('recall')) return 'recall';
  if (op.includes('archival')) return 'archival';
  return 'other';
}

function opBadge(op) {
  const cls = opClass(op);
  return `<span class="op-badge op-${cls}">${op}</span>`;
}

function shortId(id = '') { return id.slice(0, 8) + '…'; }

function fmt(ts) {
  if (!ts) return '—';
  return new Date(ts + 'Z').toLocaleString();
}

function pctCell(pct) {
  if (pct == null) return '<td>—</td>';
  const w = Math.min(100, Math.round(pct * 100));
  return `<td><div class="pct-cell">
    <div class="pct-bar-wrap"><div class="pct-bar-fill" style="width:${w}%"></div></div>
    <span>${(pct * 100).toFixed(1)}%</span>
  </div></td>`;
}

function truncate(str, n = 80) {
  if (!str) return '—';
  try { str = JSON.parse(str); if (typeof str === 'object') str = JSON.stringify(str); } catch {}
  return String(str).slice(0, n) + (String(str).length > n ? '…' : '');
}

async function api(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ── Stats / Overview ────────────────────────────────────────────
async function loadStats() {
  try {
    const s = await api('/api/stats');
    document.getElementById('stat-total').textContent = s.total_ops ?? 0;
    document.getElementById('stat-sessions').textContent = s.total_sessions ?? 0;
    document.getElementById('stat-agents').textContent = s.total_agents ?? 0;
    document.getElementById('stat-attacks').textContent = s.total_attacks ?? 0;

    const total = (s.core_ops || 0) + (s.recall_ops || 0) + (s.archival_ops || 0) || 1;
    const corePct = ((s.core_ops || 0) / total * 100).toFixed(1);
    const recallPct = ((s.recall_ops || 0) / total * 100).toFixed(1);
    const archPct = ((s.archival_ops || 0) / total * 100).toFixed(1);

    document.getElementById('bar-core').style.width = corePct + '%';
    document.getElementById('bar-recall').style.width = recallPct + '%';
    document.getElementById('bar-archival').style.width = archPct + '%';
    document.getElementById('count-core').textContent = s.core_ops ?? 0;
    document.getElementById('count-recall').textContent = s.recall_ops ?? 0;
    document.getElementById('count-archival').textContent = s.archival_ops ?? 0;
  } catch (e) {
    console.error(e);
  }
}

// ── Sessions ────────────────────────────────────────────────────
async function loadSessions() {
  const sessions = await api('/api/sessions');
  const tbody = document.getElementById('sessions-body');
  if (!sessions.length) {
    tbody.innerHTML = '<tr><td colspan="11" class="empty-state">No sessions found.</td></tr>';
    return;
  }
  tbody.innerHTML = sessions.map(s => `
    <tr class="${s.attack_ops > 0 ? 'attack-row' : ''}">
      <td title="${s.session_id}">${shortId(s.session_id)}</td>
      <td title="${s.agent_id}">${shortId(s.agent_id)}</td>
      <td>${fmt(s.started_at)}</td>
      <td>${fmt(s.last_event)}</td>
      <td>${s.total_ops}</td>
      <td style="color:var(--core)">${s.core_ops}</td>
      <td style="color:var(--recall)">${s.recall_ops}</td>
      <td style="color:var(--archival)">${s.archival_ops}</td>
      <td style="color:var(--attack)">${s.attack_ops || 0}</td>
      <td>${s.attack_scenario || '—'}</td>
      <td style="display:flex;gap:6px">
        <button class="btn-sm" onclick="viewTimeline('${s.session_id}')">Timeline</button>
        <button class="btn-sm" onclick="openLabel('${s.session_id}')">Label</button>
      </td>
    </tr>
  `).join('');
}

function viewTimeline(sessionId) {
  document.getElementById('timeline-session').value = sessionId;
  switchView('timeline', document.querySelector('[data-view="timeline"]'));
  loadTimeline();
}

// ── Timeline ────────────────────────────────────────────────────
async function loadTimeline() {
  const sid = document.getElementById('timeline-session').value.trim();
  if (!sid) return;
  const entries = await api(`/api/timeline/${encodeURIComponent(sid)}`);
  const wrap = document.getElementById('timeline-wrap');
  if (!entries.length) {
    wrap.innerHTML = '<div class="empty-state">No entries for this session.</div>';
    return;
  }
  wrap.innerHTML = entries.map(e => {
    const cls = opClass(e.operation);
    let content = '—';
    try {
      const parsed = JSON.parse(e.content);
      content = parsed.content || parsed.new_value || parsed.query || JSON.stringify(parsed);
    } catch { content = e.content; }

    const pct = e.context_window_pct != null ? Math.min(100, Math.round(e.context_window_pct * 100)) : 0;
    const hasPct = e.context_window_pct != null;

    return `
    <div class="timeline-entry">
      <div style="display:flex;flex-direction:column;align-items:center;padding-top:8px">
        <div class="tl-dot ${cls}"></div>
        <div style="flex:1;width:1px;background:var(--border);margin-top:4px;min-height:8px"></div>
      </div>
      <div class="tl-card ${e.ground_truth_label ? 'attack' : ''}" onclick="showDiff(${e.id})">
        <div class="tl-top">
          <span class="tl-op ${cls}">${e.operation}</span>
          <div style="display:flex;gap:12px;align-items:center">
            ${hasPct ? `<span class="tl-pct">
              <span class="tl-pct-bar"><span class="tl-pct-fill" style="width:${pct}%"></span></span>
              ${(e.context_window_pct * 100).toFixed(1)}% ctx
            </span>` : ''}
            <span class="tl-seq">#${e.sequence_num}</span>
            <span class="tl-time">${fmt(e.timestamp)}</span>
          </div>
        </div>
        <div class="tl-content">${truncate(content, 120)}</div>
      </div>
    </div>
  `}).join('');
}

// ── Core Writes ─────────────────────────────────────────────────
async function loadWrites() {
  const hours = document.getElementById('writes-hours').value || 1;
  const agent = document.getElementById('writes-agent').value.trim();
  let url = `/api/recent-writes?hours=${hours}`;
  if (agent) url += `&agent_id=${encodeURIComponent(agent)}`;
  const rows = await api(url);
  const tbody = document.getElementById('writes-body');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No writes found.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => {
    let content = '—', field = '—';
    try {
      const p = JSON.parse(r.content);
      content = p.content || p.new_value || JSON.stringify(p);
      field = p.field || '—';
    } catch {}
    return `<tr>
      <td>${r.sequence_num}</td>
      <td>${fmt(r.timestamp)}</td>
      <td>${opBadge(r.operation)}</td>
      <td>${field}</td>
      <td title="${content}">${truncate(content)}</td>
      <td title="${r.previous_value || ''}">${truncate(r.previous_value)}</td>
      ${pctCell(r.context_window_pct)}
      <td><button class="btn-sm" onclick="showDiff(${r.id})">Diff</button></td>
    </tr>`;
  }).join('');
}

// ── Search ──────────────────────────────────────────────────────
async function doSearch() {
  const q = document.getElementById('search-q').value.trim();
  const sid = document.getElementById('search-session').value.trim();
  if (!q) return;
  let url = `/api/search?q=${encodeURIComponent(q)}`;
  if (sid) url += `&session_id=${encodeURIComponent(sid)}`;
  const rows = await api(url);
  const tbody = document.getElementById('search-body');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No results.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => `<tr>
    <td>${r.sequence_num}</td>
    <td>${fmt(r.timestamp)}</td>
    <td>${opBadge(r.operation)}</td>
    <td title="${r.session_id}">${shortId(r.session_id || '')}</td>
    <td title="${r.content}">${truncate(r.content)}</td>
    ${pctCell(r.context_window_pct)}
  </tr>`).join('');
}

document.getElementById('search-q').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

// ── Diff Modal ──────────────────────────────────────────────────
async function showDiff(id) {
  const d = await api(`/api/diff/${id}`);
  let newContent = d.new;
  try { newContent = JSON.stringify(JSON.parse(d.new), null, 2); } catch {}

  document.getElementById('modal-body').innerHTML = `
    <div class="diff-meta">
      <div class="diff-meta-item"><span>Operation: </span>${d.operation}</div>
      <div class="diff-meta-item"><span>Time: </span>${fmt(d.timestamp)}</div>
      <div class="diff-meta-item"><span>Seq: </span>#${d.sequence_num ?? '—'}</div>
      ${d.context_window_pct != null ? `<div class="diff-meta-item"><span>Context: </span>${(d.context_window_pct * 100).toFixed(1)}%</div>` : ''}
      ${d.token_offset != null ? `<div class="diff-meta-item"><span>Tokens: </span>${d.token_offset}</div>` : ''}
    </div>
    <div class="diff-section">
      <div class="diff-label">Previous Value</div>
      <div class="diff-box prev">${d.previous || '(none)'}</div>
    </div>
    <div class="diff-section">
      <div class="diff-label">New Value</div>
      <div class="diff-box curr">${newContent || '(none)'}</div>
    </div>
  `;
  document.getElementById('modal-overlay').classList.add('open');
}

function closeModal() {
  document.getElementById('modal-overlay').classList.remove('open');
}

// ── Label Modal ─────────────────────────────────────────────────
function openLabel(sessionId) {
  document.getElementById('label-session-id').value = sessionId;
  document.getElementById('label-overlay').classList.add('open');
}

function closeLabelModal() {
  document.getElementById('label-overlay').classList.remove('open');
}

async function submitLabel() {
  const sid = document.getElementById('label-session-id').value;
  const scenario = document.getElementById('label-scenario').value;
  const gt = document.getElementById('label-gt').value;
  await fetch(`/api/label/${encodeURIComponent(sid)}?attack_scenario=${encodeURIComponent(scenario)}&ground_truth_label=${gt}`, { method: 'POST' });
  closeLabelModal();
  loadSessions();
}

// ── Init ────────────────────────────────────────────────────────
loadStats();