// Social Debate AI - frontend (streaming debate + BYOK + demo + shareable links)

const state = { topic: '', currentRound: 0, loading: false };
let elements = {};

const AGENT_META = {
  Agent_A: { type: 'support', name: 'Agent A (Support)', icon: 'fa-user-tie' },
  Agent_B: { type: 'oppose',  name: 'Agent B (Oppose)',  icon: 'fa-user-shield' },
  Agent_C: { type: 'neutral', name: 'Agent C (Neutral)', icon: 'fa-user-graduate' },
};
const agentMeta = (id) => AGENT_META[id] || { type: 'neutral', name: id, icon: 'fa-user' };

document.addEventListener('DOMContentLoaded', () => {
  elements = {
    topicInput: document.getElementById('topicInput'),
    topicDisplay: document.getElementById('topicDisplay'),
    currentRound: document.getElementById('currentRound'),
    debateStatus: document.getElementById('debateStatus'),
    debateContent: document.getElementById('debateContent'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    startBtn: document.getElementById('startBtn'),
  };
  if (elements.topicInput) {
    elements.topicInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') setTopic(); });
  }
  loadServerConfig();
  // If on /d/<id>, replay the shared debate.
  const m = window.location.pathname.match(/^\/d\/([a-f0-9]+)/);
  if (m) loadSharedDebate(m[1]);
});

// ---------- UI helpers ----------
function showLoading(text = 'Loading…') {
  if (!elements.loadingOverlay) return;
  elements.loadingText.textContent = text;
  elements.loadingOverlay.style.display = 'flex';
}
function hideLoading() { if (elements.loadingOverlay) elements.loadingOverlay.style.display = 'none'; }

function showMessage(message, type = 'info') {
  const div = document.createElement('div');
  div.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
  div.style.cssText = 'position:fixed;top:80px;right:20px;z-index:1050;min-width:300px;';
  div.innerHTML = `${message}<button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
  document.body.appendChild(div);
  setTimeout(() => div.remove(), 5000);
}

function updateUI() {
  if (elements.startBtn) elements.startBtn.disabled = state.loading || !state.topic;
  if (elements.debateStatus) {
    elements.debateStatus.textContent = state.loading ? 'Debating' : (state.topic ? 'Ready' : 'Idle');
  }
}

// ---------- Controls ----------
function setTopic() {
  const topic = (elements.topicInput.value || '').trim();
  if (!topic) { showMessage('Please enter a topic', 'warning'); return; }
  state.topic = topic;
  elements.topicDisplay.textContent = topic;
  elements.debateContent.innerHTML =
    `<div class="text-center py-5"><h4>${topic}</h4>
     <p class="text-muted">Click "Start" to begin the debate.</p></div>`;
  updateUI();
}

async function startDebate() {
  const topic = (elements.topicInput.value || state.topic || '').trim();
  if (!topic) { showMessage('Please enter a debate topic first', 'warning'); return; }
  state.topic = topic;
  elements.topicDisplay.textContent = topic;
  elements.debateContent.innerHTML = '';
  await streamDebate(topic);
}

function resetDebate() {
  state.topic = ''; state.currentRound = 0; state.loading = false;
  if (elements.topicInput) elements.topicInput.value = '';
  if (elements.topicDisplay) elements.topicDisplay.textContent = '';
  if (elements.currentRound) elements.currentRound.textContent = '0';
  elements.debateContent.innerHTML =
    `<div class="welcome-screen"><div class="welcome-icon"><i class="fas fa-robot"></i></div>
     <h3>Social Debate AI</h3><p>Enter a topic and start a streaming debate.</p></div>`;
  resetAgentStates();
  updateUI();
}

// ---------- BYOK (stored in the browser only) ----------
function byokPayload() {
  try {
    const v = JSON.parse(localStorage.getItem('sdai_byok') || '{}');
    const out = {};
    ['provider', 'model', 'base_url', 'api_key'].forEach((k) => { if (v[k]) out[k] = v[k]; });
    return Object.keys(out).length ? out : null;
  } catch { return null; }
}
function saveByok() {
  const v = ['Provider', 'Model', 'BaseUrl', 'ApiKey'].reduce((o, k) => {
    const el = document.getElementById('byok' + k);
    o[k === 'BaseUrl' ? 'base_url' : (k === 'ApiKey' ? 'api_key' : k.toLowerCase())] = el ? el.value : '';
    return o;
  }, {});
  localStorage.setItem('sdai_byok', JSON.stringify(v));
  showMessage('LLM settings saved in your browser only', 'success');
}
function clearByok() {
  localStorage.removeItem('sdai_byok');
  ['byokProvider', 'byokModel', 'byokBaseUrl', 'byokApiKey'].forEach((id) => {
    const el = document.getElementById(id); if (el) el.value = '';
  });
  showMessage('LLM settings cleared', 'info');
}

// ---------- Streaming ----------
function ensureRound(round) {
  let el = document.getElementById('round-' + round);
  if (!el) {
    el = document.createElement('div');
    el.className = 'debate-round'; el.id = 'round-' + round;
    el.innerHTML = `<div class="round-header"><div class="round-number">${round}</div><h4>Round ${round}</h4></div>`;
    elements.debateContent.appendChild(el);
  }
  return el;
}

function ensureBubble(round, agentId) {
  const id = `resp-${round}-${agentId}`;
  let el = document.getElementById(id);
  if (!el) {
    const meta = agentMeta(agentId);
    el = document.createElement('div');
    el.className = `ai-response ${meta.type}`; el.id = id;
    el.innerHTML = `
      <div class="response-header">
        <div class="agent-avatar ${meta.type}"><i class="fas ${meta.icon}"></i></div>
        <div><h5>${meta.name}</h5>
          <small class="text-muted" id="${id}-meta"><i class="fas fa-circle-notch fa-spin"></i> analyzing…</small></div>
      </div>
      <div class="response-content" id="${id}-content"></div>`;
    ensureRound(round).appendChild(el);
  }
  return el;
}

async function streamDebate(topic) {
  if (state.loading) { showMessage('A debate is already running', 'info'); return; }
  state.loading = true; updateUI();

  const body = { topic };
  const roundsEl = document.getElementById('roundsInput');
  if (roundsEl && roundsEl.value) body.max_rounds = parseInt(roundsEl.value, 10);
  const byok = byokPayload();
  if (byok) body.llm = byok;

  try {
    const resp = await fetch('/api/debate/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || ('HTTP ' + resp.status));
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop();
      for (const frame of frames) {
        const line = frame.split('\n').find((l) => l.startsWith('data:'));
        if (!line) continue;
        try { handleStreamEvent(JSON.parse(line.slice(5).trim())); } catch { /* ignore */ }
      }
    }
  } catch (e) {
    showMessage('Debate failed: ' + e.message, 'error');
  } finally {
    state.loading = false; updateUI();
  }
}

function handleStreamEvent(ev) {
  switch (ev.type) {
    case 'start':
      elements.debateContent.innerHTML = '';
      break;
    case 'turn_start':
      state.currentRound = ev.round;
      if (elements.currentRound) elements.currentRound.textContent = ev.round;
      ensureBubble(ev.round, ev.agent);
      break;
    case 'analysis': {
      const meta = document.getElementById(`resp-${ev.round}-${ev.agent}-meta`);
      if (meta) meta.innerHTML =
        `strategy: <b>${ev.strategy || '—'}</b> · evidence: ${(ev.evidence_confidence * 100).toFixed(0)}% · Δ-prob: ${(ev.delta_probability * 100).toFixed(0)}%`;
      if (ev.citations && ev.citations.length) {
        const id = `resp-${ev.round}-${ev.agent}`;
        const bubble = document.getElementById(id);
        if (bubble && !document.getElementById(id + '-cite')) {
          const cd = document.createElement('div');
          cd.id = id + '-cite';
          cd.className = 'evidence-citations small text-muted mt-2';
          cd.innerHTML = '<b>Evidence (RAG):</b><ul style="margin:.25rem 0 0 1rem">' +
            ev.citations.map((c) => `<li>${c.replace(/</g, '&lt;')}…</li>`).join('') + '</ul>';
          bubble.appendChild(cd);
        }
      }
      break;
    }
    case 'token': {
      const c = document.getElementById(`resp-${ev.round}-${ev.agent}-content`);
      if (c) { c.textContent += ev.text; elements.debateContent.scrollTop = elements.debateContent.scrollHeight; }
      break;
    }
    case 'turn_end': {
      const c = document.getElementById(`resp-${ev.round}-${ev.agent}-content`);
      if (c && ev.content) c.textContent = ev.content;
      if (ev.agent_states) updateAgentStates(ev.agent_states);
      break;
    }
    case 'summary':
      showDebateResult(ev.summary);
      showMessage('Debate complete', 'info');
      break;
    case 'saved': {
      const url = `${window.location.origin}/d/${ev.id}`;
      const div = document.createElement('div');
      div.className = 'alert alert-success mt-3';
      div.innerHTML = `Shareable link: <a href="${url}">${url}</a>`;
      elements.debateContent.appendChild(div);
      break;
    }
    case 'error':
      showMessage('LLM error: ' + (ev.message || 'unknown'), 'error');
      break;
  }
}

// ---------- Agent state bars + result ----------
function updateAgentStates(states) {
  if (!states || typeof states !== 'object') return;
  Object.entries(states).forEach(([agentId, st]) => {
    const suffix = agentId.split('_')[1];
    const stance = document.getElementById(`stance${suffix}`);
    if (stance && st.stance !== undefined) {
      stance.style.width = (((st.stance + 1) / 2) * 100).toFixed(0) + '%';
      const s = stance.querySelector('span');
      if (s) s.textContent = st.stance > 0 ? `+${st.stance.toFixed(2)}` : st.stance.toFixed(2);
    }
    const conv = document.getElementById(`conviction${suffix}`);
    if (conv && st.conviction !== undefined) {
      conv.style.width = (st.conviction * 100).toFixed(0) + '%';
      const s = conv.querySelector('span');
      if (s) s.textContent = st.conviction.toFixed(2);
    }
    if (st.has_surrendered) {
      const card = document.getElementById(`agent${suffix}`);
      if (card) card.style.opacity = '0.6';
    }
  });
}

function resetAgentStates() {
  const d = { A: ['80%', '+0.8', '70%', '0.7'], B: ['30%', '-0.6', '60%', '0.6'], C: ['50%', '0.0', '50%', '0.5'] };
  Object.entries(d).forEach(([k, v]) => {
    const st = document.getElementById('stance' + k), cv = document.getElementById('conviction' + k);
    if (st) { st.style.width = v[0]; st.querySelector('span').textContent = v[1]; }
    if (cv) { cv.style.width = v[2]; cv.querySelector('span').textContent = v[3]; }
  });
  document.querySelectorAll('.agent-card').forEach((c) => { c.style.opacity = '1'; });
}

function showDebateResult(summary) {
  if (!summary) return;
  let scores = '';
  if (summary.scores) {
    scores = Object.entries(summary.scores).sort(([, a], [, b]) => b - a)
      .map(([a, s]) => `<div class="mb-2">${a}: ${(s || 0).toFixed(1)}${a === summary.winner ? ' <span class="badge bg-warning ms-2">Winner</span>' : ''}</div>`)
      .join('');
  }
  const div = document.createElement('div');
  div.className = 'debate-result text-center py-5';
  div.innerHTML = `<h3>Result</h3><p class="lead">${summary.verdict || ''}</p>
    <div class="mt-4"><h5>Scores</h5>${scores}</div>`;
  elements.debateContent.appendChild(div);
}

// ---------- Demo + shared replay ----------
async function replayRounds(data) {
  state.topic = data.topic || 'Debate';
  if (elements.topicDisplay) elements.topicDisplay.textContent = state.topic;
  elements.debateContent.innerHTML = '';
  for (const round of (data.rounds || [])) {
    for (const r of (round.responses || [])) {
      ensureBubble(round.round, r.agent_id);
      const c = document.getElementById(`resp-${round.round}-${r.agent_id}-content`);
      if (c) c.textContent = r.content || '';
    }
    if (round.agents) updateAgentStates(round.agents);
  }
  if (data.summary) showDebateResult(data.summary);
}

async function runDemo() {
  try {
    showLoading('Loading demo…');
    const resp = await fetch('/api/demo');
    if (!resp.ok) throw new Error('Demo not available');
    const data = await resp.json();
    hideLoading();
    await replayRounds(data);
  } catch (e) { hideLoading(); showMessage('Demo failed: ' + e.message, 'error'); }
}

async function loadSharedDebate(id) {
  try {
    showLoading('Loading shared debate…');
    const resp = await fetch('/api/debate/' + id);
    if (!resp.ok) throw new Error('Debate not found');
    const data = await resp.json();
    hideLoading();
    await replayRounds(data);
  } catch (e) { hideLoading(); showMessage('Could not load shared debate: ' + e.message, 'error'); }
}

// ---------- Misc ----------
async function exportDebate() {
  try {
    const resp = await fetch('/api/export');
    const data = await resp.json();
    const blob = new Blob([JSON.stringify(data.data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'debates.json';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    showMessage('Exported recent debates', 'success');
  } catch (e) { showMessage('Export failed: ' + e.message, 'error'); }
}

async function loadServerConfig() {
  try {
    const cfg = await (await fetch('/api/config')).json();
    const badge = document.getElementById('llmBadge');
    if (badge) badge.textContent = `${cfg.provider}/${cfg.default_model}`;
  } catch { /* non-fatal */ }
}

function showStats() { showMessage('See the ablation results in docs/eval_results.md', 'info'); }
function showAbout() { showMessage('Social Debate AI — an LLM debate system built to measure (not assume) whether RAG/GNN/RL help. See docs/eval_results.md', 'info'); }
function toggleTheme() {
  document.body.classList.toggle('dark-theme');
  const icon = document.getElementById('themeIcon');
  if (icon) icon.className = document.body.classList.contains('dark-theme') ? 'fas fa-sun' : 'fas fa-moon';
}
