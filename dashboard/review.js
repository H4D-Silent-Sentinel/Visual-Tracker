/* Shared helpers + Clip Review. Boxes are drawn from saved per-frame prediction files, so
   switching model output never touches the <video>: time and play state carry over. */
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const api = async (u, body) => (await fetch(u, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : undefined)).json();
const mmss = s => Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
const hhmm = t => new Date(t * 1000).toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
const store = {get: k => {try {return localStorage.getItem(k)} catch (e) {return null}}, set: (k, v) => {try {localStorage.setItem(k, v)} catch (e) {}}};
/* true when a key press belongs to a control, not to the page shortcuts */
const inControl = e => e.metaKey || e.ctrlKey || e.altKey || e.target.isContentEditable || /^(INPUT|SELECT|TEXTAREA|BUTTON|A)$/.test(e.target.tagName) || !!document.querySelector('dialog[open]');
let toastT;
function toast(msg) {const t = $('toast'); t.textContent = msg; t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => t.hidden = true, 2600)}
const iou = (b, g) => {const x1 = Math.max(b[0], g[0]), y1 = Math.max(b[1], g[1]), x2 = Math.min(b[2], g[2]), y2 = Math.min(b[3], g[3]);
  const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1), u = (b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - i; return u > 0 ? i / u : 0};

const R = {clips: [], byKey: {}, outputs: [], demo: [], clip: null, out: null, pred: null, gt: null, st: null, fa: null, frame: 0,
  truth: true, flags: [], cache: new Map(), counts: null, worst: null, group: '', scrubbing: false, base: null};
let TAB = 'review';
/* What each version is. One place, shown in Live, Clip Review and Results. */
const VERSIONS = [
  ['v1', 'Drones only', 'Trained on 5,200 photos of drones. The starting point: it finds drones, but it has never been shown anything that is not a drone, so it also boxes birds and kites.'],
  ['v2', 'v1 + birds and kites', 'v1 trained further with 2,100 bird and kite photos marked “no drone”. Aim: fewer false alarms on birds and kites.'],
  ['v3', 'v2 + cluttered backgrounds', 'v2 trained further with 2,393 frames from 10 tracking videos where the drone flies in front of trees and buildings. Aim: keep finding the drone when it is not against open sky. Scored only on the 10 videos it never saw.'],
  ['+ hold', 'A tracker setting, not a new model', 'The same model, but once a drone is locked the tracker keeps its box through weak detections (down to 8% confidence; a new target still needs 35%) and remembers a lost drone for 2 seconds instead of 1. Aim: fewer dropped locks in front of trees.'],
  ['Stock', 'Off-the-shelf YOLO (Live tab only)', 'The model as downloaded, with no drone class. It can only call a drone “airplane”, “bird” or “kite”. For comparison.']];
function versionsHTML(training) {
  const st = Object.fromEntries((training || []).map(t => [t.model, t.state + (t.state === 'training' ? `, pass ${t.epochs_done + 1} of ${t.epochs_planned}` : '')]));
  return '<div class="tw"><table><tr><th>Version</th><th>What it is</th><th>What it does</th><th>Status</th></tr>' +
    VERSIONS.map(([id, what, does]) => `<tr><td><b>${id}</b></td><td>${what}</td><td>${does}</td><td class="mono">${esc(st[id] || (id === '+ hold' ? 'ready when its model is' : id === 'Stock' ? 'ready' : '—'))}</td></tr>`).join('') + '</table></div>';
}
function fillVersions(training) {R.training = training; document.querySelectorAll('.verBody').forEach(el => el.innerHTML = versionsHTML(training))}

/* ---------------- tabs and routing ---------------- */
function parseHash() {const h = location.hash.slice(1), [tab, qs] = h.split('?'); return {tab: ['live', 'review', 'results', 'pool', 'collect'].includes(tab) ? tab : 'review', q: new URLSearchParams(qs || '')}}
function setHash() {
  let h = '#' + TAB;
  if (TAB === 'review' && R.clip) h += `?clip=${R.clip.key}&f=${R.frame}&out=${R.out}`;
  history.replaceState(null, '', h);
}
function show(tab) {
  TAB = tab;
  document.querySelectorAll('nav button').forEach(b => b.dataset.tab === tab ? b.setAttribute('aria-current', 'page') : b.removeAttribute('aria-current'));
  ['live', 'review', 'results', 'pool', 'collect'].forEach(x => $('tab-' + x).hidden = x !== tab);
  if (tab !== 'review') $('vid').pause();
  if (tab === 'results') resultsLoad();
  if (tab === 'pool') poolLoad();
  if (tab === 'collect') collectLoad(); else if (window.collectLeave) collectLeave();
  if (tab === 'live') liveEnter(); else if (window.liveLeave) liveLeave();
  setHash();
}
function openClip(key, frame, out) {show('review'); loadClip(key, {frame, out}); scrollTo(0, 0)}

async function boot() {
  document.querySelectorAll('nav button').forEach(b => b.onclick = () => show(b.dataset.tab));
  const d = await api('/api/clips');
  R.clips = d.clips; R.outputs = d.outputs; R.demo = d.demo; R.byKey = Object.fromEntries(d.clips.map(c => [c.key, c]));
  try {R.flags = (await api('/api/collect')).flags} catch (e) {}
  undoButton(R.flags.length ? R.flags.reduce((a, b) => (b.time || 0) > (a.time || 0) ? b : a) : null);
  const groups = [...new Map(R.clips.map(c => [c.group, c.group_label])).entries()];
  $('groupSel').innerHTML = '<option value="">All types</option>' + groups.map(([g, l]) => `<option value="${g}">${esc(l)}</option>`).join('');
  $('groupSel').onchange = e => {R.group = e.target.value; renderLists()};
  wirePlayer();
  const {tab, q} = parseHash();
  const first = (q.get('clip') && R.byKey[q.get('clip')]) ? q.get('clip') : ((R.demo.find(x => R.byKey[x.clip]) || {}).clip || (R.clips[0] || {}).key);
  if (first) await loadClip(first, {frame: q.has('f') ? +q.get('f') : null, out: q.get('out')});
  else $('cTitle').textContent = 'No browser-playable clips yet. Run make_web_clips.py.';
  show(tab);
  chip(); setInterval(chip, 20000);
  addEventListener('resize', () => {drawBase(); renderFrame(R.frame)});
}
async function refreshClips() {
  const d = await api('/api/clips');
  R.clips = d.clips; R.outputs = d.outputs; R.byKey = Object.fromEntries(d.clips.map(c => [c.key, c]));
  if (R.clip) R.clip = R.byKey[R.clip.key] || R.clip;
  renderOuts(); renderLists();
}
async function chip() {
  try {
    const d = await api('/api/results'), parts = [];
    fillVersions(d.training);
    for (const t of d.training) {
      if (t.state === 'training') parts.push(`${t.label} training · pass ${t.epochs_done + 1} of ${t.epochs_planned}` + (t.eta_s != null ? ` · about ${Math.max(1, Math.round(t.eta_s / 60))} min left` : '') + (t.last_update ? ` · log updated ${hhmm(t.last_update)}` : ''));
      else if (t.state === 'queued') parts.push(`${t.label} queued`);
    }
    if (d.queue.make_review) parts.push('review files: generating or waiting for training');
    $('trainChip').textContent = parts.length ? parts.join('  |  ') : 'No training running';
    const sig = d.training.map(t => t.state).join() + '|' + JSON.stringify(d.queue);
    if (R.sig && R.sig !== sig) {await refreshClips(); if (R.clip) setOutput(R.out, true)}
    R.sig = sig;
    if (!R.lastOutSig) R.lastOutSig = '';
  } catch (e) {$('trainChip').textContent = 'Server offline'}
}

/* ---------------- clip + output loading ---------------- */
const outInfo = id => R.outputs.find(o => o.id === id);
function outReason(c, id) {
  const o = outInfo(id);
  if (c.outputs[id].available) return 'frame data';
  if (c.outputs[id].summary) return 'clip totals only';
  return 'unavailable';
}
function whyUnavailable(c, id) {
  const o = outInfo(id);
  if (!o.model_usable) return `${o.label}: the ${o.model} model is still ${o.model_state === 'training' ? 'training' : o.model_state}.`;
  return `${o.label}: boxes for each frame are not generated yet (queued after v3 finishes training, about 3:45 AM).`;
}
async function loadClip(key, opt = {}) {
  const c = R.byKey[key]; if (!c) return;
  const v = $('vid'), sameClip = R.clip && R.clip.key === key;
  R.clip = c;
  $('cTitle').textContent = c.name;
  $('cTags').innerHTML = `<span class="tag">${esc(c.group_label)}</span> <span class="tag">${esc(c.source)}</span> ` +
    (c.split ? `<span class="tag ${c.split === 'held out' ? 'held' : 'trained'}">${c.split === 'held out' ? '✓ held out' : '! v3 trained on it'}</span> ` : '') +
    `<span class="tag">${c.truth === 'boxes' ? 'truth: drone box per frame' : c.truth === 'no_drone' ? 'truth: no drone in clip' : 'no ground truth'}</span>`;
  $('player').style.aspectRatio = `${c.w} / ${c.h}`;
  $('player').style.maxWidth = `calc(${c.w >= c.h ? 60 : 70}vh * ${(c.w / c.h).toFixed(4)})`;
  $('scrub').max = c.frames - 1;
  if (!sameClip) {v.pause(); v.src = c.url; v.load(); R.frame = 0}
  R.gt = null;
  if (c.has_gt) {
    const gk = 'gt|' + key;
    if (!R.cache.has(gk)) R.cache.set(gk, await api('/api/gt?clip=' + key));
    const g = R.cache.get(gk);
    R.gt = (g.timebase_ok && g.frames === c.frames) ? g.gt : null;
  }
  const want = [opt.out, R.out, store.get('out')].find(id => id && c.outputs[id] && (c.outputs[id].available || c.outputs[id].summary))
    || R.outputs.map(o => o.id).find(id => c.outputs[id].available) || R.outputs.map(o => o.id).find(id => c.outputs[id].summary) || R.outputs[0].id;
  await setOutput(want, true);
  let f = opt.frame;
  if (f == null && !sameClip) f = 0;
  if (f != null) {
    const go = () => seekFrame(f);
    if (v.readyState >= 1) go(); else v.addEventListener('loadedmetadata', go, {once: true});
  }
  renderLists(); setHash();
}
async function setOutput(id, quiet) {
  const c = R.clip; R.out = id; store.set('out', id);
  R.pred = null;
  let problem = null;
  if (c.outputs[id].available) {
    const k = c.key + '|' + id;
    if (!R.cache.has(k)) R.cache.set(k, await api(`/api/pred?clip=${c.key}&out=${id}`));
    const p = R.cache.get(k);
    if (p.boxes && p.frames === c.frames && p.timebase_ok !== false && c.timebase_ok) R.pred = p;
    else problem = `${outInfo(id).label}: saved boxes do not line up with this video (${p.frames} vs ${c.frames} frames), so they are not shown.`;
  } else problem = whyUnavailable(c, id);
  $('pMsg').hidden = !!R.pred; $('pMsg').textContent = R.pred ? '' : problem + ' The video and scrubber still work; boxes, timeline marks and the confidence graph appear once that data exists.';
  $('confGraph').style.display = $('timeline').style.display = R.pred ? '' : 'none';
  $('rvTele').style.display = R.pred ? '' : 'none';
  $('cShowing').textContent = 'Showing: ' + outInfo(id).label + (R.pred ? '' : ' (no boxes)');
  computeStatus(); drawBase(); renderOuts(); renderFrame(R.frame); renderSummary(); renderLists();
  if (!quiet) setHash();
}

/* per-frame status: 0 no data · 1 no box · 2 box shown · 3 on target · 4 miss · 5 false alarm · 6 nothing to find, no box */
function computeStatus() {
  const c = R.clip, N = c.frames, st = new Uint8Array(N), fa = new Uint8Array(N), p = R.pred;
  const n = {vis: 0, on: 0, miss: 0, fa: 0, box: 0, longest: 0};
  let worst = null, run = 0, runStart = 0, top = 0;
  if (p) for (let k = 0; k < N; k++) {
    const bs = p.boxes[k] || [];
    if (bs.length) n.box++;
    if (c.truth === 'boxes' && R.gt) {
      const g = R.gt[k];
      if (g) {
        n.vis++;
        if (bs.some(b => iou(b, g) >= 0.3)) {st[k] = 3; n.on++; run = 0} else {
          st[k] = 4; n.miss++; if (!run) runStart = k; run++;
          if (run > n.longest) {n.longest = run; worst = runStart}
        }
        if (bs.length && !bs.some(b => iou(b, g) > 0.1)) fa[k] = 1;
      } else {run = 0; if (bs.length) {st[k] = 5; fa[k] = 1} else st[k] = 6}
    } else if (c.truth === 'no_drone') {
      if (bs.length) {st[k] = 5; fa[k] = 1; const m = Math.max(...bs.map(b => b[4])); if (m > top) {top = m; worst = k}} else st[k] = 6;
    } else {
      st[k] = bs.length ? 2 : 1;
      if (worst == null && k > 0 && st[k] === 1 && st[k - 1] === 2) worst = k;
    }
    if (fa[k]) n.fa++;
  }
  R.st = st; R.fa = fa; R.counts = n; R.worst = worst;
  /* running share of frames so far that are on target (or, without truth, have a box) + top confidence per frame */
  const good = c.truth === 'boxes' && R.gt ? 3 : 2, denomOk = k => (c.truth === 'boxes' && R.gt ? (st[k] === 3 || st[k] === 4) : true);
  const cum = new Float32Array(N), conf = new Float32Array(N); let a = 0, b = 0;
  for (let k = 0; k < N; k++) {
    if (p) {const bs = p.boxes[k] || []; conf[k] = bs.length ? Math.max(...bs.map(x => x[4])) : 0;
      if (denomOk(k)) {b++; if (st[k] === good || (c.truth === 'no_drone' && st[k] === 5)) a++}}
    cum[k] = b ? 100 * a / b : -1;
  }
  R.cum = cum; R.conf = conf;
}
const COL = {0: '#1b1e21', 1: '#3a4046', 2: '#6cc5b9', 3: '#4f8f68', 4: '#ef6b62', 5: '#e6b04a', 6: '#262a2e'};
function drawBase() {
  const cv = $('timeline'), dpr = devicePixelRatio || 1, W = Math.max(1, Math.round(cv.clientWidth * dpr)), H = Math.round(30 * dpr);
  cv.width = W; cv.height = H;
  const b = document.createElement('canvas'); b.width = W; b.height = H;
  const g = b.getContext('2d'), c = R.clip; if (!c) return;
  const N = c.frames, st = R.st, pr = [0, 2, 3, 4, 6, 5, 1];  /* draw priority: miss > false alarm > on target > box > empty */
  for (let x = 0; x < W; x++) {
    const a = Math.floor(x * N / W), z = Math.max(a + 1, Math.floor((x + 1) * N / W));
    let best = 0, anyFa = 0;
    for (let k = a; k < z && k < N; k++) {if (pr[st[k]] > pr[best]) best = st[k]; if (R.fa[k]) anyFa = 1}
    g.fillStyle = COL[best]; g.fillRect(x, 0, 1, H);
    if (anyFa && best !== 5) {g.fillStyle = COL[5]; g.fillRect(x, 0, 1, Math.round(6 * dpr))}
  }
  g.fillStyle = '#fff';
  for (const f of R.flags) if (f.clip === c.key) g.fillRect(Math.round(f.frame * W / N) - 1, H - Math.round(8 * dpr), Math.max(2, Math.round(2 * dpr)), Math.round(8 * dpr));
  R.base = b;
  /* confidence graph for the whole clip */
  const gc = $('confGraph'), gh = Math.round(44 * dpr); gc.width = W; gc.height = gh;
  const cb = document.createElement('canvas'); cb.width = W; cb.height = gh;
  const q = cb.getContext('2d');
  q.strokeStyle = '#2e3338'; q.lineWidth = 1; [0.25, 0.5, 0.75].forEach(y => {q.beginPath(); q.moveTo(0, gh * y); q.lineTo(W, gh * y); q.stroke()});
  if (R.pred) {
    q.beginPath(); q.moveTo(0, gh);
    for (let x = 0; x < W; x++) {const a = Math.floor(x * N / W), z = Math.max(a + 1, Math.floor((x + 1) * N / W)); let m = 0; for (let k = a; k < z && k < N; k++) m = Math.max(m, R.conf[k]); q.lineTo(x, gh - m * (gh - 2))}
    q.lineTo(W, gh); q.closePath(); q.fillStyle = 'rgba(108,197,185,.22)'; q.fill(); q.strokeStyle = '#6cc5b9'; q.lineWidth = 1.5 * dpr; q.stroke();
    q.fillStyle = '#9ba1a8'; q.font = `${Math.round(10 * dpr)}px ui-monospace,Menlo,monospace`; q.fillText('confidence 0-100%', 6 * dpr, 12 * dpr);
  } else {q.fillStyle = '#6f767e'; q.font = `${Math.round(11 * dpr)}px ui-monospace,Menlo,monospace`; q.fillText('confidence graph appears once per-frame boxes exist for this output', 8 * dpr, gh / 2 + 4 * dpr)}
  R.cbase = cb;
  const L = c.truth === 'boxes' ? [[3, '● on target'], [4, '✕ miss (drone in view, no box on it)'], [5, '▲ false alarm (box with no drone under it)'], [6, 'drone out of view']]
    : c.truth === 'no_drone' ? [[5, '▲ false alarm (any box; this clip has no drone)'], [6, '○ no box (correct)']] : [[2, '■ box shown'], [1, '□ no box']];
  $('legend').innerHTML = (R.pred ? L.map(([k, t]) => `<span><i style="background:${COL[k]}"></i>${t}</span>`).join('') + '<span><i style="background:#fff;width:3px"></i>flagged frame</span>'
    : '');
  drawHead();
}
function drawHead() {
  const cv = $('timeline'), g = cv.getContext('2d'); if (!R.base || !R.clip) return;
  g.drawImage(R.base, 0, 0);
  const x = Math.round((R.frame + 0.5) * cv.width / R.clip.frames), w = Math.max(2, Math.round(2 * (devicePixelRatio || 1)));
  g.fillStyle = '#000'; g.fillRect(x - w, 0, w * 2, cv.height); g.fillStyle = '#fff'; g.fillRect(x - w / 2, 0, w, cv.height);
  const gc = $('confGraph'), q = gc.getContext('2d'); if (!R.cbase) return;
  q.drawImage(R.cbase, 0, 0); q.fillStyle = '#fff'; q.fillRect(x - w / 2, 0, w, gc.height);
  if (R.pred && R.conf[R.frame] > 0) {q.beginPath(); q.arc(x, gc.height - R.conf[R.frame] * (gc.height - 2), 4 * (devicePixelRatio || 1), 0, 7); q.fillStyle = '#6cc5b9'; q.fill()}
}
const WORDS = {0: '', 1: '□ NO BOX', 2: '■ BOX SHOWN', 3: '● ON TARGET', 4: '✕ MISS: drone in view, no box on it',
  5: '▲ FALSE ALARM: box with no drone under it', 6: '○ nothing to find in this frame, no box (correct)'};
const WCLS = {3: 'ok', 4: 'miss', 5: 'fa'};
function renderFrame(k) {
  const c = R.clip; if (!c) return;
  const cv = $('ovl'), dpr = devicePixelRatio || 1, W = Math.round(cv.clientWidth * dpr), H = Math.round(cv.clientHeight * dpr);
  if (cv.width !== W || cv.height !== H) {cv.width = W; cv.height = H}
  const g = cv.getContext('2d'); g.clearRect(0, 0, W, H);
  g.font = `${Math.round(12 * dpr)}px ui-monospace,Menlo,monospace`; g.textBaseline = 'bottom';
  const label = (t, x, y, col) => {const tw = g.measureText(t).width, hh = 16 * dpr; y = Math.max(hh, y); g.fillStyle = 'rgba(20,22,24,.85)'; g.fillRect(x, y - hh, tw + 8 * dpr, hh); g.fillStyle = col; g.fillText(t, x + 4 * dpr, y - 2 * dpr)};
  if (R.truth && R.gt && R.gt[k]) {
    const b = R.gt[k], x = b[0] * W, y = b[1] * H, w = (b[2] - b[0]) * W, h = (b[3] - b[1]) * H, pad = 5 * dpr;
    g.setLineDash([5 * dpr, 4 * dpr]); g.strokeStyle = '#fff'; g.lineWidth = 1.5 * dpr; g.strokeRect(x - pad, y - pad, w + 2 * pad, h + 2 * pad); g.setLineDash([]);
    label('truth', x - pad, y + h + pad + 17 * dpr, '#fff');
  }
  const bs = R.pred ? (R.pred.boxes[k] || []) : [];
  for (const b of bs) {
    const x1 = b[0] * W, y1 = b[1] * H, x2 = b[2] * W, y2 = b[3] * H, L = Math.max(6 * dpr, Math.min(x2 - x1, y2 - y1) * 0.3);
    g.strokeStyle = '#6cc5b9'; g.lineWidth = 2.5 * dpr; g.beginPath();
    for (const [cx, cy, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x2, y2, -1, -1], [x1, y2, 1, -1]]) {g.moveTo(cx + dx * L, cy); g.lineTo(cx, cy); g.lineTo(cx, cy + dy * L)}
    g.stroke();
    label(`${outInfo(R.out).label} ${Math.round(b[4] * 100)}%` + (b[5] != null ? ` · ID ${b[5]}` : ''), x1, y1 - 4 * dpr, '#6cc5b9');
  }
  const s = R.st ? R.st[k] : 0;
  const pill = $('vPill'), topc = bs.length ? Math.round(Math.max(...bs.map(b => b[4])) * 100) + '%' : '';
  const P = {1: ['□ NO BOX', ''], 2: ['■ BOX ' + topc, 'box'], 3: ['● ON TARGET ' + topc, 'ok'], 4: ['✕ MISS', 'miss'], 5: ['▲ FALSE ALARM ' + topc, 'fa'], 6: ['○ NO DRONE IN VIEW', '']};
  pill.hidden = !R.pred || !P[s];
  if (P[s]) {pill.textContent = P[s][0]; pill.className = 'vpill ' + P[s][1]}
  $('fInfo').innerHTML = `<span class="${WCLS[s] || ''}">${WORDS[s]}</span>` + (R.fa && R.fa[k] && (s === 3 || s === 4) ? ' <span class="fa">+ ▲ false-alarm box elsewhere in the frame</span>' : '')
    + (R.flags.some(f => f.clip === c.key && f.frame === k) ? `${s ? ' · ' : ''}⚑ this frame is flagged (${R.flags.filter(f => f.clip === c.key && f.frame === k).map(f => f.kind === 'miss' ? 'miss' : 'false alarm').join(', ')})` : '');
  /* readout: the most confident box in this frame */
  const top = bs.length ? bs.reduce((a, b) => b[4] > a[4] ? b : a) : null;
  $('tConf').innerHTML = top ? Math.round(top[4] * 100) + '<small>%</small>' : '—';
  const mils = top ? (top[2] - top[0]) * (+$('tFov').value * Math.PI / 180) * 1000 : 0;
  $('tRange').innerHTML = top && mils > 0 ? '~' + Math.round(+$('tSize').value * 1000 / mils) + '<small>m</small>' : '—';
  $('tId').textContent = top && top[5] != null ? top[5] : '—';
  $('tLockK').textContent = c.truth === 'boxes' && R.gt ? 'On target so far' : c.truth === 'no_drone' ? 'False alarm so far' : 'Box on screen so far';
  $('tLock').innerHTML = R.pred && R.cum && R.cum[k] >= 0 ? R.cum[k].toFixed(1) + '<small>%</small>' : '—';
  $('read').textContent = `Frame ${k} / ${c.frames - 1} · ${(k / c.fps).toFixed(2)} s / ${(c.frames / c.fps).toFixed(1)} s`;
  if (!R.scrubbing) $('scrub').value = k;
  drawHead();
}
function renderSummary() {
  const c = R.clip, n = R.counts, o = outInfo(R.out), el = $('cSummary'), sm = c.outputs[R.out].summary;
  const pc = (a, b) => b ? (100 * a / b).toFixed(1) + '%' : '—';
  if (R.pred) {
    if (c.truth === 'boxes' && R.gt) el.innerHTML = `<b>${o.label}: on target ${pc(n.on, n.vis)}</b> of the ${n.vis} frames where the drone is in view · ${n.miss} missed frames (longest miss ${(n.longest / c.fps).toFixed(1)} s) · ${n.fa} frames with a false-alarm box. On target = a box overlapping the true drone box (IoU ≥ 0.3).`;
    else if (c.truth === 'no_drone') el.innerHTML = `<b>${o.label}: false alarm in ${pc(n.box, c.frames)}</b> of ${c.frames} frames. This clip has no drone, so every box is a false alarm.`;
    else el.innerHTML = `<b>${o.label}: box shown in ${pc(n.box, c.frames)}</b> of ${c.frames} frames. There is no ground truth for this clip, so this is not an accuracy number.`;
  } else if (sm && sm.kind === 'tracking') el.innerHTML = `<b>${o.label}, clip total from Test 2:</b> on target ${sm.on_target_pct}% of ${sm.drone_visible_frames} drone-visible frames · wrong-object box in ${sm.false_alarm_pct}% of frames · ${sm.id_switches} ID switches. Per-frame boxes are not saved yet, so nothing is drawn on the video.`;
  else if (sm) el.innerHTML = `<b>${o.label}, clip total from Test 1:</b> false alarm in ${sm.fa35}% of ${sm.frames_tested} tested frames (cutoff 0.35). Per-frame boxes are not saved yet, so nothing is drawn on the video.`;
  else el.innerHTML = `No results for ${o.label} on this clip yet.`;
}
function renderOuts() {
  const c = R.clip; if (!c) return;
  $('outs').innerHTML = R.outputs.map((o, i) => {
    const why = outReason(c, o.id), dis = why === 'unavailable';
    return `<button class="btn ${o.id === R.out ? 'on' : ''}" data-o="${o.id}" ${dis ? 'disabled' : ''} aria-pressed="${o.id === R.out}" title="${dis ? esc(whyUnavailable(c, o.id)) : 'Key ' + (i + 1)}">${esc(o.label)}${why === 'frame data' ? '' : `<small>${why}</small>`}</button>`}).join('');
  $('outs').querySelectorAll('button').forEach(b => b.onclick = () => {setOutput(b.dataset.o); b.blur()});
}
function clipStat(c) {
  const s = c.outputs[R.out] && c.outputs[R.out].summary;
  if (!s) return '—';
  return s.kind === 'tracking' ? `${s.on_target_pct}% on` : `${s.fa35}% FA`;
}
function listed() {return R.clips.filter(c => !R.group || c.group === R.group)}
function renderLists() {
  const cur = R.clip && R.clip.key;
  $('demoList').innerHTML = R.demo.filter(d => R.byKey[d.clip]).map(d => `<button class="crow" data-k="${d.clip}" aria-current="${d.clip === cur}"><span class="n">${esc(d.category)}</span><span class="s">${clipStat(R.byKey[d.clip])}</span><span class="d">${esc(R.byKey[d.clip].name)} · ${esc(d.note)}</span></button>`).join('') || '<div class="empty">No demo clips configured.</div>';
  $('clipList').innerHTML = listed().map(c => `<button class="crow" data-k="${c.key}" aria-current="${c.key === cur}"><span class="n">${esc(c.name)}</span><span class="s">${clipStat(c)}</span><span class="d">${esc(c.group_label)} · ${mmss(c.frames / c.fps)}${c.split ? ' · ' + (c.split === 'held out' ? '✓ held out' : '! v3 trained') : ''}</span></button>`).join('');
  document.querySelectorAll('#demoList .crow, #clipList .crow').forEach(b => b.onclick = () => {
    const d = R.demo.find(x => x.clip === b.dataset.k && b.closest('#demoList'));
    loadClip(b.dataset.k, {frame: d && d.frame != null ? d.frame : 0}); b.blur()});
}

/* ---------------- player ---------------- */
function frameOf(t) {return Math.max(0, Math.min(R.clip.frames - 1, Math.floor(t * R.clip.fps + 0.01)))}
function seekFrame(k) {
  const c = R.clip, v = $('vid'); k = Math.max(0, Math.min(c.frames - 1, Math.round(k)));
  R.frame = k; v.currentTime = (k + 0.5) / c.fps; renderFrame(k); setHash();
}
function togglePlay() {const v = $('vid'); if (v.paused) {if (R.frame >= R.clip.frames - 1) seekFrame(0); v.play()} else v.pause()}
function stepClip(d) {const L = listed(); if (!L.length) return; let i = L.findIndex(c => c.key === R.clip.key); i = (i + d + L.length) % L.length; loadClip(L[i].key, {frame: 0})}
function jumpWorst() {
  const d = R.demo.find(x => x.clip === R.clip.key), w = (d && d.frame != null && d.frame > 0) ? d.frame : R.worst;
  if (w == null) return toast(R.pred ? 'Nothing to jump to: no miss or false alarm for this output on this clip.' : 'Needs per-frame boxes for this output.');
  $('vid').pause(); seekFrame(R.clip.truth === 'no_drone' ? w : Math.max(0, w - Math.round(R.clip.fps)));
  toast(R.clip.truth === 'boxes' ? 'One second before the longest miss. Press Space to play into it.' : R.clip.truth === 'no_drone' ? 'Most confident false alarm.' : 'One second before the first frame where the box drops.');
}
const flagWord = f => `${f.kind === 'miss' ? '✕ miss' : '▲ false alarm'} flag on ${(R.byKey[f.clip] || {}).name || f.clip} frame ${f.frame}`;
function undoMsg(t, cls) {const m = $('undoMsg'); m.textContent = t; m.className = 'mono ' + (cls || ''); clearTimeout(R.undoT); R.undoT = setTimeout(() => {m.textContent = ''}, 8000)}
function undoButton(next) {
  R.lastFlag = next || null;
  const b = $('bUndo'); b.disabled = !next;
  b.title = next ? `U: remove the ${flagWord(next)}` : 'Nothing to undo';
  b.textContent = next ? '↶ Undo flag' : '↶ Undo flag';
}
async function flag(kind) {
  const r = await api('/api/flag', {clip: R.clip.key, frame: R.frame, kind, output: R.out});
  if (!r.ok) return toast('Could not save the flag.');
  if (r.existing) {undoMsg(`Already flagged: ${flagWord(r.flag)}. Nothing new was saved.`, 'muted'); return}
  R.flags.push(r.flag); undoButton(r.flag);
  drawBase(); renderFrame(R.frame);
  undoMsg(`⚑ Saved ${flagWord(r.flag)}. Press ↶ Undo flag (or U) to remove it.`, 'fa');
}
async function undoFlag() {
  if (!R.lastFlag) return undoMsg('Nothing to undo.', 'muted');
  undoMsg('Undoing…', 'muted');
  const r = await api('/api/flag', {undo_last: true});
  if (!r.ok) return undoMsg('✕ Undo failed: ' + (r.error || 'server error'), 'miss');
  R.flags = R.flags.filter(f => f.id !== r.removed.id);
  undoButton(r.next); drawBase(); renderFrame(R.frame);
  undoMsg(`✓ Undone at ${new Date().toLocaleTimeString([], {hour: 'numeric', minute: '2-digit', second: '2-digit'})}: removed the ${flagWord(r.removed)}.` + (r.next ? ' Press again to remove the one before it.' : ' No flags left.'), 'ok');
}
function wirePlayer() {
  const v = $('vid');
  const blurAfter = (id, fn) => $(id).onclick = () => {fn(); $(id).blur()};
  blurAfter('bPlay', togglePlay); blurAfter('bPrevF', () => {v.pause(); seekFrame(R.frame - 1)}); blurAfter('bNextF', () => {v.pause(); seekFrame(R.frame + 1)});
  blurAfter('bRestart', () => seekFrame(0)); blurAfter('bWorst', jumpWorst);
  blurAfter('bTruth', () => {R.truth = !R.truth; $('bTruth').textContent = 'Truth box: ' + (R.truth ? 'on' : 'off'); $('bTruth').setAttribute('aria-pressed', R.truth); renderFrame(R.frame)});
  blurAfter('bFlagM', () => flag('miss')); blurAfter('bFlagF', () => flag('false_alarm')); blurAfter('bUndo', undoFlag);
  $('speed').onchange = e => {v.playbackRate = +e.target.value};
  $('tSize').onchange = $('tFov').onchange = () => renderFrame(R.frame);
  v.addEventListener('play', () => {$('bPlay').textContent = '❚❚ Pause'});
  v.addEventListener('pause', () => {$('bPlay').textContent = '▶ Play'; setHash()});
  v.addEventListener('loadedmetadata', () => {v.playbackRate = +$('speed').value});
  v.addEventListener('error', () => {$('pMsg').hidden = false; $('pMsg').textContent = 'This video could not be loaded.'});
  const onT = t => {if (!R.clip) return; const k = frameOf(t); if (k !== R.frame || v.paused) {R.frame = k; renderFrame(k)}};
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {const cb = (now, m) => {onT(m.mediaTime); v.requestVideoFrameCallback(cb)}; v.requestVideoFrameCallback(cb)}
  else {v.addEventListener('timeupdate', () => onT(v.currentTime)); const loop = () => {if (!v.paused) onT(v.currentTime); requestAnimationFrame(loop)}; loop()}
  v.addEventListener('seeked', () => onT(v.currentTime));
  const sc = $('scrub');
  sc.addEventListener('input', () => {R.scrubbing = true; seekFrame(+sc.value)});
  sc.addEventListener('change', () => {R.scrubbing = false});
  const tl = $('timeline'); let drag = false;
  const at = e => {const r = tl.getBoundingClientRect(); seekFrame(Math.floor((e.clientX - r.left) / r.width * R.clip.frames))};
  tl.addEventListener('pointerdown', e => {drag = true; tl.setPointerCapture(e.pointerId); at(e)});
  tl.addEventListener('pointermove', e => {if (drag) at(e)});
  tl.addEventListener('pointerup', () => {drag = false});
  addEventListener('keydown', e => {
    if (TAB !== 'review' || !R.clip || inControl(e)) return;
    const k = e.key;
    if (k === ' ') {e.preventDefault(); togglePlay()}
    else if (k === '[') stepClip(-1); else if (k === ']') stepClip(1);
    else if (k === 'r' || k === 'R') seekFrame(0);
    else if (k === ',' || k === 'ArrowLeft') {e.preventDefault(); v.pause(); seekFrame(R.frame - 1)}
    else if (k === '.' || k === 'ArrowRight') {e.preventDefault(); v.pause(); seekFrame(R.frame + 1)}
    else if (k === 'w' || k === 'W') jumpWorst();
    else if (k === 'g' || k === 'G') $('bTruth').click();
    else if (k === 'm' || k === 'M') flag('miss'); else if (k === 'f' || k === 'F') flag('false_alarm');
    else if (k === 'u' || k === 'U') undoFlag();
    else if (/^[1-9]$/.test(k)) {const o = R.outputs[+k - 1]; if (o && outReason(R.clip, o.id) !== 'unavailable') setOutput(o.id)}
  });
}
