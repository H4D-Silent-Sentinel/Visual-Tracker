/* Collect, in three steps: 1 record our own footage, 2 label it, 3 use it (recordings, hard examples, training set). */
const C = {wired: false, preview: false, poll: null, clip: null, frame: 0, boxes: [], saved: {}, drag: null, training: false, recording: false};
const store2 = {get: k => {try {return localStorage.getItem(k)} catch (e) {return null}}, set: (k, v) => {try {localStorage.setItem(k, v)} catch (e) {}}};

async function collectLoad() {
  if (!C.wired) wireCollect();
  const d = await api('/api/collect');
  R.flags = d.flags; C.training = d.camera.training_active;
  if (window.undoButton) undoButton(d.flags.length ? d.flags.reduce((a, b) => (b.time || 0) > (a.time || 0) ? b : a) : null);

  /* 1. cameras, named, in the numbering the server opens them by */
  const sel = $('camIdx'), want = sel.value || store2.get('camIdx');
  sel.innerHTML = d.cameras.map(c => `<option value="${c.index}">${esc(c.name)}${c.builtin ? ' (this Mac)' : ' (phone or external)'}</option>`).join('');
  if (want != null && d.cameras.some(c => String(c.index) === String(want))) sel.value = want;
  if (!C.preview) camBoxOff();

  /* 2. labeler: only clips that still need labels (tracking-test clips already have the drone marked in every frame) */
  const groups = [['Our recordings', 'ours'], ['Demo flights (no labels yet)', 'demo'], ['No-drone clips (birds, kites)', 'negatives']];
  const cur = $('labClip').value || store2.get('labClip');
  $('labClip').innerHTML = groups.map(([g, k]) => {const cs = R.clips.filter(c => c.group === k); return cs.length ? `<optgroup label="${g}">${cs.map(c => `<option value="${c.key}">${esc(c.name)}${d.labeled[c.key] ? ` (${d.labeled[c.key]} labeled)` : ''}</option>`).join('')}</optgroup>` : ''}).join('');
  if (cur && [...$('labClip').options].some(o => o.value === cur)) $('labClip').value = cur;
  if (!C.clip || C.clip.key !== $('labClip').value) labLoad($('labClip').value);

  /* 3. recordings, training set, hard examples, coverage */
  $('oursList').innerHTML = d.recordings.length ? d.recordings.map(r => `<div class="flagrow"><img loading="lazy" src="/thumb?kind=clip&clip=${r.clip}&w=192" alt=""><div><b>${esc(r.name.replace('ours_', '').replace(/^(\d{4})(\d\d)(\d\d)_(\d\d)(\d\d)(\d\d)$/, '$2/$3 $4:$5'))}</b> · ${r.seconds} s · ${r.labeled ? r.labeled + ' labeled' : 'not labeled'}<div class="muted" style="font-size:12px">${esc([r.meta.location, r.meta.drone_type, r.meta.background].filter(Boolean).join(' · ') || 'no details entered')}</div></div><div><button class="btn" data-label="${r.clip}">Label</button> <button class="btn" data-review="${r.clip}">Review</button> <button class="btn" data-del="${esc(r.name)}">Delete</button></div></div>`).join('')
    : '<div class="empty">No recordings yet. Record one in step 1.</div>';
  $('oursList').querySelectorAll('[data-review]').forEach(b => b.onclick = () => openClip(b.dataset.review, 0));
  $('oursList').querySelectorAll('[data-label]').forEach(b => b.onclick = () => {$('labClip').value = b.dataset.label; labLoad(b.dataset.label); $('labBox').scrollIntoView({block: 'center', behavior: 'smooth'})});
  $('oursList').querySelectorAll('[data-del]').forEach(b => b.onclick = async () => {
    if (b.dataset.armed !== '1') {b.dataset.armed = '1'; b.textContent = 'Click again to delete'; b.classList.add('rec'); setTimeout(() => {if (b.isConnected) {b.dataset.armed = ''; b.textContent = 'Delete'; b.classList.remove('rec')}}, 4000); return}
    b.textContent = 'Deleting…'; const r = await api('/api/recording/delete', {name: b.dataset.del});
    if (!r.ok) {b.textContent = 'Delete failed'; return}
    toast(`Deleted ${b.dataset.del} and its labels.`); if (C.clip && C.clip.key === 'ours__' + b.dataset.del) C.clip = null;
    await refreshClips(); collectLoad()});

  const s = d.dataset;
  $('dsSummary').innerHTML = s.frames || s.hard_examples
    ? `${s.frames} labeled frames from ${s.clips} clip${s.clips === 1 ? '' : 's'} (${s.with_drone} with a drone, ${s.no_drone} no drone) · ${s.hard_examples} exported hard examples`
    : '<span class="muted">Nothing labeled yet.</span>';
  $('dsWrite').disabled = !(s.frames || s.hard_examples);

  $('flagList').innerHTML = d.flags.length ? d.flags.slice().reverse().map(f => {const c = R.byKey[f.clip];
    return `<div class="flagrow"><img loading="lazy" src="/thumb?kind=clip&clip=${f.clip}&f=${f.frame}&w=192" alt=""><div><b>${f.kind === 'miss' ? '✕ Miss' : '▲ False alarm'}</b> · ${esc(c ? c.name : f.clip)} · frame ${f.frame}<div class="muted" style="font-size:12px">flagged while showing ${esc(f.output || '—')}</div></div><div><button class="btn" data-open="${f.clip}|${f.frame}|${f.output || ''}">Open</button> <button class="btn" data-rm="${f.id}">Remove</button></div></div>`}).join('')
    : '<div class="empty">Nothing flagged yet. In Clip Review, pause on a bad frame and press M (miss) or F (false alarm).</div>';
  $('flagList').querySelectorAll('[data-open]').forEach(b => b.onclick = () => {const [k, f, o] = b.dataset.open.split('|'); openClip(k, +f, o || undefined)});
  $('flagList').querySelectorAll('[data-rm]').forEach(b => b.onclick = async () => {await api('/api/flag', {remove: b.dataset.rm}); collectLoad()});
  $('flagExport').disabled = !d.flags.length;

  $('covTable').innerHTML = '<tr><th>Source</th><th>Type</th><th>Background</th><th class="num">Clips</th><th class="num">Minutes</th><th class="num">Labeled frames</th></tr>' +
    d.coverage.map(r => `<tr><td>${esc(r.source)}</td><td>${esc(r.kind)}</td><td>${esc(r.background)}</td><td class="num">${r.clips}</td><td class="num">${(r.seconds / 60).toFixed(1)}</td><td class="num">${r.labeled_frames.toLocaleString()}</td></tr>`).join('') +
    (d.ours.length ? '' : '<tr><td colspan="6" class="muted">No footage of our own yet.</td></tr>');

  camState(d.camera);
  clearInterval(C.poll); C.poll = setInterval(async () => {if (TAB === 'collect') camState(await api('/api/cam'))}, 1000);
}
function collectLeave() {clearInterval(C.poll); if (C.preview && !C.recording) camPreview(false)}

/* ---------------- 1. record ---------------- */
function camBoxOff() {
  $('camBox').innerHTML = '<div><p style="margin:0 0 10px">Camera is off.</p><button class="btn on" id="camStartIn">Start camera</button></div>';
  $('camStartIn').onclick = () => camPreview(true);
}
async function camPreview(on) {
  C.preview = on;
  if (on) {
    await api('/api/cam', {index: +$('camIdx').value});
    $('camBox').innerHTML = `<img src="/cam/stream.mjpg?t=${Date.now()}" alt="Live camera picture"><div class="recbadge" id="recBadge" hidden></div>`;
  } else camBoxOff();
  $('camToggle').textContent = on ? 'Stop camera' : 'Start camera';
}
function camState(s) {
  C.recording = s.recording; C.training = s.training_active;
  const b = $('camRec');
  b.textContent = s.recording ? '■ Stop recording' : '● Record';
  b.classList.toggle('recording', s.recording);
  $('camIdx').disabled = s.recording;
  $('camOverlay').disabled = s.training_active || s.recording;
  $('camOverlay').textContent = s.training_active ? 'Model boxes: off during training' : 'Model boxes: ' + (s.overlay ? 'on' : 'off');
  $('camOverlay').setAttribute('aria-pressed', s.overlay);
  const badge = $('recBadge');
  if (badge) {badge.hidden = !s.recording; badge.textContent = s.recording ? `REC ${mmss(s.rec_seconds)}` : ''}
  let t = '', cls = '';
  if (s.error && (C.preview || s.recording)) {t = '✕ ' + s.error; cls = 'miss'}
  else if (s.recording) {t = `● Recording · ${s.rec_frames} frames`; cls = 'miss'}
  else if (C.preview) t = s.has_picture ? '● Camera on, not recording' : 'Waiting for the camera…';
  if (!C.keepMsg || t) {$('camStatus').textContent = t; $('camStatus').className = 'mono ' + cls}
  if (t) C.keepMsg = false;
}

/* ---------------- 2. label ---------------- */
async function labLoad(key) {
  const c = R.byKey[key];
  if (!c) {C.clip = null; $('labVid').removeAttribute('src'); labStatus('No clip to label yet. Record one in step 1.'); return}
  store2.set('labClip', key);
  C.clip = c; C.frame = 0;
  const v = $('labVid'); v.src = c.url; v.load();
  $('labBox').style.aspectRatio = `${c.w} / ${c.h}`; $('labBox').style.maxWidth = `calc(68vh * ${(c.w / c.h).toFixed(4)})`;  /* whole frame fits on screen */
  C.saved = (await api('/api/labels?clip=' + key)).labels || {};
  v.addEventListener('loadedmetadata', () => {
    const k = Object.keys(C.saved).map(Number), st = +$('labStride').value;
    labGo(k.length ? Math.min(c.frames - 1, Math.max(...k) + st) : 0);  /* pick up where you left off */
  }, {once: true});
}
function labGo(k) {
  const c = C.clip; if (!c) return;
  k = Math.max(0, Math.min(c.frames - 1, k)); C.frame = k;
  $('labVid').currentTime = (k + 0.5) / c.fps;
  const s = C.saved[k];
  C.boxes = s ? s.map(b => b.slice(0, 4)) : [];
  if (s) labStatus(s.length ? `✓ Saved: ${s.length} box${s.length === 1 ? '' : 'es'}` : '✓ Saved: no drone in this frame', 'saved');
  else labStatus('○ Not labeled yet. Draw a box, or press N for no drone.');
  labDraw(); labStrip();
  if (!s && $('labAuto').checked) labPropose(true);
}
function labStatus(t, cls) {
  const el = $('labStatus'); el.textContent = t; el.className = 'labstatus ' + (cls || '');
  if (C.clip) {const n = Object.keys(C.saved).length, st = +$('labStride').value, goal = Math.ceil(C.clip.frames / st);
    $('labRead').textContent = `Frame ${C.frame} / ${C.clip.frames - 1} · ${n} labeled (${Math.min(100, Math.round(100 * n / goal))}% of every-${st === 1 ? '' : st + 'th '}frame)`}
}
function labDraw() {
  const cv = $('labCv'), dpr = devicePixelRatio || 1, W = Math.round(cv.clientWidth * dpr), H = Math.round(cv.clientHeight * dpr);
  if (cv.width !== W || cv.height !== H) {cv.width = W; cv.height = H}
  const g = cv.getContext('2d'); g.clearRect(0, 0, W, H);
  const s = C.saved[C.frame], same = s && JSON.stringify(s.map(b => b.slice(0, 4))) === JSON.stringify(C.boxes);
  const all = C.drag ? [...C.boxes, C.drag] : C.boxes;
  g.lineWidth = 2 * dpr; g.font = `${Math.round(12 * dpr)}px ui-monospace,Menlo,monospace`;
  all.forEach(b => {const x = Math.min(b[0], b[2]) * W, y = Math.min(b[1], b[3]) * H, w = Math.abs(b[2] - b[0]) * W, h = Math.abs(b[3] - b[1]) * H;
    const col = same ? '#6fbf8b' : '#e6b04a';
    g.strokeStyle = col; g.setLineDash(same ? [] : [6 * dpr, 4 * dpr]); g.strokeRect(x, y, w, h); g.setLineDash([]);
    const t = same ? 'saved' : 'not saved', tw = g.measureText(t).width;
    g.fillStyle = 'rgba(20,22,24,.85)'; g.fillRect(x, Math.max(0, y - 18 * dpr), tw + 8 * dpr, 16 * dpr);
    g.fillStyle = col; g.fillText(t, x + 4 * dpr, Math.max(12 * dpr, y - 5 * dpr))});
}
function labStrip() {
  const cv = $('labStrip'), dpr = devicePixelRatio || 1, W = Math.max(1, Math.round(cv.clientWidth * dpr)), H = Math.round(18 * dpr);
  cv.width = W; cv.height = H; const g = cv.getContext('2d'), c = C.clip; if (!c) return;
  for (const [f, bs] of Object.entries(C.saved)) {g.fillStyle = bs.length ? '#6fbf8b' : '#9ba1a8'; g.fillRect(Math.floor(+f * W / c.frames), 3 * dpr, Math.max(2, Math.ceil(W / c.frames)), H - 6 * dpr)}
  g.fillStyle = '#fff'; g.fillRect(Math.round((C.frame + 0.5) * W / c.frames) - dpr, 0, 2 * dpr, H);
}
async function labPropose(quiet) {
  const k = C.frame, edits = C.edits || 0, r = await api(`/api/propose?clip=${C.clip.key}&frame=${k}`);
  if (k !== C.frame || C.saved[k] || (C.edits || 0) !== edits || C.drag) return;  /* never overwrite a box you drew meanwhile */
  if (!r.ok) return labStatus('✕ ' + r.reason, 'err');
  C.boxes = r.boxes.map(b => b.slice(0, 4)); labDraw();
  labStatus(r.boxes.length ? `Suggested (${r.source}). Not saved: press S to accept, or drag to redraw.` : `No drone suggested (${r.source}). Press N if there is none, or drag to draw it.`, 'pending');
}
function labSaveDrawn() {
  if (!C.boxes.length) return labStatus('No box to save. Drag on the picture to draw one, or press N for no drone.', 'err');
  labSave(C.boxes);
}
async function labSave(boxes) {
  const k = C.frame, r = await api('/api/label', {clip: C.clip.key, frame: k, boxes});
  if (!r.ok) return labStatus('✕ Could not save: ' + (r.error || ''), 'err');
  C.saved[k] = boxes.map(b => b.slice(0, 4));
  if (k + (+$('labStride').value) > C.clip.frames - 1) {labGo(k); return labStatus('✓ Saved. That was the last frame of this clip.', 'saved')}
  labGo(k + (+$('labStride').value));
}
async function labDelete() {
  const k = C.frame; await api('/api/label', {clip: C.clip.key, frame: k, delete: true});
  delete C.saved[k]; labGo(k); labStatus("Label deleted for this frame.");
}

function wireCollect() {
  C.wired = true;
  const post = b => api('/api/cam', b);
  $('camToggle').onclick = () => camPreview(!C.preview);
  $('camIdx').onchange = async () => {store2.set('camIdx', $('camIdx').value); if (C.preview) camPreview(true)};
  $('camOverlay').onclick = async () => {camState(await post({overlay: $('camOverlay').getAttribute('aria-pressed') !== 'true'}))};
  $('camRec').onclick = async () => {
    if (!C.recording) {
      if (!C.preview) await camPreview(true);
      const r = await post({action: 'start', location: $('mLoc').value, drone_type: $('mDrone').value, background: $('mBg').value, notes: $('mNotes').value});
      if (!r.ok) {$('camStatus').textContent = '✕ ' + r.error; $('camStatus').className = 'mono miss'; C.keepMsg = true}
    } else {
      $('camStatus').textContent = 'Saving the clip…'; C.keepMsg = true;
      const r = await post({action: 'stop'});
      $('camStatus').textContent = r.ok ? `✓ Saved (${r.seconds} s). It's ready to label below.` + (r.web_copy ? '' : ' The browser copy failed; see the Terminal.') : '✕ ' + r.error;
      $('camStatus').className = 'mono ' + (r.ok ? 'ok' : 'miss');
      if (r.ok) {await refreshClips(); await collectLoad(); if (r.web_copy) {$('labClip').value = r.clip; labLoad(r.clip)}}
    }
  };
  $('flagExport').onclick = async () => {const r = await api('/api/flags/export', {});
    $('flagMsg').textContent = r.ok ? `✓ Exported ${r.exported} frame${r.exported === 1 ? '' : 's'} to ${r.folder}.` + (r.need_labels ? ` ${r.need_labels} skipped: no known truth yet (label them in step 2, then export again).` : '') : '✕ Export failed';
    collectLoad()};
  $('dsWrite').onclick = async () => {const r = await api('/api/ours/dataset', {});
    $('dsMsg').innerHTML = r.ok ? `✓ Wrote ${esc(r.file)}. To train on it, run in Terminal:<br><code>${esc(r.command)}</code>` : '✕ ' + esc(r.error)};
  $('labClip').onchange = e => {labLoad(e.target.value); e.target.blur()};  /* so S, N, A, D work right after picking a clip */
  $('labStride').onchange = e => {labStatus($('labStatus').textContent); e.target.blur()};
  const blurAfter = (id, fn) => $(id).onclick = () => {fn(); $(id).blur()};
  const st = () => +$('labStride').value;
  blurAfter('labPrev', () => labGo(C.frame - st())); blurAfter('labNext', () => labGo(C.frame + st()));
  blurAfter('labPropose', () => labPropose()); blurAfter('labSave', labSaveDrawn); blurAfter('labNone', () => labSave([]));
  blurAfter('labClear', () => {C.boxes = []; labDraw(); labStatus('Boxes cleared (not saved).')}); blurAfter('labDelete', labDelete);
  $('labVid').addEventListener('seeked', labDraw);
  $('labStrip').addEventListener('click', e => {if (!C.clip) return; const r = $('labStrip').getBoundingClientRect(); labGo(Math.floor((e.clientX - r.left) / r.width * C.clip.frames))});
  const cv = $('labCv'), pt = e => {const r = cv.getBoundingClientRect(); return [Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)), Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))]};
  cv.addEventListener('pointerdown', e => {if (!C.clip) return; C.edits = (C.edits || 0) + 1; if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur(); const [x, y] = pt(e); C.drag = [x, y, x, y]; cv.setPointerCapture(e.pointerId)});
  cv.addEventListener('pointermove', e => {if (!C.drag) return; const [x, y] = pt(e); C.drag[2] = x; C.drag[3] = y; labDraw()});
  cv.addEventListener('pointerup', () => {const d = C.drag; C.drag = null; if (!d) return;
    const b = [Math.min(d[0], d[2]), Math.min(d[1], d[3]), Math.max(d[0], d[2]), Math.max(d[1], d[3])];
    if (b[2] - b[0] > 0.004 && b[3] - b[1] > 0.004) {C.boxes = [b]; labStatus('Box drawn, not saved. Press S to save.', 'pending')} labDraw()});
  addEventListener('resize', () => {if (TAB === 'collect') {labDraw(); labStrip()}});
  addEventListener('keydown', e => {
    if (TAB !== 'collect' || !C.clip || inControl(e)) return;
    const k = e.key.toLowerCase();
    if (k === 'a') labGo(C.frame - st()); else if (k === 'd') labGo(C.frame + st()); else if (k === 'p') labPropose();
    else if (k === 's' || k === 'enter') {e.preventDefault(); labSaveDrawn()} else if (k === 'n') labSave([]); else if (k === 'x') {C.boxes = []; labDraw()}
  });
}
