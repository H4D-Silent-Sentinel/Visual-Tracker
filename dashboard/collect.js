/* Collect: live camera + recording, model-assisted labeling, flagged hard examples, coverage. */
const C = {wired: false, preview: false, poll: null, clip: null, frame: 0, boxes: [], saved: {}, auto: false, drag: null, training: false};
async function collectLoad() {
  if (!C.wired) wireCollect();
  const d = await api('/api/collect');
  R.flags = d.flags; C.training = d.camera.training_active;
  if (window.undoButton) undoButton(d.flags.length ? d.flags.reduce((a, b) => (b.time || 0) > (a.time || 0) ? b : a) : null);
  $('flagList').innerHTML = d.flags.length ? d.flags.slice().reverse().map(f => {const c = R.byKey[f.clip];
    return `<div class="flagrow"><img loading="lazy" src="/thumb?kind=clip&clip=${f.clip}&f=${f.frame}&w=192" alt=""><div><b>${f.kind === 'miss' ? '✕ Miss' : '▲ False alarm'}</b> · ${esc(c ? c.name : f.clip)} · frame ${f.frame}<div class="muted" style="font-size:12px">flagged while showing ${esc(f.output || '—')}</div></div><div><button class="btn" data-open="${f.clip}|${f.frame}|${f.output || ''}">Open</button> <button class="btn" data-rm="${f.id}">Remove</button></div></div>`}).join('')
    : '<div class="empty">Nothing flagged yet. In Clip Review, pause on a bad frame and press M (miss) or F (false alarm).</div>';
  $('flagList').querySelectorAll('[data-open]').forEach(b => b.onclick = () => {const [k, f, o] = b.dataset.open.split('|'); openClip(k, +f, o || undefined)});
  $('flagList').querySelectorAll('[data-rm]').forEach(b => b.onclick = async () => {await api('/api/flag', {remove: b.dataset.rm}); collectLoad()});
  $('oursList').innerHTML = d.recordings.length ? d.recordings.map(r => `<div class="flagrow"><img loading="lazy" src="/thumb?kind=clip&clip=${r.clip}&w=192" alt=""><div><b>${esc(r.name.replace('ours_', ''))}</b> · ${r.seconds} s${r.labeled ? ' · ' + r.labeled + ' labeled' : ''}<div class="muted" style="font-size:12px">${esc([r.meta.location, r.meta.drone_type, r.meta.background].filter(Boolean).join(' · ') || 'no details entered')}</div></div><div><button class="btn" data-review="${r.clip}">Review</button> <button class="btn" data-label="${r.clip}">Label</button> <button class="btn" data-del="${esc(r.name)}">Delete</button></div></div>`).join('')
    : '<div class="empty">No recordings yet.</div>';
  $('oursList').querySelectorAll('[data-review]').forEach(b => b.onclick = () => openClip(b.dataset.review, 0));
  $('oursList').querySelectorAll('[data-label]').forEach(b => b.onclick = () => {$('labClip').value = b.dataset.label; labLoad(b.dataset.label); $('labBox').scrollIntoView({block: 'center'})});
  $('oursList').querySelectorAll('[data-del]').forEach(b => b.onclick = async () => {
    if (b.dataset.armed !== '1') {b.dataset.armed = '1'; b.textContent = 'Click again to delete'; b.classList.add('rec'); setTimeout(() => {if (b.isConnected) {b.dataset.armed = ''; b.textContent = 'Delete'; b.classList.remove('rec')}}, 4000); return}
    b.textContent = 'Deleting…'; const r = await api('/api/recording/delete', {name: b.dataset.del});
    if (!r.ok) {b.textContent = 'Delete failed'; return}
    toast(`Deleted ${b.dataset.del}.`); await refreshClips(); collectLoad()});
  $('covTable').innerHTML = '<tr><th>Source</th><th>Type</th><th>Background</th><th class="num">Clips</th><th class="num">Minutes</th><th class="num">Labeled frames</th></tr>' +
    d.coverage.map(r => `<tr><td>${esc(r.source)}</td><td>${esc(r.kind)}</td><td>${esc(r.background)}</td><td class="num">${r.clips}</td><td class="num">${(r.seconds / 60).toFixed(1)}</td><td class="num">${r.labeled_frames.toLocaleString()}</td></tr>`).join('') +
    (d.ours.length ? '' : '<tr><td colspan="6" class="muted">No footage of our own yet. Nothing at night, in rain, or from our own camera.</td></tr>');
  const order = [...R.clips].sort((a, b) => (a.group === 'ours' ? 0 : 1) - (b.group === 'ours' ? 0 : 1));
  const cur = $('labClip').value;
  $('labClip').innerHTML = order.map(c => `<option value="${c.key}">${esc(c.group === 'ours' ? 'Ours: ' : c.group_label + ': ')}${esc(c.name)}${d.labeled[c.key] ? ' (' + d.labeled[c.key] + ' labeled)' : ''}</option>`).join('');
  if (cur && R.byKey[cur]) $('labClip').value = cur;
  if (!C.clip || !R.byKey[C.clip.key]) labLoad($('labClip').value);
  camState(d.camera);
  clearInterval(C.poll); C.poll = setInterval(async () => {if (TAB === 'collect') camState(await api('/api/cam'))}, 1000);
}
function collectLeave() {clearInterval(C.poll); if (C.preview && !C.recording) camPreview(false)}

/* ---------------- camera ---------------- */
function camPreview(on) {
  C.preview = on;
  $('camBox').innerHTML = on ? `<img src="/cam/stream.mjpg?t=${Date.now()}" alt="Live camera picture">` : 'Camera is off.';
  $('camToggle').textContent = on ? 'Stop preview' : 'Start preview';
}
function camState(s) {
  C.recording = s.recording; C.training = s.training_active;
  $('camRec').textContent = s.recording ? '■ Stop recording' : '● Record';
  $('camOverlay').disabled = s.training_active;
  $('camOverlay').textContent = s.training_active ? 'Model boxes: off while training runs' : 'Model boxes: ' + (s.overlay ? 'on' : 'off');
  $('camOverlay').setAttribute('aria-pressed', s.overlay);
  let t = '';
  if (s.error && (C.preview || s.recording)) t = '✕ ' + s.error;
  else if (s.recording) t = `● RECORDING · ${s.rec_seconds.toFixed(0)} s · ${s.rec_frames} frames`;
  else if (C.preview) t = s.has_picture ? '● Camera live (not recording)' : 'Waiting for the camera…';
  if (t || !C.keepMsg) $('camStatus').textContent = t;
}
/* ---------------- labeler ---------------- */
async function labLoad(key) {
  const c = R.byKey[key]; if (!c) return;
  C.clip = c; C.frame = 0; C.auto = false;
  const v = $('labVid'); v.src = c.url; v.load();
  $('labBox').style.aspectRatio = `${c.w} / ${c.h}`;
  C.saved = (await api('/api/labels?clip=' + key)).labels || {};
  v.addEventListener('loadedmetadata', () => labGo(0), {once: true});
}
function labGo(k) {
  const c = C.clip; if (!c) return;
  k = Math.max(0, Math.min(c.frames - 1, k)); C.frame = k;
  $('labVid').currentTime = (k + 0.5) / c.fps;
  const s = C.saved[k];
  C.boxes = s ? s.map(b => b.slice(0, 4)) : [];
  labStatus(s ? (s.length ? `✓ Saved label: ${s.length} box${s.length === 1 ? '' : 'es'}` : '✓ Saved label: no drone in this frame') : '○ Not labeled');
  labDraw();
  if (!s && C.auto) labPropose(true);
}
function labStatus(t) {$('labStatus').textContent = t; $('labRead').textContent = C.clip ? `Frame ${C.frame} / ${C.clip.frames - 1} · ${Object.keys(C.saved).length} labeled in this clip` : ''}
function labDraw() {
  const cv = $('labCv'), dpr = devicePixelRatio || 1, W = Math.round(cv.clientWidth * dpr), H = Math.round(cv.clientHeight * dpr);
  if (cv.width !== W || cv.height !== H) {cv.width = W; cv.height = H}
  const g = cv.getContext('2d'); g.clearRect(0, 0, W, H);
  const s = C.saved[C.frame], same = s && JSON.stringify(s.map(b => b.slice(0, 4))) === JSON.stringify(C.boxes);
  const all = C.drag ? [...C.boxes, C.drag] : C.boxes;
  g.lineWidth = 2 * dpr; g.font = `${Math.round(12 * dpr)}px ui-monospace,Menlo,monospace`;
  all.forEach(b => {const x = Math.min(b[0], b[2]) * W, y = Math.min(b[1], b[3]) * H, w = Math.abs(b[2] - b[0]) * W, h = Math.abs(b[3] - b[1]) * H;
    g.strokeStyle = same ? '#6fbf8b' : '#e6b04a'; g.setLineDash(same ? [] : [6 * dpr, 4 * dpr]); g.strokeRect(x, y, w, h); g.setLineDash([]);
    g.fillStyle = same ? '#6fbf8b' : '#e6b04a'; g.fillText(same ? 'saved' : 'not saved', x, Math.max(12 * dpr, y - 4 * dpr))});
}
async function labPropose(quiet) {
  const k = C.frame, r = await api(`/api/propose?clip=${C.clip.key}&frame=${k}`);
  if (k !== C.frame) return;
  if (!r.ok) {C.auto = false; return labStatus('✕ ' + r.reason)}
  C.auto = true; C.boxes = r.boxes.map(b => b.slice(0, 4)); labDraw();
  labStatus(r.boxes.length ? `Proposed (${r.source}): ${r.boxes.length} box${r.boxes.length === 1 ? '' : 'es'}, not saved. Save to accept, or drag to redraw.` : `Proposed (${r.source}): no drone found. Press N to save “no drone”, or drag to draw the box.`);
}
function labSaveDrawn() {
  if (!C.boxes.length) return labStatus('No box to save. Drag on the picture to draw one, or press N to save “no drone”.');
  labSave(C.boxes);
}
async function labSave(boxes) {
  const k = C.frame, r = await api('/api/label', {clip: C.clip.key, frame: k, boxes});
  if (!r.ok) return labStatus('✕ Could not save: ' + (r.error || ''));
  C.saved[k] = boxes.map(b => b.slice(0, 4));
  labGo(k + (+$('labStride').value));
}
async function labDelete() {
  const k = C.frame; await api('/api/label', {clip: C.clip.key, frame: k, delete: true});
  delete C.saved[k]; labGo(k);
}
function wireCollect() {
  C.wired = true;
  const post = b => api('/api/cam', b);
  $('camToggle').onclick = async () => {if (!C.preview) await post({index: +$('camIdx').value}); camPreview(!C.preview)};
  $('camIdx').onchange = async () => {await post({index: +$('camIdx').value}); if (C.preview) camPreview(true)};
  $('camOverlay').onclick = async () => {camState(await post({overlay: $('camOverlay').getAttribute('aria-pressed') !== 'true'}))};
  $('camRec').onclick = async () => {
    if (!C.recording) {
      if (!C.preview) {await post({index: +$('camIdx').value}); camPreview(true)}
      const r = await post({action: 'start', location: $('mLoc').value, drone_type: $('mDrone').value, background: $('mBg').value, notes: $('mNotes').value});
      C.keepMsg = false; if (!r.ok) $('camStatus').textContent = '✕ ' + r.error;
    } else {
      $('camStatus').textContent = 'Saving the clip and making a browser copy…'; C.keepMsg = true;
      const r = await post({action: 'stop'});
      $('camStatus').textContent = r.ok ? `✓ Saved ${r.name} (${r.seconds} s, ${r.frames} frames)` + (r.web_copy ? '. It is now in Clip Review and the labeler.' : '. The browser copy failed; see the Terminal.') : '✕ ' + r.error;
      if (r.ok) {await refreshClips(); await collectLoad(); if (r.web_copy) {$('labClip').value = r.clip; labLoad(r.clip)}}
    }
  };
  $('flagExport').onclick = async () => {const r = await api('/api/flags/export', {}); $('flagMsg').textContent = r.ok ? `✓ Exported ${r.exported} frames to ${r.folder} (${r.with_labels} with labels, ${r.need_labels} still need a label)` : '✕ Export failed'};
  $('labClip').onchange = e => labLoad(e.target.value);
  const blurAfter = (id, fn) => $(id).onclick = () => {fn(); $(id).blur()};
  blurAfter('labPrev', () => labGo(C.frame - (+$('labStride').value))); blurAfter('labNext', () => labGo(C.frame + (+$('labStride').value)));
  blurAfter('labPropose', () => labPropose()); blurAfter('labSave', labSaveDrawn); blurAfter('labNone', () => labSave([]));
  blurAfter('labClear', () => {C.boxes = []; labDraw(); labStatus('Boxes cleared (not saved).')}); blurAfter('labDelete', labDelete);
  $('labVid').addEventListener('seeked', labDraw);
  const cv = $('labCv'), pt = e => {const r = cv.getBoundingClientRect(); return [Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)), Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))]};
  cv.addEventListener('pointerdown', e => {const [x, y] = pt(e); C.drag = [x, y, x, y]; cv.setPointerCapture(e.pointerId)});
  cv.addEventListener('pointermove', e => {if (!C.drag) return; const [x, y] = pt(e); C.drag[2] = x; C.drag[3] = y; labDraw()});
  cv.addEventListener('pointerup', () => {const d = C.drag; C.drag = null; if (!d) return;
    const b = [Math.min(d[0], d[2]), Math.min(d[1], d[3]), Math.max(d[0], d[2]), Math.max(d[1], d[3])];
    if (b[2] - b[0] > 0.004 && b[3] - b[1] > 0.004) {C.boxes = [b]; labStatus('Box drawn, not saved. Press S to save.')} labDraw()});
  addEventListener('resize', () => {if (TAB === 'collect') labDraw()});
  addEventListener('keydown', e => {
    if (TAB !== 'collect' || !C.clip || inControl(e)) return;
    const k = e.key.toLowerCase(), st = +$('labStride').value;
    if (k === 'a') labGo(C.frame - st); else if (k === 'd') labGo(C.frame + st); else if (k === 'p') labPropose();
    else if (k === 's' || k === 'enter') labSaveDrawn(); else if (k === 'n') labSave([]); else if (k === 'x') {C.boxes = []; labDraw()}
  });
}
