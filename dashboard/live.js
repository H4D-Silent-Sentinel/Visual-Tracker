/* Live: detector settings and cached verification results update while the clip plays. */
const LV = {wired: false, sources: [], paused: false, timer: null};
const lset = (k, v) => fetch(`/live/set?${k}=${encodeURIComponent(v)}`);
const liveText = (id, text) => {if ($(id).textContent !== text) $(id).textContent = text};
const lnice = s => s.replace('footage/dut/', 'Tracking: ').replace('footage/negatives/', 'No drone: ').replace('footage/ours/', 'Ours: ').replace('footage/', 'Demo: ').replace('camera:', 'Camera ');
async function liveEnter() {
  $('lvStatus').textContent = 'STARTING (loading the model)…';
  const d = await api('/live/sources');
  LV.sources = d.sources; LV.cams = d.camera_names || {};
  const groups = [['Tracking test videos', s => s.includes('/dut/')], ['Demo flights', s => s.startsWith('footage/') && !/\/(dut|negatives|ours)\//.test(s)], ['No drone: birds and kites', s => s.includes('/negatives/')], ['Our footage', s => s.includes('/ours/')], ['Live cameras', s => s.startsWith('camera:')]];
  $('lvSource').innerHTML = groups.map(([g, f]) => {const xs = d.sources.filter(f); return xs.length ? `<optgroup label="${g}">${xs.map(s => `<option value="${esc(s)}">${esc(LV.cams[s] ? LV.cams[s] + (/iphone|\(\d\) camera/i.test(LV.cams[s]) || !/macbook|facetime/i.test(LV.cams[s]) ? ' (phone)' : '') : lnice(s))}</option>`).join('')}</optgroup>` : ''}).join('');
  $('lvModel').innerHTML = d.models.map(m => `<option value="${m.id}" ${m.ok ? '' : 'disabled'}>${esc(m.label)}${m.ok ? '' : ' (not finished)'}</option>`).join('');
  if (d.training_active) $('lvNote').insertAdjacentHTML('beforeend', LV.warned ? '' : ' <span class="fa">▲ A training job is running; this view slows it down while it is open.</span>');
  LV.warned = true;
  if (!LV.wired) {
    LV.wired = true;
    $('lvSource').onchange = e => lset('source', e.target.value);
    $('lvModel').onchange = e => lset('model', e.target.value);
    $('lvConf').oninput = e => {$('lvConfV').textContent = (+e.target.value).toFixed(2); lset('conf', e.target.value)};
    $('lvImgsz').onchange = e => lset('imgsz', e.target.value);
    $('lvTracker').onchange = e => {lset('tracker', e.target.value); liveHoldNote(e.target.value)};
    $('lvVerify').onchange = e => lset('verify', e.target.checked);
    $('lvSize').onchange = e => lset('size_m', e.target.value);
    $('lvFov').onchange = e => lset('hfov', e.target.value);
    $('lvLink').onchange = e => {lset('link', e.target.checked); $('lvPose').hidden = !e.target.checked};
    for (const [id, k] of LINK_FIELDS) $(id).onchange = e => lset(k, e.target.value);
    const b = (id, fn) => $(id).onclick = () => {fn(); $(id).blur()};
    b('lvPause', livePause); b('lvPrev', () => liveStep(-1)); b('lvNext', () => liveStep(1)); b('lvRestart', () => lset('restart', 1));
    addEventListener('keydown', e => {if (TAB !== 'live' || inControl(e)) return;
      if (e.key === ' ') {e.preventDefault(); livePause()} else if (e.key === '[') liveStep(-1); else if (e.key === ']') liveStep(1); else if (e.key === 'r' || e.key === 'R') lset('restart', 1)});
  }
  $('lvImg').src = '/live/stream.mjpg?' + Date.now();
  clearTimeout(LV.timer); liveTick();
}
function liveHoldNote(t) {
  const hold = t === 'hold';
  $('lvConf').disabled = hold;
  $('lvConfV').textContent = hold ? 'set by Hold: 0.35 to start a track, 0.08 to keep one' : (+$('lvConf').value).toFixed(2);
}
function liveLeave() {$('lvImg').removeAttribute('src'); clearTimeout(LV.timer)}
function livePause() {LV.paused = !LV.paused; $('lvPause').textContent = LV.paused ? '▶ Resume' : '❚❚ Pause'; lset('paused', LV.paused)}
function liveStep(d) {const f = LV.sources.filter(s => !s.startsWith('camera:')); if (!f.length) return; let i = f.indexOf($('lvSource').value); i = (i + d + f.length) % f.length; $('lvSource').value = f[i]; lset('source', f[i])}
function lvCanvas(c) {const r = c.getBoundingClientRect(), dpr = devicePixelRatio || 1; if (c.width !== Math.round(r.width * dpr)) {c.width = Math.round(r.width * dpr); c.height = Math.round(+c.getAttribute('height') * dpr)} return dpr}
const LINK_FIELDS = [['lvYaw', 'yaw'], ['lvPitch', 'pitch'], ['lvLat', 'cam_lat'], ['lvLon', 'cam_lon'], ['lvAlt', 'cam_alt']];
function liveLink(s) {
  const L = s.link || {}, st = s.settings, on = !!st.link;
  if (document.activeElement !== $('lvLink')) $('lvLink').checked = on;
  $('lvPose').hidden = !on;
  for (const [id, k] of LINK_FIELDS) if (document.activeElement !== $(id) && +$(id).value !== st[k]) $(id).value = st[k];
  if (!on) {$('lvLinkBox').dataset.status = 'off'; liveText('lvLinkState', 'Link off'); liveText('lvLinkC2', ''); liveText('lvLinkCue', 'Turn on “Send to Silent Sentinel base” to share tracks.'); return}
  const ago = L.last_sent_s == null ? 'nothing sent yet' : `last sent ${L.last_sent_s.toFixed(1)} s ago`;
  liveText('lvLinkState', L.error ? '✕ ' + L.error : `Sending as ${L.src} to ${L.base} · ${L.sent} messages, ${ago}`);
  const C = L.c2 || {};
  let h = C.up ? 'Base link up' : C.last_heard_s == null ? 'Base not heard yet' : `✕ Base silent for ${C.last_heard_s.toFixed(0)} s`;
  if (C.up && C.heartbeat) h += ` · ESP32 ${C.esp32} · nodes ${C.nodes_online}/${C.nodes_total} online` + (C.nodes_detecting && C.nodes_detecting.length ? `, hearing drone: ${C.nodes_detecting.join(', ')}` : '');
  liveText('lvLinkC2', h);
  const c = L.cue;
  $('lvLinkBox').dataset.status = L.error ? 'rejected' : c ? 'confirmed' : 'pending';
  let t = `No cue from the base yet (listening on udp/${L.cue_port}; ${L.cues_received} received).`;
  if (c && c.rel_az_deg != null) t = `Base track ${c.track_id}: ${Math.abs(c.rel_az_deg).toFixed(0)}° ${c.rel_az_deg >= 0 ? 'right' : 'left'} of centre, ${Math.round(c.dist_m)} m away (±${Math.round(c.sigma_m)} m), ${c.in_view ? 'in view' : 'out of view'} · ${c.sources.join(', ')} · ${c.age_s.toFixed(1)} s old`;
  else if (c) t = `Base track ${c.track_id} at ${c.lat.toFixed(5)}, ${c.lon.toFixed(5)} (±${Math.round(c.sigma_m)} m). ${c.note || ''}`;
  liveText('lvLinkCue', t);
}
async function liveTick() {
  if (TAB !== 'live') return;
  try {
    const s = await api('/live/stats');
    if (s.settings) {
      const enabled = !!s.settings.verify, p = s.primary || s.candidate, v = p?.verification || s.verification || {};
      const locked = s.status === 'LOCKED' && (!enabled || s.primary?.verification?.status === 'confirmed');
      const status = s.status === 'LOCKED' && !locked ? 'UNVERIFIED' : s.status;
      const verdicts = {confirmed: 'Drone detected', rejected: 'Drone not detected', pending: 'Checking for drone', uncertain: 'Unable to check', error: 'Unable to check', unverified: v.error ? 'Unable to check' : 'Checking for drone'};
      const verdict = !enabled ? 'Verification off' : !p ? 'Drone not detected' : verdicts[v.status] || 'Checking for drone';
      const words = {LOCKED: 'Drone detected', COASTING: enabled ? verdict : 'Coasting (just lost it)', SEARCHING: enabled ? verdict : 'Searching', REVIEWING: 'Checking for drone', FALSE_REVIEW: 'Drone not detected', UNVERIFIED: verdict, VERIFY_ERROR: 'Unable to check', IDLE: 'Starting…', NO_CAMERA: 'No camera picture'};
      const el = $('lvStatus'); el.classList.toggle('locked', locked && !s.settings.paused); liveText('lvStatus', s.settings.paused ? '❚❚ PAUSED' : (words[status] || status));
      $('lvErr').hidden = !s.error; $('lvErr').textContent = s.error ? '✕ ' + s.error : '';
      el.style.color = s.settings.paused ? 'var(--muted)' : locked ? 'var(--ok)' : status === 'FALSE_REVIEW' || status === 'SEARCHING' && !p ? 'var(--miss)' : 'var(--muted)';
      $('lvTitle').textContent = LV.cams[s.settings.source] || lnice(s.settings.source);
      if (document.activeElement !== $('lvTracker') && $('lvTracker').value !== s.settings.tracker) {$('lvTracker').value = s.settings.tracker; liveHoldNote(s.settings.tracker)}
      for (const [id, k] of [['lvSource', 'source'], ['lvModel', 'model']]) if (document.activeElement !== $(id) && $(id).value !== s.settings[k]) $(id).value = s.settings[k];
      if (document.activeElement !== $('lvVerify')) $('lvVerify').checked = enabled;
      liveLink(s);
      if (s.total) {$('lvProg').style.width = (100 * s.pos / s.total) + '%'; $('lvTime').textContent = mmss(s.pos / s.source_fps) + ' / ' + mmss(s.total / s.source_fps)} else {$('lvProg').style.width = '0'; $('lvTime').textContent = 'live'}
      liveText('lvVerdict', verdict);
      $('lvVerification').dataset.status = !enabled ? 'off' : !p ? 'rejected' : v.status || 'unverified';
      const notes = {confirmed: 'VLM identified this candidate as a drone.', rejected: 'VLM rejected this YOLO candidate.', pending: 'Reviewing this YOLO candidate.', uncertain: 'Verification did not reach a clear result.', error: 'Verification is unavailable. This candidate is not confirmed.', unverified: v.error ? 'This candidate could not be checked.' : 'Waiting to review this YOLO candidate.'};
      liveText('lvVerificationNote', !enabled ? 'Enable verification to check detector candidates.' : !p ? 'YOLO has not proposed a candidate.' : notes[v.status] || 'Waiting to review this YOLO candidate.');
      $('lvObjectLabel').hidden = !enabled || !p || !v.label;
      liveText('lvObjectLabel', $('lvObjectLabel').hidden ? '' : 'Object label: ' + v.label);
      $('lvC').innerHTML = p ? Math.round(p.conf * 100) + '<small>%</small>' : '—';
      $('lvR').innerHTML = p && p.range_m ? '~' + Math.round(p.range_m) + '<small>m</small>' : '—';
      $('lvId').textContent = p && p.id != null ? p.id : '—';
      $('lvFps').textContent = s.fps.toFixed(0);
      $('lvLocked').textContent = s.locked_pct.toFixed(1) + '%'; $('lvCur').textContent = s.current_lock_s.toFixed(1) + ' s';
      $('lvLong').textContent = s.longest_lock_s.toFixed(1) + ' s'; $('lvReacq').textContent = s.reacquisitions; $('lvN').textContent = enabled ? s.targets.filter(t => t.verification?.status === 'confirmed').length : s.targets.length;
      $('lvLockedLabel').textContent = enabled ? 'Time with confirmed lock' : 'Time with a box'; $('lvNLabel').textContent = enabled ? 'Confirmed targets in view' : 'Boxes in view';
      let c = $('lvSpark'), dpr = lvCanvas(c), g = c.getContext('2d'), W = c.width, H = c.height, h = s.conf_hist; g.clearRect(0, 0, W, H);
      g.strokeStyle = '#2e3338'; g.lineWidth = 1; [0.25, 0.5, 0.75].forEach(y => {g.beginPath(); g.moveTo(0, H * y); g.lineTo(W, H * y); g.stroke()});
      if (h.length) {const st = W / 149; g.beginPath(); h.forEach((v, i) => {const x = (150 - h.length + i) * st, y = H - v * H; i ? g.lineTo(x, y) : g.moveTo(x, y)}); g.strokeStyle = '#6cc5b9'; g.lineWidth = 2 * dpr; g.stroke(); g.lineTo(W, H); g.lineTo((150 - h.length) * st, H); g.fillStyle = 'rgba(108,197,185,.15)'; g.fill()}
      c = $('lvPath'); dpr = lvCanvas(c); g = c.getContext('2d'); W = c.width; H = c.height; const t = s.trail; g.clearRect(0, 0, W, H);
      g.strokeStyle = '#2e3338'; g.lineWidth = 1; for (let i = 1; i < 4; i++) {g.beginPath(); g.moveTo(W * i / 4, 0); g.lineTo(W * i / 4, H); g.stroke(); g.beginPath(); g.moveTo(0, H * i / 4); g.lineTo(W, H * i / 4); g.stroke()}
      if (t.length) {g.lineWidth = 2 * dpr; for (let i = 1; i < t.length; i++) {g.strokeStyle = `rgba(255,107,61,${i / t.length})`; g.beginPath(); g.moveTo(t[i - 1][0] * W, t[i - 1][1] * H); g.lineTo(t[i][0] * W, t[i][1] * H); g.stroke()} const q = t[t.length - 1]; g.fillStyle = '#ff6b3d'; g.beginPath(); g.arc(q[0] * W, q[1] * H, 5 * dpr, 0, 7); g.fill()}
    }
  } catch (e) {$('lvStatus').classList.remove('locked'); $('lvStatus').style.color = 'var(--muted)'; liveText('lvStatus', 'Unable to check'); $('lvVerification').dataset.status = 'error'; liveText('lvVerdict', 'Unable to check'); liveText('lvVerificationNote', 'Server offline. The latest result is unavailable.'); $('lvObjectLabel').hidden = true; liveText('lvObjectLabel', '')}
  LV.timer = setTimeout(liveTick, 150);
}
