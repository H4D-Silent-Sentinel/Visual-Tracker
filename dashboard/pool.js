/* Data Pool: photos (filters, lazy thumbnails, enlarge) and video cards with Review clip links. */
const SPLITS = [
  {id: 'train', label: 'Training: drones', split: 'train', note: 'Source: DUT Anti-UAV detection set · split: training. Every drone hand-marked.'},
  {id: 'bird', label: 'Training: birds', split: 'train_neg', kind: 'bird', note: 'Source: COCO · split: training (v2, v3). No drone; added so birds stop being called drones.'},
  {id: 'kite', label: 'Training: kites', split: 'train_neg', kind: 'kite', note: 'Source: COCO · split: training (v2, v3). No drone; added so kites stop being called drones.'},
  {id: 'trk', label: 'Training: tracking frames', split: 'train_trk', note: 'Source: DUT Anti-UAV tracking set · split: training (v3 only). Every 4th frame of 10 tracking clips, for cluttered backgrounds.'},
  {id: 'val', label: 'Validation: drones', split: 'val', note: 'Source: DUT Anti-UAV · split: validation. Not trained on; checked after each training pass.'},
  {id: 'val_neg', label: 'Validation: birds/kites', split: 'val_neg', note: 'Source: COCO · split: validation (v2, v3). No drone. Model boxes here are false alarms.'},
  {id: 'test', label: 'Test: drones', split: 'test', note: 'Source: DUT Anti-UAV · split: test. Never used in training. Precision and recall come from these.'}];
const P = {info: null, f: SPLITS[0], page: 0, shown: 0, total: 0, busy: false, io: null};
async function poolLoad() {
  const d = await api('/api/pool');
  if (!d.ready) {$('poolStats').textContent = 'Counting images…'; setTimeout(() => TAB === 'pool' && poolLoad(), 1000); return}
  const first = !P.info; P.info = d;
  const s = d.splits, n = k => (s[k] ? s[k].images : 0), fmt = x => x.toLocaleString();
  $('poolStats').innerHTML = `<b>${fmt(n('train') + n('val') + n('test'))}</b> drone photos · <b>${fmt(n('train_neg') + n('val_neg'))}</b> bird/kite photos with no drone · <b>${fmt(n('train_trk'))}</b> tracking frames · <b>${R.clips.length}</b> video clips. All public data; our own footage will appear under Collect.`;
  const cnt = f => f.kind ? ((s[f.split] || {}).kinds || {})[f.kind] || 0 : n(f.split);
  $('splitChips').innerHTML = SPLITS.map(f => `<button class="chip" data-f="${f.id}" aria-pressed="${f === P.f}">${f.label}<span>${fmt(cnt(f))}</span></button>`).join('');
  $('splitChips').querySelectorAll('button').forEach(b => b.onclick = () => {P.f = SPLITS.find(f => f.id === b.dataset.f); $('splitChips').querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b)); syncPred(); loadImgs(true)});
  if (first) {
    $('sizeSel').innerHTML = '<option value="">any</option>' + d.sizes.map(z => `<option value="${z.id}">${esc(z.label)}</option>`).join('');
    $('sizeSel').onchange = () => loadImgs(true); $('predSel').onchange = () => {syncPred(); loadImgs(true)}; $('resSel').onchange = () => loadImgs(true);
    $('moreBtn').onclick = () => loadImgs(false);
    $('imgs').onclick = e => {const b = e.target.closest('.th'); if (!b) return; $('boxImg').src = b.dataset.u + '&w=1200'; $('boxCap').textContent = b.dataset.c; $('box').showModal()};
    $('boxClose').onclick = () => $('box').close(); $('box').onclick = e => {if (e.target === $('box')) $('box').close()};
    P.io = new IntersectionObserver(es => {if (es.some(e => e.isIntersecting) && TAB === 'pool' && P.shown < P.total) loadImgs(false)}, {rootMargin: '300px'});
    P.io.observe($('moreBtn'));
  }
  syncPred(); renderVids();
  if (first) loadImgs(true);
}
function syncPred() {
  const d = P.info, sp = P.f.split, sel = $('predSel'), cur = sel.value;
  const ms = Object.keys(d.photo_predictions).filter(m => d.photo_predictions[m].includes(sp));
  sel.innerHTML = '<option value="">off</option>' + ['v1', 'v2', 'v3'].map(m => `<option value="${m}" ${ms.includes(m) ? '' : 'disabled'}>${m}${ms.includes(m) ? '' : (['test', 'val_neg'].includes(sp) ? ' (not generated yet)' : ' (test and validation bird/kite photos only)')}</option>`).join('');
  sel.value = ms.includes(cur) ? cur : '';
  $('resSel').disabled = !sel.value; if (!sel.value) $('resSel').value = '';
}
async function loadImgs(reset) {
  if (P.busy && !reset) return;
  if (reset) {P.page = 0; P.shown = 0; $('imgs').innerHTML = ''} else P.page++;
  P.busy = true;
  const f = P.f, pred = $('predSel').value, res = $('resSel').value, size = $('sizeSel').value;
  $('imgNote').textContent = f.note + (pred ? ` Teal brackets = true drone; orange box = ${pred}'s box.` : ' Teal brackets = true drone.');
  let q = `split=${f.split}&page=${P.page}&per=48` + (f.kind ? `&kind=${f.kind}` : '') + (size ? `&size=${size}` : '') + (pred ? `&pred=${pred}` : '') + (res ? `&result=${res}` : '');
  const d = await api('/api/images?' + q);
  P.busy = false;
  if (f !== P.f) return;
  const neg = f.split.endsWith('_neg');
  $('imgs').insertAdjacentHTML('beforeend', d.items.map(i => {
    const u = `/thumb?kind=img&split=${f.split}&name=${encodeURIComponent(i.name)}` + (pred ? `&pred=${pred}` : '');
    let cap = neg ? 'no drone' : i.boxes + ' drone' + (i.boxes === 1 ? '' : 's'), cls = '';
    if (pred && i.missed != null) {if (i.missed) {cap += ' · ✕ missed'; cls = 'miss'} if (i.false_alarm) {cap += ' · ▲ false alarm'; cls = cls || 'fa'} if (!i.missed && !i.false_alarm) cap += ' · ● ok'}
    return `<button class="th" data-u="${u}" data-c="${esc(i.name)} · ${esc(f.label)} · ${cap}"><img loading="lazy" decoding="async" src="${u}&w=360" alt="${esc(cap)}"><span class="${cls}">${cap}</span></button>`}).join(''));
  P.shown += d.items.length; P.total = d.total;
  $('moreInfo').textContent = d.total ? `showing ${P.shown.toLocaleString()} of ${d.total.toLocaleString()} (random order)` : 'No photos match these filters.';
  $('moreBtn').hidden = P.shown >= d.total;
}
function renderVids() {
  const groups = [...new Map(R.clips.map(c => [c.group, c])).values()];
  $('vids').innerHTML = groups.map(g => {
    const vs = R.clips.filter(c => c.group === g.group);
    return `<h2 style="font-size:14px">${esc(g.group_label)} <span class="muted" style="font-weight:400">· ${vs.length} clips · source: ${esc(g.source)}</span></h2><div class="vids">` + vs.map(c => {
      const lines = R.outputs.map(o => {const s = c.outputs[o.id].summary; return s ? (s.kind === 'tracking' ? `${o.label}: on target ${s.on_target_pct}%` : `${o.label}: false alarm in ${s.fa35}% of frames`) : null}).filter(Boolean);
      return `<div class="vid"><img loading="lazy" decoding="async" src="/thumb?kind=clip&clip=${c.key}&w=480" alt=""><div class="b"><div class="t">${esc(c.name)}</div>
      <div class="m">${mmss(c.frames / c.fps)} · ${c.w}×${c.h} ${c.split ? `<span class="tag ${c.split === 'held out' ? 'held' : 'trained'}">${c.split === 'held out' ? '✓ held out' : '! v3 trained on it'}</span>` : ''}</div>
      <div class="m">${lines.length ? lines.join('<br>') : 'Not scored (no ground truth)'}</div><button class="btn" data-k="${c.key}">Review clip ›</button></div></div>`}).join('') + '</div>'}).join('');
  $('vids').querySelectorAll('button[data-k]').forEach(b => b.onclick = () => openClip(b.dataset.k, 0));
}
