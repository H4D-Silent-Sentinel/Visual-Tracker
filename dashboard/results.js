/* Results: did the model improve on the same footage? Every table states its clips, split and metric. */
let resTimer;
async function resultsLoad() {
  clearTimeout(resTimer);
  const d = await api('/api/results'), el = $('results');
  const outs = d.outputs, names = k => k.replace('dut__', '').replace('negatives__', '');
  const pend = why => `<span class="pending" title="${esc(why)}">pending</span>`;
  const st = Object.fromEntries(d.training.map(t => [t.model, t]));
  const whyOut = o => {const t = st[o.model]; return t.usable ? `${o.label}: the test has not been run yet` : `${o.label}: model is ${t.state}`};
  const best = (vals, low) => {const xs = vals.filter(v => v != null); return xs.length > 1 ? (low ? Math.min(...xs) : Math.max(...xs)) : null};
  const row = (name, sub, get, fmt, low) => {
    const vals = outs.map(o => {const a = d.held_out[o.id]; return a ? get(a) : null}), b = best(vals, low);
    return `<tr><td>${name}<small>${sub}</small></td>` + outs.map((o, i) => `<td class="num">${vals[i] == null ? pend(whyOut(o)) : (vals[i] === b ? '<b>' + fmt(vals[i]) + '</b> ◀' : fmt(vals[i]))}</td>`).join('') + '</tr>';
  };
  const pct = v => v.toFixed(1) + '%';
  const anyHeld = Object.values(d.held_out).find(Boolean);

  /* training status, straight from the logs */
  const status = d.training.map(t => {
    let s = `${t.label} · ${t.state}`;
    if (t.epochs_planned && !t.state.startsWith('stopped')) s += ` · pass ${t.epochs_done} of ${t.epochs_planned} done`;
    if (t.last_update) s += ` · results file last updated ${hhmm(t.last_update)}`;
    if (t.state === 'training' && t.eta_s != null) s += ` · about ${Math.max(1, Math.round(t.eta_s / 60))} min left (from ${Math.round(t.sec_per_epoch / 60 * 10) / 10} min per pass so far)`;
    if (t.state === 'training' && t.eta_s == null) s += ' · no time estimate until the first pass finishes';
    return `<div>${esc(s)}</div>`}).join('') +
    `<div>${d.queue.retest_v2 ? 'Queued: Test 1 and Test 2 on v2 as soon as it finishes. ' : ''}${d.queue.run_v3 ? 'Queued: v3 training and its tests. ' : ''}${d.queue.make_review ? 'Queued or running: per-frame review files for Clip Review. ' : ''}${!(d.queue.retest_v2 || d.queue.run_v3 || d.queue.make_review) ? 'Nothing queued.' : ''}</div>`;

  /* held-out comparison */
  const split = d.split;
  const heldTxt = split.verified ? `${split.held_out.length} held-out clips (${split.held_out.map(names).join(', ')})` : 'Held-out split not verified yet';
  /* one-paragraph answer, computed from the same numbers as the tables */
  let bottom = '';
  try {
    const H = d.held_out, F = d.false_alarms, v17 = (d.held_out_clips.find(c => c.name === 'video17') || {}).by || {};
    const done = outs.filter(o => H[o.id]), bestOn = done.reduce((a, o) => H[o.id].on_target_pct > H[a.id].on_target_pct ? o : a, done[0]);
    const fdone = Object.keys(F).filter(k => F[k]), bestF = fdone.reduce((a, k) => F[k]['false_alarm_pct@0.35'] < F[a]['false_alarm_pct@0.35'] ? k : a, fdone[0]);
    const lab = id => outs.find(o => o.id === id).label;
    bottom = `<p class="bottom"><b>Bottom line, same footage:</b> ${lab(bestF)} cut bird/kite false alarms from ${F.v1['false_alarm_pct@0.35']}% to <b>${F[bestF]['false_alarm_pct@0.35']}%</b> of frames. ` +
      `Time on target barely moved (v1 ${H.v1.on_target_pct}%); the best is <b>${bestOn.label} at ${H[bestOn.id].on_target_pct}%</b>, with ${H[bestOn.id].false_alarm_pct}% wrong-object boxes. ` +
      (v17.v1 && v17.v3 ? `Drones in front of trees are still the gap: video17 went from ${v17.v1.on_target_pct}% (v1) to ${v17.v3.on_target_pct}% (v3).` : '') + '</p>';
  } catch (e) {}
  let h = `${bottom}<h2>What each version is</h2>${versionsHTML(d.training)}<h2>Training status</h2><div class="statuslist">${status}</div>
  <h2>Did tracking improve on the same footage?</h2>
  <p class="cap"><b>${heldTxt}</b>${anyHeld ? `, ${anyHeld.frames.toLocaleString()} frames` : ''}. No model trained on these clips. Excluded: the ${split.trained.length} clips v3 trained on (${split.trained.map(names).join(', ')}); ${split.how}. “+ hold” is the same model with the tracker set to keep a locked drone through weak detections.</p>
  <div class="tw"><table><tr><th>Metric</th>${outs.map(o => `<th class="num">${esc(o.label)}</th>`).join('')}</tr>
  ${row('On target', 'share of drone-visible frames with a box on the true drone (IoU ≥ 0.3). Higher is better.', a => a.on_target_pct, pct, false)}
  ${row('Wrong-object box', 'share of all frames with a box and none of them on the drone. Lower is better.', a => a.false_alarm_pct, pct, true)}
  ${row('Track ID switches', 'times the same drone got a new ID. Lower is better.', a => a.id_switches, v => String(v), true)}
  ${row('Time to reacquire', 'average gap after losing the drone, at 30 frames per second. Lower is better.', a => a.avg_reacquire_s, v => v + ' s', true)}
  </table></div>
  <p class="cap">${Object.values(d.held_out).filter(Boolean).length < 2 ? 'The comparison is <b>pending</b>: fewer than two outputs have been tested on these clips.' : '◀ marks the best value in a row.'} These are tracking scores against marked drone positions, not precision or recall.</p>

  <h2>Held-out clips, one by one</h2>
  <p class="cap">On target % per clip (same definition as above). Click a row to open the clip in Clip Review at its worst moment, when per-frame data exists.</p>
  <div class="tw"><table><tr><th>Clip</th>${outs.map(o => `<th class="num">${esc(o.label)}</th>`).join('')}</tr>
  ${d.held_out_clips.map(c => `<tr class="link" tabindex="0" data-k="${c.key}"><td>${esc(c.name)} <span class="muted">›</span></td>` + outs.map(o => `<td class="num">${c.by[o.id] ? c.by[o.id].on_target_pct + '%' : pend(whyOut(o))}</td>`).join('') + '</tr>').join('')}
  </table></div>`;

  /* all 20, only for models that trained on none of them */
  const a20 = Object.entries(d.all20).filter(([, v]) => v);
  if (a20.length) h += `<h2>All 20 tracking clips (v1 and v2 only)</h2>
  <p class="cap">v3 and v4 are left out because they trained on 10 of these clips. Same metric definitions; ${a20[0][1].frames.toLocaleString()} frames.</p>
  <div class="tw"><table><tr><th>Output</th><th class="num">On target</th><th class="num">Wrong-object box</th><th class="num">ID switches</th><th class="num">Reacquire</th></tr>
  ${a20.map(([id, v]) => `<tr><td>${esc(outs.find(o => o.id === id).label)}</td><td class="num">${v.on_target_pct}%</td><td class="num">${v.false_alarm_pct}%</td><td class="num">${v.id_switches}</td><td class="num">${v.avg_reacquire_s ?? '—'} s</td></tr>`).join('')}</table></div>`;

  /* false alarms */
  const fo = outs.filter(o => o.id in d.false_alarms);
  h += `<h2>False alarms on birds and kites</h2>
  <p class="cap"><b>${d.false_alarm_clips.length} clips with no drone in them</b> (Wikimedia Commons; never used for training). Metric: share of tested frames (every 2nd frame) where the model drew any box. This is a share of frames with a box, not precision. Lower is better.</p>
  <div class="tw"><table><tr><th>Confidence cutoff</th>${fo.map(o => `<th class="num">${esc(o.label)}</th>`).join('')}</tr>
  ${[['0.25', 'false_alarm_pct@0.25'], ['0.35 (default)', 'false_alarm_pct@0.35'], ['0.50', 'false_alarm_pct@0.5']].map(([n, k]) => `<tr><td>${n}</td>` + fo.map(o => `<td class="num">${d.false_alarms[o.id] ? d.false_alarms[o.id][k] + '%' : pend(whyOut(o))}</td>`).join('') + '</tr>').join('')}
  </table></div>
  <div class="tw" style="margin-top:10px"><table><tr><th>Clip (cutoff 0.35)</th>${fo.map(o => `<th class="num">${esc(o.label)}</th>`).join('')}</tr>
  ${d.false_alarm_clips.map(c => `<tr class="link" tabindex="0" data-k="${c.key}"><td>${esc(c.name)} <span class="muted">›</span></td>` + fo.map(o => `<td class="num">${c.by[o.id] ? c.by[o.id].fa35 + '%' : pend(whyOut(o))}</td>`).join('') + '</tr>').join('')}
  </table></div>`;

  /* labeled precision / recall */
  const ms = ['v1', 'v2', 'v3', 'v4'];
  h += `<h2>Labeled photo test: precision and recall</h2>
  <p class="cap"><b>2,200 held-out test photos</b> (DUT Anti-UAV test split, every drone hand-marked, never trained on). Precision = when it says drone, how often it is right. Recall = share of real drones it finds. These are per-box scores on still photos, separate from the frame shares above.</p>
  <div class="tw"><table><tr><th>Model</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">mAP50</th></tr>
  ${ms.map(m => {const p = d.photo_test[m]; return `<tr><td>${m}</td>` + (p ? `<td class="num">${(p.precision * 100).toFixed(1)}%</td><td class="num">${(p.recall * 100).toFixed(1)}%</td><td class="num">${(p.map50 * 100).toFixed(1)}%</td>` : `<td class="num" colspan="3">${pend(st[m].usable ? 'runs with the review files, after training finishes' : 'model is ' + st[m].state)}</td>`) + '</tr>'}).join('')}
  </table></div>`;

  /* by drone size */
  const so = outs.filter(o => d.sizes[o.id]);
  h += `<h2>On target by drone size</h2><p class="cap">Held-out clips only. Size = longer side of the true drone box in the original frame.${so.length ? '' : ' <b>Pending</b>: needs the per-frame review files.'} Background type is not broken out because clips are not labeled by background.</p>`;
  if (so.length) {const buckets = Object.keys(d.sizes[so[0].id]);
    h += `<div class="tw"><table><tr><th>Drone size</th><th class="num">Frames</th>${so.map(o => `<th class="num">${esc(o.label)}</th>`).join('')}</tr>
    ${buckets.filter(b => d.sizes[so[0].id][b].visible > 0).map(b => `<tr><td>${b}</td><td class="num">${d.sizes[so[0].id][b].visible.toLocaleString()}</td>` + so.map(o => `<td class="num">${d.sizes[o.id][b].on_target_pct == null ? '—' : d.sizes[o.id][b].on_target_pct + '%'}</td>`).join('') + '</tr>').join('')}</table></div>`}

  /* run history */
  h += `<h2>Training runs</h2><p class="cap">Validation scores are from each run's own check photos; v2 and v3 include 300 bird/kite photos there, so compare models with the tables above, not this column.</p>
  <div class="tw"><table><tr><th>Model</th><th>Trained on</th><th>Started</th><th>State</th><th class="num">Passes</th><th class="num">Val precision</th><th class="num">Val recall</th></tr>
  ${d.training.map(t => `<tr><td>${t.label}</td><td>${esc(t.data)}</td><td>${t.started ? new Date(t.started * 1000).toLocaleString([], {month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'}) : '—'}</td><td>${esc(t.state)}</td><td class="num">${t.epochs_done}${t.epochs_planned ? ' / ' + t.epochs_planned : ''}</td><td class="num">${t.val ? (t.val.precision * 100).toFixed(1) + '%' : '—'}</td><td class="num">${t.val ? (t.val.recall * 100).toFixed(1) + '%' : '—'}</td></tr>`).join('')}
  </table></div>`;
  el.innerHTML = h;
  el.querySelectorAll('tr.link').forEach(tr => {
    const open = () => {const c = R.byKey[tr.dataset.k]; if (!c) return;
      const out = ['v3', 'v2', 'v1'].find(id => c.outputs[id] && c.outputs[id].available) || R.out;
      const w = c.outputs[out] && c.outputs[out].worst;
      openClip(c.key, w != null ? Math.max(0, w - Math.round(c.fps)) : 0, out)};
    tr.onclick = open; tr.onkeydown = e => {if (e.key === 'Enter') open()}});
  const busy = d.training.some(t => t.state === 'training') || d.queue.retest_v2 || d.queue.run_v3 || d.queue.make_review;
  if (busy) resTimer = setTimeout(() => {if (TAB === 'results') {refreshClips(); resultsLoad()}}, 30000);
}
