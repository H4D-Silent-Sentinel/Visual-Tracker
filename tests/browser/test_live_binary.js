/* Run with: node tests/browser/test_live_binary.js */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const dashboard = path.resolve(__dirname, '../../dashboard');
const html = fs.readFileSync(path.join(dashboard, 'index.html'), 'utf8');
const source = fs.readFileSync(path.join(dashboard, 'live.js'), 'utf8');
const css = fs.readFileSync(path.join(dashboard, 'style.css'), 'utf8');
const panel = html.match(/<div id="lvVerification"[\s\S]*?<h3>YOLO confidence/)[0];
assert(!/VLM confidence|lvVlmConfidence|lvVlmLabel|lv-vlm-score|calibrated|raw model score|%/i.test(panel));
assert(!/lvVlmConfidence|lvVlmLabel|v\.confidence|v\.calibrated/.test(source));
assert(!source.includes('classification'), 'Live reads the canonical VLM label directly');
assert(!html.includes('YOLO + local VLM'));
assert(!css.includes('lv-vlm-score'));
const ids = [...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]);
assert.equal(new Set(ids).size, ids.length, 'HTML IDs stay unique');
const elements = Object.fromEntries(ids.map(id => [id, {
  textContent: '', value: '', style: {}, dataset: {}, classes: new Set(),
  classList: {
    toggle(c, yes) {yes ? elements[id].classes.add(c) : elements[id].classes.delete(c)},
    remove(c) {elements[id].classes.delete(c)},
  },
  getBoundingClientRect: () => ({width: 300}), getAttribute: () => 64,
  getContext: () => new Proxy({}, {get: () => () => {}}),
  insertAdjacentHTML: () => {}, removeAttribute(name) {delete this[name]},
}]));
const requests = [], apiRequests = [], timers = new Map();
let stats, offline = false, timerId = 0, keyListeners = 0;
const context = vm.createContext({
  $: id => {assert(elements[id], `Missing element ${id}`); return elements[id]},
  document: {activeElement: null}, TAB: 'live', devicePixelRatio: 1,
  addEventListener: name => {if (name === 'keydown') keyListeners++},
  inControl: () => false, esc: String, mmss: String,
  setTimeout: fn => {timers.set(++timerId, fn); return timerId},
  clearTimeout: id => timers.delete(id),
  fetch: async url => requests.push(url),
  api: async url => {
    apiRequests.push(url);
    if (offline) throw Error('offline');
    return url === '/live/sources' ? {
      sources: ['footage/one.mp4', 'footage/two.mp4', 'camera:0'],
      models: [{id: 'v1', ok: true, label: 'v1'}], camera_names: {},
    } : stats;
  },
});
vm.runInContext(source, context);
const target = (status, extra = {}) => ({id: 7, conf: .63, range_m: 42, verification: {status, ...extra}});
const snapshot = (status, primary, candidate, verify = true) => ({
  status, primary, candidate, verification: candidate?.verification || primary?.verification,
  settings: {verify, paused: false, source: 'footage/one.mp4', tracker: 'default', model: 'v1'},
  targets: [primary || candidate].filter(Boolean), source_fps: 30, total: 0, fps: 24,
  locked_pct: 20, current_lock_s: 1, longest_lock_s: 2, reacquisitions: 0, conf_hist: [], trail: [],
});
async function tick(next) {
  stats = next;
  await context.liveTick();
  const text = ['lvVerdict', 'lvVerificationNote', 'lvObjectLabel'].map(id => elements[id].textContent).join(' ');
  assert(!text.includes('%'), 'Verification never displays percentages');
  assert(!/confidence|calibrat|raw model|unverified|false review/i.test(text), 'Verification uses plain result language');
}

(async () => {
  stats = snapshot('SEARCHING', null, null, false);
  await context.liveEnter();
  await context.liveEnter();
  assert.equal(keyListeners, 1, 'Reentering Live does not duplicate shortcuts');
  elements.lvVerify.onchange({target: {checked: true}});
  assert(requests.includes('/live/set?verify=true'));

  await tick(snapshot('SEARCHING', null, null));
  assert.equal(elements.lvVerdict.textContent, 'Drone not detected');
  assert.equal(elements.lvVerificationNote.textContent, 'YOLO has not proposed a candidate.');
  assert.equal(elements.lvStatus.textContent, 'Drone not detected');
  assert(elements.lvObjectLabel.hidden);

  const negative = target('rejected', {label: 'bird', is_drone: false});
  await tick(snapshot('FALSE_REVIEW', null, negative));
  assert.equal(elements.lvVerdict.textContent, 'Drone not detected');
  assert.equal(elements.lvVerificationNote.textContent, 'VLM rejected this YOLO candidate.');
  assert.equal(elements.lvObjectLabel.textContent, 'Object label: bird');
  assert.equal(elements.lvC.innerHTML, '63<small>%</small>', 'Detector confidence remains visible');
  assert.equal(elements.lvN.textContent, 0);
  await tick(snapshot('FALSE_REVIEW', null, target('rejected', {label: 'helicopter', is_drone: false})));
  assert.equal(elements.lvVerdict.textContent, 'Drone not detected');
  assert.equal(elements.lvObjectLabel.textContent, 'Object label: helicopter', 'Other VLM object labels pass through');

  for (const [review, status, title] of [
    ['pending', 'REVIEWING', 'Checking for drone'],
    ['unverified', 'UNVERIFIED', 'Checking for drone'],
    ['uncertain', 'UNVERIFIED', 'Unable to check'],
    ['error', 'VERIFY_ERROR', 'Unable to check'],
  ]) {
    await tick(snapshot(status, null, target(review)));
    assert.equal(elements.lvVerdict.textContent, title);
    assert.equal(elements.lvStatus.textContent, title);
    assert(!elements.lvStatus.classes.has('locked'));
    assert.equal(elements.lvN.textContent, 0);
  }
  await tick(snapshot('UNVERIFIED', null, target('unverified', {error: 'Too small'})));
  assert.equal(elements.lvVerdict.textContent, 'Unable to check');

  const positive = target('confirmed', {label: 'drone', is_drone: true});
  const mixed = snapshot('LOCKED', positive, negative); mixed.targets = [positive, negative];
  await tick(mixed);
  assert.equal(elements.lvVerdict.textContent, 'Drone detected', 'Backend verdict governs binary display');
  assert.equal(elements.lvStatus.textContent, 'Drone detected');
  assert.equal(elements.lvObjectLabel.textContent, 'Object label: drone');
  assert.equal(elements.lvVerificationNote.textContent, 'VLM identified this candidate as a drone.');
  assert(elements.lvStatus.classes.has('locked'));
  assert.equal(elements.lvN.textContent, 1);
  await tick(snapshot('LOCKED', target('pending'), null));
  assert.equal(elements.lvVerdict.textContent, 'Checking for drone');
  assert(!elements.lvStatus.classes.has('locked'));

  await tick(snapshot('LOCKED', target('unverified'), null, false));
  assert.equal(elements.lvVerdict.textContent, 'Verification off');
  assert(elements.lvStatus.classes.has('locked'), 'Detector-only mode retains locks');
  assert(elements.lvObjectLabel.hidden);
  assert.equal(elements.lvN.textContent, 1);
  elements.lvSource.value = 'footage/one.mp4'; context.liveStep(1);
  assert.equal(elements.lvSource.value, 'footage/two.mp4');
  context.liveStep(1);
  assert.equal(elements.lvSource.value, 'footage/one.mp4', 'Source stepping skips cameras');
  elements.lvSource.onchange({target: {value: 'camera:0'}});
  assert(requests.includes('/live/set?source=camera%3A0'));
  context.liveLeave();
  assert.equal(elements.lvImg.src, undefined, 'Leaving Live disconnects the stream');
  const before = apiRequests.length;
  context.TAB = 'review'; await context.liveTick();
  assert.equal(apiRequests.length, before, 'Other tabs do not poll Live stats');
  context.TAB = 'live'; offline = true; await context.liveTick();
  assert.equal(elements.lvVerdict.textContent, 'Unable to check');
  assert(!elements.lvStatus.classes.has('locked'));
  assert(elements.lvObjectLabel.hidden);
  assert(requests.every(url => !url.includes('/verify')), 'UI never schedules verification inference');
  console.log('Live binary UI checks passed.');
})().catch(error => {console.error(error); process.exitCode = 1});
