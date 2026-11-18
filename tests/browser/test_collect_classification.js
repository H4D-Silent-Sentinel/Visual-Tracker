/* Run with: node tests/browser/test_collect_classification.js */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'dashboard/index.html'), 'utf8');
const source = fs.readFileSync(path.join(root, 'dashboard/collect.js'), 'utf8');
const labels = [...fs.readFileSync(path.join(root, 'candidate_labels.py'), 'utf8').split('LABELS = (')[1].matchAll(/"([^"]+)"/g)].map(match => match[1]);
const ids = [...html.matchAll(/id="([^"]+)"/g)].map(match => match[1]);
assert.equal(ids.length, new Set(ids).size);
const drawnLabels = [], elements = {};
function element() {
  return {value: '', textContent: '', style: {}, dataset: {}, checked: false, disabled: false,
    clientWidth: 400, clientHeight: 200, readyState: 1, options: [],
    set innerHTML(text) {this.html = text; this.options = [...text.matchAll(/<option value="([^"]*)"/g)].map(match => ({value: match[1]})); if (!this.options.some(option => option.value === this.value)) this.value = this.options[0]?.value || ''},
    get innerHTML() {return this.html || ''},
    querySelectorAll: () => [], classList: {toggle() {}, add() {}, remove() {}},
    load() {}, addEventListener() {}, blur() {}, removeAttribute(name) {delete this[name]},
    setAttribute(name, value) {this[name] = value}, getAttribute(name) {return this[name]},
    getBoundingClientRect: () => ({width: 400, height: 200, left: 0, top: 0}),
    getContext: () => new Proxy({}, {get: (_, name) => name === 'measureText' ? () => ({width: 60}) : name === 'fillText' ? text => drawnLabels.push(text) : () => {}}),
  };
}
for (const id of [...ids, 'camStartIn']) elements[id] = element();
elements.labStride.value = '5'; elements.labSplit.value = 'test';
const clip = {key: 'ours__one', name: 'ours_one', group: 'ours', frames: 30, fps: 30, w: 100, h: 50, url: '/media/one.mp4'};
const other = {...clip, key: 'ours__two', name: 'ours_two'};
const box = [.1, .2, .4, .6], legacy = {'0': [[.2, .2, .5, .5]], '1': [box], '29': [box]};
const annotations = {'0': {boxes: [box], label: 'bird', no_drone: false, split: 'development', source_id: 'same-session'}};
const requests = [];
let holdProposal = false, releaseProposal, exportedTestCount = 1;
const context = vm.createContext({
  $: id => {if (id === 'recBadge') return null; assert(elements[id], `Missing ${id}`); return elements[id]},
  R: {clips: [clip, other], byKey: {[clip.key]: clip, [other.key]: other}}, TAB: 'collect',
  document: {activeElement: null}, localStorage: {getItem: () => null, setItem() {}},
  devicePixelRatio: 1, addEventListener() {}, inControl: () => false, clearInterval() {}, setInterval: () => 1, setTimeout: () => 1,
  window: {}, esc: String, mmss: String, toast() {},
  api: async (url, body) => {
    requests.push({url, body});
    if (url === '/api/collect') {
      const rows = Object.values(annotations);
      return {flags: [], camera: {training_active: false}, cameras: [], labeled: {}, class_labels: labels,
        recordings: [{clip: clip.key, name: clip.name, seconds: 1, labeled: 0, classified: rows.length, meta: {source_id: 'recording-session'}}],
        dataset: {frames: 0, hard_examples: 0}, coverage: [], ours: [clip.key], classifications: {[clip.key]: rows.length},
        classification_dataset: {frames: rows.length, development: rows.filter(row => row.split === 'development').length,
          test: rows.filter(row => row.split === 'test').length, no_drone: rows.filter(row => row.no_drone).length}};
    }
    if (url.startsWith('/api/labels?')) return {labels: legacy, classifications: annotations};
    if (url === '/api/classification/label') {
      if (body.delete) {delete annotations[body.frame]; return {ok: true}}
      const annotation = {boxes: body.boxes, label: body.object_label, no_drone: body.no_drone, split: body.split, source_id: body.source_id};
      annotations[body.frame] = annotation;
      return {ok: true, annotation};
    }
    if (url.startsWith('/api/propose?')) {
      if (holdProposal) return new Promise(resolve => {releaseProposal = resolve});
      return {ok: true, boxes: [[...box, .9]], source: 'YOLO'};
    }
    if (url === '/api/collections/export') return {ok: true, frames: 3, test_count: exportedTestCount, development_count: 2,
      test_manifest: '/tmp/frozen/test.jsonl', development_manifest: '/tmp/frozen/development.jsonl', command: 'python test_classification.py --manifest /tmp/frozen/test.jsonl'};
    throw Error('Unexpected request: ' + url);
  },
});
vm.runInContext(source, context);
const state = () => vm.runInContext('C', context);

(async () => {
  await context.collectLoad();
  await context.labLoad(clip.key);
  assert.equal(state().frame, 5, 'Classification resumes from manual class labels rather than legacy training progress');
  assert.equal(elements.labClass.options.length, labels.length + 1, 'All shared classes plus the required placeholder');
  context.labGo(0);
  assert.equal(elements.labClass.value, 'bird');
  assert.equal(elements.labSplit.value, 'development');
  assert.equal(elements.labSource.value, 'same-session');
  assert.equal(JSON.stringify(state().boxes), JSON.stringify([box]), 'Manual class boxes override legacy drone boxes');
  assert(drawnLabels.includes('bird'), 'Saved overlay retains the broad ground-truth label');
  context.labGo(1);
  assert.equal(elements.labClass.value, '', 'Legacy drone boxes never select a new true label');
  assert(elements.labDelete.disabled, 'Legacy training labels are separate from manual class deletion');

  context.labGo(5);
  await context.labPropose();
  assert.equal(elements.labClass.value, '', 'YOLO suggestions never self-label ground truth');
  let count = requests.filter(request => request.url === '/api/classification/label').length;
  await context.labSave([box]);
  assert.equal(requests.filter(request => request.url === '/api/classification/label').length, count);
  assert.match(elements.labStatus.textContent, /Choose the true object label/);
  elements.labClass.value = 'kite'; elements.labSource.value = 'same-session';
  let classBlurred = false; elements.labClass.blur = () => {classBlurred = true};
  elements.labClass.onchange({target: elements.labClass});
  assert(classBlurred, 'Choosing a class releases focus for the S shortcut');
  await context.labSave([box]);
  assert.equal(annotations['5'].label, 'kite');
  assert.equal(annotations['5'].split, 'development');
  assert.equal(state().frame, 10);
  assert.equal(elements.labClass.value, '', 'Every new frame requires an explicit class choice');
  assert(!elements.collectionExport.disabled);
  elements.labClass.value = 'drone';
  await context.labSave([]);
  assert.equal(annotations['10'].label, null);
  assert.equal(annotations['10'].no_drone, true, 'N explicitly records background with no box');
  context.labGo(5); await context.labDelete();
  assert.equal(annotations['5'], undefined);
  assert(!requests.some(request => request.url === '/api/label'), 'New Collect actions never write legacy training labels');

  count = requests.filter(request => request.url === '/api/collections/export').length;
  await elements.collectionExport.onclick();
  assert.equal(requests.filter(request => request.url === '/api/collections/export').length, count);
  elements.collectionName.value = 'fixture_test'; await elements.collectionExport.onclick();
  assert.match(elements.collectionMsg.textContent, /\/tmp\/frozen\/test.jsonl/);
  assert(!elements.collectionCommand.hidden);
  assert.match(elements.collectionCommand.textContent, /--manifest/);
  exportedTestCount = 0; await elements.collectionExport.onclick();
  assert(elements.collectionCommand.hidden, 'An empty test split is not presented as ready to run');
  assert.match(elements.collectionMsg.textContent, /No test frames yet/);
  context.camState({recording: true, overlay: true, rec_frames: 1, training_active: false});
  assert.equal(elements.camOverlay.textContent, 'Model boxes: off while recording');
  assert.equal(elements.camOverlay['aria-pressed'], false);

  context.labGo(20); holdProposal = true;
  const pending = context.labPropose();
  state().clip = other; state().boxes = [];
  releaseProposal({ok: true, boxes: [[...box, .9]], source: 'YOLO'});
  await pending;
  assert.equal(state().boxes.length, 0, 'A proposal for the previous clip cannot overwrite the new clip');
  console.log('Collect classification UI checks passed.');
})().catch(error => {console.error(error); process.exitCode = 1});
