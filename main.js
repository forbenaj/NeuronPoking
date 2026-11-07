
/* ================= constants ================= */
const GAIN = 4;                                  // gain on a neuron's activation = 1 + GAIN * value, value in [-1, 1]
const MODELS = {
  gemma: {
    name: 'Gemma 3 1B IT', id: 'gemma-3-1b-it-q4f16_1-MLC', layers: 26, neurons: 6912, columns: 96, rows: 72,
    repo: 'https://huggingface.co/mlc-ai/gemma-3-1b-it-q4f16_1-MLC',
    lib: 'https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_80/gemma-3-1b-it-q4f16_1-ctx4k_cs1k-webgpu.wasm'
  },
  qwen: {
    name: 'Qwen2.5 0.5B Instruct', id: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC', layers: 24, neurons: 4864, columns: 76, rows: 64,
    repo: 'https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    lib: 'https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_80/Qwen2-0.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm'
  }
};
const WEBLLM = 'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.80/+esm';
const KEY = 'neuron-poking-v1';
const MODE_KEY = `${KEY}:mode`;
const SETTINGS_KEY = `${KEY}:view-settings`;
const DEFAULT_SETTINGS = { max: 1, zoomLevel: 1, tool: 'brush', brushSize: 1, paintAmount: 100 };
let knobMode = false, settings = { ...DEFAULT_SETTINGS };
try {
  const savedMode = localStorage.getItem(MODE_KEY);
  knobMode = savedMode === 'true' || savedMode === 'knobs';
  const savedSettings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
  const legacyView = savedSettings[knobMode ? 'knobs' : 'pixels'];
  const legacyPixels = savedSettings.pixels;
  const max = Number.isFinite(savedSettings.max) ? savedSettings.max : legacyView?.max;
  if (Number.isFinite(max)) settings.max = Math.max(.05, Math.min(20, max));
  if (Number.isFinite(savedSettings.zoomLevel))
    settings.zoomLevel = Math.max(.25, Math.min(8, savedSettings.zoomLevel));
  const savedTool = savedSettings.tool ?? legacyPixels?.tool;
  if (['brush', 'bucket', 'rectangle'].includes(savedTool) || savedTool === 'square')
    settings.tool = savedTool === 'square' ? 'rectangle' : savedTool;
  if (Number.isInteger(savedSettings.brushSize ?? legacyPixels?.brushSize))
    settings.brushSize = Math.max(1, Math.min(32, savedSettings.brushSize ?? legacyPixels.brushSize));
  if (Number.isInteger(savedSettings.paintAmount ?? legacyPixels?.paintAmount))
    settings.paintAmount = Math.max(-100, Math.min(100, savedSettings.paintAmount ?? legacyPixels.paintAmount));
} catch (e) { console.warn('Could not load view settings:', e); }
/** Persist view preferences to local storage and report storage failures. */
function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    localStorage.setItem(MODE_KEY, String(knobMode));
  } catch (e) { console.error('Could not save view settings:', e); }
}

let editTransaction = null;
const undoStack = [], redoStack = [];
/** Start collecting the before-values for one undoable edit. */
function beginEdit() { editTransaction = new Map(); }
/** Record an index's original value once during the current edit. */
function trackEdit(index, value) {
  if (editTransaction && !editTransaction.has(index)) editTransaction.set(index, value);
}
/** Finish the current edit, record actual changes, and refresh history controls. */
function finishEdit() {
  if (!editTransaction) return;
  const changes = [];
  for (const [index, before] of editTransaction) {
    const after = store.v[index];
    if (before !== after) changes.push([index, before, after]);
  }
  editTransaction = null;
  if (changes.length) {
    undoStack.push(changes);
    redoStack.length = 0;
    requestTesterRun();
  }
  updateHistoryButtons();
}
/** Discard pending edits and clear both undo and redo history. */
function clearEditHistory() {
  editTransaction = null;
  undoStack.length = 0;
  redoStack.length = 0;
  updateHistoryButtons();
}
/** Apply one undo or redo transaction and move it to the opposite history stack. */
function applyEditHistory(source, destination, useBefore) {
  const changes = source.pop();
  if (!changes) return;
  let changed = false;
  for (const [index, before, after] of changes) {
    const oldValue = store.v[index], nextValue = useBefore ? before : after;
    if (oldValue === nextValue) continue;
    const layer = Math.floor(index / store.model.neurons);
    store.v[index] = nextValue;
    store.cnt[layer] += (nextValue !== 0) - (oldValue !== 0);
    changed = true;
  }
  destination.push(changes);
  store.save();
  updateHistoryButtons();
  schedule();
  if (changed) requestTesterRun();
}

/* ================= fp16 helpers ================= */
const fb = new Float32Array(1), ub = new Uint32Array(fb.buffer);
/** Convert a JavaScript number to its IEEE 754 binary16 bit representation. */
const toF16 = x => { fb[0] = x; const b = ub[0], s = (b >>> 16) & 0x8000, e = ((b >>> 23) & 255) - 112, m = b & 0x7fffff;
  if (e >= 31) return s | 0x7c00;
  if (e <= 0) return e < -10 ? s : s | (((m | 0x800000) + (1 << (13 - e))) >> (14 - e));
  return s | ((e << 10) + ((m + 0x1000) >> 13)); };
/** Convert IEEE 754 binary16 bits to a JavaScript number. */
const toF32 = h => { const e = (h >> 10) & 31, m = h & 1023, s = h & 0x8000 ? -1 : 1;
  return e ? (e === 31 ? s * Infinity : s * (1 + m / 1024) * 2 ** (e - 15)) : s * m * 5.960464477539063e-8; };

/* ================= STEERING STORE (model-specific state, no UI) ================= */
class SteeringStore {
  /** Load the selected model's sparse steering values from local storage. */
  constructor(model) {
    this.model = model;
    this.key = model === MODELS.gemma ? KEY : `${KEY}:${model.id}`;
    this.v = new Float32Array(model.layers * model.neurons);
    this.cnt = new Int32Array(model.layers);
    try {
      const saved = JSON.parse(localStorage.getItem(this.key) || '{}');
      for (const [l, entries] of Object.entries(saved)) {
        const layer = +l;
        if (!Number.isInteger(layer) || layer < 0 || layer >= model.layers || !Array.isArray(entries)) continue;
        for (const [i, x] of entries) {
          if (!Number.isInteger(i) || i < 0 || i >= model.neurons || !Number.isFinite(x)) continue;
          this.v[layer * model.neurons + i] = x;
          if (x !== 0) this.cnt[layer]++;
        }
      }
    } catch (e) { console.warn('Could not load saved steering settings:', e); }
  }
  /** Return one neuron's activation for a layer. */
  get(l, i) { return this.v[l * this.model.neurons + i]; }
  /** Set an activation, clamp it to the current maximum, and persist the change. */
  set(l, i, x) { const k = l * this.model.neurons + i, o = this.v[k];
    x = Math.round(Math.max(-settings.max, Math.min(settings.max, x)) * 100) / 100 + 0; if (x === o) return;
    trackEdit(k, o);
    this.v[k] = x; this.cnt[l] += (x !== 0) - (o !== 0); this.save(); }
  /** Clear all activations in one layer. */
  resetLayer(l) { const start = l * this.model.neurons, end = start + this.model.neurons;
    for (let k = start; k < end; k++) if (this.v[k] !== 0) { trackEdit(k, this.v[k]); this.v[k] = 0; }
    this.cnt[l] = 0; this.save(); }
  /** Clear all activations for this model. */
  resetAll() { for (let k = 0; k < this.v.length; k++) if (this.v[k] !== 0) { trackEdit(k, this.v[k]); this.v[k] = 0; }
    this.cnt.fill(0); this.save(); }
  /** Return the number of modified neurons across all layers. */
  total() { return this.cnt.reduce((a, b) => a + b, 0); }
  /** Return the configured activations as a layer-indexed sparse object. */
  sparse() { const out = {}; for (let l = 0; l < this.model.layers; l++) if (this.cnt[l]) { const a = [];
    for (let i = 0; i < this.model.neurons; i++) { const x = this.v[l * this.model.neurons + i]; if (x) a.push([i, x]); } out[l] = a; } return out; }
  /** Debounce persistence of the current sparse steering state. */
  save() { clearTimeout(this.t); this.t = setTimeout(() => {
    try { localStorage.setItem(this.key, JSON.stringify(this.sparse())); }
    catch (e) { console.error('Could not save steering settings:', e); }
  }, 300); }
}

/* ================= STEERING ENGINE (runtime extension for WebLLM / TVM) =================
   WebLLM compiles each model layer into fused WebGPU kernels, so there is no hook point between
   operators. For these q4f16_1 models, the up-projection is
   4-bit symmetric group-quantised: weight = (q - 7) * scale. Multiplying the fp16 scales of up-row i
   by g multiplies h_i by exactly g, i.e. a gain on that neuron's activation for every token position.
   The scale tensors live in the model's WebGPU buffers; we rewrite them in place just before a generation. */
class Steerer {
  /** Bind this steering hook to WebLLM's TVM runtime and model metadata. */
  constructor(tvm, model) { this.tvm = tvm; this.model = model; this.orig = {}; this.sig = {}; }
  /** Find and validate this model's MLP projection-scale tensors. */
  async init() {
    for (const f of ['ndarray-cache.json', 'tensor-cache.json']) {
      try { const r = await fetch(`${this.model.repo}/resolve/main/${f}`); if (!r.ok) continue;
        const recs = (await r.json()).records.flatMap(x => x.records), m = {};
        for (const rec of recs) { const t = /layers\.(\d+)\.mlp\.(gate_up|up)_proj\.q_scale$/.exec(rec.name);
          if (t) m[+t[1]] = { name: rec.name, cols: rec.shape[1], off: t[2] === 'gate_up' ? this.model.neurons : 0, rows: rec.shape[0] }; }
        if (Object.keys(m).length === this.model.layers &&
            Array.from({ length: this.model.layers }, (_, l) => m[l]).every(x => x && x.rows === x.off + this.model.neurons)) {
          this.meta = m; this.scope(() => this.arr(m[0].name)); return; }
      } catch (e) { this.err = e; }
    }
    throw new Error(`Could not locate ${this.model.layers} MLP up-projection scale tensors (${this.err?.message || 'manifest mismatch'})`);
  }
  /** Run a TVM operation inside a managed temporary-object scope. */
  scope(fn) { this.tvm.beginScope(); try { return fn(); } finally { this.tvm.endScope(); } }
  /** Resolve a named parameter through the available TVM cache accessor. */
  arr(name) { const t = this.tvm, g = t.tensorCacheGet || t.ndarrayCacheGet;
    if (!g) throw new Error('TVM runtime has no tensor-cache accessor'); const a = g.call(t, name);
    if (!a) throw new Error('parameter not found: ' + name); return a; }
  /** Read a layer's current projection scales from GPU memory as binary16 values. */
  async read(l) { const t = this.tvm; t.beginScope(); try { const a = this.arr(this.meta[l].name);
      const c = t.empty(a.shape, a.dtype, t.cpu()); c.copyFrom(a); await t.webgpu().sync();
      return new Uint16Array(c.toRawBytes().slice().buffer); } finally { t.endScope(); } }
  /** Write a layer's binary16 projection scales back to GPU memory. */
  write(l, u16) { this.scope(() => this.arr(this.meta[l].name).copyFromRawBytes(new Uint8Array(u16.buffer))); }
  /** Apply sparse activation gains and verify changed layers by reading them back. */
  async apply(sparse) {
    let neurons = 0, layers = 0, verified = true;
    for (let l = 0; l < this.model.layers; l++) {
      const list = sparse[l] || [], sig = JSON.stringify(list);
      if (list.length) { neurons += list.length; layers++; }
      if (sig === (this.sig[l] || '[]')) continue;
      const m = this.meta[l]; if (!this.orig[l]) this.orig[l] = await this.read(l);
      const o = this.orig[l], w = o.slice();
      for (const [i, v] of list) { const g = 1 + GAIN * v, b = (m.off + i) * m.cols;
        for (let j = 0; j < m.cols; j++) w[b + j] = toF16(toF32(o[b + j]) * g); }
      this.write(l, w); this.sig[l] = sig;
      if (list.length) { const back = await this.read(l); for (let k = 0; k < w.length; k++) if (back[k] !== w[k]) { verified = false; break; } }
    }
    return { neurons, layers, verified };
  }
}

/* ================= MODEL RUNTIME (WebLLM wrapper, no UI) ================= */
class ModelRuntime {
  ready = false; steerer = null; steerErr = 'model not loaded';
  /** Load WebLLM and attach the experimental tensor steering hook if available. */
  async load(model, onProgress) {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    const webllm = await import(WEBLLM);
    this.engine = await webllm.CreateMLCEngine(model.id, {
      appConfig: { model_list: Object.values(MODELS).map(m => ({
        model: m.repo, model_id: m.id, model_lib: m.lib, required_features: ['shader-f16']
      })) },
      initProgressCallback: onProgress });
    this.ready = true;
    try { const mp = this.engine.loadedModelIdToPipeline, p = mp.get(model.id) || [...mp.values()][0];
      this.steerer = new Steerer(p.tvm, model); await this.steerer.init(); this.steerErr = null;
    } catch (e) { this.steerer = null; this.steerErr = e.message; }
  }
  /** Apply sparse neuron gains using the attached steering hook. */
  applySteering(sparse) { return this.steerer.apply(sparse); }
  /** Reset chat/KV state, stream a completion, and return its complete text. */
  async generate(messages, { temperature, max_tokens }, onToken) {
    await this.engine.resetChat();   // drop the KV cache: cached tokens would carry the previous steering
    const stream = await this.engine.chat.completions.create({ messages, stream: true, temperature, max_tokens });
    let full = '';
    for await (const c of stream) { const d = c.choices[0]?.delta?.content; if (d) { full += d; onToken(d); } }
    return full;
  }
  /** Interrupt the currently running WebLLM generation, if any. */
  stop() { this.engine?.interruptGenerate(); }
  /** Release the loaded model and reset runtime state. */
  async unload() {
    if (!this.engine) return;
    await this.engine.unload();
    this.engine = null;
    this.ready = false;
    this.steerer = null;
    this.steerErr = 'model not loaded';
  }
}

/* ================= UI: knob panel ================= */
/** Look up a required application element by its DOM id. */
const $ = id => document.getElementById(id);
let activeModel = MODELS.gemma, store;
const rt = new ModelRuntime();
let S = 1, fitS = 1;
const R = 8.5, GREEN = '#00a651', RED = '#e00000';
const cv = $('cv'), ctx = cv.getContext('2d'), sc = $('sc'), sp = $('sp'), left = $('left');
let layer = 0, cols = 1, rows = 1, offX = 0, offY = 0, W = 0, H = 0, canvasW = 0, canvasH = 0, contentW = 0, contentH = 0, dpr = 1, hover = -1, raf = 0, drag = null;
const layerButtons = [];

/** Recompute the grid's fit/zoom geometry and resize its drawing surface. */
function layout() {
  W = sc.clientWidth; H = sc.clientHeight; dpr = devicePixelRatio || 1;
  cols = activeModel.columns; rows = activeModel.rows;
  const fitS = Math.max(.1, Math.min(W / cols, H / rows));
  S = fitS * settings.zoomLevel;
  canvasW = cols * S; canvasH = rows * S;
  contentW = Math.max(W, canvasW); contentH = Math.max(H, canvasH);
  offX = (contentW - canvasW) / 2; offY = (contentH - canvasH) / 2;
  sp.style.width = contentW + 'px';
  sp.style.height = contentH + 'px';
  cv.width = W * dpr; cv.height = H * dpr;
  cv.style.width = W + 'px'; cv.style.height = H + 'px';
  $('zoom-info').textContent = `Zoom ${Math.round(settings.zoomLevel * 100)}%`;
  $('zoom-reset').hidden = Math.abs(settings.zoomLevel - 1) < .001;
  draw();
}
/** Draw the visible portion of the active layer and its interaction overlays. */
function draw() {
  raf = 0; const top = sc.scrollTop, base = layer * activeModel.neurons;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  const r0 = Math.max(0, Math.floor((top - offY) / S)), r1 = Math.min(rows - 1, Math.floor((top + H - offY) / S));
  for (let r = r0; r <= r1; r++) {
    const cellY = offY + r * S - top, y = cellY + S / 2;
    const c0 = Math.max(0, Math.floor((sc.scrollLeft - offX) / S));
    const c1 = Math.min(cols - 1, Math.floor((sc.scrollLeft + W - offX) / S));
    for (let c = c0; c <= c1; c++) { const i = r * cols + c; if (i >= activeModel.neurons) break;
      const cellX = offX + c * S - sc.scrollLeft, x = cellX + S / 2, v = store.v[base + i];
      if (!knobMode) {
        const strength = Math.min(1, Math.abs(v) / settings.max);
        const color = v > 0 ? `rgba(0,166,81,${strength})` : v < 0 ? `rgba(224,0,0,${strength})` : '#fff';
        ctx.fillStyle = color;
        ctx.fillRect(cellX + .5, cellY + .5, S - 1, S - 1);
        ctx.strokeStyle = v ? color : '#ddd';
        ctx.lineWidth = v ? 1.5 : 1;
        ctx.strokeRect(cellX + .5, cellY + .5, S - 1, S - 1);
      } else {
        const radius = Math.min(R, S * .38);
        ctx.beginPath(); ctx.arc(x, y, radius, 0, 6.2832);
        ctx.lineWidth = v ? 2.25 : 1; ctx.strokeStyle = v > 0 ? GREEN : v < 0 ? RED : '#000'; ctx.stroke();
        const a = Math.max(-1, Math.min(1, v / settings.max)) * 2.356, sn = Math.sin(a), cs = Math.cos(a);
        ctx.beginPath(); ctx.moveTo(x + sn * radius * .25, y - cs * radius * .25); ctx.lineTo(x + sn * radius * .85, y - cs * radius * .85);
        ctx.lineWidth = 1.5; ctx.strokeStyle = '#000'; ctx.stroke();
      } } }
  if (drag?.tool === 'rectangle') {
    const x0 = drag.i % cols, y0 = Math.floor(drag.i / cols);
    const end = drag.end, x1 = end % cols, y1 = Math.floor(end / cols);
    const left = Math.min(x0, x1), right = Math.max(x0, x1);
    const top = Math.min(y0, y1), bottom = Math.max(y0, y1);
    const value = drag.value;
    ctx.fillStyle = value > 0 ? 'rgba(0,166,81,.25)' : value < 0 ? 'rgba(224,0,0,.25)' : 'rgba(0,0,0,.12)';
    ctx.fillRect(offX + left * S - sc.scrollLeft, offY + top * S - sc.scrollTop, (right - left + 1) * S, (bottom - top + 1) * S);
    const strength = Math.min(1, Math.abs(value) / settings.max);
    ctx.strokeStyle = value > 0 ? `rgba(0,166,81,${strength})` :
      value < 0 ? `rgba(224,0,0,${strength})` : '#000';
    ctx.lineWidth = 2;
    ctx.strokeRect(offX + left * S - sc.scrollLeft + 1, offY + top * S - sc.scrollTop + 1,
      (right - left + 1) * S - 2, (bottom - top + 1) * S - 2);
  }
  if (hover >= 0) { const r = Math.floor(hover / cols), c = hover % cols;
    const v = store.v[base + hover];
    const strength = Math.min(1, Math.abs(v) / settings.max);
    ctx.lineWidth = 1;
    ctx.strokeStyle = !knobMode && v > 0 ? `rgba(0,166,81,${strength})` :
      !knobMode && v < 0 ? `rgba(224,0,0,${strength})` : '#000';
    const size = !knobMode && settings.tool === 'brush' ? settings.brushSize : 1;
    const left = c - Math.floor(size / 2), brushTop = r - Math.floor(size / 2);
    const right = Math.min(cols - 1, left + size - 1);
    const bottom = Math.min(rows - 1, brushTop + size - 1);
    const visibleLeft = Math.max(0, left), visibleTop = Math.max(0, brushTop);
    ctx.strokeRect(offX + visibleLeft * S - sc.scrollLeft + .5, offY + visibleTop * S - top + .5,
      (right - visibleLeft + 1) * S - 1, (bottom - visibleTop + 1) * S - 1); }
}
/** Schedule one canvas redraw and status refresh for the next animation frame. */
const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { draw(); head(); }); };
/** Map a pointer event's viewport coordinates to a neuron index or -1. */
function idxAt(e) { const b = cv.getBoundingClientRect(), c = Math.floor((e.clientX - b.left + sc.scrollLeft - offX) / S), r = Math.floor((e.clientY - b.top + sc.scrollTop - offY) / S);
  if (c < 0 || c >= cols || r < 0 || r >= rows) return -1; const i = r * cols + c; return i < activeModel.neurons ? i : -1; }
/** Update the readout for the hovered neuron or the current interaction mode. */
function readout() { const el = $('ro'); if (hover < 0) {
    el.textContent = knobMode ? 'drag a knob vertically · shift = fine · double-click = reset' : 'paint pixels · choose a tool and activation intensity'; return; }
  const v = store.get(layer, hover);
  el.textContent = `L${layer} · N${hover} · ${v > 0 ? '+' : ''}${v.toFixed(2)} · gain ×${(1 + GAIN * v).toFixed(2)}`; }
/** Convert the paint dial percentage to the current activation range. */
function paintActivation() { return Math.round(settings.paintAmount / 100 * settings.max * 100) / 100; }
/** Set one valid neuron to an activation value. */
function paintCell(i, value = paintActivation()) { if (i >= 0 && i < activeModel.neurons) store.set(layer, i, value); }
/** Paint a square brush centered on a neuron. */
function paintBrush(i, value) {
  const size = settings.brushSize, left = i % cols - Math.floor(size / 2), top = Math.floor(i / cols) - Math.floor(size / 2);
  for (let y = top; y < top + size; y++) for (let x = left; x < left + size; x++)
    if (x >= 0 && x < cols && y >= 0 && y < rows) paintCell(y * cols + x, value);
}
/** Interpolate a stroke between two neurons to avoid gaps during fast pointer movement. */
function paintLine(from, to, value) {
  const x0 = from % cols, y0 = Math.floor(from / cols), x1 = to % cols, y1 = Math.floor(to / cols);
  const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0));
  for (let n = 0; n <= steps; n++) paintBrush(Math.round(y0 + (y1 - y0) * (n / (steps || 1))) * cols + Math.round(x0 + (x1 - x0) * (n / (steps || 1))), value);
}
/** Fill the rectangular region between two neuron indices. */
function paintRectangle(from, to, value) {
  const x0 = from % cols, y0 = Math.floor(from / cols), x1 = to % cols, y1 = Math.floor(to / cols);
  for (let y = Math.min(y0, y1); y <= Math.max(y0, y1); y++)
    for (let x = Math.min(x0, x1); x <= Math.max(x0, x1); x++) paintCell(y * cols + x, value);
}
/** Replace the connected region of matching values starting at one neuron. */
function paintBucket(start, replacement) {
  const target = store.get(layer, start);
  if (target === replacement) return;
  const queue = [start], seen = new Set(queue);
  while (queue.length) {
    const i = queue.pop(), x = i % cols, y = Math.floor(i / cols);
    if (store.get(layer, i) !== target) continue;
    store.set(layer, i, replacement);
    for (const n of [x > 0 ? i - 1 : -1, x + 1 < cols ? i + 1 : -1, y > 0 ? i - cols : -1, y + 1 < rows ? i + cols : -1])
      if (n >= 0 && n < activeModel.neurons && !seen.has(n) && store.get(layer, n) === target) { seen.add(n); queue.push(n); }
  }
}
/** Handle the start of a canvas painting or knob-drag gesture. */
cv.onpointerdown = e => {
  const i = idxAt(e); if (i < 0) return;
  if (knobMode && e.button !== 0) return;
  cv.setPointerCapture(e.pointerId); e.preventDefault();
  beginEdit();
  if (!knobMode) {
    const tool = settings.tool, value = e.button === 2 ? 0 : paintActivation();
    drag = { i, last: i, tool, value };
    if (tool === 'brush') paintBrush(i, value);
    else if (tool === 'rectangle') drag.end = i;
    else paintBucket(i, value);
  } else drag = { i, y: e.clientY, v: store.get(layer, i) };
  hover = i; readout(); schedule();
};
/** Update the active gesture or hover state as the pointer moves. */
cv.onpointermove = e => { const i = idxAt(e);
  if (drag && !knobMode && i >= 0) {
  if (drag.tool === 'brush' && i !== drag.last) paintLine(drag.last, i, drag.value);
    if (drag.tool === 'rectangle') drag.end = i;
    drag.last = i; hover = i;
  } else if (drag && knobMode) { store.set(layer, drag.i, drag.v + (drag.y - e.clientY) / (e.shiftKey ? 1500 : 150)); hover = drag.i; }
  else hover = i;
  readout(); schedule();
};
/** Commit the current gesture, applying a rectangle on pointer release. */
cv.onpointerup = e => {
  if (drag?.tool === 'rectangle') {
    const end = idxAt(e);
    paintRectangle(drag.i, end >= 0 ? end : drag.end, drag.value);
  }
  drag = null; finishEdit(); schedule();
};
/** End an interrupted pointer gesture and commit changes made before cancellation. */
cv.onpointercancel = () => { drag = null; finishEdit(); schedule(); };
/** Prevent the browser context menu from interrupting pixel-mode erasing. */
cv.oncontextmenu = e => { if (!knobMode) e.preventDefault(); };
/** Clear the hover highlight when the pointer leaves the grid. */
cv.onpointerleave = () => { if (!drag) { hover = -1; readout(); schedule(); } };
/** Reset one knob on a double-click in knob mode. */
cv.ondblclick = e => { const i = idxAt(e); if (i >= 0 && knobMode) {
  beginEdit(); store.set(layer, i, 0); finishEdit(); readout(); schedule();
} };
sc.onscroll = schedule; new ResizeObserver(layout).observe(sc);
/** Change zoom while keeping the grid point under the cursor stationary. */
function setZoom(nextZoom, clientX, clientY) {
  nextZoom = Math.max(.25, Math.min(8, nextZoom));
  if (Math.abs(nextZoom - settings.zoomLevel) < .001) return;
  const bounds = sc.getBoundingClientRect();
  const x = Math.max(0, Math.min(sc.clientWidth, clientX - bounds.left));
  const y = Math.max(0, Math.min(sc.clientHeight, clientY - bounds.top));
  const gridX = (sc.scrollLeft + x - offX) / S;
  const gridY = (sc.scrollTop + y - offY) / S;
  settings.zoomLevel = nextZoom;
  layout();
  sc.scrollLeft = gridX * S + offX - x;
  sc.scrollTop = gridY * S + offY - y;
  saveSettings();
}
/** Apply Alt+wheel zoom while preserving normal scrolling otherwise. */
left.addEventListener('wheel', e => {
  if (!e.altKey) return;
  e.preventDefault();
  setZoom(settings.zoomLevel * Math.exp(-e.deltaY * .001), e.clientX, e.clientY);
}, { passive: false });
/** Zoom relative to the center of the grid viewport. */
function zoomAtCenter(factor) {
  const bounds = sc.getBoundingClientRect();
  setZoom(settings.zoomLevel * factor, bounds.left + sc.clientWidth / 2, bounds.top + sc.clientHeight / 2);
}
/** Bind the canvas zoom controls to centered zoom operations. */
$('zoom-in').onclick = () => zoomAtCenter(1.2);
$('zoom-out').onclick = () => zoomAtCenter(1 / 1.2);
$('zoom-reset').onclick = () => zoomAtCenter(1 / settings.zoomLevel);

/* view settings */
/** Synchronize settings controls and the canvas cursor with persisted preferences. */
function syncSettingsUI() {
  $('knob-mode-toggle').setAttribute('aria-pressed', String(knobMode));
  $('max-activation').value = settings.max;
  $('max-activation-value').textContent = settings.max.toFixed(2);
  $('paint-tools').hidden = knobMode;
  $('cv').style.cursor = knobMode ? 'ns-resize' : 'crosshair';
  document.querySelectorAll('[data-paint-tool]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.paintTool === settings.tool));
  });
  updatePaintDial($('brush-size'), settings.brushSize, 1, 32);
  $('brush-size-value').textContent = settings.brushSize;
  updatePaintDial($('paint-value'), settings.paintAmount, -100, 100);
  $('brush-size-wrap').hidden = settings.tool !== 'brush';
  $('paint-value-label').textContent = formatPaintAmount();
}
/** Format the selected paint amount using the current activation maximum. */
function formatPaintAmount() {
  const value = settings.paintAmount / 100 * settings.max;
  return `${value > 0 ? '+' : ''}${value.toFixed(2)}`;
}
/** Draw an accessible activation/brush dial and update its ARIA value. */
function updatePaintDial(canvas, value, min, max) {
  const scale = devicePixelRatio || 1, size = 28;
  canvas.width = size * scale; canvas.height = size * scale;
  const context = canvas.getContext('2d');
  context.setTransform(scale, 0, 0, scale, 0, 0);
  const center = size / 2, radius = 10.5;
  context.beginPath(); context.arc(center, center, radius, 0, Math.PI * 2);
  context.lineWidth = 1; context.strokeStyle = '#000'; context.stroke();
  const amount = (value - min) / (max - min) * 2 - 1, angle = amount * 2.356;
  const sn = Math.sin(angle), cs = Math.cos(angle);
  context.beginPath();
  context.moveTo(center + sn * radius * .25, center - cs * radius * .25);
  context.lineTo(center + sn * radius * .85, center - cs * radius * .85);
  context.lineWidth = 1.5; context.strokeStyle = '#000'; context.stroke();
  canvas.setAttribute('aria-valuenow', value);
  canvas.setAttribute('aria-valuetext', canvas.id === 'brush-size' ? `${value} pixels` : formatPaintAmount());
}
/** Bind pointer and keyboard controls for a custom vertical dial. */
function bindPaintDial(id, min, max, step, getValue, setValue) {
  const canvas = $(id);
  let dialDrag = null;
  /** Clamp and apply a dial value, then redraw its indicator. */
  const update = value => {
    const next = Math.max(min, Math.min(max, Math.round((value - min) / step) * step + min));
    if (next === getValue()) return;
    setValue(next);
    updatePaintDial(canvas, next, min, max);
  };
  canvas.onpointerdown = e => {
    if (e.button !== 0) return;
    canvas.setPointerCapture(e.pointerId);
    dialDrag = { pointerId: e.pointerId, y: e.clientY, value: getValue() };
    e.preventDefault();
  };
  canvas.onpointermove = e => {
    if (!dialDrag || dialDrag.pointerId !== e.pointerId) return;
    const sensitivity = e.shiftKey ? 1500 : 150;
    update(dialDrag.value - (e.clientY - dialDrag.y) * (max - min) / sensitivity);
  };
  canvas.onpointerup = canvas.onpointercancel = () => { dialDrag = null; };
  canvas.onkeydown = e => {
    const current = getValue();
    if (e.key === 'Home') update(min);
    else if (e.key === 'End') update(max);
    else if (e.key === 'ArrowUp' || e.key === 'ArrowRight') update(current + step);
    else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') update(current - step);
    else return;
    e.preventDefault();
  };
}
bindPaintDial('brush-size', 1, 32, 1, () => settings.brushSize, value => {
  settings.brushSize = value;
  $('brush-size-value').textContent = value; saveSettings(); schedule();
});
bindPaintDial('paint-value', -100, 100, 1, () => settings.paintAmount, value => {
  settings.paintAmount = value;
  $('paint-value-label').textContent = formatPaintAmount(); saveSettings(); schedule();
});
/** Open the settings dialog and bind live preference updates. */
$('settings-open').onclick = () => $('settings-dialog').showModal();
/** Toggle knob/pixel interactions and persist the selected mode. */
function toggleKnobMode() {
  knobMode = !knobMode;
  saveSettings(); syncSettingsUI(); layout(); hover = -1; readout(); schedule();
}
$('knob-mode-toggle').onclick = toggleKnobMode;
/** Keep the activation maximum, display, and persisted preference synchronized. */
$('max-activation').oninput = e => {
  settings.max = +e.target.value;
  $('max-activation-value').textContent = settings.max.toFixed(2);
  $('paint-value-label').textContent = formatPaintAmount();
  schedule();
};
$('max-activation').onchange = () => { saveSettings(); schedule(); };
/** Persist the selected paint tool and update its related controls. */
document.querySelectorAll('[data-paint-tool]').forEach(button => {
  button.onclick = () => {
    settings.tool = button.dataset.paintTool;
    saveSettings(); syncSettingsUI(); schedule();
  };
});
/* layer selector */
/** Rebuild the layer buttons for the active model. */
function buildLayers() {
  $('layers').replaceChildren(); layerButtons.length = 0;
  for (let l = 0; l < activeModel.layers; l++) {
    const b = document.createElement('button'); b.className = 'L'; b.textContent = l; b.title = 'Layer ' + l;
    b.onclick = () => selectLayer(l); $('layers').append(b); layerButtons.push(b);
  }
}
/** Select a model and rebuild its model-specific steering/UI state. */
function selectModel(key) {
  if (rt.ready) return;
  clearEditHistory();
  activeModel = MODELS[key];
  store = new SteeringStore(activeModel);
  layer = 0; hover = -1; sc.scrollTop = 0;
  $('model-info').textContent = `${activeModel.name} · MLP hidden units · ${activeModel.neurons.toLocaleString()} per layer · ${activeModel.layers} layers`;
  $('st').textContent = `No model loaded. ${activeModel.name} selected; choose Load model.`;
  buildLayers(); readout(); head(); layout();
}
/** Select a layer, reset scrolling, and refresh its readout. */
function selectLayer(l) { layer = (l + activeModel.layers) % activeModel.layers; sc.scrollTop = 0; hover = -1; readout(); schedule(); }
/** Refresh layer highlighting, steering counts, and the steering status message. */
function head() { layerButtons.forEach((b, l) => { b.classList.toggle('on', l === layer); b.classList.toggle('mod', store.cnt[l] > 0); });
  $('info').textContent = `Layer ${layer} · ${store.cnt[layer]} modified · ${store.total()} total`;
  const n = store.total(); if (!busy) $('sv').textContent = !rt.ready ? 'Model not loaded.' :
    n ? `Steering configured: ${n} neurons. Applied on Generate.` : 'Steering: none configured.'; }
/** Bind layer reset controls to undoable reset transactions. */
$('rl').onclick = () => { beginEdit(); store.resetLayer(layer); finishEdit(); schedule(); };
$('ra').onclick = () => { if (confirm(`Reset every neuron in all ${activeModel.layers} layers?`)) {
  beginEdit(); store.resetAll(); finishEdit(); schedule();
} };
/** Enable or disable undo/redo buttons based on the edit-history stacks. */
function updateHistoryButtons() {
  const undo = $('undo'), redo = $('redo');
  if (!undo || !redo) return;
  undo.disabled = undoStack.length === 0;
  redo.disabled = redoStack.length === 0;
}
/** Restore the previous neuron-edit transaction. */
function undoEdit() { applyEditHistory(undoStack, redoStack, true); }
/** Reapply the most recently undone neuron-edit transaction. */
function redoEdit() { applyEditHistory(redoStack, undoStack, false); }
$('undo').onclick = undoEdit;
$('redo').onclick = redoEdit;
/** Handle history shortcuts and layer navigation outside text-editing controls. */
addEventListener('keydown', e => {
  const target = e.target;
  const isEditingText = target.isContentEditable || ['TEXTAREA', 'INPUT', 'SELECT'].includes(target.tagName);
  if ((e.ctrlKey || e.metaKey) && !isEditingText) {
    const key = e.key.toLowerCase();
    if (key === 'z') {
      e.preventDefault();
      if (e.shiftKey) redoEdit(); else undoEdit();
      return;
    }
    if (key === 'y') { e.preventDefault(); redoEdit(); return; }
  }
  if (isEditingText) return;
  if (e.key === '[') selectLayer(layer - 1); if (e.key === ']') selectLayer(layer + 1); });

/* ================= UI: chat ================= */
const history = []; let busy = false, testerTimer = 0, testerPending = false, testerRunning = false, testerRunId = 0;
const log = $('log');
/** Append a safely rendered chat message and return its text span. */
function addMsg(role, text) { const d = document.createElement('div'); d.className = 'm';
  d.innerHTML = '<i>' + (role === 'user' ? 'YOU' : activeModel.name.toUpperCase()) + '</i><span></span>'; d.lastChild.textContent = text; log.append(d); log.scrollTop = log.scrollHeight; return d.lastChild; }
/** Debounce a tester run after a steering edit. */
function requestTesterRun() {
  testerRunId++;
  clearTimeout(testerTimer);
  if (testerRunning) { testerPending = true; rt.stop(); }
  if (!$('tester-enabled').checked || !rt.ready) return;
  testerTimer = setTimeout(runTester, 350);
}
/** Generate a short deterministic response for the current tester prompt. */
async function runTester() {
  if (!$('tester-enabled').checked || !rt.ready) return;
  if (busy) { testerPending = true; return; }
  const prompt = $('tester-prompt').value.trim();
  if (!prompt) { $('tester-response').textContent = 'Enter a test prompt.'; return; }
  const model = activeModel, runId = testerRunId;
  busy = true;
  testerRunning = true;
  testerPending = false;
  $('send').disabled = true;
  $('stop').disabled = false;
  $('unload').disabled = true;
  $('tester-response').textContent = '…';
  try {
    const sparse = store.sparse();
    if (store.total() && !rt.steerer)
      throw new Error('Steering is configured but unavailable: ' + rt.steerErr);
    if (rt.steerer) await rt.applySteering(sparse);
    if (runId !== testerRunId) return;
    let output = '';
    const result = await rt.generate([{ role: 'user', content: prompt }],
      { temperature: 0, max_tokens: 16 }, token => {
        output += token;
        if (runId === testerRunId && $('tester-enabled').checked && model === activeModel && $('tester-prompt').value.trim() === prompt)
          $('tester-response').textContent = output;
      });
    if (runId === testerRunId && $('tester-enabled').checked && model === activeModel && $('tester-prompt').value.trim() === prompt)
      $('tester-response').textContent = result || '(No response)';
  } catch (e) {
    if (runId === testerRunId && $('tester-enabled').checked && model === activeModel)
      $('tester-response').textContent = 'Test failed: ' + e.message;
  } finally {
    busy = false;
    testerRunning = false;
    $('send').disabled = !rt.ready;
    $('stop').disabled = true;
    $('unload').disabled = !rt.ready;
    if (testerPending) requestTesterRun();
  }
}
/** Update the temperature display as its slider moves. */
$('temp').oninput = e => $('tv').textContent = (+e.target.value).toFixed(2);
/** Toggle a floating panel and keep its button's accessible state synchronized. */
function togglePanel(panelId, buttonId) {
  const panel = $(panelId), button = $(buttonId), open = panel.hidden;
  if (open && innerWidth <= 700) {
    for (const [otherId, otherButtonId] of [
      ['chat', 'chat-toggle'], ['tester', 'tester-toggle']
    ]) if (otherId !== panelId) {
      $(otherId).hidden = true;
      $(otherButtonId).setAttribute('aria-expanded', 'false');
    }
  }
  panel.hidden = !open;
  button.setAttribute('aria-expanded', String(open));
}
/** Bind the Chat and Tester panel toggles. */
$('chat-toggle').onclick = () => togglePanel('chat', 'chat-toggle');
$('tester-toggle').onclick = () => togglePanel('tester', 'tester-toggle');
const testerTop = $('tester-top');
let testerDrag = null;
/** Position the Tester panel within the visible browser viewport. */
function positionTester(left, top) {
  const panel = $('tester');
  const maxLeft = Math.max(0, innerWidth - panel.offsetWidth);
  const maxTop = Math.max(0, innerHeight - panel.offsetHeight);
  panel.style.left = `${Math.max(0, Math.min(maxLeft, left))}px`;
  panel.style.top = `${Math.max(0, Math.min(maxTop, top))}px`;
  panel.style.right = 'auto';
  panel.style.bottom = 'auto';
}
/** Begin dragging the Tester panel from its header. */
testerTop.onpointerdown = e => {
  if (e.button !== 0 || e.target.closest('input, button, label')) return;
  const bounds = $('tester').getBoundingClientRect();
  testerDrag = { pointerId: e.pointerId, x: e.clientX, y: e.clientY, left: bounds.left, top: bounds.top };
  testerTop.setPointerCapture(e.pointerId);
  e.preventDefault();
};
/** Move the Tester panel with the captured pointer while dragging its header. */
testerTop.onpointermove = e => {
  if (!testerDrag || testerDrag.pointerId !== e.pointerId) return;
  positionTester(testerDrag.left + e.clientX - testerDrag.x, testerDrag.top + e.clientY - testerDrag.y);
};
/** End a Tester drag when the pointer is released or canceled. */
testerTop.onpointerup = testerTop.onpointercancel = e => {
  if (testerDrag?.pointerId === e.pointerId) testerDrag = null;
};
/** Move the Tester panel with the keyboard when its header has focus. */
testerTop.onkeydown = e => {
  const moves = { ArrowLeft: [-10, 0], ArrowRight: [10, 0], ArrowUp: [0, -10], ArrowDown: [0, 10] };
  const move = moves[e.key];
  if (!move) return;
  const bounds = $('tester').getBoundingClientRect();
  positionTester(bounds.left + move[0], bounds.top + move[1]);
  e.preventDefault();
};
/** Keep a manually positioned Tester panel on-screen when the viewport changes. */
addEventListener('resize', () => {
  const panel = $('tester');
  if (panel.style.left) {
    const bounds = panel.getBoundingClientRect();
    positionTester(bounds.left, bounds.top);
  }
});
/** Start or cancel debounced tester runs when its enabled state changes. */
$('tester-enabled').onchange = e => {
  clearTimeout(testerTimer);
  testerPending = false;
  if (!e.target.checked && testerRunning) { testerRunId++; rt.stop(); }
  if (e.target.checked) requestTesterRun();
};
/** Schedule a tester run when its prompt is changed. */
$('tester-prompt').oninput = requestTesterRun;
/** Clear visible and conversational chat history when generation is idle. */
$('clr').onclick = () => { if (!busy) { history.length = 0; log.innerHTML = ''; } };
/** Interrupt an active model generation. */
$('stop').onclick = () => rt.stop();
/** Select the requested model until a runtime has been loaded. */
$('model').onchange = e => selectModel(e.target.value);
/** Clamp the chat token request to the range exposed by the numeric control. */
function getMaxTokens() {
  const input = $('mx'), raw = input.value.trim();
  const parsed = raw === '' ? 256 : Number(raw);
  const tokens = Number.isFinite(parsed) ? Math.trunc(parsed) : 256;
  input.value = String(Math.max(1, Math.min(2048, tokens)));
  return Number(input.value);
}
// Native number inputs do not enforce min/max unless a form is submitted.
$('mx').onchange = getMaxTokens;
/** Load the selected model and expose chat/tester controls on success. */
$('load').onclick = async () => {
  $('load').disabled = true; $('model').disabled = true; $('unload').disabled = true;
  $('st').textContent = `Loading ${activeModel.name}…`;
  try { await rt.load(activeModel, p => $('st').textContent = p.text.slice(0, 120));
    $('st').textContent = `${activeModel.name} ready.` + (rt.steerer ? ' Steering runtime attached.' : ' Steering UNAVAILABLE: ' + rt.steerErr);
    $('load').hidden = true; $('unload').hidden = false; $('unload').disabled = false; $('send').disabled = false;
    $('chat').classList.remove('unloaded'); $('tester').classList.remove('unloaded');
    requestTesterRun();
  } catch (e) {
    $('st').textContent = `No model loaded. Load failed: ${e.message}`;
    $('load').disabled = false; $('model').disabled = false; $('unload').disabled = true;
  }
};
/** Unload the runtime and return the controls to the unloaded state. */
$('unload').onclick = async () => {
  if (!rt.ready || busy) return;
  $('unload').disabled = true;
  $('st').textContent = `Unloading ${activeModel.name}…`;
  try {
    clearTimeout(testerTimer); testerPending = false;
    await rt.unload();
    history.length = 0; log.replaceChildren();
    $('load').hidden = false; $('load').disabled = false;
    $('unload').hidden = true;
    $('model').disabled = false;
    $('send').disabled = true; $('stop').disabled = true;
    $('chat').classList.add('unloaded'); $('tester').classList.add('unloaded');
    $('tester-response').textContent = 'Load a model to test.';
    $('st').textContent = `No model loaded. ${activeModel.name} can be loaded again or choose another model.`;
    head();
  } catch (e) {
    $('st').textContent = `Unload failed: ${e.message}`;
    $('unload').disabled = false;
  }
};
/** Submit the chat prompt, apply configured steering, and stream the answer. */
async function generate() {
  const text = $('in').value.trim(); if (!text || busy || !rt.ready) return;
  const sparse = store.sparse(), want = store.total();
  if (want && !rt.steerer) { $('sv').textContent = 'Not generating: steering is configured but the runtime hook is unavailable (' + rt.steerErr + ').'; return; }
  busy = true; $('send').disabled = true; $('stop').disabled = false; $('unload').disabled = true; $('in').value = '';
  history.push({ role: 'user', content: text }); addMsg('user', text);
  try {
    $('sv').textContent = 'Applying steering to model weights…';
    const r = rt.steerer ? await rt.applySteering(sparse) : { neurons: 0, layers: 0, verified: true };
    $('sv').textContent = r.neurons ? `Applied ${r.neurons} neurons across ${r.layers} layers` + (r.verified ? ' (read back from GPU: verified).' : ' (WARNING: GPU read-back mismatch).') : 'Steering: none (unmodified model).';
    const out = addMsg('assistant', '');
    const full = await rt.generate(history, { temperature: +$('temp').value, max_tokens: getMaxTokens() }, t => { out.textContent += t; log.scrollTop = log.scrollHeight; });
    history.push({ role: 'assistant', content: full });
  } catch (e) { addMsg('assistant', '[error] ' + e.message); history.pop(); }
  busy = false; $('send').disabled = !rt.ready; $('stop').disabled = true; $('unload').disabled = !rt.ready;
  if (testerPending) requestTesterRun();
}
$('send').onclick = generate;
$('in').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); generate(); } };
syncSettingsUI(); buildLayers(); selectModel($('model').value); layout();
