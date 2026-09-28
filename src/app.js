/**
 * DTLN lab - watch and hear the model work.
 *
 * Two modes over the same worklet:
 *   live    mic -> worklet, spectrograms updating as you talk, optionally
 *           monitored through your headphones
 *   record  capture a clip, denoise it offline, then A/B the two side by side
 *
 * Both draw the input and the output spectrogram so you can see what the model
 * removed, not just hear it. The spectrograms are tapped BEFORE the makeup
 * gain (see MAKEUP_GAIN) so the two panels share one scale whatever it is.
 */

const SR = 16000;
const FFT_SIZE = 512; // 31 Hz per bin at 16 kHz
const BINS = FFT_SIZE / 2;

/**
 * Mirrors awful's DtlnProcessor.OUTPUT_COMPENSATION, so "after" sounds as
 * loud here as on a call. 1 since awful levels the mic with AGC before the
 * worklet (see MIC); it was 3 while that path ran with AGC off.
 */
const MAKEUP_GAIN = 1.0;

/**
 * Measured, by rendering a tone burst through the worklet offline and finding
 * where the output rises: 697 samples at 16 kHz. That is this worklet's block
 * buffering plus dtln_denoise's own frame delay. Record mode trims it so the
 * before/after clips line up; being a little off only shifts the comparison
 * by a few milliseconds.
 */
const LATENCY_SAMPLES = 697;

const MIN_DB = -95;
const MAX_DB = -15;

const el = (id) => document.getElementById(id);
const ui = {
  tabs: [...document.querySelectorAll("[data-mode]")],
  action: el("action"),
  status: el("status"),
  gate: el("gate"),
  gateValue: el("gateValue"),
  monitorRow: el("monitorRow"),
  monitor: el("monitor"),
  players: el("players"),
  rawAudio: el("rawAudio"),
  denoisedAudio: el("denoisedAudio"),
  canvasIn: el("specIn"),
  canvasOut: el("specOut"),
};

let mode = "live";
let running = false;
let busy = false;
let ctx = null;
let worklet = null;
let stream = null;
let recorder = null;
let chunks = [];
let raf = 0;
let graph = null;
let lastClip = null; // decoded AudioBuffer, kept so the gate can be re-applied
let urls = [];

const status = (text, kind = "") => {
  ui.status.textContent = text;
  ui.status.className = kind;
};

/** Slider is in dBFS because the threshold is perceptually logarithmic; the
 *  worklet wants a linear RMS amplitude. -54 dB is the 0.002 default. */
const gateDb = () => Number(ui.gate.value);
const gateAmp = () => (gateDb() <= -90 ? 0 : Math.pow(10, gateDb() / 20));

// ─── spectrogram ──────────────────────────────────────────────────────────

const LOG2 = Math.log2(FFT_SIZE);
const reverse = new Uint16Array(FFT_SIZE);
for (let i = 0; i < FFT_SIZE; i++) {
  let r = 0;
  for (let b = 0; b < LOG2; b++) r |= ((i >> b) & 1) << (LOG2 - 1 - b);
  reverse[i] = r;
}
const hann = new Float32Array(FFT_SIZE);
for (let i = 0; i < FFT_SIZE; i++) {
  hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1));
}

/** In-place iterative radix-2 FFT. Only used for recorded clips; live mode
 *  gets its magnitudes from an AnalyserNode for free. */
function fft(re, im) {
  for (let i = 0; i < FFT_SIZE; i++) {
    const j = reverse[i];
    if (j > i) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  for (let len = 2; len <= FFT_SIZE; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < FFT_SIZE; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k];
        const ai = im[i + k];
        const br = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const bi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ar + br;
        im[i + k] = ai + bi;
        re[i + k + len / 2] = ar - br;
        im[i + k + len / 2] = ai - bi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

const fre = new Float32Array(FFT_SIZE);
const fim = new Float32Array(FFT_SIZE);
const mags = new Float32Array(BINS);

/**
 * dB magnitudes for one window starting at `offset`. Normalised by the
 * window's coherent gain (Hann = 0.5) as well as the transform length, so a
 * full-scale sine reads ~0 dB and a recorded panel is directly comparable to
 * a live one.
 */
function spectrumAt(samples, offset) {
  for (let i = 0; i < FFT_SIZE; i++) {
    const s = offset + i < samples.length ? samples[offset + i] : 0;
    fre[i] = s * hann[i];
    fim[i] = 0;
  }
  fft(fre, fim);
  for (let i = 0; i < BINS; i++) {
    const m = Math.sqrt(fre[i] * fre[i] + fim[i] * fim[i]) / (BINS * 0.5);
    mags[i] = 20 * Math.log10(m + 1e-12);
  }
  return mags;
}

// Continuous ramp: each stop starts where the previous ended, so the picture
// has no false contour lines at the segment boundaries.
const STOPS = [
  [0.0, [8, 10, 35]],
  [0.3, [20, 60, 160]],
  [0.55, [30, 170, 190]],
  [0.78, [240, 220, 90]],
  [1.0, [255, 255, 235]],
];

function color(db) {
  let t = (db - MIN_DB) / (MAX_DB - MIN_DB);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1];
      const [t1, c1] = STOPS[i];
      const k = (t - t0) / (t1 - t0);
      return [
        c0[0] + k * (c1[0] - c0[0]),
        c0[1] + k * (c1[1] - c0[1]),
        c0[2] + k * (c1[2] - c0[2]),
      ];
    }
  }
  return STOPS[STOPS.length - 1][1];
}

const BG = "rgb(8,10,35)";

function drawColumn(c2d, x, data, height) {
  for (let y = 0; y < height; y++) {
    // bottom of the canvas is 0 Hz, top is 8 kHz
    const bin = Math.floor(((height - 1 - y) / height) * BINS);
    const [r, g, b] = color(data[bin]);
    c2d.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
    c2d.fillRect(x, y, 1, 1);
  }
}

function clear(canvas) {
  const c2d = canvas.getContext("2d");
  c2d.fillStyle = BG;
  c2d.fillRect(0, 0, canvas.width, canvas.height);
}

/** Scroll one pixel left and draw the newest column on the right. */
function pushColumn(canvas, data) {
  const c2d = canvas.getContext("2d");
  c2d.drawImage(canvas, -1, 0);
  drawColumn(c2d, canvas.width - 1, data, canvas.height);
}

/** Whole clip at once, one column per pixel of width. */
function paintBuffer(canvas, samples) {
  const c2d = canvas.getContext("2d");
  clear(canvas);
  const hop = Math.max(1, Math.floor(samples.length / canvas.width));
  for (let x = 0; x < canvas.width; x++) {
    drawColumn(c2d, x, spectrumAt(samples, x * hop), canvas.height);
  }
}

/** Match the backing store to the element's real size, or every column is
 *  resampled and the picture blurs. Resizing clears, so callers repaint. */
function sizeCanvases() {
  for (const canvas of [ui.canvasIn, ui.canvasOut]) {
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * devicePixelRatio));
    const h = Math.max(1, Math.round(rect.height * devicePixelRatio));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      clear(canvas);
    }
  }
}

const byteToDb = (v) => MIN_DB + (v / 255) * (MAX_DB - MIN_DB);

// ─── worklet plumbing ─────────────────────────────────────────────────────

/**
 * The worklet posts "ready" only once its WASM runtime is up and the denoiser
 * exists, and posts nothing at all if that fails - so every wait needs a
 * timeout, here as much as in the app.
 */
function waitUntilReady(node, ms = 15000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("worklet ready timeout")), ms);
    node.onprocessorerror = () => reject(new Error("worklet crashed"));
    node.port.onmessage = (e) => {
      if (e.data === "ready") {
        clearTimeout(timer);
        resolve();
      }
    };
  });
}

function makeNode(context) {
  return new AudioWorkletNode(context, "NoiseSuppressionWorker", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    channelCount: 1,
    channelCountMode: "explicit",
  });
}

/**
 * Only live mode needs this. A failure must not be cached: the guard used to
 * be `if (ctx) return`, which meant one timeout poisoned the page and every
 * later attempt reported success over a worklet that was never ready.
 */
async function ensureContext() {
  if (ctx && worklet) return;
  status("loading the model (8 MB, first run only)…");
  try {
    ctx = new AudioContext({ sampleRate: SR });
    await ctx.audioWorklet.addModule("audio-worklet.js");
    const node = makeNode(ctx);
    await waitUntilReady(node);
    worklet = node;
    worklet.port.postMessage({ noise_gate: gateAmp() });
  } catch (e) {
    worklet = null;
    try {
      await ctx?.close();
    } catch {}
    ctx = null;
    throw e;
  }
}

// AGC on, as on an awful call: the gate's threshold is absolute, so judging
// it on an unleveled mic judges a level no call will ever send it.
const MIC = {
  channelCount: 1,
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: true,
};

function friendly(e) {
  if (e.name === "NotAllowedError") return "microphone permission denied";
  if (e.name === "NotFoundError") return "no microphone found";
  if (/ready timeout/.test(e.message)) return "the model did not load - reload and try again";
  return e.message;
}

function dropMic() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
}

// ─── live mode ────────────────────────────────────────────────────────────

async function startLive() {
  await ensureContext();
  if (ctx.state === "suspended") await ctx.resume();
  status("waiting for the microphone…");
  stream = await navigator.mediaDevices.getUserMedia({ audio: MIC });

  const source = ctx.createMediaStreamSource(stream);
  const analyserIn = ctx.createAnalyser();
  const analyserOut = ctx.createAnalyser();
  for (const a of [analyserIn, analyserOut]) {
    a.fftSize = FFT_SIZE;
    a.smoothingTimeConstant = 0;
    // Without these the byte data is mapped over the default -100..-30 dB and
    // MIN_DB/MAX_DB silently do nothing to the live panels.
    a.minDecibels = MIN_DB;
    a.maxDecibels = MAX_DB;
  }

  const makeup = ctx.createGain();
  makeup.gain.value = MAKEUP_GAIN;
  const monitorGain = ctx.createGain();
  monitorGain.gain.value = ui.monitor.checked ? 1 : 0;
  // A silent sink keeps the analysers being pulled even when nothing is
  // monitored - a node the graph never reaches produces no data at all.
  const silent = ctx.createGain();
  silent.gain.value = 0;

  source.connect(analyserIn);
  analyserIn.connect(silent);
  silent.connect(ctx.destination);

  // Measure pre-gain, listen post-gain.
  source.connect(worklet);
  worklet.connect(analyserOut);
  analyserOut.connect(makeup);
  makeup.connect(monitorGain);
  monitorGain.connect(ctx.destination);

  graph = { source, analyserIn, analyserOut, makeup, monitorGain, silent };

  sizeCanvases();
  clear(ui.canvasIn);
  clear(ui.canvasOut);
  const bytesIn = new Uint8Array(BINS);
  const bytesOut = new Uint8Array(BINS);
  const dbIn = new Float32Array(BINS);
  const dbOut = new Float32Array(BINS);

  // One column per analysis hop rather than per animation frame, so the time
  // axis means the same thing on a 60 Hz and a 144 Hz display.
  const HOP_MS = (FFT_SIZE / SR) * 1000;
  let last = performance.now();
  let owed = 0;

  const tick = () => {
    const now = performance.now();
    owed += (now - last) / HOP_MS;
    last = now;
    const columns = Math.min(Math.floor(owed), 8);
    owed -= columns;
    if (columns > 0) {
      analyserIn.getByteFrequencyData(bytesIn);
      analyserOut.getByteFrequencyData(bytesOut);
      for (let i = 0; i < BINS; i++) {
        dbIn[i] = byteToDb(bytesIn[i]);
        dbOut[i] = byteToDb(bytesOut[i]);
      }
      for (let c = 0; c < columns; c++) {
        pushColumn(ui.canvasIn, dbIn);
        pushColumn(ui.canvasOut, dbOut);
      }
    }
    raf = requestAnimationFrame(tick);
  };
  tick();
  announceLive();
}

const announceLive = () =>
  status(
    ui.monitor.checked
      ? "listening - you are hearing the denoised signal"
      : "listening - monitoring is off"
  );

function stopLive() {
  cancelAnimationFrame(raf);
  raf = 0;
  if (graph) {
    for (const n of Object.values(graph)) n.disconnect();
    // The worklet is shared with the rest of the page, so detach only this edge.
    try {
      worklet?.disconnect(graph.analyserOut);
    } catch {}
    graph = null;
  }
  dropMic();
  status("stopped");
}

// ─── record mode ──────────────────────────────────────────────────────────

/** Record mode never uses the realtime worklet - only an offline render - so
 *  it decodes on a context that opens no output device. */
const decodeContext = () => new OfflineAudioContext(1, 1, SR);

async function startRecording() {
  status("waiting for the microphone…");
  stream = await navigator.mediaDevices.getUserMedia({ audio: MIC });
  chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => e.data.size > 0 && chunks.push(e.data);
  recorder.onstop = () => {
    processRecording().finally(() => {
      busy = false;
      ui.action.disabled = false;
      setTabsEnabled(true);
    });
  };
  recorder.start();
  ui.players.hidden = true;
  sizeCanvases();
  clear(ui.canvasIn);
  clear(ui.canvasOut);
  status("recording - say something, then stop (a few seconds is plenty)");
}

/** Returns once the clip is captured; processing continues in recorder.onstop,
 *  which is why the button stays disabled until that settles. */
function stopRecording() {
  busy = true;
  ui.action.disabled = true;
  setTabsEnabled(false);
  if (recorder && recorder.state !== "inactive") recorder.stop();
  dropMic();
  status("processing…");
}

async function processRecording() {
  try {
    const blob = new Blob(chunks, { type: recorder?.mimeType || "audio/webm" });
    chunks = [];
    lastClip = await decodeContext().decodeAudioData(await blob.arrayBuffer());
    await renderAndShow();
  } catch (e) {
    console.error(e);
    status(`failed: ${friendly(e)}`, "error");
  }
}

async function renderAndShow() {
  const raw = lastClip.getChannelData(0);
  const denoised = await renderDenoised(lastClip);

  // The players carry the makeup gain (that is what a call sends); the
  // spectrograms deliberately do not.
  const loud = new Float32Array(denoised.length);
  let clipped = 0;
  for (let i = 0; i < denoised.length; i++) {
    const v = denoised[i] * MAKEUP_GAIN;
    if (v > 1 || v < -1) clipped++;
    loud[i] = v;
  }

  revokeUrls();
  ui.rawAudio.src = track(wav(raw, lastClip.sampleRate));
  ui.denoisedAudio.src = track(wav(loud, lastClip.sampleRate));
  ui.players.hidden = false;

  paintBuffer(ui.canvasIn, raw);
  paintBuffer(ui.canvasOut, denoised);
  status(
    `${lastClip.duration.toFixed(1)}s - compare below` +
      (clipped ? ` (${clipped} samples clipped by the ${MAKEUP_GAIN}x makeup gain)` : "")
  );
}

async function renderDenoised(buffer) {
  // Render past the end and drop the head, so the output lines up with the
  // input instead of arriving ~44 ms late and losing its tail.
  const length = buffer.length + LATENCY_SAMPLES;
  const offline = new OfflineAudioContext(1, length, buffer.sampleRate);
  await offline.audioWorklet.addModule("audio-worklet.js");
  const node = makeNode(offline);
  await waitUntilReady(node);
  node.port.postMessage({ noise_gate: gateAmp() });

  const mono = offline.createBuffer(1, buffer.length, buffer.sampleRate);
  mono.copyToChannel(buffer.getChannelData(0), 0);
  const source = offline.createBufferSource();
  source.buffer = mono;
  source.connect(node);
  node.connect(offline.destination);
  source.start();

  const rendered = (await offline.startRendering()).getChannelData(0);
  return rendered.subarray(LATENCY_SAMPLES, LATENCY_SAMPLES + buffer.length);
}

function track(blob) {
  const url = URL.createObjectURL(blob);
  urls.push(url);
  return url;
}

function revokeUrls() {
  for (const u of urls) URL.revokeObjectURL(u);
  urls = [];
}

/** Mono 16-bit PCM WAV. */
function wav(samples, rate) {
  const bytes = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(bytes);
  const str = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  str(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  str(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([bytes], { type: "audio/wav" });
}

// ─── wiring ───────────────────────────────────────────────────────────────

function setTabsEnabled(on) {
  ui.tabs.forEach((t) => {
    t.disabled = !on;
  });
}

let gateTimer = 0;
ui.gate.addEventListener("input", () => {
  ui.gateValue.textContent = `${gateDb()} dB`;
  worklet?.port.postMessage({ noise_gate: gateAmp() });
  // In record mode the threshold only takes effect when the clip is rendered,
  // so re-render the clip we already have - otherwise judging a threshold
  // would mean recording a new take for every value.
  if (mode === "record" && lastClip && !running && !busy) {
    clearTimeout(gateTimer);
    gateTimer = setTimeout(async () => {
      busy = true;
      status("re-processing…");
      try {
        await renderAndShow();
      } catch (e) {
        status(`failed: ${friendly(e)}`, "error");
      } finally {
        busy = false;
      }
    }, 350);
  }
});

ui.monitor.addEventListener("change", () => {
  if (graph) graph.monitorGain.gain.value = ui.monitor.checked ? 1 : 0;
  if (running) announceLive();
});

ui.tabs.forEach((tab) =>
  tab.addEventListener("click", () => {
    if (running || busy) return;
    mode = tab.dataset.mode;
    ui.tabs.forEach((t) => {
      const on = t === tab;
      t.classList.toggle("on", on);
      t.setAttribute("aria-selected", String(on));
    });
    ui.monitorRow.hidden = mode !== "live";
    ui.players.hidden = true;
    ui.action.textContent = mode === "live" ? "Start listening" : "Record";
    sizeCanvases();
    clear(ui.canvasIn);
    clear(ui.canvasOut);
    status("");
  })
);

ui.action.addEventListener("click", async () => {
  if (busy) return;
  ui.action.disabled = true;
  setTabsEnabled(false);
  try {
    if (!running) {
      await (mode === "live" ? startLive() : startRecording());
      running = true;
      ui.action.textContent = mode === "live" ? "Stop" : "Stop and process";
    } else {
      running = false;
      ui.action.textContent = mode === "live" ? "Start listening" : "Record";
      if (mode === "live") stopLive();
      else stopRecording(); // keeps the button disabled until processing ends
    }
  } catch (e) {
    console.error(e);
    dropMic(); // a throw after getUserMedia would otherwise leave it recording
    status(`failed: ${friendly(e)}`, "error");
    running = false;
    ui.action.textContent = mode === "live" ? "Start listening" : "Record";
  } finally {
    if (!busy) {
      ui.action.disabled = false;
      setTabsEnabled(!running);
    }
  }
});

addEventListener("resize", () => {
  // Resizing wipes the backing store; live mode refills it within a second.
  sizeCanvases();
});

sizeCanvases();
clear(ui.canvasIn);
clear(ui.canvasOut);
ui.gateValue.textContent = `${gateDb()} dB`;
