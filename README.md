# dtln worklet

Real-time speech noise suppression as a single AudioWorklet. This is a fork of
[dtln-rs-demo](https://github.com/DataDog/dtln-rs-demo) reworked from a
record-then-process demo into the streaming worklet that
[awful](https://github.com/awful-org/awful.chat) uses on its voice calls; the
model is [dtln-rs](https://github.com/DataDog/dtln-rs) (two stacked LSTMs)
compiled to WebAssembly.

## What it does

The worklet registers as `NoiseSuppressionWorker` and denoises mono audio in
place - connect it between a mic source and whatever consumes the result.

```
mic ──▶ [ anti-alias + decimate to 16 kHz ]      only when the context
                      │                          isn't already at 16 kHz
                      ▼
             512-sample blocks (32 ms)
                      │
                      ▼
              dtln_denoise (WASM)
                      │
                      ▼
              noise gate (see below)
                      │
                      ▼
         [ interpolate back to native rate ]
                      │
                      ▼
             output ring ──▶ downstream
```

- **Sample rate.** The model is 16 kHz. Give it an `AudioContext({ sampleRate:
  16000 })` and the resampling stages collapse to an identity pass - the
  browser's own resampler handles the mic, which is better than anything the
  worklet can do inline. At other rates it falls back to an internal path: a
  4th-order Butterworth low-pass at 7 kHz before decimation, linear
  interpolation on the way back. That path only handles rates *above* 16 kHz;
  a context below it is not supported (nothing filters the way back out).
- **Latency.** ~44 ms added at a 16 kHz context, measured by rendering a tone
  burst through it (697 samples). Only 24 ms of that is this worklet's own
  buffering - a 512-sample block minus one 128-sample render quantum - the
  rest is `dtln_denoise`'s own frame delay. On top of the graph's I/O latency.
- **Gain.** Unity, and the model barely touches clean speech (~0.4 dB), so a
  leveled input needs no makeup gain. Level the mic *before* the worklet -
  awful asks for the browser's `autoGainControl` - rather than boosting after
  it: the gate below has a fixed threshold, and an unleveled quiet mic sits
  right on it. Any makeup gain belongs to the consumer, one visible knob
  outside this worklet rather than two multiplying constants in different
  repos.
- **CPU.** The WASM is SIMD-vectorized and single-threaded; one inference runs
  inline every 4th render quantum at 16 kHz. Measured on a desktop (headless
  Chromium, offline render): ~12 ms of inference per 32 ms block, so roughly
  40% of one core while a mic is live. Fine there - close enough to the budget
  that it is worth measuring on low-end mobile rather than assuming.

### The noise gate

DTLN suppresses noise but does not silence a quiet channel, so a gate runs on
its output: block RMS into an envelope follower (instant attack, ~80 ms
release), a threshold with 6 dB of hysteresis, then a gain that opens with a
~27 ms time constant and closes with a ~300 ms one (about 64 ms and 700 ms to
settle), ramped per sample so transitions don't click. Fast opening is the
point - a slow one eats the first syllable after every pause.

Two things keep it from chopping speech. It holds open for ~256 ms after the
level last cleared the close threshold, so the dips between syllables don't
start it closing. And closed means -20 dB, not silence: on top of the model's
own suppression that still reads as quiet between phrases, while a threshold
set too high for someone's mic dims their soft syllables instead of deleting
them. Neither fixes an input that is simply too quiet for the threshold -
that is what leveling the mic is for (see *Gain*). The gate was a hard mute
with no hold until a quiet headset mic on awful, with AGC off, came through
as a spotty voice.

Set the threshold with `port.postMessage({ noise_gate })` (RMS, `0` disables
the gate). The default is `0.002`.

## Using it

```js
const ctx = new AudioContext({ sampleRate: 16000 });
await ctx.audioWorklet.addModule("/audio-worklet.js");
const node = new AudioWorkletNode(ctx, "NoiseSuppressionWorker", {
  numberOfInputs: 1,
  numberOfOutputs: 1,
  outputChannelCount: [1],
  channelCount: 1,
  channelCountMode: "explicit",
});

// Wait for "ready" before sending audio through it.
await new Promise((resolve, reject) => {
  const timer = setTimeout(() => reject(new Error("DTLN ready timeout")), 15000);
  node.onprocessorerror = () => reject(new Error("DTLN processor crashed"));
  node.port.onmessage = (e) =>
    e.data === "ready" && (clearTimeout(timer), resolve());
});
node.port.postMessage({ noise_gate: 0.002 });
```

**The `"ready"` handshake is not optional.** The worklet posts it only once the
WASM runtime has initialized *and* the denoiser exists; until then it outputs
silence. A denoiser that cannot be created, or that traps mid-stream, throws
from `process()` instead - that is `processorerror`, never a silent node - so
always pair the wait with a timeout and an `onprocessorerror` handler, and
fall back to the browser's own `noiseSuppression` constraint when either
fires.

**After a `processorerror`, rebuild on a new `AudioContext`.** Every node in a
context shares one WASM instance, and its heap is a fixed 16 MB that
`dtln_destroy` does not give back: one instance can create exactly two
denoisers, ever. A third `dtln_create()` returned a handle that trapped on its
first block, or never returned and hung the audio thread, so the worklet now
refuses it (with a `processorerror` saying so). Closing the context and making
a new one gets a new instance.

## Building

```sh
npm install
npm run build    # -> dist/audio-worklet.js (~8 MB: the models are embedded)
npm run sync     # build + install it into the sibling awful checkout
npm run serve    # serve dist/ at http://localhost:8080 (build first)
```

`npm run sync` copies the bundle to `frontend/public/audio-worklet.js` and
regenerates `frontend/src/lib/audio/worklet-url.ts` with a content hash
(`/audio-worklet.js?v=<hash>`). That query is load-bearing: awful's service
worker caches the worklet forever, so without a changing URL a returning user
would keep the first build they ever downloaded. **Always sync; never copy the
bundle by hand.** It looks for `../awful.chat` (then `../awful2`, `../awful`),
or takes a path:

```sh
node sync-to-awful.mjs ../wherever/awful2     # or set AWFUL_DIR
```

## Trying it and verifying it

Two pages, both built into `dist/` and served by `npm run serve`:

**`index.html` - the lab.** Input and output spectrograms (0-8 kHz) side by
side, so you can see what the model removed rather than only hearing it. Two
modes: *live* runs your mic through the worklet with the spectrograms scrolling
and optional monitoring (headphones - speakers feed back), and *record*
captures a clip, denoises it offline and gives you before/after players to A/B.
The noise-gate slider (in dBFS; the -54 dB default is the worklet's `0.002`)
is live in both modes - in record mode, moving it re-processes the clip you
already captured, so judging a threshold does not mean recording a new take
per value. That matters because `0.002` is a guess until someone listens to
their own voice through it.

The mic is captured with `autoGainControl` on, as awful does, so the level
the gate sees here is the level it sees on a call. The lab runs at 16 kHz,
though, and awful at the hardware rate - so the lab does not exercise the
worklet's resampling path; a call does.

**`check.html` - the self-check.** Renders synthetic noise, speech and digital
silence through the worklet and asserts what comes back:

```
worklet ready in 19 ms
PASS  noise is suppressed  -27.9 dB (want <= -12)
PASS  speech survives       -0.4 dB (want >= -3)
PASS  silence stays silent  rms 0.0e+0
PASS  output is finite      0 non-finite samples
```

Thresholds are loose deliberately: this catches *broken*, not "1 dB worse". It
needs no microphone and no user gesture, so it runs to completion on load.

Both pages are published to GitHub Pages on every push to `main`
(`.github/workflows/pages.yml`): the lab at
<https://flaviozanoni.github.io/dtln-rs-web/> and the self-check at
<https://flaviozanoni.github.io/dtln-rs-web/check.html>.

Run the self-check before syncing a worklet change, and the lab when the change
is one that only ears can judge. The easiest thing to ship here is a **silent**
worklet, and no amount of unit testing with fake audio nodes catches that - the
WASM never runs in them.

The trap that makes it necessary: `dtln_create()` returns incrementing ids and
the first one is **0**. Testing the handle for truthiness (`!handle`) therefore
reads a perfectly good denoiser as a missing one and mutes the worklet
completely - silently, since nothing throws. Compare against `undefined`.

## Layout

| Path | |
| --- | --- |
| `src/audio-worklet/main.ts` | the worklet: resampling, blocking, gate, ring buffer |
| `src/audio-worklet/dtln.js` | emscripten glue + embedded model (generated; only its hand-written tail is edited) |
| `src/app.js`, `index.html` | the lab: spectrograms, live monitor, A/B record |
| `src/test.js`, `check.html` | the self-check page |
| `sync-to-awful.mjs` | build artifact → awful, with cache-busting hash |
| `.github/workflows/pages.yml` | builds `dist/` and publishes both pages |

## Attribution

Apache-2.0, from Datadog's [dtln-rs-demo](https://github.com/DataDog/dtln-rs-demo);
see `LICENSE`, `NOTICE`, and `LICENSE-3rdparty.csv`. The model is
[dtln-rs](https://github.com/DataDog/dtln-rs) (MIT), itself a port of
[breizhn/DTLN](https://github.com/breizhn/DTLN).
