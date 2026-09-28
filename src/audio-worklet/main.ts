import type { DtlnPluginOpaqueHandle } from "dtln-rs";
import dtln from "./dtln.js";

const DTLN_FIXED_BUFFER_SIZE = 512;
const DTLN_SAMPLE_RATE = 16000;
const RING_SIZE = DTLN_FIXED_BUFFER_SIZE * 8;

// Gate time constants, per 512-sample block (31.25 blocks/s at 16 kHz).
// The envelope attack is instant - anything under one block collapses to
// instant at this rate anyway - and the release is ~80 ms.
const GATE_BLOCK_RATE = DTLN_SAMPLE_RATE / DTLN_FIXED_BUFFER_SIZE;
const GATE_ENV_RELEASE = 1.0 - Math.exp(-1.0 / (GATE_BLOCK_RATE * 0.08));
// Gain smoothing: open fast (~91% in 2 blocks / 64 ms) so speech onsets
// survive the gate, close slow (~300 ms) so tails fade instead of chopping.
const GATE_OPEN_COEFF = 0.7;
const GATE_CLOSE_COEFF = 0.1;
// Hold the gate open ~256 ms after the level last cleared the close
// threshold. Without it the gate started closing inside every dip between
// syllables, and on a quiet mic (no AGC on the DTLN path) that chopped
// speech into pieces - the "spotty" voice that only a threshold of 0 fixed.
const GATE_HOLD_BLOCKS = Math.round(GATE_BLOCK_RATE * 0.256);
// A closed gate attenuates by 20 dB instead of muting. On top of the model's
// own suppression that still reads as silence between phrases, and a
// threshold set too high for someone's mic now dims soft syllables instead
// of deleting them.
const GATE_FLOOR = 0.1;

interface AudioWorkletProcessor {
  readonly port: MessagePort;
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>,
  ): boolean;
}
declare function registerProcessor(
  name: string,
  processorCtor: (new (
    options?: AudioWorkletNodeOptions,
  ) => AudioWorkletProcessor) & {
    parameterDescriptors?: any[];
  },
): void;
declare let AudioWorkletProcessor: {
  prototype: AudioWorkletProcessor;
  new (options?: AudioWorkletNodeOptions): AudioWorkletProcessor;
};

// The WASM runtime finishes initializing asynchronously, after this module
// has evaluated. Hooking postRun here is race-free: instantiation waits on a
// promise that cannot resolve before module evaluation completes.
let wasmReady = false;
const onWasmReady: Array<() => void> = [];
dtln.postRun = [
  () => {
    wasmReady = true;
    for (const cb of onWasmReady) cb();
    onWasmReady.length = 0;
  },
];

// The WASM heap is a fixed 16 MB (the build never grows it), each denoiser
// allocates two tflite interpreters, and dtln_destroy does not hand that
// memory back. So one instance - one AudioWorkletGlobalScope, i.e. one
// AudioContext - can create exactly two denoisers in its lifetime. A third
// dtln_create() either returns a handle whose first denoise traps ("table
// index is out of bounds") or never returns at all, which would hang the
// audio thread with no error anywhere. Refuse instead; a fresh context gets
// a fresh instance and a fresh budget.
const MAX_DENOISERS_PER_INSTANCE = 2;
let denoisers_created = 0;

class NoiseSuppressionWorker extends AudioWorkletProcessor {
  private dtln_handle: DtlnPluginOpaqueHandle | undefined;
  // Set when this processor can never produce audio; process() throws it so
  // the node fires processorerror and the consumer rebuilds on a new context.
  private fatal: unknown;

  // resampling
  private native_rate = 0;
  private resample_ratio = 1;
  private downsample_frac = 0;
  private upsample_frac = 0;
  private upsample_last = 0;

  // anti-alias low-pass before decimation; only used when ratio > 1, which
  // is every context not at 16 kHz - awful's included, it runs at the
  // hardware rate
  private aa_coeffs: Float64Array | undefined;
  private aa_state = new Float64Array(8); // x1,x2,y1,y2 per stage
  private aa_buf: Float32Array | undefined;

  // input accumulator at 16kHz
  private input_buf = new Float32Array(DTLN_FIXED_BUFFER_SIZE);
  private input_index = 0;

  // dtln scratch
  private dtln_out = new Float32Array(DTLN_FIXED_BUFFER_SIZE);

  // output ring at native rate
  private ring = new Float32Array(RING_SIZE);
  private ring_read = 0;
  private ring_write = 0;
  private ring_count = 0;

  // gate
  private gate_threshold = 0.002;
  private gate_envelope = 0;
  private gate_target = 0;
  private gate_open = 0;
  private gate_hold = 0;

  constructor() {
    super();
    this.port.onmessage = (event) => {
      if (typeof event.data?.noise_gate === "number") {
        this.gate_threshold = event.data.noise_gate;
      }
    };
    // Only announce readiness once the runtime is up and the denoiser
    // exists: consumers gate audio on this message, so process() can assume
    // a live handle, and the tflite interpreter construction happens here
    // instead of inside a render quantum.
    const announce = () => {
      if (denoisers_created >= MAX_DENOISERS_PER_INSTANCE) {
        this.fatal = new Error(
          `[DTLN] this AudioContext already created ${denoisers_created} ` +
            "denoisers, the most its WASM heap holds - use a new AudioContext",
        );
        return;
      }
      denoisers_created++;
      try {
        this.dtln_handle = dtln.dtln_create();
        this.port.postMessage("ready");
      } catch (e) {
        this.fatal = e;
      }
    };
    if (wasmReady) announce();
    else onWasmReady.push(announce);
  }

  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    _parameters: Record<string, Float32Array>,
  ): boolean {
    // No "ready" was sent. Throwing fires processorerror, which a consumer
    // waiting on the handshake sees at once instead of at its timeout.
    if (this.fatal !== undefined) throw this.fatal;

    const input = inputs?.[0]?.[0];
    const output = outputs?.[0]?.[0];
    // Compare against undefined, never truthiness: dtln_create() hands out
    // incrementing ids and the FIRST one is 0, so `!handle` reads a perfectly
    // good denoiser as a missing one and mutes the whole worklet.
    if (!input || !output || this.dtln_handle === undefined) {
      output?.fill(0);
      return true;
    }

    if (this.native_rate === 0) {
      this.native_rate = (globalThis as any).sampleRate ?? 48000;
      this.resample_ratio = this.native_rate / DTLN_SAMPLE_RATE;
      if (this.resample_ratio > 1) {
        this.initAntiAlias();
        // Each finished block bursts ~512*ratio samples into the ring; the
        // default size only holds ratio <= 8 (128 kHz). Grow it for exotic
        // device rates (176.4/192 kHz) instead of silently overwriting.
        const needed =
          DTLN_FIXED_BUFFER_SIZE * Math.ceil(this.resample_ratio) * 2;
        if (needed > this.ring.length) this.ring = new Float32Array(needed);
      }
    }

    const ratio = this.resample_ratio;
    const src_buf = ratio > 1 ? this.antiAlias(input) : input;

    // downsample input to 16kHz
    let src = this.downsample_frac;
    while (src < input.length) {
      this.input_buf[this.input_index++] =
        src_buf[Math.min(Math.floor(src), input.length - 1)];
      src += ratio;

      if (this.input_index >= DTLN_FIXED_BUFFER_SIZE) {
        // A throw here is a WASM trap, and nothing in this instance can be
        // trusted after one - recreating the handle in place used up the
        // instance's denoiser budget (see MAX_DENOISERS_PER_INSTANCE) and
        // then hung or trapped again. Let it propagate: processorerror tells
        // the consumer to rebuild on a new context, where a silent zombie
        // node would have told nobody.
        dtln.dtln_denoise(this.dtln_handle, this.input_buf, this.dtln_out);

        this.applyGate();

        this.input_index = 0;

        // upsample back to native rate with linear interpolation
        let frac = this.upsample_frac;
        for (let i = 0; i < DTLN_FIXED_BUFFER_SIZE; i++) {
          const next = this.dtln_out[i];
          frac += ratio;
          const steps = Math.floor(frac);
          frac -= steps;
          for (let s = 0; s < steps; s++) {
            const t = steps > 1 ? s / steps : 0;
            this.ring[this.ring_write] =
              this.upsample_last + t * (next - this.upsample_last);
            this.ring_write = (this.ring_write + 1) % this.ring.length;
            this.ring_count++;
          }
          this.upsample_last = next;
        }
        this.upsample_frac = frac;
      }
    }
    this.downsample_frac = src - input.length;

    // drain ring into output
    if (this.ring_count >= output.length) {
      for (let i = 0; i < output.length; i++) {
        output[i] = this.ring[this.ring_read];
        this.ring_read = (this.ring_read + 1) % this.ring.length;
      }
      this.ring_count -= output.length;
    } else {
      output.fill(0);
    }

    return true;
  }

  private applyGate(): void {
    // compute block RMS
    let rms = 0;
    for (let i = 0; i < DTLN_FIXED_BUFFER_SIZE; i++) {
      rms += this.dtln_out[i] * this.dtln_out[i];
    }
    rms = Math.sqrt(rms / DTLN_FIXED_BUFFER_SIZE);

    // envelope follower - instant attack, slow release
    if (rms > this.gate_envelope) {
      this.gate_envelope = rms;
    } else {
      this.gate_envelope += GATE_ENV_RELEASE * (rms - this.gate_envelope);
    }

    // hysteresis: open at the threshold, close 6 dB below it, so a level
    // hovering at the threshold doesn't flutter the gate - and only once the
    // hold has run out, so the gaps inside a sentence don't close it
    if (this.gate_envelope > this.gate_threshold) {
      this.gate_target = 1.0;
      this.gate_hold = GATE_HOLD_BLOCKS;
    } else if (this.gate_envelope >= this.gate_threshold * 0.5) {
      if (this.gate_target === 1.0) this.gate_hold = GATE_HOLD_BLOCKS;
    } else if (this.gate_hold > 0) {
      this.gate_hold--;
    } else {
      this.gate_target = 0.0;
    }

    // asymmetric smoothing (see the coefficients up top)
    const prev = this.gate_open;
    const coeff = this.gate_target > prev ? GATE_OPEN_COEFF : GATE_CLOSE_COEFF;
    this.gate_open = prev + coeff * (this.gate_target - prev);

    // ramp the gain across the block - a per-block constant steps the gain
    // at 31 Hz, which is audible zipper noise at every transition
    const from = GATE_FLOOR + (1 - GATE_FLOOR) * prev;
    const to = GATE_FLOOR + (1 - GATE_FLOOR) * this.gate_open;
    const step = (to - from) / DTLN_FIXED_BUFFER_SIZE;
    let g = from;
    for (let i = 0; i < DTLN_FIXED_BUFFER_SIZE; i++) {
      g += step;
      this.dtln_out[i] *= g;
    }
  }

  // 4th-order Butterworth low-pass at 7 kHz (two RBJ biquads), applied at
  // native rate before decimation so 8..native/2 kHz content doesn't fold
  // into the speech band the model expects.
  private initAntiAlias(): void {
    const f0 = 7000;
    const qs = [0.5412, 1.3066]; // Butterworth 4th-order stage Qs
    const coeffs = new Float64Array(10);
    for (let s = 0; s < 2; s++) {
      const w = (2 * Math.PI * f0) / this.native_rate;
      const alpha = Math.sin(w) / (2 * qs[s]);
      const cosw = Math.cos(w);
      const a0 = 1 + alpha;
      coeffs[s * 5 + 0] = (1 - cosw) / 2 / a0;
      coeffs[s * 5 + 1] = (1 - cosw) / a0;
      coeffs[s * 5 + 2] = (1 - cosw) / 2 / a0;
      coeffs[s * 5 + 3] = (-2 * cosw) / a0;
      coeffs[s * 5 + 4] = (1 - alpha) / a0;
    }
    this.aa_coeffs = coeffs;
  }

  private antiAlias(input: Float32Array): Float32Array {
    if (!this.aa_buf || this.aa_buf.length < input.length) {
      this.aa_buf = new Float32Array(input.length);
    }
    const c = this.aa_coeffs!;
    const st = this.aa_state;
    const out = this.aa_buf;
    for (let i = 0; i < input.length; i++) {
      let x = input[i];
      for (let s = 0; s < 2; s++) {
        const o = s * 4;
        const k = s * 5;
        const y =
          c[k] * x +
          c[k + 1] * st[o] +
          c[k + 2] * st[o + 1] -
          c[k + 3] * st[o + 2] -
          c[k + 4] * st[o + 3];
        st[o + 1] = st[o];
        st[o] = x;
        st[o + 3] = st[o + 2];
        st[o + 2] = y;
        x = y;
      }
      out[i] = x;
    }
    return out;
  }
}

registerProcessor("NoiseSuppressionWorker", NoiseSuppressionWorker);
