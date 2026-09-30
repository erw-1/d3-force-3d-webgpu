import {STORAGE, UNIFORM, COPY_DST} from "../constants.js";

// Largest finite value we hand to a shader in place of Infinity: WGSL does not
// guarantee that infinities survive a round trip through a uniform.
export var BIG = 3e38;

export function finite(x) {
  return x > BIG ? BIG : x < -BIG ? -BIG : x;
}

// A read-only storage buffer holding `array` (at least 16 bytes: zero-size bindings are
// not allowed).
export function storage(device, array) {
  var buffer = device.createBuffer({
    size: Math.max(16, (array.byteLength + 3) & ~3),
    usage: STORAGE | COPY_DST
  });
  if (array.byteLength) device.queue.writeBuffer(buffer, 0, array);
  return buffer;
}

// f32 view over a JS array of numbers; anything non-numeric becomes 0 so that a stray
// undefined can never poison the simulation with NaN.
export function floats(values, n) {
  var out = new Float32Array(n);
  for (var i = 0; i < n; ++i) out[i] = finite(+values[i] || 0);
  return out;
}

// A small uniform block that is only rewritten when a value actually changes.
export function params(device, size) {
  var buffer = device.createBuffer({size: size, usage: UNIFORM | COPY_DST}),
      data = new Float32Array(size / 4),
      view = new Uint32Array(data.buffer),
      dirty = true;

  return {
    buffer: buffer,
    data: data,
    view: view,
    // values: array of numbers written as f32 from word 0; ints: {word: uint value}
    set: function(values, ints) {
      var i;
      for (i = 0; i < values.length; ++i) {
        var v = finite(+values[i]);
        if (data[i] !== v && !(v !== v && data[i] !== data[i])) data[i] = v, dirty = true;
      }
      if (ints) for (var word in ints) {
        if (view[word] !== ints[word]) view[word] = ints[word], dirty = true;
      }
      if (dirty) device.queue.writeBuffer(buffer, 0, data), dirty = false;
    },
    destroy: function() { buffer.destroy(); }
  };
}

// Small graphs give a GPU too few threads (one per node) to hide memory latency. Kernels
// that loop over all nodes therefore let SPLIT threads share each node's loop and add
// their partial sums; big graphs already have threads to spare and use SPLIT = 1.
var TARGET_THREADS = 1 << 18, MAX_SPLIT = 16, TILE_SIZE = 128;
export function splitFor(n, forced) {
  var split = 1;
  if (forced > 0) { // the caller's choice: a power of two, at most one node per workgroup
    while (split * 2 <= Math.min(forced, TILE_SIZE)) split *= 2;
    return split;
  }
  while (split < MAX_SPLIT && n * split < TARGET_THREADS) split *= 2;
  return split;
}

export function ceilDiv(a, b) {
  return Math.max(1, Math.ceil(a / b));
}
