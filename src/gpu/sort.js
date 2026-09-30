import {STORAGE, UNIFORM, COPY_DST, COPY_SRC} from "./constants.js";
import {PRELUDE} from "./wgsl.js";

// Stable least-significant-digit radix sort of (key, value) pairs of u32, 8 bits per
// pass. Every pass takes three dispatches:
//
//   count     each tile of TILE pairs counts its digits into hist, digit by digit
//   scan      one workgroup per digit turns its row of hist into running totals over
//             the tiles, and records the digit's total
//   scatter   every pair goes to its position: the start of its digit (from the totals),
//             plus its tile's running total for that digit, plus the number of earlier
//             pairs of the tile with the same digit.
//             That rank comes from a bit mask per digit (one bit per thread), which keeps
//             the sort stable without subgroup operations.
//
// No global atomics: summing the counts with them serialises on 256 addresses, which
// cost more than the rest of the sort for a million keys.
//
// Pairs sit in two buffers used in turn; after an even number of passes the result is
// back in the first one.

var THREADS = 256, PER_THREAD = 4;
export var SORT_TILE = THREADS * PER_THREAD;

var code = PRELUDE + /* wgsl */`
struct SortParams { n: u32, tiles: u32, shift: u32, pad: u32 }
@group(1) @binding(0) var<uniform> sp: SortParams;
@group(1) @binding(1) var<storage, read> src: array<vec2<u32>>;
@group(1) @binding(2) var<storage, read_write> dst: array<vec2<u32>>;
@group(1) @binding(3) var<storage, read_write> hist: array<u32>;
@group(1) @binding(4) var<storage, read_write> totals: array<u32, 256>;

const THREADS = ${THREADS}u;
const PER_THREAD = ${PER_THREAD}u;
const TILE = ${THREADS * PER_THREAD}u;

var<workgroup> counts: array<atomic<u32>, 256>;
var<workgroup> sums: array<u32, 256>;
var<workgroup> masks: array<atomic<u32>, 2048>; // digit * 8 + thread / 32
var<workgroup> starts: array<u32, 256>;

fn digitOf(key: u32) -> u32 {
  return (key >> sp.shift) & 255u;
}

@compute @workgroup_size(${THREADS})
fn count(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  atomicStore(&counts[li], 0u);
  workgroupBarrier();
  for (var k = 0u; k < PER_THREAD; k++) {
    let i = wid.x * TILE + k * THREADS + li;
    if (i < sp.n) { atomicAdd(&counts[digitOf(src[i].x)], 1u); }
  }
  workgroupBarrier();
  hist[li * sp.tiles + wid.x] = atomicLoad(&counts[li]);
}

// Inclusive prefix sum of sums[0..256) (Hillis-Steele).
fn scanShared(li: u32) {
  for (var o = 1u; o < 256u; o <<= 1u) {
    var v = 0u;
    if (li >= o) { v = sums[li - o]; }
    workgroupBarrier();
    sums[li] += v;
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn scan(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let d = wid.x;
  let row = d * sp.tiles;
  var running = 0u;
  for (var base = 0u; base < sp.tiles; base += 256u) {
    let t = base + li;
    var v = 0u;
    if (t < sp.tiles) { v = hist[row + t]; }
    sums[li] = v;
    workgroupBarrier();
    scanShared(li);
    if (t < sp.tiles) { hist[row + t] = running + sums[li] - v; }
    running += sums[255];
    workgroupBarrier();
  }
  if (li == 0u) { totals[d] = running; }
}

@compute @workgroup_size(${THREADS})
fn scatter(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  sums[li] = totals[li];
  workgroupBarrier();
  scanShared(li);
  starts[li] = sums[li] - totals[li] + hist[li * sp.tiles + wid.x];
  let word = li >> 5u;
  let bit = 1u << (li & 31u);

  for (var k = 0u; k < PER_THREAD; k++) {
    for (var w = 0u; w < 8u; w++) { atomicStore(&masks[li * 8u + w], 0u); }
    workgroupBarrier();

    let i = wid.x * TILE + k * THREADS + li;
    let valid = i < sp.n;
    var e = vec2<u32>(0u);
    var d = 0u;
    if (valid) {
      e = src[i];
      d = digitOf(e.x);
      atomicOr(&masks[d * 8u + word], bit);
    }
    workgroupBarrier();

    if (valid) {
      var rank = countOneBits(atomicLoad(&masks[d * 8u + word]) & (bit - 1u));
      for (var w = 0u; w < word; w++) { rank += countOneBits(atomicLoad(&masks[d * 8u + w])); }
      dst[starts[d] + rank] = e;
    }
    workgroupBarrier();

    var c = 0u;
    for (var w = 0u; w < 8u; w++) { c += countOneBits(atomicLoad(&masks[li * 8u + w])); }
    starts[li] += c;
    workgroupBarrier();
  }
}
`;

var spec = ["uniform", "ro", "rw", "rw", "rw"];

export var pipelines = {
  sortCount: {code: code, entry: "count", spec: spec},
  sortScan: {code: code, entry: "scan", spec: spec},
  sortScatter: {code: code, entry: "scatter", spec: spec}
};

// A sorter for `n` pairs. The caller fills pairs[0] (vec2<u32> per pair: key, value) and
// encodes the sort: one pass per entry of `shifts`, sorting on the 8-bit digit of the key
// at that bit, least significant first. After p passes the pairs are in pairs[p & 1], so
// the key can be changed between passes (to sort on more than 32 bits).
export function createSort(engine, n, shifts) {
  var device = engine.device,
      tiles = Math.max(1, Math.ceil(n / SORT_TILE)),
      slot = Math.max(256, device.limits.minUniformBufferOffsetAlignment | 0),
      make = function(size, usage) { return device.createBuffer({size: Math.max(16, size), usage: usage}); },
      pairs = [make(8 * n, STORAGE | COPY_DST | COPY_SRC), make(8 * n, STORAGE | COPY_SRC)],
      hist = make(1024 * tiles, STORAGE),
      totals = make(1024, STORAGE),
      passes = shifts.length,
      params = make(slot * passes, UNIFORM | COPY_DST),
      data = new Uint32Array(slot * passes / 4),
      groups = [], p;

  for (p = 0; p < passes; ++p) {
    data.set([n, tiles, shifts[p], 0], p * slot / 4);
    groups.push(device.createBindGroup({
      layout: engine.pipeline("sortCount").getBindGroupLayout(1),
      entries: [
        {binding: 0, resource: {buffer: params, offset: p * slot, size: 16}},
        {binding: 1, resource: {buffer: pairs[p & 1]}},
        {binding: 2, resource: {buffer: pairs[1 - (p & 1)]}},
        {binding: 3, resource: {buffer: hist}},
        {binding: 4, resource: {buffer: totals}}
      ]
    }));
  }
  device.queue.writeBuffer(params, 0, data);

  return {
    pairs: pairs,
    // where the sorted pairs end up
    result: pairs[passes & 1],

    // passes [from, to), all of them by default
    encode: function(pass, from, to) {
      for (var p = from || 0, end = to === undefined ? passes : to; p < end; ++p) {
        pass.setBindGroup(1, groups[p]);
        pass.setPipeline(engine.pipeline("sortCount"));
        pass.dispatchWorkgroups(tiles);
        pass.setPipeline(engine.pipeline("sortScan"));
        pass.dispatchWorkgroups(256);
        pass.setPipeline(engine.pipeline("sortScatter"));
        pass.dispatchWorkgroups(tiles);
      }
    },

    destroy: function() {
      pairs.forEach(function(b) { b.destroy(); });
      hist.destroy(), totals.destroy(), params.destroy();
    }
  };
}
