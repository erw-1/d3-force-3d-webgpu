import {PRELUDE} from "../wgsl.js";
import {STORAGE, UNIFORM, COPY_DST, INDIRECT} from "../constants.js";
import {createSort, pipelines as sortPipelines} from "../sort.js";
import {ceilDiv} from "./util.js";

// Many-body force with a Barnes-Hut tree that reproduces the one manyBody.js builds with
// d3-octree (d3-quadtree in 2D, d3-binarytree in 1D), so the GPU gives the same forces as
// the CPU, theta and all.
//
// The d3 tree: a cube [x0, x0 + size) where x0 = floor(min) and size is the smallest power
// of two that covers the max; cells split at their midpoints until every leaf holds one
// point (or coincident points). A cell is approximated when w * w / theta2 < l (w: its
// width, l: squared distance to its centre of mass), and its strength is the sum of its
// children's, times sqrt(4 / numChildren) (0.71 in 3D, 1 in 2D, 1.41 in 1D). A point's
// strength therefore reaches a cell k levels above its leaf scaled by c^k:
//
//     value(cell at depth D) = sum over its points of strength * c^(leafDepth - D)
//
// and the cell's centre of mass weighs each point by |strength| * c^leafDepth.
//
// On the GPU (every step, in this order):
//
//   bbox      bounding box of the positions (atomics on order-preserving float bits)
//   cube      the d3 cube, from the box (one thread; also empties the box for the next step).
//             Up to BOX_MAX nodes, bboxCube does both in one workgroup.
//   keys      every node's Morton code in the cube: LEVELS levels, in two u32 halves
//   sort      radix sort of (code, node) pairs on the high halves only; then fix orders
//             the runs of points sharing a high half on their low halves (longRuns, a
//             workgroup each, the rare runs of more than RUN points). Up to RANK_MAX
//             nodes, rank sorts them on the whole codes in one step instead.
//   prep      per sorted point: its leaf depth (from the codes around it), and the bottom
//             of a pyramid of partial sums over runs of sorted points: strength * c^depth,
//             its absolute value w, and w * (position - p0), p0 being the run's first
//             point (sums about a nearby point keep f32 precision far from the origin)
//   up        the higher levels of the pyramid
//   build     a binary radix tree over the codes (Karras 2012). Each of its nodes stands
//             for the d3 cells between its parent's split and its own, which all hold the
//             same points; their sums come from the pyramid.
//   collapse  every node lists its children in the d3 tree: the first nodes or points
//             below it that are in deeper cells (up to 2^DIMS of them, at most DIMS levels
//             down the binary tree). The binary nodes in between are never visited.
//   walk      for every point, the d3 traversal: a node is tested at each of the cell
//             depths it stands for, and approximated at the first one that passes.
//
// Points closer than the smallest cell (size / 2^LEVELS, 20 levels in 3D and 24 in 1D and
// 2D) share a leaf cell here, where d3 would split further; they still interact exactly
// with each other.

var WG = 64, REDUCE = 512, RUN = 32, LEVELS = [0, 24, 24, 20];

// Up to RANK_MAX nodes, a one-step rank sort (n^2 comparisons) beats the radix sort's
// dispatches; up to SHARED_MAX nodes, walking the tree with 8 threads per node beats one;
// up to BOX_MAX nodes, one workgroup finds the box faster than a dispatch more.
var RANK_MAX = 8192, SHARED_MAX = 32768, BOX_MAX = 16384;

// Digits of the high half of the codes that the sort has to look at (the code sits in its
// high bits: 12, 24 or 30 bits in 1D, 2D and 3D).
var SHIFTS = [null, [16, 24], [8, 16, 24], [0, 8, 16, 24]];

var params = /* wgsl */`
override DIMS: u32 = 3u;
override LEVELS: u32 = 20u;
override LOGC: f32 = -0.5; // log2(sqrt(4 / 2^DIMS))
override HALF: u32 = 30u;  // bits of code in each half: LEVELS / 2 * DIMS

struct BH {
  n: u32,
  base: u32,    // up: the pyramid level it starts from
  inner: u32,   // nodes of the binary tree: max(1, n - 1)
  total: u32,   // pyramid entries
  theta2: f32,
  distanceMin2: f32,
  distanceMax2: f32,
  pad1: f32,
  offs: array<vec4<u32>, 8>, // where each pyramid level starts
}

@group(1) @binding(0) var<uniform> bh: BH;

fn levelStart(k: u32) -> u32 { return bh.offs[k >> 2u][k & 3u]; }
fn levelCount(k: u32) -> u32 { return (bh.n + (1u << k) - 1u) >> k; }

// c^k, c = sqrt(4 / 2^DIMS): a power of sqrt(2), so exact up to the rounding of sqrt(2)
fn damp(k: i32) -> f32 {
  if (LOGC == 0.0) { return 1.0; }
  let half = k >> 1u;
  let odd = select(1.0, exp2(LOGC), (k & 1) != 0);
  return ldexp(odd, select(-half, half, LOGC > 0.0));
}

// Depth of the d3 cell that splits between codes sharing prefix bits.
fn depthOf(prefix: i32) -> u32 {
  return min(u32(prefix) / DIMS, LEVELS);
}
`;

// For the kernels that use the sorted points (layouts A and B).
var lookups = /* wgsl */`
// Pyramid entry k: its weighted sums (four f32 from 4k) and its strength (at 4 total + k).
// Plain f32s, so that threads writing neighbouring entries never share a vector.
fn pyrW(k: u32) -> vec4<f32> { return vec4<f32>(pyr[4u * k], pyr[4u * k + 1u], pyr[4u * k + 2u], pyr[4u * k + 3u]); }
fn pyrA(k: u32) -> f32 { return pyr[4u * bh.total + k]; }

// Length of the common prefix of the codes at sorted positions a and b (-1 if b is out of
// range). The sorted pairs hold the high halves; the low halves are looked up only when
// those are equal. Equal codes are told apart by their position, as in Karras's
// construction.
fn delta(a: u32, b: i32) -> i32 {
  if (b < 0 || b >= i32(bh.n)) { return -1; }
  let pa = pairs[a];
  let pb = pairs[u32(b)];
  if (pa.x != pb.x) { return i32(countLeadingZeros(pa.x ^ pb.x)); }
  let la = codes[pa.y].y;
  let lb = codes[pb.y].y;
  if (la != lb) { return i32(HALF + countLeadingZeros(la ^ lb)); }
  return 64 + i32(countLeadingZeros(a ^ u32(b)));
}
`;

// Kernels that build the pyramid (bind group layout A).
var build = PRELUDE + /* wgsl */`
// box: min x, y, z, -, max x, y, z, -. cube: origin, size. shift: log2(2^LEVELS / size)
struct Misc { box: array<atomic<u32>, 8>, cube: vec4<f32>, shift: i32, pad0: i32, pad1: i32, pad2: i32 }
@group(1) @binding(1) var<storage, read_write> misc: Misc;
@group(1) @binding(2) var<storage, read_write> pairs: array<vec2<u32>>;
@group(1) @binding(3) var<storage, read> strength: array<f32>;
@group(1) @binding(4) var<storage, read_write> leaves: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> pyr: array<f32>;
@group(1) @binding(6) var<storage, read_write> codes: array<vec2<u32>>; // by node: high, low half
` + params + lookups + /* wgsl */`
// Floats as u32 with the same order, for atomicMin / atomicMax.
fn ordered(f: f32) -> u32 {
  let b = bitcast<u32>(f);
  return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}
fn unordered(u: u32) -> f32 {
  return bitcast<f32>(select(~u, u & 0x7fffffffu, (u & 0x80000000u) != 0u));
}

var<workgroup> boxLo: array<vec3<f32>, 256>;
var<workgroup> boxHi: array<vec3<f32>, 256>;

@compute @workgroup_size(256)
fn bbox(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(num_workgroups) groups: vec3<u32>) {
  var lo = vec3<f32>(3.4e38);
  var hi = vec3<f32>(-3.4e38);
  for (var i = gid.x; i < bh.n; i += groups.x * 256u) {
    let p = pos[i].xyz;
    lo = min(lo, p);
    hi = max(hi, p);
  }
  boxLo[li] = lo;
  boxHi[li] = hi;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) {
      boxLo[li] = min(boxLo[li], boxLo[li + s]);
      boxHi[li] = max(boxHi[li], boxHi[li + s]);
    }
    workgroupBarrier();
  }
  if (li < 3u) {
    atomicMin(&misc.box[li], ordered(boxLo[0][li]));
    atomicMax(&misc.box[4u + li], ordered(boxHi[0][li]));
  }
}

// d3's cover(): x0 = floor(min), then double the (cubic) extent until it holds the max.
fn cover(lo: vec3<f32>, hi: vec3<f32>) {
  let origin = floor(lo);
  var size = 1.0;
  var log2size = 0;
  for (var k = 0u; k < 128u; k++) {
    if (all(hi < origin + size)) { break; }
    size *= 2.0;
    log2size += 1;
  }
  misc.cube = vec4<f32>(origin, size);
  misc.shift = i32(LEVELS) - log2size;
}

// A single thread: every thread reading the box atomically would queue on six addresses.
@compute @workgroup_size(1)
fn cube() {
  cover(vec3<f32>(unordered(atomicLoad(&misc.box[0])), unordered(atomicLoad(&misc.box[1])), unordered(atomicLoad(&misc.box[2]))),
        vec3<f32>(unordered(atomicLoad(&misc.box[4])), unordered(atomicLoad(&misc.box[5])), unordered(atomicLoad(&misc.box[6]))));
  for (var j = 0u; j < 8u; j++) { atomicStore(&misc.box[j], select(0u, 0xffffffffu, j < 4u)); }
}

@compute @workgroup_size(256)
fn bboxCube(@builtin(local_invocation_index) li: u32) {
  var lo = vec3<f32>(3.4e38);
  var hi = vec3<f32>(-3.4e38);
  for (var i = li; i < bh.n; i += 256u) {
    let p = pos[i].xyz;
    lo = min(lo, p);
    hi = max(hi, p);
  }
  boxLo[li] = lo;
  boxHi[li] = hi;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) {
      boxLo[li] = min(boxLo[li], boxLo[li + s]);
      boxHi[li] = max(boxHi[li], boxHi[li + s]);
    }
    workgroupBarrier();
  }
  if (li == 0u) { cover(boxLo[0], boxHi[0]); }
}

var<workgroup> redW: array<vec4<f32>, 256>;
var<workgroup> redA: array<f32, 256>;
var<workgroup> redP: array<vec3<f32>, 256>; // first point of each run

// Two consecutive runs as one: (sum of w * (p - p0), sum of w) about the first run's p0.
fn join(w1: vec4<f32>, w2: vec4<f32>, p1: vec3<f32>, p2: vec3<f32>) -> vec4<f32> {
  if (w2.w == 0.0) { return w1; }
  return vec4<f32>(w1.xyz + w2.xyz + w2.w * (p2 - p1), w1.w + w2.w);
}

fn setPyr(k: u32, w: vec4<f32>, a: f32) {
  pyr[4u * k] = w.x;
  pyr[4u * k + 1u] = w.y;
  pyr[4u * k + 2u] = w.z;
  pyr[4u * k + 3u] = w.w;
  pyr[4u * bh.total + k] = a;
}

// This thread holds block (wid * 256 + li) of pyramid level base + 1, whose first point is
// p; write it, and the blocks of the eight levels above it that this workgroup completes.
fn reduceUp(base: u32, wid: u32, li: u32, w: vec4<f32>, a: f32, p: vec3<f32>) {
  var b = wid * 256u + li;
  if (b < levelCount(base + 1u)) { setPyr(levelStart(base + 1u) + b, w, a); }
  redW[li] = w;
  redA[li] = a;
  redP[li] = p;
  workgroupBarrier();
  for (var k = 1u; k <= 8u; k++) {
    let half = 1u << (k - 1u);
    if ((li & ((half << 1u) - 1u)) == 0u) {
      let sw = join(redW[li], redW[li + half], redP[li], redP[li + half]);
      let sa = redA[li] + redA[li + half];
      redW[li] = sw;
      redA[li] = sa;
      let level = base + 1u + k;
      b = (wid * 256u + li) >> k;
      if (b < levelCount(level)) { setPyr(levelStart(level) + b, sw, sa); }
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn prep(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  var w = vec4<f32>(0.0);
  var a = 0.0;
  var first = vec3<f32>(0.0);
  for (var h = 0u; h < 2u; h++) {
    let l = wid.x * ${REDUCE}u + li * 2u + h;
    if (l < bh.n) {
      let node = pairs[l].y;
      let p = pos[node].xyz;
      let s = strength[node];
      leaves[l] = vec4<f32>(p, s);
      var depth = 0u; // a lone point is the root
      if (bh.n > 1u) { depth = depthOf(max(delta(l, i32(l) - 1), delta(l, i32(l) + 1))) + 1u; }
      let al = s * damp(i32(depth));
      let e = vec4<f32>(0.0, 0.0, 0.0, abs(al));
      setPyr(l, e, al);
      if (h == 0u) { first = p; w = e; } else { w = join(w, e, first, p); }
      a += al;
    }
  }
  reduceUp(0u, wid.x, li, w, a, first);
}

@compute @workgroup_size(256)
fn up(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let base = bh.base;
  let count = levelCount(base);
  let start = levelStart(base);
  var w = vec4<f32>(0.0);
  var a = 0.0;
  var first = vec3<f32>(0.0);
  for (var h = 0u; h < 2u; h++) {
    let e = wid.x * ${REDUCE}u + li * 2u + h;
    if (e < count) {
      let p = leaves[e << base].xyz;
      if (h == 0u) { first = p; w = pyrW(start + e); } else { w = join(w, pyrW(start + e), first, p); }
      a += pyrA(start + e);
    }
  }
  reduceUp(base, wid.x, li, w, a, first);
}
`;

// Kernels that put the points in the order of their codes (bind group layout S; longRuns:
// S2, the same with the run list read-only). The radix sort sorts on the high halves
// only, which leaves the points of a cell LEVELS / 2 levels down (runs sharing a high
// half) in node order. Sorting each run on the low halves then gives exactly the order of
// a sort on the whole codes. Runs are short: in one of up to RUN points, every point finds
// its place by comparing its low half with the others' (fix); a longer run is listed for
// longRuns, which sorts it with a workgroup.
var order = PRELUDE + /* wgsl */`
struct Misc { box: array<atomic<u32>, 8>, cube: vec4<f32>, shift: i32, pad0: i32, pad1: i32, pad2: i32 }
@group(1) @binding(1) var<storage, read_write> misc: Misc;
@group(1) @binding(2) var<storage, read_write> src: array<vec2<u32>>;    // keys: the sort's input; fix: its output; longRuns: scratch
@group(1) @binding(3) var<storage, read_write> dst: array<vec2<u32>>;    // (high half, node), in the final order
@group(1) @binding(4) var<storage, read_write> codes: array<vec2<u32>>;  // by node: high, low half
@group(1) @binding(5) var<storage, read_write> runs: array<atomic<u32>>; // longRuns' dispatch (x, y, z), then the long runs' starts from 4
@group(1) @binding(5) var<storage, read> runList: array<u32>;            // the same, for longRuns, which is dispatched from it
` + params + /* wgsl */`
const RUN = ${RUN}u;

fn spread3(v: u32) -> u32 { // 10 bits -> every third bit
  var x = v & 0x3ffu;
  x = (x | (x << 16u)) & 0x030000ffu;
  x = (x | (x << 8u)) & 0x0300f00fu;
  x = (x | (x << 4u)) & 0x030c30c3u;
  x = (x | (x << 2u)) & 0x09249249u;
  return x;
}
fn spread2(v: u32) -> u32 { // 16 bits -> every other bit
  var x = v & 0xffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}

// Morton code of LEVELS / 2 levels, most significant bits first, with d3's child order
// (z, y, x) at each level, in the high bits of a u32.
fn morton(q: vec3<u32>) -> u32 {
  var m = q.x;
  if (DIMS == 3u) { m = (spread3(q.z) << 2u) | (spread3(q.y) << 1u) | spread3(q.x); }
  if (DIMS == 2u) { m = (spread2(q.y) << 1u) | spread2(q.x); }
  return m << (32u - HALF);
}

// floor((x - origin) / cell) for a cell of 2^-k, without rounding: x - origin is not exact
// in f32, and a point put on the wrong side of a cell boundary lands at another depth,
// which changes its weight in every cell above it. origin is an integer, so either
// origin * 2^k is one (k >= 0) or floor(x) - origin is exact and 2^-k a whole number.
fn cellOf(x: f32, origin: f32, k: i32) -> f32 {
  if (k >= 0) { return floor(ldexp(x, k)) - ldexp(origin, k); }
  return floor(ldexp(floor(x) - origin, k));
}

@compute @workgroup_size(${WG})
fn keys(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i == 0u) { atomicStore(&runs[0], 0u); } // no long run listed yet this step
  if (i >= bh.n) { return; }
  let origin = misc.cube.xyz;
  let k = misc.shift;
  let p = pos[i].xyz;
  let top = f32((1u << LEVELS) - 1u);
  let q = vec3<u32>(clamp(vec3<f32>(cellOf(p.x, origin.x, k), cellOf(p.y, origin.y, k), cellOf(p.z, origin.z, k)), vec3<f32>(0.0), vec3<f32>(top)));
  let half = LEVELS / 2u;
  let code = vec2<u32>(morton(q >> vec3<u32>(half)), morton(q & vec3<u32>((1u << half) - 1u)));
  codes[i] = code;
  src[i] = vec2<u32>(code.x, i);
}

// For small graphs, a sort in one step: every code's rank is the number of codes before it
// (smaller, or equal with a smaller node index: the radix sort's order). RANK_SPLIT threads
// share each code's count.
override RANK_SPLIT: u32 = 8u;
var<workgroup> rankTile: array<vec2<u32>, 256>;
var<workgroup> rankPart: array<u32, 256>;

@compute @workgroup_size(256)
fn rank(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let lane = li % RANK_SPLIT;
  let i = wid.x * (256u / RANK_SPLIT) + li / RANK_SPLIT;
  let live = i < bh.n;
  var mine = vec2<u32>(0u);
  if (live) { mine = codes[i]; }
  var r = 0u;
  for (var base = 0u; base < bh.n; base += 256u) {
    if (base + li < bh.n) { rankTile[li] = codes[base + li]; }
    workgroupBarrier();
    let count = min(256u, bh.n - base);
    for (var k = lane; k < count; k += RANK_SPLIT) {
      let c = rankTile[k];
      if (c.x < mine.x || (c.x == mine.x && (c.y < mine.y || (c.y == mine.y && base + k < i)))) { r += 1u; }
    }
    workgroupBarrier();
  }
  rankPart[li] = r;
  workgroupBarrier();
  if (lane == 0u && live) {
    for (var s = 1u; s < RANK_SPLIT; s++) { r += rankPart[li + s]; }
    dst[r] = vec2<u32>(mine.x, i);
  }
}

@compute @workgroup_size(${WG})
fn fix(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= bh.n) { return; }
  let e = src[s];
  // this point's run of equal high halves, looked for no further than RUN points either way
  var first = s;
  loop {
    if (first == 0u || s - first > RUN || src[first - 1u].x != e.x) { break; }
    first -= 1u;
  }
  var last = s;
  loop {
    if (last + 1u >= bh.n || last - s > RUN || src[last + 1u].x != e.x) { break; }
    last += 1u;
  }
  if (first == last) { // alone in its cell: already in place
    dst[s] = e;
    return;
  }
  if (last - first + 1u > RUN) { // a long run: copied as it is, for longRuns
    dst[s] = e;
    if (first == s) {
      let k = atomicAdd(&runs[0], 1u);
      atomicStore(&runs[4u + k], s);
    }
    return;
  }
  // this point's place in its run: after the points with a smaller low half, and, the run
  // being in node order, after those with the same one that come before it
  let lo = codes[e.y].y;
  var place = 0u;
  for (var j = first; j <= last; j++) {
    let other = codes[src[j].y].y;
    if (other < lo || (other == lo && j < s)) { place += 1u; }
  }
  dst[first + place] = e;
}

var<workgroup> lrCount: array<atomic<u32>, 256>;
var<workgroup> lrSums: array<u32, 256>;
var<workgroup> lrMasks: array<atomic<u32>, 2048>; // digit * 8 + thread / 32
var<workgroup> lrStarts: array<u32, 256>;
var<workgroup> lrEnd: atomic<u32>;
var<workgroup> lrEndAt: u32;

// Inclusive prefix sum of lrSums (Hillis-Steele).
fn lrScan(li: u32) {
  for (var o = 1u; o < 256u; o <<= 1u) {
    var v = 0u;
    if (li >= o) { v = lrSums[li - o]; }
    workgroupBarrier();
    lrSums[li] += v;
    workgroupBarrier();
  }
}

fn lrRead(i: u32, even: bool) -> vec2<u32> {
  if (even) { return dst[i]; }
  return src[i];
}

fn lrWrite(i: u32, even: bool, e: vec2<u32>) {
  if (even) { src[i] = e; } else { dst[i] = e; }
}

// One long run per workgroup: a stable radix sort of dst[first, end) on the low halves,
// 8 bits per pass, through src and back.
@compute @workgroup_size(256)
fn longRuns(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let first = runList[4u + wid.x];
  let high = dst[first].x;
  if (li == 0u) { atomicStore(&lrEnd, bh.n); }
  workgroupBarrier();
  var end = bh.n;
  for (var base = first; base < bh.n; base += 256u) {
    let s = base + li;
    if (s < bh.n && dst[s].x != high) { atomicMin(&lrEnd, s); }
    workgroupBarrier();
    if (li == 0u) { lrEndAt = atomicLoad(&lrEnd); }
    end = workgroupUniformLoad(&lrEndAt);
    if (end < bh.n) { break; }
  }
  let count = end - first;
  let word = li >> 5u;
  let bit = 1u << (li & 31u);

  for (var sweep = 0u; sweep < 4u; sweep++) {
    let shift = 8u * sweep;
    let even = (sweep & 1u) == 0u;
    atomicStore(&lrCount[li], 0u);
    workgroupBarrier();
    for (var i = li; i < count; i += 256u) {
      let e = lrRead(first + i, even);
      atomicAdd(&lrCount[(codes[e.y].y >> shift) & 255u], 1u);
    }
    workgroupBarrier();
    lrSums[li] = atomicLoad(&lrCount[li]);
    workgroupBarrier();
    lrScan(li);
    lrStarts[li] = lrSums[li] - atomicLoad(&lrCount[li]);
    workgroupBarrier();

    for (var b = 0u; b < count; b += 256u) {
      for (var w = 0u; w < 8u; w++) { atomicStore(&lrMasks[li * 8u + w], 0u); }
      workgroupBarrier();
      let i = b + li;
      let valid = i < count;
      var e = vec2<u32>(0u);
      var d = 0u;
      if (valid) {
        e = lrRead(first + i, even);
        d = (codes[e.y].y >> shift) & 255u;
        atomicOr(&lrMasks[d * 8u + word], bit);
      }
      workgroupBarrier();
      if (valid) {
        var rank = countOneBits(atomicLoad(&lrMasks[d * 8u + word]) & (bit - 1u));
        for (var w = 0u; w < word; w++) { rank += countOneBits(atomicLoad(&lrMasks[d * 8u + w])); }
        lrWrite(first + lrStarts[d] + rank, even, e);
      }
      workgroupBarrier();
      var c = 0u;
      for (var w = 0u; w < 8u; w++) { c += countOneBits(atomicLoad(&lrMasks[li * 8u + w])); }
      lrStarts[li] += c;
      workgroupBarrier();
    }
    storageBarrier(); // this sweep's writes, for the next sweep's reads
  }
}
`;

// Kernels that build and walk the tree (bind group layout B). The tree lives in one buffer
// of vec4<u32>, in three parts (inner = n - 1 nodes):
//
//   cells  [0, inner)             centre of mass and strength, as f32 bits
//   links  [inner, 2 inner)       the binary tree: left, right, and the node's depth
//   kids   [2 inner, 4 inner)     two entries per node: its children in the d3 tree
//
// A child reference holds a node or sorted position (INDEX bits), LEAF for a point, and,
// in kids, a node's depth from bit 24. Unused kids are NONE. The walk reads 16 bytes per
// child it visits.
var walk = PRELUDE + /* wgsl */`
struct Misc { box: array<u32, 8>, cube: vec4<f32>, shift: i32, pad0: i32, pad1: i32, pad2: i32 }
@group(1) @binding(1) var<storage, read_write> misc: Misc;
@group(1) @binding(2) var<storage, read_write> pairs: array<vec2<u32>>;
@group(1) @binding(3) var<storage, read_write> leaves: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read_write> pyr: array<f32>;
@group(1) @binding(5) var<storage, read_write> tree: array<vec4<u32>>;
@group(1) @binding(6) var<storage, read_write> codes: array<vec2<u32>>;
` + params + lookups + /* wgsl */`
const INDEX = 0x7fffffu;
const LEAF = 0x800000u;
const NONE = 0xffffffffu;
const STACK = 128u;

struct Sum { w: vec4<f32>, a: f32 }

// Sum of the pyramid over sorted positions [first, last], about the first point (p0).
fn rangeSum(first: u32, last: u32, p0: vec3<f32>) -> Sum {
  var s = Sum(vec4<f32>(0.0), 0.0);
  var lo = first;
  var hi = last + 1u;
  var k = 0u;
  loop {
    if (lo >= hi) { break; }
    let start = levelStart(k);
    if ((lo & 1u) != 0u) {
      let w = pyrW(start + lo);
      s.w += vec4<f32>(w.xyz + w.w * (leaves[lo << k].xyz - p0), w.w);
      s.a += pyrA(start + lo);
      lo += 1u;
    }
    if ((hi & 1u) != 0u) {
      hi -= 1u;
      let w = pyrW(start + hi);
      s.w += vec4<f32>(w.xyz + w.w * (leaves[hi << k].xyz - p0), w.w);
      s.a += pyrA(start + hi);
    }
    lo >>= 1u;
    hi >>= 1u;
    k += 1u;
  }
  return s;
}

@compute @workgroup_size(${WG})
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = i32(bh.n);
  let i = i32(gid.x);
  if (i >= n - 1) { return; }
  let ui = u32(i);

  // direction and extent of the range of codes this node covers
  let before = delta(ui, i - 1);
  let after = delta(ui, i + 1);
  let d = select(-1, 1, after > before);
  let floorPrefix = min(before, after);
  var span = 2;
  while (span < (1 << 30) && delta(ui, i + span * d) > floorPrefix) { span *= 2; }
  var len = 0;
  for (var t = span >> 1; t >= 1; t >>= 1) {
    if (delta(ui, i + (len + t) * d) > floorPrefix) { len += t; }
  }
  let j = i + len * d;
  let prefix = delta(ui, j);

  // where its two halves split
  var s = 0;
  var stride = len;
  loop {
    stride = (stride + 1) >> 1;
    if (delta(ui, i + (s + stride) * d) > prefix) { s += stride; }
    if (stride <= 1) { break; }
  }
  let split = i + s * d + min(d, 0);
  let first = min(i, j);
  let last = max(i, j);
  var left = u32(split);
  if (first == split) { left |= LEAF; }
  var right = u32(split + 1);
  if (last == split + 1) { right |= LEAF; }

  let depth = depthOf(prefix);
  let p0 = leaves[first].xyz;
  let sum = rangeSum(u32(first), u32(last), p0);
  var com = p0;
  if (sum.w.w > 0.0) { com += sum.w.xyz / sum.w.w; }
  // d3 skips a cell whose strengths add up to exactly 0 (and all it holds); here the sum
  // carries f32 rounding, so a cancellation leaves a residue of the order of the terms
  var value = sum.a * damp(-i32(depth));
  if (abs(sum.a) <= sum.w.w * 0x1p-20f) { value = 0.0; }
  tree[ui] = bitcast<vec4<u32>>(vec4<f32>(com, value));
  tree[bh.inner + ui] = vec4<u32>(left, right, depth, 0u);
}

fn depthAt(node: u32) -> u32 {
  return tree[bh.inner + node].z;
}

@compute @workgroup_size(${WG})
fn collapse(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i + 1u >= bh.n) { return; }
  let link = tree[bh.inner + i];
  var list = array<u32, 8>(link.x, link.y, NONE, NONE, NONE, NONE, NONE, NONE);
  var count = 2u;
  for (var level = 1u; level < DIMS; level++) {
    var next = array<u32, 8>(NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE);
    var m = 0u;
    for (var c = 0u; c < count; c++) {
      let k = list[c];
      if ((k & LEAF) == 0u && depthAt(k) == link.z) { // the same cell: look through it
        let below = tree[bh.inner + k];
        next[m] = below.x;
        next[m + 1u] = below.y;
        m += 2u;
      } else {
        next[m] = k;
        m += 1u;
      }
    }
    list = next;
    count = m;
  }
  for (var c = 0u; c < count; c++) {
    if ((list[c] & LEAF) == 0u) { list[c] |= depthAt(list[c]) << 24u; }
  }
  tree[2u * bh.inner + 2u * i] = vec4<u32>(list[0], list[1], list[2], list[3]);
  tree[2u * bh.inner + 2u * i + 1u] = vec4<u32>(list[4], list[5], list[6], list[7]);
}

// The centres of mass of the nodes at depth bh.base, weighing each child as d3 does: by the
// absolute value of its strength (a point's own, or a cell's at the depth below this one).
// The children are deeper, so already done.
@compute @workgroup_size(${WG})
fn recentre(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i + 1u >= bh.n || depthAt(i) != bh.base) { return; }
  let at = 2u * bh.inner + 2u * i;
  let kids = array<vec4<u32>, 2>(tree[at], tree[at + 1u]);
  var p0 = vec3<f32>(0.0);
  var sum = vec4<f32>(0.0); // w * (p - p0), w
  for (var c = 0u; c < 8u; c++) {
    let k = kids[c >> 2u][c & 3u];
    if (k == NONE) { break; }
    var q: vec4<f32>;
    if ((k & LEAF) != 0u) {
      q = leaves[k & INDEX];
    } else {
      q = bitcast<vec4<f32>>(tree[k & INDEX]);
      q.w *= damp(i32(((k >> 24u) & 31u) - bh.base - 1u));
    }
    let w = abs(q.w);
    if (w == 0.0) { continue; }
    if (sum.w == 0.0) { p0 = q.xyz; }
    sum += vec4<f32>(w * (q.xyz - p0), w);
  }
  if (sum.w > 0.0) {
    let cell = bitcast<vec4<f32>>(tree[i]);
    tree[i] = bitcast<vec4<u32>>(vec4<f32>(p0 + sum.xyz / sum.w, cell.w));
  }
}

// One point's force from a leaf point: exactly as manyBody.js does for points it visits.
fn pointForce(me: vec3<f32>, mine: u32, j: u32) -> vec3<f32> {
  let q = leaves[j];
  var d = q.xyz - me;
  var l = dot(d, d);
  if (l >= bh.distanceMax2) { return vec3<f32>(0.0); }
  if (l == 0.0) {
    d = coincident(pairs[mine].y, pairs[j].y);
    l = dot(d, d);
  }
  if (l < bh.distanceMin2) { l = sqrt(bh.distanceMin2 * l); }
  return d * (q.w * sim.alpha / l);
}

struct Visit { accepted: bool, force: vec3<f32> }

// Whether point t can approximate node cur, and with what force. The node stands for the
// d3 cells from depth top down to depth, which all hold the same points: d3 approximates
// at the first (widest) one that is far enough, if any.
fn visit(cur: u32, top: u32, depth: u32, me: vec3<f32>, t: u32, size: f32) -> Visit {
  if (top > depth) { return Visit(false, vec3<f32>(0.0)); } // points sharing the smallest cell
  let cell = bitcast<vec4<f32>>(tree[cur]);
  if (cell.w == 0.0) { return Visit(true, vec3<f32>(0.0)); } // d3 skips cells with no strength
  var d = cell.xyz - me;
  var l = dot(d, d);
  let reach = bh.theta2 * l;
  var w = ldexp(size, -i32(top));
  var D = top;
  loop {
    if (D > depth || w * w < reach) { break; }
    w *= 0.5;
    D += 1u;
  }
  if (D > depth) { return Visit(false, vec3<f32>(0.0)); }
  if (l >= bh.distanceMax2) { return Visit(true, vec3<f32>(0.0)); }
  if (l == 0.0) {
    d = coincident(pairs[t].y, bh.n + cur);
    l = dot(d, d);
  }
  if (l < bh.distanceMin2) { l = sqrt(bh.distanceMin2 * l); }
  return Visit(true, d * (cell.w * damp(i32(depth - D)) * sim.alpha / l));
}

@compute @workgroup_size(${WG})
fn walk(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= bh.n || bh.n < 2u) { return; }
  let me = leaves[t].xyz;
  let size = misc.cube.w;
  var acc = vec3<f32>(0.0);

  // depth-first over the d3 tree. A stack entry is a node to open (INDEX bits) and its
  // depth (from bit 26); opening it handles all its children, and stacks those that must be
  // opened in turn (at most 2^DIMS - 1 more per level).
  var stack: array<u32, STACK>;
  var sp = 0u;
  let rootDepth = min(depthAt(0u), 31u);
  let root = visit(0u, 0u, rootDepth, me, t, size);
  acc += root.force;
  if (!root.accepted) {
    stack[0] = rootDepth << 26u;
    sp = 1u;
  }
  // a tree node is opened at most once: the bound only guards against a malformed tree
  // hanging the GPU (a hung GPU takes the whole device down)
  for (var guard = 0u; guard < bh.n; guard++) {
    if (sp == 0u) { break; }
    sp -= 1u;
    let e = stack[sp];
    let base = 2u * bh.inner + 2u * (e & INDEX);
    let top = (e >> 26u) + 1u;
    let kids = array<vec4<u32>, 2>(tree[base], tree[base + 1u]);
    for (var c = 0u; c < 8u; c++) {
      let k = kids[c >> 2u][c & 3u];
      if (k == NONE) { break; }
      if ((k & LEAF) != 0u) {
        if ((k & INDEX) != t) { acc += pointForce(me, t, k & INDEX); }
        continue;
      }
      let depth = (k >> 24u) & 31u;
      let v = visit(k & INDEX, top, depth, me, t, size);
      acc += v.force;
      if (!v.accepted && sp < STACK) {
        stack[sp] = (k & INDEX) | (depth << 26u);
        sp += 1u;
      }
    }
  }

  let b = pairs[t].y;
  let v = vel[b];
  vel[b] = vec4<f32>(v.xyz + acc, v.w);
}

// The same walk for small graphs, where one thread per point leaves the GPU idle and the
// walks are long chains of dependent reads: LANES threads share each point. Its nodes to
// open are on a stack in workgroup memory; every round, each lane takes one (from the top)
// and handles its children, then the nodes they open are stacked in lane order.
const LANES = 8u;
const BODIES = 8u; // 64 / LANES
const CAP = 256u;  // stack entries per point
var<workgroup> coStack: array<u32, 2048>; // BODIES * CAP
var<workgroup> coSize: array<u32, 8>;
var<workgroup> coPush: array<u32, 64>;
var<workgroup> coAcc: array<vec3<f32>, 64>;
var<workgroup> coLeft: u32;

@compute @workgroup_size(64)
fn walkShared(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let b = li / LANES;
  let lane = li % LANES;
  let t = wid.x * BODIES + b;
  let live = t < bh.n && bh.n > 1u;
  var me = vec3<f32>(0.0);
  if (live) { me = leaves[t].xyz; }
  let size = misc.cube.w;
  var acc = vec3<f32>(0.0);

  if (lane == 0u) {
    var s = 0u;
    if (live) {
      let rootDepth = min(depthAt(0u), 31u);
      let root = visit(0u, 0u, rootDepth, me, t, size);
      acc += root.force;
      if (!root.accepted) { coStack[b * CAP] = rootDepth << 26u; s = 1u; }
    }
    coSize[b] = s;
  }
  for (var guard = 0u; guard < bh.n; guard++) {
    workgroupBarrier();
    if (li == 0u) {
      var left = 0u;
      for (var k = 0u; k < BODIES; k++) { left += coSize[k]; }
      coLeft = left;
    }
    if (workgroupUniformLoad(&coLeft) == 0u) { break; }

    let size0 = coSize[b];
    var e = NONE;
    if (lane < size0) { e = coStack[b * CAP + size0 - 1u - lane]; }
    workgroupBarrier();

    var opened = array<u32, 8>(NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE);
    var m = 0u;
    if (e != NONE) {
      let base = 2u * bh.inner + 2u * (e & INDEX);
      let top = (e >> 26u) + 1u;
      let kids = array<vec4<u32>, 2>(tree[base], tree[base + 1u]);
      for (var c = 0u; c < 8u; c++) {
        let k = kids[c >> 2u][c & 3u];
        if (k == NONE) { break; }
        if ((k & LEAF) != 0u) {
          if ((k & INDEX) != t) { acc += pointForce(me, t, k & INDEX); }
          continue;
        }
        let depth = (k >> 24u) & 31u;
        let v = visit(k & INDEX, top, depth, me, t, size);
        acc += v.force;
        if (!v.accepted) {
          opened[m] = (k & INDEX) | (depth << 26u);
          m += 1u;
        }
      }
    }
    coPush[li] = m;
    workgroupBarrier();

    let kept = size0 - min(size0, LANES);
    var at = kept;
    for (var k = 0u; k < lane; k++) { at += coPush[b * LANES + k]; }
    for (var j = 0u; j < m; j++) {
      if (at + j < CAP) { coStack[b * CAP + at + j] = opened[j]; }
    }
    if (lane == LANES - 1u) { coSize[b] = min(at + m, CAP); }
  }

  coAcc[li] = acc;
  workgroupBarrier();
  if (lane == 0u && live) {
    for (var k = 1u; k < LANES; k++) { acc += coAcc[li + k]; }
    let node = pairs[t].y;
    let v = vel[node];
    vel[node] = vec4<f32>(v.xyz + acc, v.w);
  }
}
`;

var specA = ["uniform", "rw", "rw", "ro", "rw", "rw", "rw"],
    specS = ["uniform", "rw", "rw", "rw", "rw", "rw"],
    specS2 = ["uniform", "rw", "rw", "rw", "rw", "ro"],
    specB = ["uniform", "rw", "rw", "rw", "rw", "rw", "rw"];

// (specialised: only ever compiled with constants, see treeVariants)
export var pipelines = Object.assign({
  bhBbox: {code: build, entry: "bbox", spec: specA, specialised: true},
  bhCube: {code: build, entry: "cube", spec: specA, specialised: true},
  bhBboxCube: {code: build, entry: "bboxCube", spec: specA, specialised: true},
  bhPrep: {code: build, entry: "prep", spec: specA, specialised: true},
  bhUp: {code: build, entry: "up", spec: specA, specialised: true},
  bhKeys: {code: order, entry: "keys", spec: specS, specialised: true},
  bhRank: {code: order, entry: "rank", spec: specS, specialised: true},
  bhFix: {code: order, entry: "fix", spec: specS, specialised: true},
  bhLongRuns: {code: order, entry: "longRuns", spec: specS2, specialised: true},
  bhBuild: {code: walk, entry: "build", spec: specB, specialised: true},
  bhCollapse: {code: walk, entry: "collapse", spec: specB, specialised: true},
  bhRecentre: {code: walk, entry: "recentre", spec: specB, specialised: true},
  bhWalk: {code: walk, entry: "walk", spec: specB, specialised: true},
  bhWalkShared: {code: walk, entry: "walkShared", spec: specB, specialised: true}
}, sortPipelines);

function treeConstants(dims) {
  return {DIMS: dims, LEVELS: LEVELS[dims], LOGC: 0.5 * (2 - dims), HALF: LEVELS[dims] / 2 * dims};
}

// The rank sort's constants: threads per code for ~2^17 threads in all.
function rankConstants(n, dims) {
  return Object.assign({RANK_SPLIT: Math.min(32, Math.max(1, 1 << Math.round(Math.log2(Math.max(1, (1 << 17) / n)))))}, treeConstants(dims));
}

// The pipelines ([id, constants]) a tree over n nodes in dims dimensions uses.
export function treeVariants(n, dims) {
  var c = treeConstants(dims);
  return (n <= BOX_MAX ? ["bhBboxCube"] : ["bhBbox", "bhCube"]).concat(["bhKeys", "bhPrep", "bhUp", "bhBuild", "bhCollapse", "bhRecentre"]).map(function(id) {
    return [id, c];
  }).concat(n <= RANK_MAX ? [["bhRank", rankConstants(n, dims)]] : [["bhFix", c], ["bhLongRuns", c]]).concat([
    [n <= SHARED_MAX ? "bhWalkShared" : "bhWalk", c]
  ]);
}

export function createBarnesHut(engine) {
  var device = engine.device,
      n = engine.n,
      dims = engine.nDim,
      inner = Math.max(1, n - 1),
      constants = treeConstants(dims),
      ranking = rankConstants(n, dims),
      shifts = SHIFTS[dims],
      slot = Math.max(256, device.limits.minUniformBufferOffsetAlignment | 0),
      count = function(k) { return Math.ceil(n / Math.pow(2, k)); },
      ups = [], offsets = [], total = 0, from, k;

  // prep writes levels 0..9, each up the nine above the level it starts from
  for (from = 9; count(from) > 1; from += 9) ups.push(from);
  for (k = 0; k <= 9 * (ups.length + 1); ++k) offsets.push(total), total += count(k);

  var make = function(size, usage) { return device.createBuffer({size: Math.max(16, size), usage: usage}); },
      sort = createSort(engine, n, shifts), // on the high halves of the codes
      sorted = sort.pairs[1 - (shifts.length & 1)], // (high half, node) in the final order
      misc = make(64, STORAGE | COPY_DST),
      leaves = make(16 * n, STORAGE),
      pyr = make(20 * total, STORAGE),
      codes = make(8 * n, STORAGE),
      tree = make(64 * inner, STORAGE),
      runs = make(16 + 4 * (Math.floor(n / (RUN + 1)) + 1), STORAGE | INDIRECT | COPY_DST),
      // uniform slots: the main one, one per up, one per depth for recentre
      slots = 1 + ups.length + LEVELS[dims],
      uniforms = make(slot * slots, UNIFORM | COPY_DST),
      data = new ArrayBuffer(slot * slots),
      strengths = null,
      mixed = false,
      groupsA = null,
      groupB = null,
      groupsRecentre = null,
      groupKeys, groupSort, groupLongRuns,
      last = null;

  // an empty box for the first step; longRuns' dispatch is (x, 1, 1)
  device.queue.writeBuffer(misc, 0, new Uint32Array([-1, -1, -1, -1, 0, 0, 0, 0]));
  device.queue.writeBuffer(runs, 0, new Uint32Array([0, 1, 1, 0]));

  function pipeline(id) {
    return engine.pipeline(id, constants);
  }

  function groupBAt(s) {
    return device.createBindGroup({
      layout: pipeline("bhBuild").getBindGroupLayout(1), // the same layout B as every kernel here
      entries: [
        {binding: 0, resource: {buffer: uniforms, offset: s * slot, size: 160}},
        {binding: 1, resource: {buffer: misc}},
        {binding: 2, resource: {buffer: sorted}},
        {binding: 3, resource: {buffer: leaves}},
        {binding: 4, resource: {buffer: pyr}},
        {binding: 5, resource: {buffer: tree}},
        {binding: 6, resource: {buffer: codes}}
      ]
    });
  }

  // layout S (or S2, from bhLongRuns): from, into (distinct buffers: two writable bindings
  // of one buffer are invalid)
  function groupS(layoutOf, from, into) {
    return device.createBindGroup({
      layout: pipeline(layoutOf).getBindGroupLayout(1),
      entries: [
        {binding: 0, resource: {buffer: uniforms, offset: 0, size: 160}},
        {binding: 1, resource: {buffer: misc}},
        {binding: 2, resource: {buffer: from}},
        {binding: 3, resource: {buffer: into}},
        {binding: 4, resource: {buffer: codes}},
        {binding: 5, resource: {buffer: runs}}
      ]
    });
  }

  groupKeys = groupS("bhKeys", sort.pairs[0], sort.pairs[1]);
  groupSort = groupS("bhKeys", sort.result, sorted);
  if (n > RANK_MAX) groupLongRuns = groupS("bhLongRuns", sort.result, sorted);

  function writeParams(desc) {
    var u32 = new Uint32Array(data), f32 = new Float32Array(data);
    for (var s = 0; s < slots; ++s) {
      var w = s * slot / 4;
      u32[w] = n;
      u32[w + 1] = s > ups.length ? s - ups.length - 1 : s ? ups[s - 1] : 0;
      u32[w + 2] = inner;
      u32[w + 3] = total;
      f32[w + 4] = desc.theta2;
      f32[w + 5] = desc.distanceMin2;
      f32[w + 6] = Math.min(3e38, desc.distanceMax2);
      for (var k = 0; k < offsets.length; ++k) u32[w + 8 + k] = offsets[k];
    }
    device.queue.writeBuffer(uniforms, 0, data);
  }

  return {
    // strengthBuffer: the force's per-node strengths (indexed by node); mixed: whether they
    // have both signs
    update: function(desc, strengthBuffer, isMixed) {
      var key = [desc.theta2, desc.distanceMin2, desc.distanceMax2].join();
      if (key !== last) last = key, writeParams(desc);
      mixed = isMixed;
      if (strengthBuffer !== strengths) {
        strengths = strengthBuffer;
        groupsA = [];
        for (var s = 0; s <= ups.length; ++s) {
          groupsA.push(device.createBindGroup({
            layout: pipeline("bhPrep").getBindGroupLayout(1), // layout A
            entries: [
              {binding: 0, resource: {buffer: uniforms, offset: s * slot, size: 160}},
              {binding: 1, resource: {buffer: misc}},
              {binding: 2, resource: {buffer: sorted}},
              {binding: 3, resource: {buffer: strengths}},
              {binding: 4, resource: {buffer: leaves}},
              {binding: 5, resource: {buffer: pyr}},
              {binding: 6, resource: {buffer: codes}}
            ]
          }));
        }
        groupB = groupB || groupBAt(0);
      }
      if (mixed && !groupsRecentre) {
        groupsRecentre = [];
        for (var d = 0; d < LEVELS[dims]; ++d) groupsRecentre.push(groupBAt(1 + ups.length + d));
      }
    },

    encode: function(pass) {
      if (n < 2) return; // a lone node feels nothing
      pass.setBindGroup(1, groupsA[0]);
      if (n <= BOX_MAX) {
        pass.setPipeline(pipeline("bhBboxCube"));
        pass.dispatchWorkgroups(1);
      } else {
        pass.setPipeline(pipeline("bhBbox"));
        pass.dispatchWorkgroups(Math.min(256, ceilDiv(n, 256 * 8)));
        pass.setPipeline(pipeline("bhCube"));
        pass.dispatchWorkgroups(1);
      }
      pass.setBindGroup(1, groupKeys);
      pass.setPipeline(pipeline("bhKeys"));
      pass.dispatchWorkgroups(ceilDiv(n, WG));
      if (n <= RANK_MAX) {
        pass.setBindGroup(1, groupSort);
        pass.setPipeline(engine.pipeline("bhRank", ranking));
        pass.dispatchWorkgroups(ceilDiv(n * ranking.RANK_SPLIT, 256));
      } else {
        sort.encode(pass);
        pass.setBindGroup(1, groupSort);
        pass.setPipeline(pipeline("bhFix"));
        pass.dispatchWorkgroups(ceilDiv(n, WG));
        pass.setBindGroup(1, groupLongRuns);
        pass.setPipeline(pipeline("bhLongRuns"));
        pass.dispatchWorkgroupsIndirect(runs, 0); // usually none
      }
      pass.setBindGroup(1, groupsA[0]);
      pass.setPipeline(pipeline("bhPrep"));
      pass.dispatchWorkgroups(ceilDiv(n, REDUCE));
      for (var s = 0; s < ups.length; ++s) {
        pass.setBindGroup(1, groupsA[s + 1]);
        pass.setPipeline(pipeline("bhUp"));
        pass.dispatchWorkgroups(ceilDiv(count(ups[s]), REDUCE));
      }
      pass.setBindGroup(1, groupB);
      pass.setPipeline(pipeline("bhBuild"));
      pass.dispatchWorkgroups(ceilDiv(n - 1, WG));
      pass.setPipeline(pipeline("bhCollapse"));
      pass.dispatchWorkgroups(ceilDiv(n - 1, WG));
      if (mixed) {
        // deepest first; the smallest cells, where d3 would split further, keep their centres
        pass.setPipeline(pipeline("bhRecentre"));
        for (var d = LEVELS[dims] - 1; d >= 0; --d) {
          pass.setBindGroup(1, groupsRecentre[d]);
          pass.dispatchWorkgroups(ceilDiv(n - 1, WG));
        }
        pass.setBindGroup(1, groupB);
      }
      if (n <= SHARED_MAX) {
        pass.setPipeline(pipeline("bhWalkShared"));
        pass.dispatchWorkgroups(ceilDiv(n, 8));
      } else {
        pass.setPipeline(pipeline("bhWalk"));
        pass.dispatchWorkgroups(ceilDiv(n, WG));
      }
    },

    destroy: function() {
      sort.destroy();
      [misc, leaves, pyr, codes, tree, runs, uniforms].forEach(function(b) { b.destroy(); });
    }
  };
}
