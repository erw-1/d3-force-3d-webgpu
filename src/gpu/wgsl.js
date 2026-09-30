// WGSL shared by every kernel. Bind group 0 is identical for all of them (one layout
// object), so it stays bound while the pass switches pipelines; group 1 carries the
// per-force buffers.
//
//   pos   xyz = position                       (w unused)
//   vel   xyz = velocity                       (w unused)
//
// Unused dimensions (numDimensions < 3) stay exactly 0 in pos and vel. Kernels that read
// other nodes' predicted positions (link, collide) get the snapshot, pos + vel refreshed
// just before them, in group 1.

export var PRELUDE = /* wgsl */`
struct Sim {
  alpha: f32,
  decay: f32,   // velocity multiplier, i.e. 1 - velocityDecay
  n: u32,
  nDim: u32,
  step: u32,
  seed: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4<f32>>;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

// Same distribution as d3's jiggle: (random() - 0.5) * 1e-6. The value depends only on
// (a, b, axis, step, seed), so both ends of a pair compute the same number.
fn jiggle(a: u32, b: u32, axis: u32) -> f32 {
  let h = pcg(a ^ pcg(b ^ pcg(sim.seed ^ pcg(sim.step * 3u + axis))));
  return (f32(h) * (1.0 / 4294967296.0) - 0.5) * 1e-6;
}

fn activeDims() -> vec3<bool> {
  return vec3<bool>(true, sim.nDim > 1u, sim.nDim > 2u);
}

// Displacement between two coincident points, antisymmetric in (i, j).
fn coincident(i: u32, j: u32) -> vec3<f32> {
  let lo = min(i, j);
  let hi = max(i, j);
  var r = vec3<f32>(jiggle(lo, hi, 0u), jiggle(lo, hi, 1u), jiggle(lo, hi, 2u));
  if (i > j) { r = -r; }
  return select(vec3<f32>(0.0), r, activeDims());
}
`;

// pos + vel -> snap
export var SNAPSHOT = PRELUDE + /* wgsl */`
@group(1) @binding(0) var<storage, read_write> snap: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sim.n) { return; }
  snap[i] = vec4<f32>(pos[i].xyz + vel[i].xyz, 0.0);
}
`;

// Velocity decay, fixed positions, position update. Mirrors the loop at the end of
// simulation.tick():   x += vx *= velocityDecay   /   x = fx, vx = 0
export var INTEGRATE = PRELUDE + /* wgsl */`
@group(1) @binding(0) var<storage, read> fixPos: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> fixMask: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sim.n) { return; }
  let m = fixMask[i];
  let pinned = vec3<bool>((m & 1u) != 0u, (m & 2u) != 0u, (m & 4u) != 0u);
  let v = vel[i].xyz * sim.decay;
  let p = pos[i].xyz + v;
  pos[i] = vec4<f32>(select(p, fixPos[i].xyz, pinned), 0.0);
  vel[i] = vec4<f32>(select(v, vec3<f32>(0.0), pinned), 0.0);
}
`;
