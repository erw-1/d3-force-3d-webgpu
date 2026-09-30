import {PRELUDE} from "../wgsl.js";
import {STORAGE, UNIFORM, COPY_DST} from "../constants.js";
import {createSort, pipelines as sortPipelines} from "../sort.js";
import {ceilDiv} from "./util.js";

// Collision force for large graphs: the same pairs as the all-pairs kernel of collide.js,
// found with a uniform grid. Cells are twice the largest radius wide, so two nodes that
// overlap are in the same or neighbouring cells (3^dims of them). Every step:
//
//   keys    each node's cell, hashed into a table of (a power of two >= 2n) slots
//   sort    radix sort of (hash, node) pairs: a cell's nodes become a run
//   starts  where each run starts, and the sorted nodes' (position, radius)
//   query   every node looks through the runs of its neighbouring cells
//
// A hash shared by two cells only costs distance tests, except that the two runs are then
// one: a node skips a neighbouring cell whose hash it has already looked through. The
// start of a run is not cleared between steps; it is checked against the sorted hashes.

var WG = 64;

var code = PRELUDE + /* wgsl */`
struct G {
  n: u32,
  mask: u32,     // table slots - 1
  pad0: u32,
  pad1: u32,
  strength: f32,
  inv: f32,      // 1 / cell width
  pad2: f32,
  pad3: f32,
}
@group(1) @binding(0) var<uniform> g: G;
@group(1) @binding(1) var<storage, read> radius: array<f32>;
@group(1) @binding(2) var<storage, read> snap: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> pairs: array<vec2<u32>>;
@group(1) @binding(4) var<storage, read_write> starts: array<u32>;
@group(1) @binding(5) var<storage, read_write> sorted: array<vec4<f32>>;

fn cellOf(p: vec3<f32>) -> vec3<i32> {
  return select(vec3<i32>(0), vec3<i32>(floor(p * g.inv)), activeDims());
}

fn hash(c: vec3<i32>) -> u32 {
  return ((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u)) & g.mask;
}

@compute @workgroup_size(${WG})
fn keys(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= g.n) { return; }
  pairs[i] = vec2<u32>(hash(cellOf(snap[i].xyz)), i);
}

@compute @workgroup_size(${WG})
fn begin(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= g.n) { return; }
  let e = pairs[s];
  if (s == 0u || pairs[s - 1u].x != e.x) { starts[e.x] = s; }
  sorted[s] = vec4<f32>(snap[e.y].xyz, radius[e.y]);
}

@compute @workgroup_size(${WG})
fn query(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= g.n) { return; }
  let me = sorted[s];
  let i = pairs[s].y;
  let ri2 = me.w * me.w;
  let home = cellOf(me.xyz);
  let reach = vec3<i32>(select(vec3<i32>(0), vec3<i32>(1), activeDims()));
  var seen: array<u32, 27>;
  var count = 0u;
  var acc = vec3<f32>(0.0);

  for (var dz = -reach.z; dz <= reach.z; dz++) {
    for (var dy = -reach.y; dy <= reach.y; dy++) {
      for (var dx = -reach.x; dx <= reach.x; dx++) {
        let key = hash(home + vec3<i32>(dx, dy, dz));
        var again = false;
        for (var k = 0u; k < count; k++) { if (seen[k] == key) { again = true; } }
        if (again) { continue; }
        seen[count] = key;
        count += 1u;

        let start = starts[key];
        if (start >= g.n || pairs[start].x != key || (start > 0u && pairs[start - 1u].x == key)) { continue; }
        for (var t = start; t < g.n && pairs[t].x == key; t++) {
          if (t == s) { continue; }
          let q = sorted[t];
          let rj = q.w;
          let r = me.w + rj;
          var d = me.xyz - q.xyz;
          var l = dot(d, d);
          if (l >= r * r) { continue; }
          if (l == 0.0) {
            d = coincident(i, pairs[t].y);
            l = dot(d, d);
          }
          let len = sqrt(l);
          let rj2 = rj * rj;
          acc += d * ((r - len) / len * g.strength * rj2 / (ri2 + rj2));
        }
      }
    }
  }

  let v = vel[i];
  vel[i] = vec4<f32>(v.xyz + acc, v.w);
}
`;

var spec = ["uniform", "ro", "ro", "rw", "rw", "rw"];

export var pipelines = Object.assign({
  gridKeys: {code: code, entry: "keys", spec: spec},
  gridBegin: {code: code, entry: "begin", spec: spec},
  gridQuery: {code: code, entry: "query", spec: spec}
}, sortPipelines);

export function createGrid(engine) {
  var device = engine.device,
      n = engine.n,
      bits = Math.max(8, Math.ceil(Math.log2(2 * n))),
      shifts = [],
      make = function(size, usage) { return device.createBuffer({size: Math.max(16, size), usage: usage}); },
      uniforms = make(32, UNIFORM | COPY_DST),
      data = new ArrayBuffer(32),
      sort,
      starts = make(4 << bits, STORAGE),
      sorted = make(16 * n, STORAGE),
      groups = null,
      radii = null,
      last = null;

  for (var b = 0; b < bits; b += 8) shifts.push(b);
  sort = createSort(engine, n, shifts);

  return {
    // radiusBuffer: per-node radii; rmax: the largest
    update: function(desc, radiusBuffer, rmax) {
      var key = [desc.strength, rmax].join();
      if (key !== last) {
        last = key;
        var u32 = new Uint32Array(data), f32 = new Float32Array(data);
        u32[0] = n, u32[1] = (1 << bits) - 1 >>> 0;
        f32[4] = desc.strength, f32[5] = rmax > 0 ? 1 / (2 * rmax) : 0;
        device.queue.writeBuffer(uniforms, 0, data);
      }
      if (radiusBuffer !== radii) {
        radii = radiusBuffer;
        groups = [sort.pairs[0], sort.result].map(function(pairs) {
          return device.createBindGroup({
            layout: engine.pipeline("gridKeys").getBindGroupLayout(1),
            entries: [
              {binding: 0, resource: {buffer: uniforms}},
              {binding: 1, resource: {buffer: radii}},
              {binding: 2, resource: {buffer: engine.snap}},
              {binding: 3, resource: {buffer: pairs}},
              {binding: 4, resource: {buffer: starts}},
              {binding: 5, resource: {buffer: sorted}}
            ]
          });
        });
      }
    },

    // one collision round, after engine.snapshot()
    encode: function(pass) {
      pass.setBindGroup(1, groups[0]);
      pass.setPipeline(engine.pipeline("gridKeys"));
      pass.dispatchWorkgroups(ceilDiv(n, WG));
      sort.encode(pass);
      pass.setBindGroup(1, groups[1]);
      pass.setPipeline(engine.pipeline("gridBegin"));
      pass.dispatchWorkgroups(ceilDiv(n, WG));
      pass.setPipeline(engine.pipeline("gridQuery"));
      pass.dispatchWorkgroups(ceilDiv(n, WG));
    },

    destroy: function() {
      sort.destroy();
      [uniforms, starts, sorted].forEach(function(b) { b.destroy(); });
    }
  };
}
