import {PRELUDE} from "../wgsl.js";
import {storage, floats, params, ceilDiv, splitFor} from "./util.js";
import {createGrid, pipelines as gridPipelines} from "./collideGrid.js";

// Collision force. collide.js visits every pair once and pushes both nodes apart; here
// every node computes its own share of every overlapping pair instead, which gives the
// same numbers without any node writing to another node's velocity:
//
//     d = pi - pj   (positions are pos + vel, from the snapshot)
//     if |d| < ri + rj:  vi += d * (ri + rj - |d|) / |d| * strength * rj^2 / (ri^2 + rj^2)
//
// The CPU force resolves pairs one after another, so a node already pushed by an earlier
// pair is seen at its new position by later ones; here all pairs are resolved against the
// same snapshot. Use `iterations` to tighten dense packings, as with the CPU force.
//
// As in manyBody.js, SPLIT threads share each node's loop over the sources. Large graphs
// find the same pairs with a grid instead (collideGrid.js).

var TILE = 128;

// From this many nodes, the grid's few dispatches beat testing every pair.
var GRID_MIN = 8192;

var code = PRELUDE + /* wgsl */`
struct P { strength: f32, pad0: f32, pad1: f32, pad2: f32 }
@group(1) @binding(0) var<uniform> p: P;
@group(1) @binding(1) var<storage, read> radius: array<f32>;
@group(1) @binding(2) var<storage, read> snap: array<vec4<f32>>;

override SPLIT: u32 = 1u;
var<workgroup> tile: array<vec4<f32>, ${TILE}>;
var<workgroup> partial: array<vec4<f32>, ${TILE}>;

@compute @workgroup_size(${TILE})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = sim.n;
  let lane = li % SPLIT;
  let i = wid.x * (${TILE}u / SPLIT) + li / SPLIT;
  let live = i < n;
  var me = vec3<f32>(0.0);
  var ri = 0.0;
  if (live) { me = snap[i].xyz; ri = radius[i]; }
  let ri2 = ri * ri;
  var acc = vec3<f32>(0.0);

  for (var base = 0u; base < n; base += ${TILE}u) {
    let j = base + li;
    var t = vec4<f32>(0.0);
    if (j < n) { t = vec4<f32>(snap[j].xyz, radius[j]); }
    tile[li] = t;
    workgroupBarrier();

    let count = min(${TILE}u, n - base);
    for (var k = lane; k < count; k += SPLIT) {
      let q = tile[k];
      let rj = q.w;
      let r = ri + rj;
      var d = me - q.xyz;
      var l = dot(d, d);
      if (l >= r * r || base + k == i) { continue; }
      if (l == 0.0) {
        d = coincident(i, base + k);
        l = dot(d, d);
      }
      let s = sqrt(l);
      let rj2 = rj * rj;
      acc += d * ((r - s) / s * p.strength * rj2 / (ri2 + rj2));
    }
    workgroupBarrier();
  }

  partial[li] = vec4<f32>(acc, 0.0);
  workgroupBarrier();
  if (lane == 0u && live) {
    var total = vec3<f32>(0.0);
    for (var s = 0u; s < SPLIT; s++) { total += partial[li + s].xyz; }
    let v = vel[i];
    vel[i] = vec4<f32>(v.xyz + total, v.w);
  }
}
`;

export default {
  type: "collide",
  pipelines: Object.assign({collide: {code: code, entry: "main", spec: ["uniform", "ro", "ro"], specialised: true}}, gridPipelines),

  variants: function(n, nDim, split) {
    return [["collide", {SPLIT: splitFor(n, split)}]];
  },

  create: function(engine) {
    var device = engine.device,
        split = {SPLIT: splitFor(engine.n, engine.forcedSplit)},
        uniforms = params(device, 16),
        radii = null,
        bindGroup = null,
        version = -1,
        iterations = 1,
        rmax = 0,
        grid = engine.n >= GRID_MIN ? createGrid(engine) : null;

    return {
      update: function(desc) {
        if (desc.version !== version) {
          version = desc.version;
          if (radii) radii.destroy();
          var values = floats(desc.radii, engine.n);
          radii = storage(device, values);
          rmax = 0;
          for (var i = 0; i < values.length; ++i) if (values[i] > rmax) rmax = values[i];
          bindGroup = null;
        }
        iterations = rmax > 0 ? desc.iterations : 0; // nothing can collide
        if (grid) return grid.update(desc, radii, rmax);
        uniforms.set([desc.strength]);
        if (!bindGroup) bindGroup = device.createBindGroup({
          layout: engine.pipeline("collide", split).getBindGroupLayout(1),
          entries: [
            {binding: 0, resource: {buffer: uniforms.buffer}},
            {binding: 1, resource: {buffer: radii}},
            {binding: 2, resource: {buffer: engine.snap}}
          ]
        });
      },

      encode: function(pass) {
        for (var k = 0; k < iterations; ++k) {
          engine.snapshot(pass);
          if (grid) { grid.encode(pass); continue; }
          pass.setPipeline(engine.pipeline("collide", split));
          pass.setBindGroup(1, bindGroup);
          pass.dispatchWorkgroups(ceilDiv(engine.n * split.SPLIT, TILE));
        }
      },

      destroy: function() {
        uniforms.destroy();
        if (radii) radii.destroy();
        if (grid) grid.destroy();
      }
    };
  }
};
