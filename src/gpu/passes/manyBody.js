import {PRELUDE} from "../wgsl.js";
import {storage, floats, params, ceilDiv, splitFor} from "./util.js";
import {createBarnesHut, pipelines as treePipelines, treeVariants} from "./barnesHut.js";

// Many-body force. With theta > 0 (0.9 by default) it is the Barnes-Hut tree of
// barnesHut.js, which gives the same forces as the CPU. With theta ~ 0 it is this exact
// all-pairs kernel: each workgroup streams the whole node list through shared memory one
// tile at a time, so global memory traffic is n * n / TILE reads rather than n * n.
//
// SPLIT threads share each node: thread `lane` of a node takes every SPLIT-th source of
// each tile, and the partial sums are added at the end. That multiplies the threads in
// flight for small graphs, where one thread per node leaves most of a GPU idle.
//
// For every other node j (same arithmetic as manyBody.js, minus the tree):
//     if l < distanceMax2:  if l < distanceMin2: l = sqrt(distanceMin2 * l)
//                           v += (pj - pi) * strength[j] * alpha / l

var TILE = 128;

// Below this theta^2 the tree would open every cell anyway: sum all pairs instead.
var EXACT_THETA2 = 1e-6;

var code = PRELUDE + /* wgsl */`
struct P { distanceMin2: f32, distanceMax2: f32, pad0: f32, pad1: f32 }
@group(1) @binding(0) var<uniform> p: P;
@group(1) @binding(1) var<storage, read> strength: array<f32>;

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
  if (live) { me = pos[i].xyz; }
  var acc = vec3<f32>(0.0);

  for (var base = 0u; base < n; base += ${TILE}u) {
    let j = base + li;
    var t = vec4<f32>(0.0);
    if (j < n) { t = vec4<f32>(pos[j].xyz, strength[j]); }
    tile[li] = t;
    workgroupBarrier();

    let count = min(${TILE}u, n - base);
    for (var k = lane; k < count; k += SPLIT) {
      let q = tile[k];
      var d = q.xyz - me;
      var l = dot(d, d);
      if (l >= p.distanceMax2 || base + k == i) { continue; }
      if (l == 0.0) {
        d = coincident(i, base + k);
        l = dot(d, d);
      }
      if (l < p.distanceMin2) { l = sqrt(p.distanceMin2 * l); }
      acc += d * (q.w * sim.alpha / l);
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
  type: "manyBody",
  pipelines: Object.assign({manyBody: {code: code, entry: "main", spec: ["uniform", "ro"], specialised: true}}, treePipelines),

  // the kernels, with their constants, that n nodes in nDim dimensions use
  variants: function(n, nDim, split) {
    return [["manyBody", {SPLIT: splitFor(n, split)}]].concat(treeVariants(n, nDim));
  },

  create: function(engine) {
    var device = engine.device,
        split = {SPLIT: splitFor(engine.n, engine.forcedSplit)},
        uniforms = params(device, 16),
        strengths = null,
        bindGroup = null,
        version = -1,
        tree = null,
        mixed = false, // strengths of both signs (the tree then needs d3's centres of mass)
        exact = true;

    return {
      update: function(desc) {
        if (desc.version !== version) {
          version = desc.version;
          if (strengths) strengths.destroy();
          var values = floats(desc.strengths, engine.n), negative = false, positive = false;
          for (var i = 0; i < values.length; ++i) negative = negative || values[i] < 0, positive = positive || values[i] > 0;
          mixed = negative && positive;
          strengths = storage(device, values);
          bindGroup = null;
        }
        exact = !(desc.theta2 > EXACT_THETA2);
        if (!exact) return (tree || (tree = createBarnesHut(engine))).update(desc, strengths, mixed);
        uniforms.set([desc.distanceMin2, desc.distanceMax2]);
        if (!bindGroup) bindGroup = device.createBindGroup({
          layout: engine.pipeline("manyBody", split).getBindGroupLayout(1),
          entries: [
            {binding: 0, resource: {buffer: uniforms.buffer}},
            {binding: 1, resource: {buffer: strengths}}
          ]
        });
      },

      encode: function(pass) {
        if (!exact) return tree.encode(pass);
        pass.setPipeline(engine.pipeline("manyBody", split));
        pass.setBindGroup(1, bindGroup);
        pass.dispatchWorkgroups(ceilDiv(engine.n * split.SPLIT, TILE));
      },

      destroy: function() {
        uniforms.destroy();
        if (strengths) strengths.destroy();
        if (tree) tree.destroy();
      }
    };
  }
};
