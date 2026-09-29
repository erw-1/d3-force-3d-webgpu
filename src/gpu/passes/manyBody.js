import {PRELUDE} from "../wgsl.js";
import {storage, floats, params, ceilDiv} from "./util.js";

// Exact all-pairs many-body force. Each workgroup streams the whole node list through
// shared memory one tile at a time, so global memory traffic is n * n / TILE reads per
// tile rather than n * n. Because it is exact, `theta` has no effect here.
//
// For every other node j (same arithmetic as manyBody.js, minus the tree):
//     if l < distanceMax2:  if l < distanceMin2: l = sqrt(distanceMin2 * l)
//                           v += (pj - pi) * strength[j] * alpha / l

var TILE = 128;

var code = PRELUDE + /* wgsl */`
struct P { distanceMin2: f32, distanceMax2: f32, pad0: f32, pad1: f32 }
@group(1) @binding(0) var<uniform> p: P;
@group(1) @binding(1) var<storage, read> strength: array<f32>;

var<workgroup> tile: array<vec4<f32>, ${TILE}>;

@compute @workgroup_size(${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = sim.n;
  let i = gid.x;
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
    for (var k = 0u; k < count; k++) {
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

  if (live) {
    let v = vel[i];
    vel[i] = vec4<f32>(v.xyz + acc, v.w);
  }
}
`;

export default {
  type: "manyBody",
  pipelines: {manyBody: {code: code, entry: "main", spec: ["uniform", "ro"]}},

  create: function(engine) {
    var device = engine.device,
        uniforms = params(device, 16),
        strengths = null,
        bindGroup = null,
        version = -1;

    return {
      update: function(desc) {
        if (desc.version !== version) {
          version = desc.version;
          if (strengths) strengths.destroy();
          strengths = storage(device, floats(desc.strengths, engine.n));
          bindGroup = null;
        }
        uniforms.set([desc.distanceMin2, desc.distanceMax2]);
        if (!bindGroup) bindGroup = device.createBindGroup({
          layout: engine.pipeline("manyBody").getBindGroupLayout(1),
          entries: [
            {binding: 0, resource: {buffer: uniforms.buffer}},
            {binding: 1, resource: {buffer: strengths}}
          ]
        });
      },

      encode: function(pass) {
        pass.setPipeline(engine.pipeline("manyBody"));
        pass.setBindGroup(1, bindGroup);
        pass.dispatchWorkgroups(ceilDiv(engine.n, TILE));
      },

      destroy: function() {
        uniforms.destroy();
        if (strengths) strengths.destroy();
      }
    };
  }
};
