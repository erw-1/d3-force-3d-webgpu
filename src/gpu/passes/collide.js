import {PRELUDE} from "../wgsl.js";
import {storage, floats, params, ceilDiv} from "./util.js";

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

var TILE = 128;

var code = PRELUDE + /* wgsl */`
struct P { strength: f32, pad0: f32, pad1: f32, pad2: f32 }
@group(1) @binding(0) var<uniform> p: P;
@group(1) @binding(1) var<storage, read> radius: array<f32>;

var<workgroup> tile: array<vec4<f32>, ${TILE}>;

@compute @workgroup_size(${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = sim.n;
  let i = gid.x;
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
    for (var k = 0u; k < count; k++) {
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

  if (live) {
    let v = vel[i];
    vel[i] = vec4<f32>(v.xyz + acc, v.w);
  }
}
`;

export default {
  type: "collide",
  pipelines: {collide: {code: code, entry: "main", spec: ["uniform", "ro"]}},

  create: function(engine) {
    var device = engine.device,
        uniforms = params(device, 16),
        radii = null,
        bindGroup = null,
        version = -1,
        iterations = 1;

    return {
      update: function(desc) {
        if (desc.version !== version) {
          version = desc.version;
          if (radii) radii.destroy();
          radii = storage(device, floats(desc.radii, engine.n));
          bindGroup = null;
        }
        iterations = desc.iterations;
        uniforms.set([desc.strength]);
        if (!bindGroup) bindGroup = device.createBindGroup({
          layout: engine.pipeline("collide").getBindGroupLayout(1),
          entries: [
            {binding: 0, resource: {buffer: uniforms.buffer}},
            {binding: 1, resource: {buffer: radii}}
          ]
        });
      },

      encode: function(pass) {
        for (var k = 0; k < iterations; ++k) {
          engine.snapshot(pass);
          pass.setPipeline(engine.pipeline("collide"));
          pass.setBindGroup(1, bindGroup);
          pass.dispatchWorkgroups(ceilDiv(engine.n, TILE));
        }
      },

      destroy: function() {
        uniforms.destroy();
        if (radii) radii.destroy();
      }
    };
  }
};
