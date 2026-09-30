import {PRELUDE} from "../wgsl.js";
import {STORAGE} from "../constants.js";
import {params, ceilDiv} from "./util.js";

// Centering force: shift every node so that the mean position sits at (x, y, z).
// The mean is a two-level parallel reduction (workgroup partial sums, then one workgroup
// summing the partials into partial[0]), followed by a per-node shift. Up to 256 partial
// sums (65536 nodes), every workgroup of the shift sums them itself, the same way, which
// spares a dispatch.

var code = PRELUDE + /* wgsl */`
struct P { center: vec3<f32>, strength: f32 }
@group(1) @binding(0) var<uniform> p: P;
@group(1) @binding(1) var<storage, read_write> partial: array<vec4<f32>>;

var<workgroup> sh: array<vec4<f32>, 256>;

fn reduceShared(li: u32) {
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (li < s) { sh[li] = sh[li] + sh[li + s]; }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32,
          @builtin(workgroup_id) wid: vec3<u32>) {
  var v = vec4<f32>(0.0);
  if (gid.x < sim.n) { v = vec4<f32>(pos[gid.x].xyz, 0.0); }
  sh[li] = v;
  workgroupBarrier();
  reduceShared(li);
  if (li == 0u) { partial[wid.x] = sh[0]; }
}

@compute @workgroup_size(256)
fn combine(@builtin(local_invocation_index) li: u32) {
  let groups = (sim.n + 255u) / 256u;
  var v = vec4<f32>(0.0);
  for (var g = li; g < groups; g += 256u) { v += partial[g]; }
  sh[li] = v;
  workgroupBarrier();
  reduceShared(li);
  if (li == 0u) { partial[0] = sh[0] / f32(sim.n); }
}

fn shiftBy(i: u32, mean: vec3<f32>) {
  let shift = select(vec3<f32>(0.0), (mean - p.center) * p.strength, activeDims());
  let q = pos[i];
  pos[i] = vec4<f32>(q.xyz - shift, q.w);
}

@compute @workgroup_size(256)
fn apply(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sim.n) { return; }
  shiftBy(i, partial[0].xyz);
}

@compute @workgroup_size(256)
fn combineApply(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let groups = (sim.n + 255u) / 256u;
  var v = vec4<f32>(0.0);
  if (li < groups) { v += partial[li]; }
  sh[li] = v;
  workgroupBarrier();
  reduceShared(li);
  let mean = sh[0] / f32(sim.n);
  if (gid.x >= sim.n) { return; }
  shiftBy(gid.x, mean.xyz);
}
`;

var spec = ["uniform", "rw"];

export default {
  type: "center",
  pipelines: {
    centerReduce: {code: code, entry: "reduce", spec: spec},
    centerCombine: {code: code, entry: "combine", spec: spec},
    centerApply: {code: code, entry: "apply", spec: spec},
    centerCombineApply: {code: code, entry: "combineApply", spec: spec}
  },

  create: function(engine) {
    var device = engine.device,
        uniforms = params(device, 16),
        groups = ceilDiv(engine.n, 256),
        partial = device.createBuffer({size: groups * 16, usage: STORAGE}),
        bindGroup = device.createBindGroup({
          layout: engine.pipeline("centerReduce").getBindGroupLayout(1),
          entries: [
            {binding: 0, resource: {buffer: uniforms.buffer}},
            {binding: 1, resource: {buffer: partial}}
          ]
        });

    return {
      update: function(desc) {
        uniforms.set([desc.x, desc.y, desc.z, desc.strength]);
      },

      encode: function(pass) {
        pass.setBindGroup(1, bindGroup);
        pass.setPipeline(engine.pipeline("centerReduce"));
        pass.dispatchWorkgroups(groups);
        if (groups <= 256) {
          pass.setPipeline(engine.pipeline("centerCombineApply"));
        } else {
          pass.setPipeline(engine.pipeline("centerCombine"));
          pass.dispatchWorkgroups(1);
          pass.setPipeline(engine.pipeline("centerApply"));
        }
        pass.dispatchWorkgroups(groups);
      },

      destroy: function() {
        uniforms.destroy();
        partial.destroy();
      }
    };
  }
};
