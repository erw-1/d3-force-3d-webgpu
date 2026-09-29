import {PRELUDE} from "../wgsl.js";
import {WORKGROUP} from "../constants.js";
import {storage, floats, params, ceilDiv} from "./util.js";

// forceRadial: pull every node towards the sphere of `radius[i]` around (x, y, z).
// As in radial.js, a component that is exactly zero is replaced by 1e-6 so a node sitting
// on the centre still gets a direction.

var code = PRELUDE + /* wgsl */`
struct P { center: vec3<f32>, pad0: f32 }
@group(1) @binding(0) var<uniform> p: P;
@group(1) @binding(1) var<storage, read> radiuses: array<f32>;
@group(1) @binding(2) var<storage, read> weight: array<f32>;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sim.n) { return; }
  var d = pos[i].xyz - p.center;
  d = select(d, vec3<f32>(1e-6), d == vec3<f32>(0.0));
  d = select(vec3<f32>(0.0), d, activeDims());
  let r = length(d);
  let k = (radiuses[i] - r) * weight[i] * sim.alpha / r;
  let v = vel[i];
  vel[i] = vec4<f32>(v.xyz + d * k, v.w);
}
`;

export default {
  type: "radial",
  pipelines: {radial: {code: code, entry: "main", spec: ["uniform", "ro", "ro"]}},

  create: function(engine) {
    var device = engine.device,
        uniforms = params(device, 16),
        buffers = [],
        bindGroup = null,
        version = -1;

    return {
      update: function(desc) {
        uniforms.set([desc.x, desc.y, desc.z]);
        if (desc.version !== version) {
          version = desc.version;
          buffers.forEach(function(b) { b.destroy(); });
          buffers = [storage(device, floats(desc.radiuses, engine.n)), storage(device, floats(desc.strengths, engine.n))];
          bindGroup = device.createBindGroup({
            layout: engine.pipeline("radial").getBindGroupLayout(1),
            entries: [
              {binding: 0, resource: {buffer: uniforms.buffer}},
              {binding: 1, resource: {buffer: buffers[0]}},
              {binding: 2, resource: {buffer: buffers[1]}}
            ]
          });
        }
      },

      encode: function(pass) {
        pass.setPipeline(engine.pipeline("radial"));
        pass.setBindGroup(1, bindGroup);
        pass.dispatchWorkgroups(ceilDiv(engine.n, WORKGROUP));
      },

      destroy: function() {
        uniforms.destroy();
        buffers.forEach(function(b) { b.destroy(); });
      }
    };
  }
};
