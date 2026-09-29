import {PRELUDE} from "../wgsl.js";
import {WORKGROUP} from "../constants.js";
import {storage, floats, params, ceilDiv} from "./util.js";

// forceX / forceY / forceZ:   v[axis] += (goal[i] - p[axis]) * weight[i] * alpha

var code = PRELUDE + /* wgsl */`
struct P { axis: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(1) @binding(0) var<uniform> p: P;
@group(1) @binding(1) var<storage, read> goal: array<f32>;
@group(1) @binding(2) var<storage, read> weight: array<f32>;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sim.n) { return; }
  var v = vel[i];
  v[p.axis] += (goal[i] - pos[i][p.axis]) * weight[i] * sim.alpha;
  vel[i] = v;
}
`;

export default {
  type: "position",
  pipelines: {position: {code: code, entry: "main", spec: ["uniform", "ro", "ro"]}},

  create: function(engine) {
    var device = engine.device,
        uniforms = params(device, 16),
        buffers = [],
        bindGroup = null,
        version = -1,
        axis = 0;

    return {
      update: function(desc) {
        axis = desc.axis;
        uniforms.set([], {0: axis});
        if (desc.version !== version) {
          version = desc.version;
          buffers.forEach(function(b) { b.destroy(); });
          buffers = [storage(device, floats(desc.targets, engine.n)), storage(device, floats(desc.strengths, engine.n))];
          bindGroup = device.createBindGroup({
            layout: engine.pipeline("position").getBindGroupLayout(1),
            entries: [
              {binding: 0, resource: {buffer: uniforms.buffer}},
              {binding: 1, resource: {buffer: buffers[0]}},
              {binding: 2, resource: {buffer: buffers[1]}}
            ]
          });
        }
      },

      encode: function(pass) {
        if (axis >= engine.nDim) return; // forceZ on a 2D simulation has nothing to act on
        pass.setPipeline(engine.pipeline("position"));
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
