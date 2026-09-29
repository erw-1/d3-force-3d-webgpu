import {PRELUDE} from "../wgsl.js";
import {WORKGROUP} from "../constants.js";
import {storage, ceilDiv} from "./util.js";

// Link force. link.js walks the links in order and adds to both endpoints, which cannot
// be parallelised over links without write conflicts. Instead every node gathers the
// contribution of each link it belongs to, using a CSR adjacency list built on the CPU
// whenever the links change:
//
//     d = (target + v) - (source + v)      (from the snapshot)
//     d *= (|d| - distance) / |d| * alpha * strength
//     target -= d * bias        source += d * (1 - bias)
//
// The CPU force applies links sequentially, so later links already see the velocities
// changed by earlier ones; here they all see the same snapshot. `iterations` refines it.

var code = PRELUDE + /* wgsl */`
struct Link { src: u32, dst: u32, distance: f32, strength: f32, bias: f32 }

@group(1) @binding(0) var<storage, read> links: array<Link>;
@group(1) @binding(1) var<storage, read> adjacency: array<u32>;   // link * 2 + (node is target)
@group(1) @binding(2) var<storage, read> rowStart: array<u32>;    // n + 1 offsets into adjacency

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sim.n) { return; }
  var acc = vec3<f32>(0.0);

  for (var e = rowStart[i]; e < rowStart[i + 1u]; e++) {
    let code = adjacency[e];
    let id = code >> 1u;
    let link = links[id];
    var d = snap[link.dst].xyz - snap[link.src].xyz;
    if (d.x == 0.0) { d.x = jiggle(id, 0u, 0u); }
    if (sim.nDim > 1u && d.y == 0.0) { d.y = jiggle(id, 0u, 1u); }
    if (sim.nDim > 2u && d.z == 0.0) { d.z = jiggle(id, 0u, 2u); }
    let l = length(d);
    d *= (l - link.distance) / l * sim.alpha * link.strength;
    if ((code & 1u) == 1u) { acc -= d * link.bias; } else { acc += d * (1.0 - link.bias); }
  }

  let v = vel[i];
  vel[i] = vec4<f32>(v.xyz + acc, v.w);
}
`;

export default {
  type: "link",
  pipelines: {link: {code: code, entry: "main", spec: ["ro", "ro", "ro"]}},

  create: function(engine) {
    var device = engine.device,
        buffers = [],
        bindGroup = null,
        version = -1,
        iterations = 1;

    function release() {
      buffers.forEach(function(b) { b.destroy(); });
      buffers = [];
    }

    function build(desc) {
      var n = engine.n,
          links = desc.links,
          m = links.length,
          packed = new ArrayBuffer(Math.max(1, m) * 20),
          u32 = new Uint32Array(packed),
          f32 = new Float32Array(packed),
          rowStart = new Uint32Array(n + 1),
          adjacency = new Uint32Array(Math.max(1, 2 * m)),
          i, k, s, t;

      for (k = 0; k < m; ++k) {
        s = links[k].source.index, t = links[k].target.index;
        u32[5 * k] = s;
        u32[5 * k + 1] = t;
        f32[5 * k + 2] = +desc.distances[k];
        f32[5 * k + 3] = +desc.strengths[k];
        f32[5 * k + 4] = +desc.bias[k];
        ++rowStart[s + 1], ++rowStart[t + 1];
      }
      for (i = 0; i < n; ++i) rowStart[i + 1] += rowStart[i];

      var cursor = rowStart.slice(0, n);
      for (k = 0; k < m; ++k) {
        adjacency[cursor[links[k].source.index]++] = 2 * k;
        adjacency[cursor[links[k].target.index]++] = 2 * k + 1;
      }

      release();
      buffers = [storage(device, packed), storage(device, adjacency), storage(device, rowStart)];
      bindGroup = device.createBindGroup({
        layout: engine.pipeline("link").getBindGroupLayout(1),
        entries: buffers.map(function(buffer, binding) {
          return {binding: binding, resource: {buffer: buffer}};
        })
      });
    }

    return {
      update: function(desc) {
        if (desc.version !== version) version = desc.version, build(desc);
        iterations = desc.links.length ? desc.iterations : 0;
      },

      encode: function(pass) {
        for (var k = 0; k < iterations; ++k) {
          engine.snapshot(pass);
          pass.setPipeline(engine.pipeline("link"));
          pass.setBindGroup(1, bindGroup);
          pass.dispatchWorkgroups(ceilDiv(engine.n, WORKGROUP));
        }
      },

      destroy: release
    };
  }
};
