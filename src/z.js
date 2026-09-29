import constant from "./constant.js";

export default function(z) {
  var strength = constant(0.1),
      nodes,
      strengths,
      zz,
      version = 0;

  if (typeof z !== "function") z = constant(z == null ? 0 : +z);

  function force(alpha) {
    for (var i = 0, n = nodes.length, node; i < n; ++i) {
      node = nodes[i], node.vz += (zz[i] - node.z) * strengths[i] * alpha;
    }
  }

  function initialize() {
    if (!nodes) return;
    var i, n = nodes.length;
    strengths = new Array(n);
    zz = new Array(n);
    for (i = 0; i < n; ++i) {
      strengths[i] = isNaN(zz[i] = +z(nodes[i], i, nodes)) ? 0 : +strength(nodes[i], i, nodes);
    }
    ++version;
  }

  force.initialize = function(_) {
    nodes = _;
    initialize();
  };

  force.strength = function(_) {
    return arguments.length ? (strength = typeof _ === "function" ? _ : constant(+_), initialize(), force) : strength;
  };

  force.z = function(_) {
    return arguments.length ? (z = typeof _ === "function" ? _ : constant(+_), initialize(), force) : z;
  };

  // Description of this force for the WebGPU simulation.
  force.gpu = function() {
    return {type: "position", axis: 2, version: version, targets: zz, strengths: strengths};
  };

  return force;
}
