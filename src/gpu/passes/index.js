import center from "./center.js";
import collide from "./collide.js";
import link from "./link.js";
import manyBody from "./manyBody.js";
import position from "./position.js";
import radial from "./radial.js";

// A force takes part in a GPU simulation by exposing force.gpu(), which returns a
// descriptor {type, version, ...}. `type` selects the pass below; `version` changes
// whenever the force recomputes its per-node values, telling the pass to re-upload them.
var passes = {};
[center, collide, link, manyBody, position, radial].forEach(function(pass) {
  passes[pass.type] = pass;
});

export default passes;
