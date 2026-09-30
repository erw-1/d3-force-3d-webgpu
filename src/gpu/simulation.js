import {dispatch} from "d3-dispatch";
import {timer} from "d3-timer";
import lcg from "../lcg.js";
import {isWebGPUAvailable, requestDevice} from "./device.js";
import createEngine, {prepare, unsupported} from "./engine.js";

// forceSimulation() with the physics running on the GPU (WebGPU compute shaders).
//
// The API is that of forceSimulation(nodes, numDimensions), plus a few additions (see
// the end of the returned object). The differences to keep in mind:
//
//   * Node positions are copied back from the GPU asynchronously, so after tick() the
//     nodes are up to date one readback later (about a frame); "tick" events fire once
//     the copy has landed. To step and then read synchronously, use `await tickAsync(n)`.
//   * Forces are described by force.gpu(); every force in this package has one. If the
//     simulation contains any other force, or WebGPU is unavailable, it runs on the CPU
//     exactly like forceSimulation().

var MAX_DIMENSIONS = 3;

var initialRadius = 10,
    initialAngleRoll = Math.PI * (3 - Math.sqrt(5)), // Golden ratio angle
    initialAngleYaw = Math.PI * 20 / (9 + Math.sqrt(221)); // Markov irrational number

export default function(nodes, numDimensions, options) {
  numDimensions = numDimensions || 2;
  options = options || {};

  // options.device    a GPUDevice to run on (default: requested from navigator.gpu)
  // options.gpu       a GPU object to request the device from (e.g. from Node's `webgpu` package)
  // options.split    threads per node in the GPU's all-pairs kernels (a power of two; by
  //                   default chosen from the number of nodes, more for small graphs)
  // options.readback  false: do not copy positions back into the nodes on every tick, for
  //                   renderers that draw straight from gpuBuffers(). Nodes then only
  //                   update when sync() or tickAsync() is called.
  var readback = options.readback !== false;

  var nDim = Math.min(MAX_DIMENSIONS, Math.max(1, Math.round(numDimensions))),
      simulation,
      alpha = 1,
      alphaMin = 0.001,
      alphaDecay = 1 - Math.pow(alphaMin, 1 / 300),
      alphaTarget = 0,
      velocityDecay = 0.6,
      forces = new Map(),
      stepper = timer(step),
      event = dispatch("tick", "end"),
      random = lcg();

  var engine = null,        // GPU state, once the device is ready
      ready,                // Promise<boolean>: settles when GPU setup has finished
      initializing = false, // waiting for the device; the timer idles meanwhile
      usable = false,       // engine exists and every force can run on it
      oversized = false,    // too many nodes for this device
      running = true,       // the timer is active (cleared by stop() and by "end")
      frame = false,        // a timer-driven GPU step + readback is under way
      flushing = null,      // readback loop started by tick()
      batch = false,        // tick() has already been called in this synchronous run
      handover = null,      // GPU -> CPU state transfer in progress
      destroyed = false,
      warned = null;

  if (nodes == null) nodes = [];

  // --- CPU path (identical to forceSimulation) -----------------------------------------

  function cpuTick(iterations) {
    var i, n = nodes.length, node;

    if (handover) return simulation;
    if (iterations === undefined) iterations = 1;

    for (var k = 0; k < iterations; ++k) {
      alpha += (alphaTarget - alpha) * alphaDecay;

      forces.forEach(function (force) {
        force(alpha);
      });

      for (i = 0; i < n; ++i) {
        node = nodes[i];
        if (node.fx == null) node.x += node.vx *= velocityDecay;
        else node.x = node.fx, node.vx = 0;
        if (nDim > 1) {
          if (node.fy == null) node.y += node.vy *= velocityDecay;
          else node.y = node.fy, node.vy = 0;
        }
        if (nDim > 2) {
          if (node.fz == null) node.z += node.vz *= velocityDecay;
          else node.z = node.fz, node.vz = 0;
        }
      }
    }

    return simulation;
  }

  // --- GPU path ------------------------------------------------------------------------

  function gpuFail(error) {
    if (engine) engine.destroy();
    engine = null, usable = false, frame = false;
    if (!destroyed) console.warn("WebGPU simulation failed; continuing on the CPU.", error);
  }

  // (Re)build the GPU pass list from the current forces, or hand over to the CPU if a
  // force cannot run on the GPU.
  function refresh() {
    if (!engine || oversized) return;
    var name = unsupported(forces);

    if (name === null) {
      warned = null;
      engine.setForces(Array.from(forces.values()));
      if (!usable) usable = true, engine.markStale(); // the CPU may have moved the nodes
    } else {
      if (warned !== name) {
        warned = name;
        console.warn("Force \"" + name + "\" cannot run on the GPU (it has no force.gpu()): simulating on the CPU.");
      }
      if (usable) {
        usable = false;
        handover = engine.sync().then(function() { handover = null; }, function(error) { handover = null, gpuFail(error); });
      }
    }
  }

  // Submit `iterations` steps to the GPU (without waiting for them).
  function advance(iterations) {
    var alphas = new Array(iterations);
    for (var k = 0; k < iterations; ++k) {
      alpha += (alphaTarget - alpha) * alphaDecay;
      alphas[k] = alpha;
    }
    engine.enqueue(alphas, velocityDecay);
  }

  function gpuStep() {
    frame = true;
    advance(1);
    (readback ? engine.sync() : engine.idle()).then(function() {
      frame = false;
      if (!running) return;
      event.call("tick", simulation);
      if (alpha < alphaMin) {
        running = false;
        stepper.stop();
        event.call("end", simulation);
      }
    }, gpuFail);
  }

  function init() {
    var gpu = options.gpu || null;
    if (!options.device && !gpu && !isWebGPUAvailable()) return Promise.resolve(false);

    initializing = true;
    return (options.device ? Promise.resolve(options.device) : requestDevice(gpu)).then(function(device) {
      // the forces set so far (the usual chained .force() calls have run by now)
      var types = [];
      forces.forEach(function(force) { if (typeof force.gpu === "function") types.push((force.gpu() || {}).type); });
      return prepare(device, nodes.length, nDim, options.split, types).then(function() { return device; });
    }).then(function(device) {
      if (destroyed) return false;
      engine = createEngine(device, gpuFail, {split: options.split});
      engine.setSeed(random);
      reset();
      return usable;
    }).catch(function(error) {
      console.warn("WebGPU unavailable; simulating on the CPU.", error);
      return false;
    }).then(function(ok) {
      initializing = false;
      return ok;
    });
  }

  // --- simulation ----------------------------------------------------------------------

  function step() {
    if (initializing || handover) return;
    if (usable) return frame || gpuStep();

    cpuTick();
    event.call("tick", simulation);
    if (alpha < alphaMin) {
      running = false;
      stepper.stop();
      event.call("end", simulation);
    }
  }

  // Keep the JS nodes following the GPU: one readback loop at a time.
  function flush() {
    if (flushing) return;
    flushing = engine.sync().then(function() { flushing = null; }, function(error) { flushing = null, gpuFail(error); });
  }

  function tick(iterations) {
    if (iterations === undefined) iterations = 1;
    if (!usable) return cpuTick(iterations);

    // A tick() per frame must not pile work up when the GPU cannot keep pace, so it is
    // skipped while the GPU is behind (as the CPU simulation would simply run slower).
    // Further calls within the same synchronous run (e.g. a warm-up loop) are a batch,
    // not a frame: those always run.
    var frameCall = !batch;
    if (frameCall) batch = true, Promise.resolve().then(function() { batch = false; });
    if (frameCall && iterations === 1 && engine.busy()) return simulation;

    advance(iterations);
    if (readback) flush();
    return simulation;
  }

  function initializeNodes() {
    for (var i = 0, n = nodes.length, node; i < n; ++i) {
      node = nodes[i], node.index = i;
      if (node.fx != null) node.x = node.fx;
      if (node.fy != null) node.y = node.fy;
      if (node.fz != null) node.z = node.fz;
      if (isNaN(node.x) || (nDim > 1 && isNaN(node.y)) || (nDim > 2 && isNaN(node.z))) {
        var radius = initialRadius * (nDim > 2 ? Math.cbrt(0.5 + i) : (nDim > 1 ? Math.sqrt(0.5 + i) : i)),
          rollAngle = i * initialAngleRoll,
          yawAngle = i * initialAngleYaw;

        if (nDim === 1) {
          node.x = radius;
        } else if (nDim === 2) {
          node.x = radius * Math.cos(rollAngle);
          node.y = radius * Math.sin(rollAngle);
        } else { // 3 dimensions: use spherical distribution along 2 irrational number angles
          node.x = radius * Math.sin(rollAngle) * Math.cos(yawAngle);
          node.y = radius * Math.cos(rollAngle);
          node.z = radius * Math.sin(rollAngle) * Math.sin(yawAngle);
        }
      }
      if (isNaN(node.vx) || (nDim > 1 && isNaN(node.vy)) || (nDim > 2 && isNaN(node.vz))) {
        node.vx = 0;
        if (nDim > 1) { node.vy = 0; }
        if (nDim > 2) { node.vz = 0; }
      }
    }
  }

  function initializeForce(force) {
    if (force.initialize) force.initialize(nodes, random, nDim);
    return force;
  }

  function reset() {
    if (!engine) return;
    if (oversized = !engine.fits(nodes.length)) {
      engine.setNodes([], nDim); // free the buffers
      usable = false;
      console.warn(nodes.length + " nodes exceed this GPU's buffer limits: simulating on the CPU.");
      return;
    }
    engine.setNodes(nodes, nDim);
    refresh();
  }

  initializeNodes();
  ready = init();

  return simulation = {
    tick: tick,

    restart: function() {
      return running = true, stepper.restart(step), simulation;
    },

    stop: function() {
      return running = false, stepper.stop(), simulation;
    },

    numDimensions: function(_) {
      return arguments.length
          ? (nDim = Math.min(MAX_DIMENSIONS, Math.max(1, Math.round(_))), forces.forEach(initializeForce), reset(), simulation)
          : nDim;
    },

    nodes: function(_) {
      return arguments.length ? (nodes = _, initializeNodes(), forces.forEach(initializeForce), reset(), simulation) : nodes;
    },

    alpha: function(_) {
      return arguments.length ? (alpha = +_, simulation) : alpha;
    },

    alphaMin: function(_) {
      return arguments.length ? (alphaMin = +_, simulation) : alphaMin;
    },

    alphaDecay: function(_) {
      return arguments.length ? (alphaDecay = +_, simulation) : +alphaDecay;
    },

    alphaTarget: function(_) {
      return arguments.length ? (alphaTarget = +_, simulation) : alphaTarget;
    },

    velocityDecay: function(_) {
      return arguments.length ? (velocityDecay = 1 - _, simulation) : 1 - velocityDecay;
    },

    randomSource: function(_) {
      return arguments.length ? (random = _, forces.forEach(initializeForce), engine && engine.setSeed(random), simulation) : random;
    },

    force: function(name, _) {
      if (arguments.length < 2) return forces.get(name);
      if (_ == null) forces.delete(name);
      else forces.set(name, initializeForce(_));
      refresh();
      return simulation;
    },

    find: function() {
      var args = Array.prototype.slice.call(arguments);
      var x = args.shift() || 0,
          y = (nDim > 1 ? args.shift() : null) || 0,
          z = (nDim > 2 ? args.shift() : null) || 0,
          radius = args.shift() || Infinity;

      var i = 0,
          n = nodes.length,
          dx,
          dy,
          dz,
          d2,
          node,
          closest;

      radius *= radius;

      for (i = 0; i < n; ++i) {
        node = nodes[i];
        dx = x - node.x;
        dy = y - (node.y || 0);
        dz = z - (node.z ||0);
        d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < radius) closest = node, radius = d2;
      }

      return closest;
    },

    on: function(name, _) {
      return arguments.length > 1 ? (event.on(name, _), simulation) : event.on(name);
    },

    // --- additions to the forceSimulation API ---

    // Promise for the simulation once `iterations` ticks have run and the nodes reflect
    // them. This is the way to run a layout to completion (the synchronous tick() cannot
    // wait for the GPU).
    tickAsync: function(iterations) {
      if (iterations === undefined) iterations = 1;
      return ready.then(function() { return handover; }).then(function() {
        if (!usable) return cpuTick(iterations);
        advance(iterations);
        return engine.sync().then(function() { return simulation; });
      });
    },

    // Promise for the simulation once the nodes reflect everything the GPU has computed.
    sync: function() {
      return ready.then(function() { return handover; }).then(function() {
        return usable && engine.sync();
      }).then(function() { return simulation; });
    },

    // True while the physics runs on the GPU.
    isGPUEnabled: function() {
      return usable;
    },

    // Promise for whether GPU setup succeeded (false: the simulation runs on the CPU).
    gpuReady: function() {
      return ready.then(function() { return usable; });
    },

    // The GPU-resident state, {positions, velocities, count, stride}: GPUBuffers holding a
    // vec4<f32> per node (xyz used). Lets a WebGPU renderer draw the layout without ever
    // reading it back. null until the GPU is ready.
    gpuBuffers: function() {
      return engine && usable ? engine.buffers() : null;
    },

    // Release GPU resources and stop the simulation.
    destroy: function() {
      destroyed = true, running = false, usable = false;
      stepper.stop();
      if (engine) engine.destroy(), engine = null;
      return simulation;
    }
  };
}
