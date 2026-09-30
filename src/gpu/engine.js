import {COPY_DST, COPY_SRC, MAP_READ, MAP_MODE_READ, STORAGE, UNIFORM, SIM_BYTES, WORKGROUP} from "./constants.js";
import {layout0, pipeline, prewarm} from "./shared.js";
import {SNAPSHOT, INTEGRATE} from "./wgsl.js";
import {ceilDiv} from "./passes/util.js";
import passes from "./passes/index.js";

// Steps encoded into one command buffer. Each gets its own slot of the per-step uniform
// block (alpha changes from step to step), selected with a dynamic offset.
var MAX_BATCH = 64;

// Submissions the queue may hold before Engine.busy() reports true. Completion is only
// acknowledged asynchronously, so leave room for a frame or two of jitter.
var MAX_PENDING = 3;

var definitions = {
  snapshot: {code: SNAPSHOT, entry: "main", spec: ["rw"]},
  integrate: {code: INTEGRATE, entry: "main", spec: ["ro", "ro"]}
};
Object.keys(passes).forEach(function(type) {
  Object.assign(definitions, passes[type].pipelines);
});

// Compile ahead of time, and in parallel, the kernels a simulation of `n` nodes in `nDim`
// dimensions uses with forces of the given types (all of them by default): the specialised
// variants the passes ask for (the many-body tree for these dimensions, say), and their
// other kernels. Others compile when first needed. Rejects on any shader error.
export function prepare(device, n, nDim, split, types) {
  var jobs = [["snapshot"], ["integrate"]], seen = {};
  (types || Object.keys(passes)).forEach(function(type) {
    var pass = passes[type];
    if (!pass || seen[type]) return;
    seen[type] = true;
    if (pass.variants) jobs = jobs.concat(pass.variants(n, nDim, split));
    Object.keys(pass.pipelines).forEach(function(id) {
      if (!pass.pipelines[id].specialised) jobs.push([id]);
    });
  });
  return prewarm(device, definitions, jobs);
}

// The name of the first force that cannot run on the GPU, or null.
export function unsupported(forces) {
  var found = null;
  forces.forEach(function(force, name) {
    if (found === null && !(typeof force.gpu === "function" && passes[(force.gpu() || {}).type])) found = name;
  });
  return found;
}

// Calls onLost if the device is lost while the engine is alive. The promise outlives the
// engine, so its callback must not close over the engine's state (nodes and their copies):
// it only holds this small object, which destroy() empties.
function watchLoss(device, onLost) {
  var watch = {onLost: onLost};
  device.lost.then(function(info) {
    if (watch.onLost) watch.onLost(info);
  });
  return watch;
}

function num(v) {
  return (v = +v) === v ? v : 0; // undefined / NaN -> 0
}

// Mirrors a node array on the GPU.
//
//   * The GPU holds the truth for position and velocity while a simulation runs.
//   * Nodes are written back asynchronously by readback().
//   * Changes made on the JS side in the meantime -- a moved x/y/z, a pinned fx/fy/fz --
//     are detected (by comparing against what was last exchanged) and uploaded at the
//     start of the next step. A node the user has touched is never overwritten by a
//     readback that was already in flight.
export default function(device, onLost, options) {
  var engine,
      nodes = [],
      n = 0,
      nDim = 2,
      plan = [],
      alive = true,
      seed = 0,
      stepCount = 0,
      enqueued = 0,   // steps submitted
      landed = 0,     // steps reflected in the JS nodes
      pending = 0,    // submissions the GPU has not finished
      epoch = 0,      // bumped whenever the buffers are replaced or overwritten
      stale = false,
      inflight = null,
      buffers = null;

  var alignment = Math.max(256, device.limits.minUniformBufferOffsetAlignment | 0),
      slotWords = alignment / 4,
      simData = new ArrayBuffer(MAX_BATCH * alignment),
      simF32 = new Float32Array(simData),
      simU32 = new Uint32Array(simData);

  // What the GPU was last told / told us about every node (exact JS doubles).
  var shadow = new Float64Array(0),   // x, y, z
      pinned = new Uint8Array(0),     // fx/fy/fz mask
      pinnedAt = new Float64Array(0); // fx, fy, fz

  var loss = watchLoss(device, onLost);

  function release() {
    plan.forEach(function(entry) { entry.pass.destroy(); });
    plan = [];
    if (buffers) {
      ++epoch;
      buffers.forEach(function(b) { b.destroy(); });
      buffers = null;
    }
  }

  // --- JS -> GPU ---------------------------------------------------------------------

  function uploadAll() {
    var pos = new Float32Array(4 * n),
        vel = new Float32Array(4 * n),
        fixPos = new Float32Array(4 * n),
        fixMask = new Uint32Array(n),
        i, j, node, mask;

    for (i = 0; i < n; ++i) {
      node = nodes[i], j = 3 * i;
      shadow[j] = pos[4 * i] = num(node.x);
      shadow[j + 1] = pos[4 * i + 1] = nDim > 1 ? num(node.y) : 0;
      shadow[j + 2] = pos[4 * i + 2] = nDim > 2 ? num(node.z) : 0;
      vel[4 * i] = num(node.vx);
      if (nDim > 1) vel[4 * i + 1] = num(node.vy);
      if (nDim > 2) vel[4 * i + 2] = num(node.vz);
      mask = 0;
      if (node.fx != null) mask |= 1, pinnedAt[j] = fixPos[4 * i] = +node.fx;
      if (nDim > 1 && node.fy != null) mask |= 2, pinnedAt[j + 1] = fixPos[4 * i + 1] = +node.fy;
      if (nDim > 2 && node.fz != null) mask |= 4, pinnedAt[j + 2] = fixPos[4 * i + 2] = +node.fz;
      pinned[i] = fixMask[i] = mask;
    }

    var queue = device.queue;
    queue.writeBuffer(buffers[0], 0, pos);
    queue.writeBuffer(buffers[1], 0, vel);
    queue.writeBuffer(buffers[4], 0, fixPos);
    queue.writeBuffer(buffers[5], 0, fixMask);
    ++epoch; // a readback still in flight predates this state
    stale = false;
    landed = enqueued;
  }

  function uploadPositions(a, b) {
    var out = new Float32Array(4 * (b - a)), i, j, node;
    for (i = a; i < b; ++i) {
      node = nodes[i], j = 3 * i;
      shadow[j] = out[4 * (i - a)] = num(node.x);
      shadow[j + 1] = out[4 * (i - a) + 1] = nDim > 1 ? num(node.y) : 0;
      shadow[j + 2] = out[4 * (i - a) + 2] = nDim > 2 ? num(node.z) : 0;
    }
    device.queue.writeBuffer(buffers[0], 16 * a, out);
  }

  function uploadPinned(a, b) {
    var pos = new Float32Array(4 * (b - a)), mask = new Uint32Array(b - a), i, j, node, m;
    for (i = a; i < b; ++i) {
      node = nodes[i], j = 3 * i, m = 0;
      if (node.fx != null) m |= 1, pinnedAt[j] = pos[4 * (i - a)] = +node.fx;
      if (nDim > 1 && node.fy != null) m |= 2, pinnedAt[j + 1] = pos[4 * (i - a) + 1] = +node.fy;
      if (nDim > 2 && node.fz != null) m |= 4, pinnedAt[j + 2] = pos[4 * (i - a) + 2] = +node.fz;
      pinned[i] = mask[i - a] = m;
    }
    device.queue.writeBuffer(buffers[4], 16 * a, pos);
    device.queue.writeBuffer(buffers[5], 4 * a, mask);
  }

  function syncFromJS() {
    var posRun = -1, pinRun = -1, i, j, node, x, y, z, m, posDirty, pinDirty;

    for (i = 0; i <= n; ++i) {
      posDirty = pinDirty = false;
      if (i < n) {
        node = nodes[i], j = 3 * i;
        x = num(node.x), y = nDim > 1 ? num(node.y) : 0, z = nDim > 2 ? num(node.z) : 0;
        posDirty = x !== shadow[j] || y !== shadow[j + 1] || z !== shadow[j + 2];

        m = (node.fx != null ? 1 : 0) | (nDim > 1 && node.fy != null ? 2 : 0) | (nDim > 2 && node.fz != null ? 4 : 0);
        pinDirty = m !== pinned[i]
            || (m & 1) !== 0 && +node.fx !== pinnedAt[j]
            || (m & 2) !== 0 && +node.fy !== pinnedAt[j + 1]
            || (m & 4) !== 0 && +node.fz !== pinnedAt[j + 2];
      }

      if (posDirty) { if (posRun < 0) posRun = i; }
      else if (posRun >= 0) uploadPositions(posRun, i), posRun = -1;

      if (pinDirty) { if (pinRun < 0) pinRun = i; }
      else if (pinRun >= 0) uploadPinned(pinRun, i), pinRun = -1;
    }
  }

  // --- GPU -> JS ---------------------------------------------------------------------

  function land(data) {
    var i, j, k, node, x, y, z, v = 4 * n;
    for (i = 0; i < n; ++i) {
      node = nodes[i], j = 3 * i, k = 4 * i;
      x = num(node.x), y = nDim > 1 ? num(node.y) : 0, z = nDim > 2 ? num(node.z) : 0;

      // Untouched since the last exchange: take the GPU's position. Otherwise the user
      // has moved it and syncFromJS() will push their value at the next step.
      if (x === shadow[j] && y === shadow[j + 1] && z === shadow[j + 2]) {
        node.x = shadow[j] = data[k];
        if (nDim > 1) node.y = shadow[j + 1] = data[k + 1];
        if (nDim > 2) node.z = shadow[j + 2] = data[k + 2];
      }

      node.vx = data[v + k];
      if (nDim > 1) node.vy = data[v + k + 1];
      if (nDim > 2) node.vz = data[v + k + 2];
    }
  }

  function readback() {
    if (inflight) return inflight;
    if (!alive || !n || landed === enqueued) return Promise.resolve();

    var target = enqueued, era = epoch, staging = buffers[6], bytes = 16 * n,
        encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffers[0], 0, staging, 0, bytes);
    encoder.copyBufferToBuffer(buffers[1], 0, staging, bytes, bytes);
    device.queue.submit([encoder.finish()]);

    return inflight = staging.mapAsync(MAP_MODE_READ).then(function() {
      if (era === epoch) land(new Float32Array(staging.getMappedRange())), landed = target;
      staging.unmap();
    }, function(error) {
      if (era === epoch && alive) throw error; // otherwise the buffers were replaced under us
    }).then(function() {
      inflight = null;
    }, function(error) {
      inflight = null;
      throw error;
    });
  }

  // --- setup -------------------------------------------------------------------------

  return engine = {
    device: device,

    get n() { return n; },
    // threads per node in the all-pairs kernels, when the caller chose one (else automatic)
    get forcedSplit() { return options && options.split; },
    get nDim() { return nDim; },

    pipeline: function(id, constants) {
      return pipeline(device, id, definitions[id], constants);
    },

    // pos + vel, for kernels that read other nodes' predicted positions. snapshot()
    // refreshes it and leaves bind group 1 for the caller to set.
    get snap() { return buffers && buffers[2]; },

    snapshot: function(pass) {
      pass.setPipeline(engine.pipeline("snapshot"));
      pass.setBindGroup(1, buffers.bindGroupSnapshot);
      pass.dispatchWorkgroups(ceilDiv(n, WORKGROUP));
    },

    // Replace the node set: (re)allocate all buffers and upload the nodes. Forces must be
    // passed to setForces() again afterwards.
    setNodes: function(_nodes, _nDim) {
      release();
      nodes = _nodes, n = nodes.length, nDim = _nDim;
      enqueued = landed = 0;
      shadow = new Float64Array(3 * n);
      pinned = new Uint8Array(n);
      pinnedAt = new Float64Array(3 * n);
      if (!n) return engine;

      var vec4 = 16 * n,
          state = STORAGE | COPY_DST | COPY_SRC,
          make = function(size, usage) { return device.createBuffer({size: size, usage: usage}); };

      buffers = [
        make(vec4, state),                    // 0 pos
        make(vec4, state),                    // 1 vel
        make(vec4, STORAGE),                  // 2 snap
        make(MAX_BATCH * alignment, UNIFORM | COPY_DST), // 3 per-step uniforms
        make(vec4, STORAGE | COPY_DST),       // 4 pinned positions
        make(Math.max(16, 4 * n), STORAGE | COPY_DST), // 5 pinned mask
        make(2 * vec4, MAP_READ | COPY_DST)   // 6 readback staging
      ];
      buffers.bindGroup0 = device.createBindGroup({
        layout: layout0(device),
        entries: [
          {binding: 0, resource: {buffer: buffers[3], size: SIM_BYTES}},
          {binding: 1, resource: {buffer: buffers[0]}},
          {binding: 2, resource: {buffer: buffers[1]}}
        ]
      });
      buffers.bindGroupSnapshot = device.createBindGroup({
        layout: engine.pipeline("snapshot").getBindGroupLayout(1),
        entries: [{binding: 0, resource: {buffer: buffers[2]}}]
      });
      buffers.bindGroupIntegrate = device.createBindGroup({
        layout: engine.pipeline("integrate").getBindGroupLayout(1),
        entries: [
          {binding: 0, resource: {buffer: buffers[4]}},
          {binding: 1, resource: {buffer: buffers[5]}}
        ]
      });
      uploadAll();
      return engine;
    },

    // Build the pass list from the forces (in application order). Returns false, leaving
    // the engine untouched, when a force has no GPU implementation.
    setForces: function(forces) {
      var descs = forces.map(function(force) {
        return typeof force.gpu === "function" ? force.gpu() : null;
      });
      if (descs.some(function(d) { return !d || !passes[d.type]; })) return false;

      // A force that is still in the list keeps its pass (and its uploaded buffers).
      var previous = new Map(plan.map(function(entry) { return [entry.force, entry]; }));
      plan = n ? forces.map(function(force, i) {
        var entry = previous.get(force);
        if (entry && entry.type === descs[i].type) return previous.delete(force), entry;
        return {force: force, type: descs[i].type, pass: passes[descs[i].type].create(engine)};
      }) : [];
      previous.forEach(function(entry) { entry.pass.destroy(); });
      plan.forEach(function(entry, i) { entry.pass.update(descs[i]); });
      return true;
    },

    // Whether `count` nodes fit within this device's buffer and dispatch limits. The
    // largest buffer is the many-body tree's, 64 bytes per node.
    fits: function(count) {
      var limits = device.limits;
      return 64 * count <= Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize)
          && Math.ceil(count / WORKGROUP) <= limits.maxComputeWorkgroupsPerDimension;
    },

    // Seed for the shaders' coincident-node jiggle, drawn from the simulation's random source.
    setSeed: function(random) {
      seed = Math.floor(random() * 4294967296) >>> 0;
    },

    // The JS nodes have been changed behind the engine's back (e.g. by CPU ticks):
    // re-upload everything at the next step.
    markStale: function() {
      stale = true;
    },

    // Submit `alphas.length` steps. `decay` is the velocity multiplier, 1 - velocityDecay.
    enqueue: function(alphas, decay) {
      if (!alive || !n) return;

      if (stale) uploadAll();
      else syncFromJS();
      plan.forEach(function(entry) { entry.pass.update(entry.force.gpu()); });

      var total = alphas.length, integrate = engine.pipeline("integrate"),
          groups = ceilDiv(n, WORKGROUP), start, count, s, w, encoder, pass;

      for (start = 0; start < total; start += MAX_BATCH) {
        count = Math.min(MAX_BATCH, total - start);
        for (s = 0; s < count; ++s) {
          w = s * slotWords;
          simF32[w] = alphas[start + s];
          simF32[w + 1] = decay;
          simU32[w + 2] = n;
          simU32[w + 3] = nDim;
          simU32[w + 4] = stepCount++ >>> 0;
          simU32[w + 5] = seed;
        }
        device.queue.writeBuffer(buffers[3], 0, simF32, 0, count * slotWords);

        encoder = device.createCommandEncoder();
        pass = encoder.beginComputePass();
        for (s = 0; s < count; ++s) {
          pass.setBindGroup(0, buffers.bindGroup0, [s * alignment]);
          for (w = 0; w < plan.length; ++w) plan[w].pass.encode(pass);
          pass.setPipeline(integrate);
          pass.setBindGroup(1, buffers.bindGroupIntegrate);
          pass.dispatchWorkgroups(groups);
        }
        pass.end();
        device.queue.submit([encoder.finish()]);

        ++pending;
        device.queue.onSubmittedWorkDone().then(function() { --pending; }, function() { pending = 0; });
      }
      enqueued += total;
    },

    // True when the GPU is still working through earlier submissions.
    busy: function() {
      return pending >= MAX_PENDING;
    },

    // Copy the latest state into the JS nodes (one readback at a time).
    readback: readback,

    // Resolves once the GPU has finished everything submitted so far (no readback).
    idle: function() {
      return device.queue.onSubmittedWorkDone();
    },

    // Resolves once the JS nodes reflect every step submitted so far.
    sync: function() {
      function next() {
        return readback().then(function() {
          if (alive && n && landed !== enqueued) return next();
        });
      }
      return next();
    },

    // The GPU-resident state, for rendering straight from the simulation's buffers:
    // vec4<f32> per node, xyz used.
    buffers: function() {
      return buffers && {positions: buffers[0], velocities: buffers[1], count: n, stride: 16};
    },

    destroy: function() {
      alive = false;
      loss.onLost = null;
      release();
      nodes = [], shadow = pinnedAt = new Float64Array(0), pinned = new Uint8Array(0);
    }
  };
}
