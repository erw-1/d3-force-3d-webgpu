import {COMPUTE, SIM_BYTES} from "./constants.js";

// Bind group layouts and compute pipelines are created once per device and reused by
// every simulation on it.

var caches = new WeakMap();

function cacheFor(device) {
  var cache = caches.get(device);
  if (!cache) {
    cache = {layout0: null, layouts: new Map(), pipelines: new Map()};
    caches.set(device, cache);
  }
  return cache;
}

var BUFFER_TYPES = {
  uniform: {type: "uniform"},
  ro: {type: "read-only-storage"},
  rw: {type: "storage"}
};

// Group 0: per-step uniforms (dynamic offset, so one buffer holds a whole batch of
// steps) and the node state buffers.
export function layout0(device) {
  var cache = cacheFor(device);
  return cache.layout0 || (cache.layout0 = device.createBindGroupLayout({
    entries: [
      {binding: 0, visibility: COMPUTE, buffer: {type: "uniform", hasDynamicOffset: true, minBindingSize: SIM_BYTES}},
      {binding: 1, visibility: COMPUTE, buffer: BUFFER_TYPES.rw},
      {binding: 2, visibility: COMPUTE, buffer: BUFFER_TYPES.rw}
    ]
  }));
}

// Group 1: described by a list such as ["uniform", "ro", "ro"].
export function layout1(device, spec) {
  var cache = cacheFor(device), key = spec.join(","), layout = cache.layouts.get(key);
  if (!layout) {
    cache.layouts.set(key, layout = device.createBindGroupLayout({
      entries: spec.map(function(kind, binding) {
        return {binding: binding, visibility: COMPUTE, buffer: BUFFER_TYPES[kind]};
      })
    }));
  }
  return layout;
}

// A pipeline description is {code, entry, spec}; spec is null for kernels that only use
// group 0. `constants` sets the shader's `override` values, specialising it (a kernel
// tuned to the problem size, say) without another copy of the source.
function descriptor(device, def, constants) {
  var layouts = [layout0(device)];
  if (def.spec) layouts.push(layout1(device, def.spec));
  var compute = {module: device.createShaderModule({code: def.code}), entryPoint: def.entry};
  if (constants) compute.constants = constants;
  return {layout: device.createPipelineLayout({bindGroupLayouts: layouts}), compute: compute};
}

function keyOf(id, constants) {
  return constants ? id + JSON.stringify(constants) : id;
}

export function pipeline(device, id, def, constants) {
  var cache = cacheFor(device), key = keyOf(id, constants), p = cache.pipelines.get(key);
  if (!p) cache.pipelines.set(key, p = device.createComputePipeline(descriptor(device, def, constants)));
  return p;
}

// Compile pipelines up front, off the main thread: `jobs` is a list of [id, constants].
// Rejects if a pipeline is invalid (a shader error, say); no error scope is used, so
// validation errors of the application's own work on a shared device are never captured.
export function prewarm(device, defs, jobs) {
  var cache = cacheFor(device), queued = {};
  return Promise.all(jobs.filter(function(job) {
    var key = keyOf(job[0], job[1]);
    return !cache.pipelines.has(key) && !queued[key] && (queued[key] = true);
  }).map(function(job) {
    return device.createComputePipelineAsync(descriptor(device, defs[job[0]], job[1])).then(function(p) {
      cache.pipelines.set(keyOf(job[0], job[1]), p);
    });
  }));
}
