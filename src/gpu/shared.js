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
      {binding: 2, visibility: COMPUTE, buffer: BUFFER_TYPES.rw},
      {binding: 3, visibility: COMPUTE, buffer: BUFFER_TYPES.rw}
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
// group 0.
function descriptor(device, def) {
  var layouts = [layout0(device)];
  if (def.spec) layouts.push(layout1(device, def.spec));
  return {
    layout: device.createPipelineLayout({bindGroupLayouts: layouts}),
    compute: {module: device.createShaderModule({code: def.code}), entryPoint: def.entry}
  };
}

export function pipeline(device, id, def) {
  var cache = cacheFor(device), p = cache.pipelines.get(id);
  if (!p) cache.pipelines.set(id, p = device.createComputePipeline(descriptor(device, def)));
  return p;
}

// Compile everything up front, off the main thread. Rejects if a pipeline is invalid (a
// shader error, say); no error scope is used, so validation errors of the application's
// own work on a shared device are never captured here.
export function prewarm(device, defs) {
  var cache = cacheFor(device);
  return Promise.all(Object.keys(defs).filter(function(id) {
    return !cache.pipelines.has(id);
  }).map(function(id) {
    return device.createComputePipelineAsync(descriptor(device, defs[id])).then(function(p) {
      cache.pipelines.set(id, p);
    });
  }));
}
