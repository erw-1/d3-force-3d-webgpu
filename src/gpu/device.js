// Adapter / device acquisition. A single device is shared by all simulations unless
// one is passed in explicitly (e.g. to share buffers with a WebGPU renderer).

var devices = new WeakMap(); // GPU -> Promise<GPUDevice>

export function isWebGPUAvailable() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}

export function checkWebGPUSupport() {
  if (!isWebGPUAvailable()) return Promise.resolve(false);
  return navigator.gpu.requestAdapter().then(function(adapter) {
    return !!adapter;
  }, function() {
    return false;
  });
}

export function requestDevice(gpu) {
  if (!gpu) gpu = isWebGPUAvailable() ? navigator.gpu : null;
  if (!gpu) return Promise.reject(new Error("WebGPU is not available"));

  var promise = devices.get(gpu);
  if (!promise) {
    devices.set(gpu, promise = gpu.requestAdapter({powerPreference: "high-performance"}).then(function(adapter) {
      if (!adapter) throw new Error("No WebGPU adapter found");
      // the largest buffers the adapter allows: big graphs need more than the defaults
      return adapter.requestDevice({requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize
      }});
    }).then(function(device) {
      device.lost.then(function() { if (devices.get(gpu) === promise) devices.delete(gpu); });
      return device;
    }));
    promise.catch(function() { if (devices.get(gpu) === promise) devices.delete(gpu); });
  }
  return promise;
}
