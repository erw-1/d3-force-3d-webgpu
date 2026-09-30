import assert from "assert";
import {forceSimulation, forceSimulationGPU} from "../src/index.js";

// A headless WebGPU device (Dawn, via the `webgpu` package). null when the package cannot
// be loaded or there is no adapter, in which case the GPU tests are skipped.
var deviceP, keepAlive = []; // Dawn crashes if the GPU/adapter objects are garbage collected
export function getDevice() {
  return deviceP || (deviceP = (async function() {
    try {
      var gpu = (await import("webgpu")).create([]), adapter = await gpu.requestAdapter();
      if (!adapter) return null;
      var device = await adapter.requestDevice();
      keepAlive.push(gpu, adapter, device);
      device.errors = [];
      device.addEventListener("uncapturederror", function(event) {
        device.errors.push(event.error.message);
      });
      return device;
    } catch (error) {
      return null;
    }
  })());
}

// Use inside a describe(): resolves the shared device and skips the suite without one.
export function useDevice() {
  var context = {device: null};
  before(async function() {
    context.device = await getDevice();
    if (!context.device) this.skip();
  });
  afterEach(function() {
    if (context.device && context.device.errors.length) {
      var errors = context.device.errors.splice(0);
      assert.fail("WebGPU errors: " + errors.join("; "));
    }
  });
  return context;
}

// Deterministic pseudo-random nodes in a cube of side `size`.
export function makeNodes(n, dims, size, seed) {
  var s = seed || 12345, nodes = [];
  // f32-exact coordinates: the GPU stores f32, so the comparison then tests the arithmetic, not the rounding of the input
  function rnd() { return Math.fround((s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296); }
  for (var i = 0; i < n; ++i) {
    var node = {x: Math.fround((rnd() - 0.5) * (size || 200))};
    if (dims > 1) node.y = Math.fround((rnd() - 0.5) * (size || 200));
    if (dims > 2) node.z = Math.fround((rnd() - 0.5) * (size || 200));
    nodes.push(node);
  }
  return nodes;
}

export function fields(dims) {
  return ["x", "y", "z"].slice(0, dims).concat(["vx", "vy", "vz"].slice(0, dims));
}

// `outliers`: how many nodes may miss the tolerance (they must still be finite). The
// Barnes-Hut tree needs a few: a theta test that f64 decides by less than f32 can resolve
// (a relative margin of ~1e-7) can go the other way, and changes that node's force by up to
// 41% of the cell's in 3D.
export function assertNodesClose(actual, expected, dims, options) {
  var rtol = options && options.rtol || 1e-4, atol = options && options.atol || 1e-4,
      outliers = options && options.outliers || 0, worst = 0, missed = [];
  assert.strictEqual(actual.length, expected.length);
  actual.forEach(function(a, i) {
    var miss = null;
    fields(dims).forEach(function(f) {
      var e = expected[i][f], tolerance = atol + rtol * Math.max(1, Math.abs(a[f]), Math.abs(e)), d = Math.abs(a[f] - e);
      assert(isFinite(a[f]), "node " + i + "." + f + " is " + a[f]);
      worst = Math.max(worst, d / tolerance);
      if (!(d <= tolerance) && !miss) miss = "node " + i + "." + f + ": " + a[f] + " vs " + e + " (|diff| " + d + " > " + tolerance + ")";
    });
    if (miss) missed.push(miss);
  });
  assert(missed.length <= outliers, missed.length + " node(s) off (" + outliers + " allowed): " + missed.slice(0, 3).join("; "));
  return worst;
}

// Outliers allowed when comparing n nodes against the CPU's Barnes-Hut tree: 1 per 1000.
export function treeOutliers(n) {
  return Math.max(1, Math.ceil(n / 1000));
}

// Run the same forces for `ticks` on the CPU and the GPU from the same start.
// `makeForces` returns a fresh [[name, force], ...] on every call.
export async function runBoth(device, options) {
  var dims = options.dims || 2,
      cpuNodes = (options.makeNodes || makeNodes)(options.n, dims),
      gpuNodes = (options.makeNodes || makeNodes)(options.n, dims),
      cpu = forceSimulation(cpuNodes, dims).stop(),
      gpu = forceSimulationGPU(gpuNodes, dims, {device: device, split: options.split}).stop();

  options.makeForces().forEach(function(d) { cpu.force(d[0], d[1]); });
  options.makeForces().forEach(function(d) { gpu.force(d[0], d[1]); });
  if (options.setup) options.setup(cpu, cpuNodes), options.setup(gpu, gpuNodes);

  assert.strictEqual(await gpu.gpuReady(), true, "the simulation should run on the GPU");
  cpu.tick(options.ticks || 1);
  await gpu.tickAsync(options.ticks || 1);
  gpu.destroy();
  return {cpu: cpu, gpu: gpu, cpuNodes: cpuNodes, gpuNodes: gpuNodes};
}
