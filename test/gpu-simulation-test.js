import assert from "assert";
import {forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, forceSimulationGPU, forceX, forceZ} from "../src/index.js";
import {assertNodesClose, makeNodes, useDevice} from "./gpu-helpers.js";
import {assertNodeEqual} from "./asserts.js";

var GPU_KEYS = ["destroy", "gpuBuffers", "gpuReady", "isGPUEnabled", "sync", "tickAsync"];

function delay(ms) {
  return new Promise(function(resolve) { setTimeout(resolve, ms); });
}

// Silence and record console.warn while `fn` runs.
async function withWarnings(fn) {
  var warn = console.warn, warnings = [];
  console.warn = function() { warnings.push(Array.prototype.join.call(arguments, " ")); };
  try { await fn(warnings); } finally { console.warn = warn; }
  return warnings;
}

describe("forceSimulationGPU without WebGPU", function() {
  // Node has no navigator.gpu and no device is passed, so this is the CPU fallback.
  it("has the forceSimulation API plus the GPU additions", function() {
    var cpu = Object.keys(forceSimulation().stop()).sort(),
        gpu = Object.keys(forceSimulationGPU().stop()).sort();
    assert.deepStrictEqual(gpu, cpu.concat(GPU_KEYS).sort());
  });

  it("reports that the GPU is not in use", async function() {
    var sim = forceSimulationGPU().stop();
    assert.strictEqual(sim.isGPUEnabled(), false);
    assert.strictEqual(await sim.gpuReady(), false);
    assert.strictEqual(sim.gpuBuffers(), null);
  });

  it("initializes nodes exactly like forceSimulation", function() {
    var f = forceSimulationGPU().stop(), a = {}, b = {}, c = {};
    f.nodes([a, b, c]);
    assertNodeEqual(a, {index: 0, x: 7.0710678118654755, y: 0, vy: 0, vx: 0});
    assertNodeEqual(b, {index: 1, x: -9.03088751750192, y: 8.27303273571596, vy: 0, vx: 0});
    assertNodeEqual(c, {index: 2, x: 1.3823220809823638, y: -15.750847141167634, vy: 0, vx: 0});
  });

  it("simulates identically to forceSimulation (it is the CPU simulation)", async function() {
    var links = function() { return Array.from({length: 40}, function(_, i) { return {source: i, target: (i * 7 + 1) % 40}; }); },
        setup = function(sim) {
          return sim.force("link", forceLink(links())).force("charge", forceManyBody()).force("center", forceCenter());
        },
        a = Array.from({length: 40}, function() { return {}; }),
        b = Array.from({length: 40}, function() { return {}; });
    setup(forceSimulation(a, 3).stop()).tick(25);
    var gpu = setup(forceSimulationGPU(b, 3).stop());
    await gpu.tickAsync(25);
    assert.deepStrictEqual(b.map(function(n) { return [n.x, n.y, n.z, n.vx, n.vy, n.vz]; }),
                           a.map(function(n) { return [n.x, n.y, n.z, n.vx, n.vy, n.vz]; }));
  });
});

describe("forceSimulationGPU", function() {
  this.timeout(30000);
  var gpu = useDevice();

  // A test that is not about the CPU fallback must actually run on the GPU: otherwise it
  // would compare the CPU with itself and pass without testing anything.
  var expectCPU, fellBack;
  beforeEach(function() { expectCPU = false, fellBack = []; });
  afterEach(function() {
    assert.deepStrictEqual(fellBack, [], "a simulation fell back to the CPU: " + fellBack.join("; "));
  });

  function make(nodes, dims) {
    var sim = forceSimulationGPU(nodes, dims, {device: gpu.device});
    sim.gpuReady().then(function(ok) { if (!ok && !expectCPU) fellBack.push("gpuReady() was false"); });
    return sim;
  }

  it("runs on the GPU when a device is available", async function() {
    var sim = make(makeNodes(10, 3), 3).stop();
    assert.strictEqual(sim.isGPUEnabled(), false, "not until the device is ready");
    assert.strictEqual(await sim.gpuReady(), true);
    assert.strictEqual(sim.isGPUEnabled(), true);
    sim.destroy();
  });

  it("advances alpha exactly like forceSimulation", async function() {
    var cpu = forceSimulation(makeNodes(20, 2), 2).stop().alphaTarget(0.1).alphaDecay(0.05),
        sim = make(makeNodes(20, 2), 2).stop().alphaTarget(0.1).alphaDecay(0.05);
    cpu.force("charge", forceManyBody());
    sim.force("charge", forceManyBody());
    cpu.tick(17);
    await sim.tickAsync(17);
    assert.strictEqual(sim.alpha(), cpu.alpha());
    sim.destroy();
  });

  it("tickAsync() resolves with the simulation once the nodes are up to date", async function() {
    var nodes = makeNodes(50, 3), before = nodes.map(function(n) { return n.x; }),
        sim = make(nodes, 3).stop().force("charge", forceManyBody());
    assert.strictEqual(await sim.tickAsync(3), sim);
    assert(nodes.some(function(n, i) { return n.x !== before[i]; }), "nodes moved");
    sim.destroy();
  });

  it("tick() is asynchronous: nodes are current after sync()", async function() {
    var nodes = makeNodes(50, 3), cpuNodes = makeNodes(50, 3),
        sim = make(nodes, 3).stop().force("charge", forceManyBody().theta(1e-6)),
        cpu = forceSimulation(cpuNodes, 3).stop().force("charge", forceManyBody().theta(1e-6));
    await sim.gpuReady();
    assert.strictEqual(sim.tick(4), sim);
    cpu.tick(4);
    await sim.sync();
    assertNodesClose(nodes, cpuNodes, 3, {rtol: 1e-3});
    sim.destroy();
  });

  it("emits tick events with up-to-date nodes, then end", async function() {
    var nodes = makeNodes(30, 2), ticks = 0, ends = 0, snapshots = [],
        sim = make(nodes, 2).alphaDecay(0.3).force("charge", forceManyBody());
    sim.on("tick", function() { ++ticks; snapshots.push(nodes[0].x); });
    sim.on("end", function() { ++ends; });
    await delay(1500);
    assert.strictEqual(ends, 1, "end fires once");
    assert(sim.alpha() < sim.alphaMin());
    assert(ticks > 3 && ticks <= 40, "ticks: " + ticks);
    assert(snapshots[snapshots.length - 1] !== snapshots[0], "positions change from tick to tick");
    sim.destroy();
  });

  it("stop() and restart() control the timer", async function() {
    var nodes = makeNodes(30, 2), ticks = 0,
        sim = make(nodes, 2).force("charge", forceManyBody()).on("tick", function() { ++ticks; });
    await sim.gpuReady();
    await delay(100);
    sim.stop();
    await delay(100);
    var stopped = ticks;
    await delay(200);
    assert.strictEqual(ticks, stopped, "no ticks while stopped");
    sim.alpha(1).restart();
    await delay(200);
    assert(ticks > stopped, "ticks resume");
    sim.destroy();
  });

  it("picks up force parameter changes made after the simulation started", async function() {
    var nodes = makeNodes(30, 3), charge = forceManyBody().theta(1e-6),
        sim = make(nodes, 3).stop().force("charge", charge),
        cpuNodes = makeNodes(30, 3), cpuCharge = forceManyBody().theta(1e-6),
        cpu = forceSimulation(cpuNodes, 3).stop().force("charge", cpuCharge);
    await sim.tickAsync(2), cpu.tick(2);
    charge.strength(function(d, i) { return -10 - i; }).distanceMin(3);
    cpuCharge.strength(function(d, i) { return -10 - i; }).distanceMin(3);
    await sim.tickAsync(2), cpu.tick(2);
    assertNodesClose(nodes, cpuNodes, 3, {rtol: 1e-3});
    sim.destroy();
  });

  it("supports adding and removing forces", async function() {
    var nodes = makeNodes(40, 3), cpuNodes = makeNodes(40, 3),
        sim = make(nodes, 3).stop().force("charge", forceManyBody().theta(1e-6)),
        cpu = forceSimulation(cpuNodes, 3).stop().force("charge", forceManyBody().theta(1e-6));
    await sim.tickAsync(2), cpu.tick(2);
    sim.force("x", forceX(20).strength(0.2)).force("z", forceZ(-10));
    cpu.force("x", forceX(20).strength(0.2)).force("z", forceZ(-10));
    await sim.tickAsync(2), cpu.tick(2);
    sim.force("charge", null);
    cpu.force("charge", null);
    await sim.tickAsync(2), cpu.tick(2);
    assertNodesClose(nodes, cpuNodes, 3, {rtol: 1e-3, atol: 1e-3});
    assert.strictEqual(sim.force("charge"), undefined);
    assert.strictEqual(sim.isGPUEnabled(), true);
    sim.destroy();
  });

  it("replaces the nodes with nodes()", async function() {
    var first = makeNodes(30, 2), second = makeNodes(70, 2, 100, 999),
        sim = make(first, 2).stop().force("charge", forceManyBody());
    await sim.tickAsync(2);
    var frozen = first.map(function(n) { return n.x; });
    sim.nodes(second);
    await sim.tickAsync(2);
    assert.deepStrictEqual(first.map(function(n) { return n.x; }), frozen, "old nodes are left alone");
    assert.strictEqual(second[69].index, 69);
    assert(second.every(function(n) { return isFinite(n.x) && isFinite(n.y); }));
    sim.destroy();
  });

  it("changes dimensionality with numDimensions()", async function() {
    var nodes = makeNodes(30, 3), sim = make(nodes, 3).stop().force("charge", forceManyBody());
    await sim.tickAsync(2);
    assert(nodes.every(function(n) { return isFinite(n.z); }));
    sim.numDimensions(2);
    var z = nodes.map(function(n) { return n.z; });
    await sim.tickAsync(2);
    assert.deepStrictEqual(nodes.map(function(n) { return n.z; }), z, "z is left alone in 2D");
    assert.strictEqual(sim.isGPUEnabled(), true);
    sim.destroy();
  });

  describe("edits to the nodes while the simulation runs", function() {
    it("respects a node pinned with fx/fy/fz, and releases it", async function() {
      var nodes = makeNodes(30, 3), sim = make(nodes, 3).stop().force("charge", forceManyBody());
      await sim.tickAsync(2);
      nodes[3].fx = 11, nodes[3].fy = 12, nodes[3].fz = 13;
      await sim.tickAsync(2);
      assert.deepStrictEqual([nodes[3].x, nodes[3].y, nodes[3].z, nodes[3].vx], [11, 12, 13, 0]);
      nodes[3].fx = 20; // dragged
      await sim.tickAsync(1);
      assert.strictEqual(nodes[3].x, 20);
      nodes[3].fx = nodes[3].fy = nodes[3].fz = null; // released
      await sim.tickAsync(5);
      assert.notStrictEqual(nodes[3].x, 20, "moves again once released");
      sim.destroy();
    });

    it("moves a node whose x/y/z the user has set", async function() {
      var nodes = makeNodes(30, 3), sim = make(nodes, 3).stop().force("x", forceX(0).strength(0));
      await sim.tickAsync(2);
      nodes[5].x = 1000, nodes[5].y = -1000, nodes[5].z = 500;
      await sim.tickAsync(1);
      assert.deepStrictEqual([nodes[5].x, nodes[5].y, nodes[5].z], [1000, -1000, 500], "no forces: it stays where it was put");
      sim.destroy();
    });

    it("keeps a user's edit even if it lands while a readback is in flight", async function() {
      var nodes = makeNodes(30, 3), sim = make(nodes, 3).stop().force("x", forceX(0).strength(0));
      await sim.tickAsync(1);
      sim.tick(); // readback now in flight
      nodes[7].x = -777;
      await sim.sync();
      assert.strictEqual(nodes[7].x, -777, "not overwritten by the older GPU state");
      await sim.tickAsync(1);
      assert.strictEqual(nodes[7].x, -777);
      sim.destroy();
    });
  });

  describe("forces that cannot run on the GPU", function() {
    function gravity() {
      var nodes;
      function force(alpha) { for (var i = 0; i < nodes.length; ++i) nodes[i].vy -= 0.5 * alpha; }
      force.initialize = function(_) { nodes = _; };
      return force;
    }

    it("makes the simulation run on the CPU, with a warning", async function() {
      expectCPU = true;
      var nodes = makeNodes(20, 2), sim, warnings = await withWarnings(async function() {
        sim = make(nodes, 2).stop().force("gravity", gravity());
        assert.strictEqual(await sim.gpuReady(), false);
        await sim.tickAsync(3);
      });
      assert.strictEqual(sim.isGPUEnabled(), false);
      assert.strictEqual(warnings.length, 1);
      assert(/gravity/.test(warnings[0]), warnings[0]);
      assert(nodes.every(function(n) { return n.vy < 0; }), "the custom force ran");
    });

    it("moves to the GPU when a GPU-capable set of forces is restored (and back)", async function() {
      var nodes = makeNodes(40, 3), cpuNodes = makeNodes(40, 3), sim;
      await withWarnings(async function() {
        sim = make(nodes, 3).stop().force("charge", forceManyBody().theta(1e-6));
        var cpu = forceSimulation(cpuNodes, 3).stop().force("charge", forceManyBody().theta(1e-6));
        await sim.gpuReady();
        await sim.tickAsync(3), cpu.tick(3);
        assert.strictEqual(sim.isGPUEnabled(), true);

        sim.force("gravity", gravity());            // -> CPU (GPU state is handed over first)
        cpu.force("gravity", gravity());
        assert.strictEqual(sim.isGPUEnabled(), false);
        await sim.tickAsync(3), cpu.tick(3);

        sim.force("gravity", null);                 // -> GPU again, from the CPU-moved nodes
        cpu.force("gravity", null);
        assert.strictEqual(sim.isGPUEnabled(), true);
        await sim.tickAsync(3), cpu.tick(3);
        assertNodesClose(nodes, cpuNodes, 3, {rtol: 2e-3, atol: 1e-3});
      });
      sim.destroy();
    });
  });

  describe("edge cases", function() {
    it("resolves links by id, like forceLink on the CPU", async function() {
      var make2 = function(sim) {
            return sim.force("link", forceLink([{source: "a", target: "b"}, {source: "b", target: "c"}]).id(function(d) { return d.name; }).distance(20));
          },
          nodes = [{name: "a"}, {name: "b"}, {name: "c"}], cpuNodes = [{name: "a"}, {name: "b"}, {name: "c"}],
          sim = make2(make(nodes, 3).stop()), cpu = make2(forceSimulation(cpuNodes, 3).stop());
      await sim.tickAsync(200), cpu.tick(200);
      var dist = function(a, b) { return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z); };
      assert(Math.abs(dist(nodes[0], nodes[1]) - 20) < 1, "a-b is " + dist(nodes[0], nodes[1]));
      assert(Math.abs(dist(nodes[1], nodes[2]) - 20) < 1);
      assert(Math.abs(dist(nodes[0], nodes[1]) - dist(cpuNodes[0], cpuNodes[1])) < 1, "same length as on the CPU");
      sim.destroy();
    });

    it("runs a link force with no links", async function() {
      var nodes = makeNodes(10, 2), sim = make(nodes, 2).stop().force("link", forceLink([])).force("charge", forceManyBody());
      await sim.tickAsync(3);
      assert.strictEqual(sim.isGPUEnabled(), true);
      assert(nodes.every(function(n) { return isFinite(n.x) && isFinite(n.y); }));
      sim.destroy();
    });

    it("runs with no nodes, and with nodes added afterwards", async function() {
      var sim = make([], 3).stop().force("charge", forceManyBody()).force("center", forceCenter());
      assert.strictEqual(await sim.gpuReady(), true);
      await sim.tickAsync(3);
      var nodes = makeNodes(20, 3);
      sim.nodes(nodes);
      await sim.tickAsync(3);
      assert(nodes.every(function(n) { return isFinite(n.x) && isFinite(n.z); }));
      sim.nodes([]);
      await sim.tickAsync(1);
      sim.destroy();
    });

    it("falls back to the CPU, with a warning, when the nodes exceed the device's limits", async function() {
      var limits = gpu.device.limits, nodes = makeNodes(30, 2), sim;
      expectCPU = true;
      // pretend the device can only hold 10 nodes
      Object.defineProperty(gpu.device, "limits", {configurable: true, value: new Proxy(limits, {get: function(target, key) {
        return key === "maxStorageBufferBindingSize" ? 16 * 10 : target[key];
      }})});
      try {
        var warnings = await withWarnings(async function() {
          sim = make(nodes, 2).stop().force("charge", forceManyBody());
          assert.strictEqual(await sim.gpuReady(), false);
          sim.force("center", forceCenter()); // must not re-enable the GPU
          await sim.tickAsync(2);
        });
        assert.strictEqual(sim.isGPUEnabled(), false);
        assert(/exceed/.test(warnings.join()), warnings.join());
        assert(nodes.every(function(n) { return isFinite(n.x); }), "still simulated, on the CPU");
        assert.notStrictEqual(nodes[0].vx, 0);
      } finally {
        Object.defineProperty(gpu.device, "limits", {configurable: true, value: limits}); // the real ones again
      }
      sim.destroy();
    });

    it("keeps the passes of the forces that stay when another is added", async function() {
      var nodes = makeNodes(30, 3), cpuNodes = makeNodes(30, 3),
          links = function() { return Array.from({length: 15}, function(_, i) { return {source: 2 * i, target: 2 * i + 1}; }); }, // disjoint: exact parity
          sim = make(nodes, 3).stop().force("link", forceLink(links())).force("charge", forceManyBody().theta(1e-6)),
          cpu = forceSimulation(cpuNodes, 3).stop().force("link", forceLink(links())).force("charge", forceManyBody().theta(1e-6));
      await sim.tickAsync(2), cpu.tick(2);
      sim.force("center", forceCenter()), cpu.force("center", forceCenter()); // link and charge are kept
      await sim.tickAsync(2), cpu.tick(2);
      assertNodesClose(nodes, cpuNodes, 3, {rtol: 2e-3, atol: 1e-3});
      sim.destroy();
    });
  });

  it("exposes its GPU buffers for rendering", async function() {
    var nodes = makeNodes(25, 3), sim = make(nodes, 3).stop().force("charge", forceManyBody());
    assert.strictEqual(sim.gpuBuffers(), null, "not before the GPU is ready");
    await sim.gpuReady();
    await sim.tickAsync(2);
    var b = sim.gpuBuffers(), device = gpu.device;
    assert.strictEqual(b.count, 25);
    assert.strictEqual(b.stride, 16);

    var staging = device.createBuffer({size: 25 * 16, usage: 1 | 8});  // MAP_READ | COPY_DST
    var encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(b.positions, 0, staging, 0, 25 * 16);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(1);
    var p = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap(), staging.destroy();
    nodes.forEach(function(n, i) {
      assert.strictEqual(p[4 * i], n.x);
      assert.strictEqual(p[4 * i + 1], n.y);
      assert.strictEqual(p[4 * i + 2], n.z);
    });
    sim.destroy();
  });

  it("with readback: false, keeps the nodes untouched until sync()", async function() {
    var nodes = makeNodes(40, 3), start = nodes.map(function(n) { return n.x; }), ticks = 0,
        sim = forceSimulationGPU(nodes, 3, {device: gpu.device, readback: false})
            .force("charge", forceManyBody()).on("tick", function() { ++ticks; });
    await delay(300);
    assert(ticks > 2, "tick events still fire (" + ticks + ")");
    assert.deepStrictEqual(nodes.map(function(n) { return n.x; }), start, "JS nodes are not updated");
    sim.stop();
    await sim.sync();
    assert(nodes.some(function(n, i) { return n.x !== start[i]; }), "sync() brings them up to date");
    sim.destroy();
  });

  it("is deterministic for a given randomSource (coincident nodes)", async function() {
    var run = async function() {
      var nodes = Array.from({length: 20}, function() { return {x: 0, y: 0, z: 0}; }),
          sim = make(nodes, 3).stop().force("charge", forceManyBody()).force("collide", forceCollide(3));
      await sim.tickAsync(10);
      sim.destroy();
      return nodes.map(function(n) { return [n.x, n.y, n.z]; });
    };
    assert.deepStrictEqual(await run(), await run());
  });

  it("runs every tick() of a synchronous batch (e.g. a warm-up loop)", async function() {
    var nodes = makeNodes(40, 3), cpuNodes = makeNodes(40, 3),
        sim = make(nodes, 3).stop().force("charge", forceManyBody().theta(1e-6)),
        cpu = forceSimulation(cpuNodes, 3).stop().force("charge", forceManyBody().theta(1e-6));
    await sim.gpuReady();
    for (var i = 0; i < 100; ++i) sim.tick(), cpu.tick();
    assert.strictEqual(sim.alpha(), cpu.alpha(), "all 100 ticks were taken");
    await sim.sync();
    assertNodesClose(nodes, cpuNodes, 3, {rtol: 5e-3, atol: 5e-3});
    sim.destroy();
  });

  it("takes a tick() per task even when calls are spread over many tasks", async function() {
    var nodes = makeNodes(40, 3), sim = make(nodes, 3).stop().force("charge", forceManyBody());
    await sim.gpuReady();
    for (var i = 0; i < 20; ++i) sim.tick(), await delay(25); // about a frame
    // The GPU is idle between calls, so nothing is skipped.
    assert(Math.abs(sim.alpha() - Math.pow(1 - sim.alphaDecay(), 20)) < 1e-12, "alpha " + sim.alpha());
    sim.destroy();
  });

  it("destroy() stops everything and is safe to call more than once", async function() {
    var nodes = makeNodes(20, 2), sim = make(nodes, 2).force("charge", forceManyBody());
    await sim.gpuReady();
    sim.destroy();
    assert.strictEqual(sim.isGPUEnabled(), false);
    sim.destroy();
    await delay(60);
    assert.doesNotThrow(function() { sim.tick(); });
  });
});
