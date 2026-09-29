import assert from "assert";
import {forceCenter, forceCollide, forceLink, forceManyBody, forceRadial, forceX, forceY, forceZ} from "../src/index.js";
import {assertNodesClose, makeNodes, runBoth, useDevice} from "./gpu-helpers.js";

// Each GPU force is compared with the CPU force on identical input. The GPU works in
// f32, so agreement is to ~1e-4 relative, not bit-for-bit.

describe("WebGPU forces match the CPU forces", function() {
  this.timeout(30000);
  var gpu = useDevice();

  [1, 2, 3].forEach(function(dims) {
    describe(dims + "D", function() {
      it("forceManyBody (all pairs, several workgroups)", async function() {
        var r = await runBoth(gpu.device, {dims: dims, n: 300, ticks: 1, makeForces: function() {
          // A tiny theta makes the CPU's Barnes-Hut tree evaluate every pair exactly.
          return [["charge", forceManyBody().theta(1e-6)]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, dims);
      });

      it("forceManyBody with per-node strength, distanceMin and distanceMax", async function() {
        var r = await runBoth(gpu.device, {dims: dims, n: 200, ticks: 3, makeForces: function() {
          return [["charge", forceManyBody().theta(1e-6)
              .strength(function(d, i) { return i % 3 ? -20 - i % 7 : 15; })
              .distanceMin(5).distanceMax(150)]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, dims, {rtol: 1e-3});
      });

      it("forceCenter", async function() {
        var r = await runBoth(gpu.device, {dims: dims, n: 1000, ticks: 2, makeNodes: function(n, d) {
          // off-centre cloud, more than 256 nodes so the reduction spans workgroups
          return makeNodes(n, d, 100).map(function(node) {
            node.x = Math.fround(node.x + 40); if (d > 1) node.y = Math.fround(node.y - 25); if (d > 2) node.z = Math.fround(node.z + 10);
            return node;
          });
        }, makeForces: function() {
          return [["center", forceCenter(3, dims > 1 ? 4 : 0, dims > 2 ? 5 : 0).strength(0.7)]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, dims, {rtol: 1e-3});
      });

      it("forceRadial", async function() {
        var r = await runBoth(gpu.device, {dims: dims, n: 200, ticks: 3, makeForces: function() {
          return [["radial", forceRadial(function(d, i) { return 30 + i % 5 * 10; }, 5, dims > 1 ? -5 : 0, dims > 2 ? 2 : 0).strength(0.3)]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, dims, {rtol: 1e-3});
      });

      it("forceX / forceY / forceZ", async function() {
        var r = await runBoth(gpu.device, {dims: dims, n: 200, ticks: 3, makeForces: function() {
          var forces = [["x", forceX(function(d, i) { return i % 10; }).strength(function(d, i) { return i % 4 ? 0.2 : 0.05; })]];
          if (dims > 1) forces.push(["y", forceY(-20).strength(0.3)]);
          if (dims > 2) forces.push(["z", forceZ(15).strength(0.1)]);
          return forces;
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, dims, {rtol: 1e-3});
      });

      it("forceLink on disjoint links (no node shared, so order cannot matter)", async function() {
        var r = await runBoth(gpu.device, {dims: dims, n: 200, ticks: 5, makeForces: function() {
          var links = [];
          for (var i = 0; i < 200; i += 2) links.push({source: i, target: i + 1});
          return [["link", forceLink(links).distance(function(l, i) { return 10 + i % 5; }).strength(0.8)]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, dims, {rtol: 1e-3});
      });

      it("forceLink with several iterations on disjoint links", async function() {
        var r = await runBoth(gpu.device, {dims: dims, n: 100, ticks: 3, makeForces: function() {
          var links = [];
          for (var i = 0; i < 100; i += 2) links.push({source: i, target: i + 1});
          return [["link", forceLink(links).distance(25).strength(1).iterations(3)]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, dims, {rtol: 1e-3});
      });

      it("forceCollide on separated pairs (no node touches more than one other)", async function() {
        // pairs of overlapping nodes, far apart from each other
        var pairs = function(n, d) {
          var nodes = [];
          for (var i = 0; i < n; i += 2) {
            var a = {x: i * 20}, b = {x: i * 20 + 3 + (i % 5)};
            if (d > 1) a.y = b.y = (i % 7) * 3;
            if (d > 2) a.z = b.z = 0;
            nodes.push(a, b);
          }
          return nodes;
        };
        var r = await runBoth(gpu.device, {dims: dims, n: 100, ticks: 4, makeNodes: pairs, makeForces: function() {
          return [["collide", forceCollide(function(d, i) { return 6 + i % 3; }).strength(0.9)]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, dims, {rtol: 1e-3});
      });
    });
  });

  it("combines forces in the order they were added", async function() {
    // center moves positions, x/charge move velocities: the order is observable
    var r = await runBoth(gpu.device, {dims: 3, n: 250, ticks: 5, makeNodes: function(n, d) {
      return makeNodes(n, d, 120).map(function(node) { node.x = Math.fround(node.x + 30); return node; });
    }, makeForces: function() {
      return [
        ["charge", forceManyBody().theta(1e-6).strength(-15)],
        ["x", forceX(10).strength(0.05)],
        ["center", forceCenter()],
        ["radial", forceRadial(60).strength(0.02)]
      ];
    }});
    assertNodesClose(r.gpuNodes, r.cpuNodes, 3, {rtol: 2e-3, atol: 1e-3});
  });

  it("honours fixed positions (fx, fy, fz)", async function() {
    var r = await runBoth(gpu.device, {dims: 3, n: 120, ticks: 5, setup: function(sim, nodes) {
      nodes[0].fx = 5, nodes[0].fy = -6, nodes[0].fz = 7;
      nodes[1].fx = 100; // only x pinned
    }, makeForces: function() {
      return [["charge", forceManyBody().theta(1e-6)], ["center", forceCenter()]];
    }});
    assert.strictEqual(r.gpuNodes[0].x, 5);
    assert.strictEqual(r.gpuNodes[0].y, -6);
    assert.strictEqual(r.gpuNodes[0].z, 7);
    assert.strictEqual(r.gpuNodes[0].vx, 0);
    assert.strictEqual(r.gpuNodes[1].x, 100);
    assertNodesClose(r.gpuNodes, r.cpuNodes, 3, {rtol: 2e-3, atol: 1e-3});
  });

  it("separates coincident nodes", async function() {
    var coincident = function(n) {
      var nodes = [];
      for (var i = 0; i < n; ++i) nodes.push({x: 1, y: 2, z: 3});
      return nodes;
    };
    var r = await runBoth(gpu.device, {dims: 3, n: 10, ticks: 30, makeNodes: coincident, makeForces: function() {
      return [["charge", forceManyBody()], ["collide", forceCollide(2)]];
    }});
    r.gpuNodes.forEach(function(a, i) {
      assert(isFinite(a.x) && isFinite(a.y) && isFinite(a.z), "node " + i + " is finite");
      r.gpuNodes.slice(i + 1).forEach(function(b) {
        assert(Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) > 0.5, "nodes should have been pushed apart");
      });
    });
  });
});
