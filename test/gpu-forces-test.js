import assert from "assert";
import {forceCenter, forceCollide, forceLink, forceManyBody, forceRadial, forceSimulation, forceSimulationGPU, forceX, forceY, forceZ} from "../src/index.js";
import {assertNodesClose, makeNodes, runBoth, treeOutliers, useDevice} from "./gpu-helpers.js";

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

  describe("the all-pairs kernels, however many threads share a node", function() {
    // 1 = one thread per node; 128 = a whole workgroup per node; 3 and 1000 are rounded
    // down to a power of two and to 128.
    [1, 2, 4, 8, 16, 32, 64, 128, 3, 1000].forEach(function(split) {
      it("forceManyBody matches the CPU with split " + split, async function() {
        var r = await runBoth(gpu.device, {dims: 3, n: 300, ticks: 2, split: split, makeForces: function() {
          return [["charge", forceManyBody().theta(1e-6).strength(function(d, i) { return i % 5 ? -30 : 20; }).distanceMin(4)]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, 3, {rtol: 1e-3});
      });
    });

    [1, 8, 128].forEach(function(split) {
      it("forceCollide matches the CPU with split " + split, async function() {
        var pairs = function(n) {
          var nodes = [];
          for (var i = 0; i < n; i += 2) nodes.push({x: i * 20, y: i % 7, z: 0}, {x: i * 20 + 3 + i % 5, y: i % 7, z: 0});
          return nodes;
        };
        var r = await runBoth(gpu.device, {dims: 3, n: 200, ticks: 3, split: split, makeNodes: pairs, makeForces: function() {
          return [["collide", forceCollide(function(d, i) { return 6 + i % 3; }).strength(0.9)]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, 3, {rtol: 1e-3});
      });
    });

    // Sizes around the workgroup width, where a partial last workgroup and node counts
    // that are not a multiple of the number of nodes per workgroup are easy to get wrong.
    [1, 2, 3, 7, 8, 9, 127, 128, 129, 255, 257].forEach(function(n) {
      it("forceManyBody with the automatic split matches the CPU for " + n + " node(s)", async function() {
        var r = await runBoth(gpu.device, {dims: 3, n: n, ticks: 2, makeForces: function() {
          return [["charge", forceManyBody().theta(1e-6)]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, 3, {rtol: 1e-3, atol: 1e-3});
      });
    });
  });

  describe("forceManyBody with theta > 0: the same Barnes-Hut tree as the CPU", function() {
    // d3-force-3d scales a cell's strength by sqrt(4 / numChildren) per level, so theta
    // shapes the layout (3D layouts are more compact than with the exact sum). The GPU tree
    // reproduces that. 1D amplifies every difference (x1.41 per level), so it gets more slack.
    // A few nodes may differ more: see treeOutliers.
    [[1, 2e-2], [2, 1e-3], [3, 1e-3]].forEach(function(c) {
      it(c[0] + "D, 500 nodes", async function() {
        var r = await runBoth(gpu.device, {dims: c[0], n: 500, ticks: 1, makeForces: function() {
          return [["charge", forceManyBody()]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, c[0], {rtol: c[1], outliers: treeOutliers(500)});
      });
    });

    [0.5, 1.5].forEach(function(theta) {
      it("3D, theta " + theta, async function() {
        var r = await runBoth(gpu.device, {dims: 3, n: 2000, ticks: 1, makeForces: function() {
          return [["charge", forceManyBody().theta(theta)]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, 3, {rtol: 1e-3, outliers: treeOutliers(2000)});
      });
    });

    it("with per-node strength, distanceMin and distanceMax", async function() {
      var r = await runBoth(gpu.device, {dims: 3, n: 2000, ticks: 1, makeForces: function() {
        return [["charge", forceManyBody().strength(function(d, i) { return -10 - i % 7; }).distanceMin(3).distanceMax(60)]];
      }});
      assertNodesClose(r.gpuNodes, r.cpuNodes, 3, {rtol: 1e-3, outliers: treeOutliers(2000)});
    });

    it("with strengths of both signs (and cells where they cancel out)", async function() {
      for (var dims = 2; dims <= 3; ++dims) {
        var r = await runBoth(gpu.device, {dims: dims, n: 1000, ticks: 1, makeForces: function() {
          return [["charge", forceManyBody().strength(function(d, i) { return i % 3 ? -30 : 20; })]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, dims, {rtol: 1e-3, outliers: treeOutliers(1000)});
      }
    });

    it("with outliers far away (a huge cube)", async function() {
      var r = await runBoth(gpu.device, {dims: 3, n: 1000, ticks: 1, makeNodes: function(n, d) {
        var nodes = makeNodes(n, d);
        nodes[0].x = 1e5, nodes[1].y = -2e5;
        return nodes;
      }, makeForces: function() {
        return [["charge", forceManyBody()]];
      }});
      assertNodesClose(r.gpuNodes, r.cpuNodes, 3, {rtol: 1e-3, outliers: treeOutliers(1000)});
    });

    it("far from the origin", async function() {
      // f32 positions near 5e4 are 0.004 apart, and so are centres of mass: enough to tip a
      // few theta tests the other way than in f64, so a few forces differ by a few percent
      var r = await runBoth(gpu.device, {dims: 3, n: 1000, ticks: 1, makeNodes: function(n, d) {
        return makeNodes(n, d).map(function(node) {
          node.x = Math.fround(node.x + 5e4), node.y = Math.fround(node.y - 3e4), node.z = Math.fround(node.z + 1e4);
          return node;
        });
      }, makeForces: function() {
        return [["charge", forceManyBody()]];
      }});
      assertNodesClose(r.gpuNodes, r.cpuNodes, 3, {rtol: 1e-2, outliers: treeOutliers(1000)});
    });

    [2, 3, 1025].forEach(function(n) {
      it("with " + n + " nodes", async function() {
        var r = await runBoth(gpu.device, {dims: 3, n: n, ticks: 2, makeForces: function() {
          return [["charge", forceManyBody()]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, 3, {rtol: 1e-3, outliers: treeOutliers(n)});
      });
    });

    // around the sizes where the sort (8192) and the walk (32768) change method
    [8192, 8193, 32769].forEach(function(n) {
      it("with " + n + " nodes", async function() {
        var r = await runBoth(gpu.device, {dims: 3, n: n, ticks: 1, makeNodes: function(n, d) {
          return makeNodes(n, d, 20 * Math.cbrt(n));
        }, makeForces: function() {
          return [["charge", forceManyBody()]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, 3, {rtol: 1e-3, outliers: treeOutliers(n)});
      });
    });

    // Nodes that share their cells down to half the tree's depth sort by the rest of their
    // codes: in runs of up to 32 nodes, or with a sort of their own past that. A clump of
    // side 2 and an outlier: at 120, the clump spans thousands of those cells; at 2000, a
    // few (2D) or one (3D).
    [[2, 120, "runs of a few nodes"], [3, 120, "runs of a few nodes"], [2, 2000, "long runs"], [3, 2000, "one run"]].forEach(function(c) {
      it(c[0] + "D, 9000 nodes sharing the tree's top levels: " + c[2], async function() {
        var r = await runBoth(gpu.device, {dims: c[0], n: 9000, ticks: 1, makeNodes: function(n, d) {
          var nodes = makeNodes(n, d, 2);
          nodes[0].x = c[1];
          return nodes;
        }, makeForces: function() {
          return [["charge", forceManyBody()]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, c[0], {rtol: 1e-3, outliers: treeOutliers(9000)});
      });
    });

    it("over 20 ticks", async function() {
      var r = await runBoth(gpu.device, {dims: 3, n: 1000, ticks: 20, makeForces: function() {
        return [["charge", forceManyBody()], ["center", forceCenter()]];
      }});
      assertNodesClose(r.gpuNodes, r.cpuNodes, 3, {rtol: 1e-3, outliers: treeOutliers(1000)});
    });

    it("gives 3D layouts the size of the CPU's", async function() {
      // A whole layout diverges node by node (the link force runs in another order on the
      // GPU), but its size must not: the exact sum makes it ~1.5x larger.
      var size = async function(sim, nodes) {
        var s = 42, links = [];
        function rnd() { return (s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296; }
        for (var i = 1; i < nodes.length; ++i) links.push({source: i, target: Math.floor(rnd() * i)});
        sim.force("link", forceLink(links).distance(30)).force("charge", forceManyBody()).force("center", forceCenter());
        if (sim.tickAsync) await sim.tickAsync(300); else sim.tick(300);
        return Math.sqrt(nodes.reduce(function(sum, d) { return sum + d.x * d.x + d.y * d.y + d.z * d.z; }, 0) / nodes.length);
      };
      var cpuNodes = makeNodes(500, 3), gpuNodes = makeNodes(500, 3),
          cpu = await size(forceSimulation(cpuNodes, 3).stop(), cpuNodes),
          sim = forceSimulationGPU(gpuNodes, 3, {device: gpu.device}).stop(),
          ratio = await size(sim, gpuNodes) / cpu;
      assert(sim.isGPUEnabled(), "ran on the GPU");
      sim.destroy();
      assert(Math.abs(ratio - 1) < 0.03, "GPU layout / CPU layout size: " + ratio);
    });
  });

  describe("forceCollide on large graphs (a grid finds the pairs)", function() {
    [1, 2, 3].forEach(function(dims) {
      it(dims + "D, 10000 nodes in separated pairs", async function() {
        var pairs = function(n, d) {
          var nodes = [], side = Math.ceil(Math.pow(n / 2, 1 / d));
          for (var i = 0; i < n; i += 2) {
            var k = i / 2, a = {x: (k % side) * 40}, b = {x: (k % side) * 40 + 3 + i % 5};
            if (d > 1) a.y = b.y = Math.floor(k / side) % side * 40 + i % 7;
            if (d > 2) a.z = b.z = Math.floor(k / side / side) * 40;
            nodes.push(a, b);
          }
          return nodes;
        };
        var r = await runBoth(gpu.device, {dims: dims, n: 10000, ticks: 3, makeNodes: pairs, makeForces: function() {
          return [["collide", forceCollide(function(d, i) { return 6 + i % 3; }).strength(0.9)]];
        }});
        assertNodesClose(r.gpuNodes, r.cpuNodes, dims, {rtol: 1e-3});
      });
    });

    it("pushes apart overlapping and coincident nodes", async function() {
      var clump = function(n) {
        var nodes = makeNodes(n, 3, 60);
        for (var i = 0; i < 50; ++i) nodes[i].x = nodes[i].y = nodes[i].z = 0;
        return nodes;
      };
      var r = await runBoth(gpu.device, {dims: 3, n: 10000, ticks: 1, makeNodes: clump, makeForces: function() {
        return [["collide", forceCollide(2)]];
      }});
      var closest = Infinity;
      for (var i = 0; i < 50; ++i) for (var j = i + 1; j < 50; ++j) {
        var a = r.gpuNodes[i], b = r.gpuNodes[j];
        closest = Math.min(closest, Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z));
      }
      assert(closest > 0, "coincident nodes separated");
      r.gpuNodes.forEach(function(d, i) { assert(isFinite(d.x + d.y + d.z), "node " + i + " finite"); });
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
