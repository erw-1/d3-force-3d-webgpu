# GPU vs CPU: what's different

`forceSimulationGPU` runs the same forces as `forceSimulation`. Every force is tested against its CPU twin on the same input, in 1D, 2D and 3D. They agree to about 1e-4 (that's the limit of 32-bit floats).

Some things are different on purpose. Here they are.

## Nodes update asynchronously

A browser can't wait for the GPU synchronously. So `simulation.tick()` doesn't leave your nodes up to date.

* **Timer-driven** (the default): nothing changes for you. `"tick"` fires once the new positions are in your nodes.
* **`tick()` by hand**: the steps go to the GPU, and the nodes follow about a frame later.
  * One `tick()` per frame is fine. If the GPU is still busy, that tick is skipped.
  * A synchronous loop of `tick()` calls (a warm-up) is fine too. All of them run.
* **Run a layout, then read it**: `await simulation.tickAsync(300)`.
* **`readback: false`**: skips the copy on every tick. Use it when you draw from `gpuBuffers()`. Call `sync()` when you need the nodes.

Edits you make while it runs are picked up: pin with `fx`/`fy`/`fz` (dragging), or set `x`/`y`/`z`. Velocities are copied back to `vx`/`vy`/`vz`, but the GPU owns them. Writing them does nothing.

## Many-body: the same tree as the CPU

The CPU force approximates far-away nodes with a Barnes-Hut tree (d3-octree in 3D, d3-quadtree in 2D, d3-binarytree in 1D). The GPU builds that same tree every tick, and walks it the same way.

That matters in 3D. d3-force-3d scales the strength of every aggregated cell by `sqrt(4 / numChildren)`: 0.71 per octree level. So far-field repulsion is weaker than the exact sum, and layouts get more compact as `theta` grows. The GPU reproduces it. RMS radius of a 1,000-node random graph after 300 ticks:

| | `theta` 0.9 (default) | 0.5 | 0.2 | 0 (exact) |
|---|---:|---:|---:|---:|
| 2D, CPU | 552 | 552 | 552 | 552 |
| 2D, GPU | 555 | 555 | 554 | 554 |
| 3D, CPU | 372 | 418 | 499 | 548 |
| 3D, GPU | 374 | 422 | 503 | 551 |

How it's built, per tick: a bounding box, the tree's cube (d3's: `floor` of the minimum, doubled until it holds the maximum), a 64-bit Morton code per node, a radix sort, a binary radix tree over the codes (Karras 2012), strengths and centres of mass per cell, then one walk down the tree per node. Each tree node stands for the chain of d3 cells between two splits; the walk tests each of them, widest first, exactly like d3 does.

Where it can differ:

* **Very close nodes.** The codes have 20 levels in 3D, 24 in 1D and 2D. Nodes closer than the cube's size / 2^20 share a cell here where d3 would split further. They still push each other exactly. Their strength seen from afar can differ a little.
* **Coincident nodes.** d3 pushes a group of coincident nodes apart in one random direction; the GPU picks one per pair.
* **Positive and negative strengths that cancel out exactly.** d3 skips a cell whose strengths add up to 0. Whether a sum lands on exactly 0 depends on 64-bit rounding; the GPU treats a sum that is zero to f32 precision as zero.
* **1D** amplifies aggregated cells by 1.41 per level, so any f32 difference grows: a few forces can be ~1% off.
* **Far from the origin**, f32 positions are coarse (0.004 apart near 50,000), and a few `theta` tests can tip the other way.

With `theta(0)` the GPU sums every pair exactly, like the CPU with `theta(0)`, but fast: that's the quickest option below ~5,000 nodes.

## Link and collide resolve all at once

The CPU visits links and pairs one after another. Later ones see the velocity changes of earlier ones. The GPU can't. Each node computes its response against a snapshot of positions + velocities.

* Each link or pair pushes exactly as much as on the CPU.
* Links and pairs that share no node match the CPU (to f32 precision).
* With shared nodes, layouts converge to the same quality. In 2D, mean link length of a 1,000-node random graph after 300 ticks: 191 (CPU), 192 (GPU).
* From 8,192 nodes, pairs are found with a grid (cells twice the largest radius wide). Same pairs, same result.
* Dense collisions are jitterier. Resolving every contact at once overshoots more than resolving them in sequence.
  * Moderate density: same residual speed as the CPU from `iterations(2)`.
  * Jammed packings: still 2-4× higher at any `iterations`, with a similar amount of overlap.
  * Averaging the pushes over contacts removes the jitter but leaves noticeably more overlap. So the exact pairwise rule stayed.

## f32, not f64

Positions and velocities are 32-bit floats on the GPU. Coincident nodes are pushed apart with a hash-based jiggle. `randomSource` seeds it, so a given seed is reproducible, but it isn't the CPU's sequence.

## Forces on the GPU: `force.gpu()`

A force runs on the GPU if it has a `force.gpu()` method. It returns `{type, version, ...}`:

* `type` picks the shader (`manyBody`, `link`, `collide`, `center`, `radial`, `position`).
* `version` changes whenever the force recomputes its per-node values. The GPU re-uploads then.

Every force in this package has one. A custom force written as a plain function doesn't, so the whole simulation runs on the CPU (with a warning that names it).

## Not done yet

* **Sequential collide.** A coloured sweep over the grid would remove the jitter of dense packings.
* **Custom CPU forces next to GPU ones.** Possible by round-tripping through the CPU each tick.

Feel free to open an issue if one of these matters to you.
