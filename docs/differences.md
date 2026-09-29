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

## Many-body is exact, `theta` is ignored

The CPU force uses a Barnes-Hut tree. The GPU force sums every pair.

* **1D, 2D**: the tree is a close approximation of the exact sum. Results match.
* **3D**: it's different. d3-force-3d's octree scales the strength of every aggregated cell by `sqrt(4 / numChildren)`. That's 0.71 per octree level. So far-field repulsion is deliberately weaker than the exact sum, and the layout gets more compact as `theta` grows.

RMS radius of a 1,000-node random graph after 300 ticks:

| | `theta` 0.9 (default) | 0.5 | 0.2 | ≈ 0 (exact) | GPU |
|---|---:|---:|---:|---:|---:|
| 2D | 536 | 536 | 536 | 536 | 536 |
| 3D | 377 | 425 | 506 | 551 | 553 |

The GPU matches the CPU's exact result. So **at default settings, 3D layouts come out larger on the GPU**: about 1.5× at 1,000 nodes, more as the graph grows. Scale `forceManyBody().strength()` down to compensate, or take it as the physically exact answer.

## Link and collide resolve all at once

The CPU visits links and pairs one after another. Later ones see the velocity changes of earlier ones. The GPU can't. Each node computes its response against a snapshot of positions + velocities.

* Each link or pair pushes exactly as much as on the CPU.
* Links and pairs that share no node match the CPU (to f32 precision).
* With shared nodes, layouts converge to the same quality. In 2D, mean link length of a 1,000-node random graph after 300 ticks: 191 (CPU), 192 (GPU).
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

* **GPU Barnes-Hut** that reproduces d3-octree, including the damping. It would make 3D layouts match the CPU, and cut many-body from O(n²) to about O(n log n).
* **Spatial hash for collide.** Fixes the O(n²) cost and, with a coloured sweep, the jitter.
* **Custom CPU forces next to GPU ones.** Possible by round-tripping through the CPU each tick.

Feel free to open an issue if one of these matters to you.
