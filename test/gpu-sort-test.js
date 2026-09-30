import assert from "assert";
import {pipeline, layout0} from "../src/gpu/shared.js";
import {SIM_BYTES} from "../src/gpu/constants.js";
import {pipelines, createSort, SORT_TILE} from "../src/gpu/sort.js";
import {useDevice} from "./gpu-helpers.js";

// The radix sort on its own: sorted by key, and stable (equal keys keep their order).

async function gpuSort(device, keys, passes, firstBit) {
  var n = keys.length,
      engine = {device: device, pipeline: function(id, c) { return pipeline(device, id, pipelines[id], c); }},
      sorter = createSort(engine, n, Array.from({length: passes}, function(_, p) { return firstBit + 8 * p; })),
      data = new Uint32Array(2 * n),
      sim = device.createBuffer({size: 256, usage: 0x40 | 0x8}),
      state = device.createBuffer({size: 16, usage: 0x80}),
      state2 = device.createBuffer({size: 16, usage: 0x80}),
      readback = device.createBuffer({size: 8 * n, usage: 0x1 | 0x8}),
      group0 = device.createBindGroup({layout: layout0(device), entries: [
        {binding: 0, resource: {buffer: sim, size: SIM_BYTES}},
        {binding: 1, resource: {buffer: state}},
        {binding: 2, resource: {buffer: state2}}
      ]});

  keys.forEach(function(key, i) { data[2 * i] = key, data[2 * i + 1] = i; });
  device.queue.writeBuffer(sorter.pairs[0], 0, data);
  var encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setBindGroup(0, group0, [0]);
  sorter.encode(pass);
  pass.end();
  encoder.copyBufferToBuffer(sorter.result, 0, readback, 0, 8 * n);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(0x1);
  var out = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  [readback, sim, state, state2].forEach(function(b) { b.destroy(); });
  sorter.destroy();
  return out;
}

function expected(keys, passes, firstBit) {
  var mask = passes * 8 + firstBit >= 32 ? 0xffffffff : (2 ** (passes * 8 + firstBit)) - 1;
  return keys.map(function(key, i) { return [key, i]; }).sort(function(a, b) {
    var ka = (a[0] & mask) >>> firstBit, kb = (b[0] & mask) >>> firstBit;
    return ka - kb || a[1] - b[1];
  });
}

describe("the GPU radix sort", function() {
  this.timeout(30000);
  var gpu = useDevice();

  function random(n, spread, seed) {
    var s = seed || 7, keys = new Array(n);
    for (var i = 0; i < n; ++i) keys[i] = (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) % spread;
    return keys;
  }

  [1, 2, 3, 255, 256, 257, SORT_TILE - 1, SORT_TILE, SORT_TILE + 1, 3 * SORT_TILE + 17, 100000].forEach(function(n) {
    it("sorts " + n + " random 32-bit keys, stably", async function() {
      var keys = random(n, 4294967296), out = await gpuSort(gpu.device, keys, 4, 0);
      expected(keys, 4, 0).forEach(function(pair, i) {
        assert.strictEqual(out[2 * i], pair[0], "key at " + i);
        assert.strictEqual(out[2 * i + 1], pair[1], "value at " + i);
      });
    });
  });

  it("keeps equal keys in their original order", async function() {
    var keys = random(20000, 37), out = await gpuSort(gpu.device, keys, 4, 0);
    expected(keys, 4, 0).forEach(function(pair, i) {
      assert.strictEqual(out[2 * i + 1], pair[1], "value at " + i);
    });
  });

  it("sorts on the requested digits only", async function() {
    var keys = random(5000, 4294967296), out = await gpuSort(gpu.device, keys, 2, 8);
    expected(keys, 2, 8).forEach(function(pair, i) {
      assert.strictEqual(out[2 * i + 1], pair[1], "value at " + i);
    });
  });
});
