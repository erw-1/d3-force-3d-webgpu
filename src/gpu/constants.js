// Numeric values of the WebGPU flag enums, so the library does not depend on the
// GPUBufferUsage / GPUShaderStage / GPUMapMode globals (absent in Node, for instance).

export var MAP_READ = 0x0001,
    COPY_SRC = 0x0004,
    COPY_DST = 0x0008,
    UNIFORM = 0x0040,
    STORAGE = 0x0080;

export var COMPUTE = 0x4;

export var MAP_MODE_READ = 0x1;

// Threads per workgroup for the one-thread-per-node kernels.
export var WORKGROUP = 64;

// Size of the per-step uniform block (see wgsl.js: struct Sim).
export var SIM_BYTES = 32;
