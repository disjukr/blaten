// Core facilities shared by backend implementations; no GPU dependencies.
export { lower } from './lowering.js';
export type { Program } from './lowering.js';
export { topological } from './ir.js';
export type { Node, Op } from './ir.js';
export { sizeOf, isHostStorage, isFloat, makeStorage } from './types.js';
export type { Shape, Signature, DType, HostStorage, Scalar } from './types.js';
export type { KernelFunction } from './kernel.js';
export type { Backend, Executable } from './runtime.js';
export type { DeviceStorage } from './tensor.js';

export { signatureOf } from './kernel.js';
