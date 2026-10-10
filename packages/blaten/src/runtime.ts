import type { KernelFunction } from './kernel.js';
import type { Shape, DType, Scalar } from './types.js';
import type { Tensor } from './tensor.js';
export interface Executable {
  readonly dispatchCount: number;
  run(...inputs: (Tensor<Shape, DType> | Scalar)[]): Tensor<Shape, DType> | Tensor<Shape, DType>[];
  dispose(): void;
}
export interface Backend {
  compile(fn: KernelFunction): Promise<Executable>;
}
export function compile(fn: KernelFunction, backend: Backend): Promise<Executable> {
  return backend.compile(fn);
}
