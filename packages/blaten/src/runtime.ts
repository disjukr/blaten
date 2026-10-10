import type { KernelFunction } from './kernel.js';
import type { Shape, DType, Scalar } from './types.js';
import type { Tensor } from './tensor.js';
export interface Executable {
  run(...inputs: (Tensor<Shape, DType> | Scalar)[]): Tensor<Shape, DType> | Tensor<Shape, DType>[];
  dispose(): void;
}
export type TensorOrScalar = Tensor<Shape, DType> | Scalar;
export interface CompileOptions { readonly inputs: readonly TensorOrScalar[] }
export interface Backend {
  readonly capabilities: { readonly dtypes: readonly DType[] };
  compile(fn: KernelFunction, options?: CompileOptions): Promise<Executable>;
  /** Copy into this backend's environment, preserving shape/dtype and source ownership. */
  copyFrom<S extends Shape, D extends DType>(tensor: Tensor<S, D>): Promise<Tensor<S, D>>;
  synchronize(): Promise<void>;
  dispose(): void;
}
export function compile(fn: KernelFunction, backend: Backend, options?: CompileOptions): Promise<Executable> {
  return backend.compile(fn, options);
}
