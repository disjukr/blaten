import { f32, bool, i64, sizeOf, isDType, isHostStorage, storageDType, makeStorage, promoteTypes, scalarDType, type DType, type Shape, type HostStorage, type Scalar, type ScalarOf } from './types.js';
import { trackTensor } from './scope.js';
import * as intrinsics from './intrinsics.js';
import type { Operand, OperandShape, BinaryDType, Broadcast, Floating, Summed, MatmulShape, Transposed, Reduced, Promote } from './tensor-types.js';

export interface DeviceStorage<D extends DType = DType> {
  readonly owner: object;
  readonly buffer: unknown;
  read(): Promise<HostStorage<D>>;
  dispose(): void;
}
const storages = new WeakMap<Tensor, HostStorage | DeviceStorage>();
/** Backend-only access; storage is deliberately absent from Tensor's public API. */
export function storageOf<D extends DType>(tensor: Tensor<Shape, D>): HostStorage<D> | DeviceStorage<D> {
  tensor.assertAlive();
  const storage = storages.get(tensor);
  if (!storage) throw new Error('Unknown tensor storage');
  return storage as HostStorage<D> | DeviceStorage<D>;
}
export function hostData<D extends DType>(tensor: Tensor<Shape, D>): HostStorage<D> {
  const storage = storageOf(tensor);
  if (!isHostStorage(storage)) throw new Error('Tensor is not host-resident; use read() to obtain host values');
  return storage;
}
export function deviceStorageOf<D extends DType>(tensor: Tensor<Shape, D>): DeviceStorage<D> {
  const storage = storageOf(tensor);
  if (isHostStorage(storage)) throw new Error('Tensor is host-resident');
  return storage;
}
type Nested = Scalar | readonly Nested[];
type NestedOf<S extends Scalar> = S | readonly NestedOf<S>[];
export type TensorBool<S extends Shape = Shape> = Tensor<S, 'bool'>;
export type TensorI8<S extends Shape = Shape> = Tensor<S, 'i8'>;
export type TensorU8<S extends Shape = Shape> = Tensor<S, 'u8'>;
export type TensorI16<S extends Shape = Shape> = Tensor<S, 'i16'>;
export type TensorU16<S extends Shape = Shape> = Tensor<S, 'u16'>;
export type TensorI32<S extends Shape = Shape> = Tensor<S, 'i32'>;
export type TensorU32<S extends Shape = Shape> = Tensor<S, 'u32'>;
export type TensorI64<S extends Shape = Shape> = Tensor<S, 'i64'>;
export type TensorU64<S extends Shape = Shape> = Tensor<S, 'u64'>;
export type TensorF16<S extends Shape = Shape> = Tensor<S, 'f16'>;
export type TensorF32<S extends Shape = Shape> = Tensor<S, 'f32'>;
export type TensorF64<S extends Shape = Shape> = Tensor<S, 'f64'>;
export class Tensor<S extends Shape = Shape, D extends DType = DType> {
  readonly shape: Readonly<S>;
  readonly strides: readonly number[];
  readonly size: number;
  private disposed = false;
  constructor(storage: HostStorage<D> | DeviceStorage<D>, shape: S, readonly dtype: D = f32 as D) {
    if (!isDType(dtype)) throw new Error(`Unknown dtype ${dtype}`);
    this.size = sizeOf(shape);
    this.shape = Object.freeze([...shape]) as unknown as Readonly<S>;
    let stride = 1;
    this.strides = Object.freeze([...shape].reverse().map(n => {
      const result = stride; stride *= n; return result;
    }).reverse());
    if (isHostStorage(storage) && storage.length !== this.size) {
      throw new Error(`Data length ${storage.length} does not match shape [${shape}]`);
    }
    if (isHostStorage(storage) && storageDType(storage) !== (dtype === bool ? 'u8' : dtype)) throw new Error('Storage array does not match tensor dtype');
    if (dtype === bool && isHostStorage(storage) && Array.from(storage as any).some(value => value !== 0 && value !== 1)) throw new Error('bool storage must contain only 0 or 1');
    storages.set(this, storage);
    trackTensor(this);
  }
  get rank(): S['length'] { return this.shape.length }
  get isDisposed() { return this.disposed }
  assertAlive() { if (this.disposed) throw new Error('Tensor is disposed') }
  async read(): Promise<HostStorage<D>> {
    const storage = storageOf(this);
    return (isHostStorage(storage) ? storage.slice() : await storage.read()) as HostStorage<D>;
  }
  async item(): Promise<ScalarOf<D>> {
    if (this.size !== 1) throw new Error('item() requires a one-element tensor');
    const value = (await this.read())[0]!;
    return (this.dtype === bool ? Boolean(value) : value) as ScalarOf<D>;
  }
  add<X extends Operand>(x: X): Tensor<Broadcast<S, OperandShape<X>>, BinaryDType<Tensor<S, D>, X>> { return intrinsics.add(this, x) as Tensor<Broadcast<S, OperandShape<X>>, BinaryDType<Tensor<S, D>, X>> }
  sub<X extends Operand>(x: X): Tensor<Broadcast<S, OperandShape<X>>, BinaryDType<Tensor<S, D>, X>> { return intrinsics.sub(this, x) as Tensor<Broadcast<S, OperandShape<X>>, BinaryDType<Tensor<S, D>, X>> }
  mul<X extends Operand>(x: X): Tensor<Broadcast<S, OperandShape<X>>, BinaryDType<Tensor<S, D>, X>> { return intrinsics.mul(this, x) as Tensor<Broadcast<S, OperandShape<X>>, BinaryDType<Tensor<S, D>, X>> }
  div<X extends Operand>(x: X): Tensor<Broadcast<S, OperandShape<X>>, Floating<BinaryDType<Tensor<S, D>, X>>> { return intrinsics.div(this, x) as Tensor<Broadcast<S, OperandShape<X>>, Floating<BinaryDType<Tensor<S, D>, X>>> }
  gt<X extends Operand>(x: X): Tensor<Broadcast<S, OperandShape<X>>, typeof bool> { return intrinsics.gt(this, x) as Tensor<Broadcast<S, OperandShape<X>>, typeof bool> }
  and<X extends Tensor<Shape, typeof bool> | boolean>(x: X): Tensor<Broadcast<S, OperandShape<X>>, typeof bool> { return intrinsics.and(this, x) as Tensor<Broadcast<S, OperandShape<X>>, typeof bool> }
  or<X extends Tensor<Shape, typeof bool> | boolean>(x: X): Tensor<Broadcast<S, OperandShape<X>>, typeof bool> { return intrinsics.or(this, x) as Tensor<Broadcast<S, OperandShape<X>>, typeof bool> }
  not(): Tensor<S, typeof bool> { return intrinsics.not(this) as Tensor<S, typeof bool> }
  neg(): Tensor<S, D> { return intrinsics.neg(this) as Tensor<S, D> }
  exp(): Tensor<S, Floating<D>> { return intrinsics.exp(this) as Tensor<S, Floating<D>> }
  log(): Tensor<S, Floating<D>> { return intrinsics.log(this) as Tensor<S, Floating<D>> }
  tanh(): Tensor<S, Floating<D>> { return intrinsics.tanh(this) as Tensor<S, Floating<D>> }
  relu(): Tensor<S, D> { return intrinsics.relu(this) as Tensor<S, D> }
  softmax(): Tensor<S, Floating<D>> { return intrinsics.softmax(this) as Tensor<S, Floating<D>> }
  cast<E extends DType>(dtype: E): Tensor<S, E> { return intrinsics.cast(this, dtype) }
  matmul<R extends Shape, E extends DType>(x: Tensor<R, E>): Tensor<MatmulShape<S, R>, Promote<D, E>> { return intrinsics.matmul(this, x) as Tensor<MatmulShape<S, R>, Promote<D, E>> }
  transpose(): Tensor<Transposed<S>, D> { return intrinsics.transpose(this) }
  reshape<const R extends Shape>(shape: R): Tensor<R, D> { return intrinsics.reshape(this, shape) }
  sum<const A extends number | readonly number[] | undefined = undefined, K extends boolean = false>(axis?: A, keepDims?: K): Tensor<Reduced<S, A, K>, Summed<D>> { return intrinsics.sum(this, axis, keepDims) as Tensor<Reduced<S, A, K>, Summed<D>> }
  mean<const A extends number | readonly number[] | undefined = undefined, K extends boolean = false>(axis?: A, keepDims?: K): Tensor<Reduced<S, A, K>, Floating<D>> { return intrinsics.mean(this, axis, keepDims) as Tensor<Reduced<S, A, K>, Floating<D>> }
  dispose() {
    const storage = storages.get(this);
    if (!this.disposed && storage && !isHostStorage(storage)) storage.dispose();
    this.disposed = true;
  }
  [Symbol.dispose](): void { this.dispose() }
}
function flatten(data: Nested): { values: Scalar[]; shape: number[] } {
  if (!Array.isArray(data)) return { values: [data as Scalar], shape: [] };
  if (data.length === 0) throw new Error('Empty tensors are not supported yet');
  const parts = data.map(flatten);
  const shape = parts[0]!.shape;
  if (parts.some(p => String(p.shape) !== String(shape))) throw new Error('Ragged tensor data');
  return { values: parts.flatMap(p => p.values), shape: [parts.length, ...shape] };
}
export function tensor<const S extends Shape, D extends DType>(data: Nested | HostStorage, options: { shape: S; dtype: D }): Tensor<S, D>;
export function tensor<const S extends Shape>(data: NestedOf<boolean>, options: { shape: S }): Tensor<S, typeof bool>;
export function tensor<const S extends Shape>(data: NestedOf<bigint>, options: { shape: S }): Tensor<S, typeof i64>;
export function tensor<const S extends Shape>(data: Float32Array | NestedOf<number>, options: { shape: S }): Tensor<S, typeof f32>;
export function tensor<const S extends Shape>(data: Nested | HostStorage, options: { shape: S }): Tensor<S, DType>;
export function tensor<D extends DType>(data: Nested | HostStorage, options: { shape?: Shape; dtype: D }): Tensor<Shape, D>;
export function tensor(data: NestedOf<boolean>, options?: { shape?: Shape }): Tensor<Shape, typeof bool>;
export function tensor(data: NestedOf<bigint>, options?: { shape?: Shape }): Tensor<Shape, typeof i64>;
export function tensor(data: Float32Array | NestedOf<number>, options?: { shape?: Shape }): Tensor<Shape, typeof f32>;
export function tensor(data: HostStorage, options?: { shape?: Shape }): Tensor<Shape, DType>;
export function tensor(data: Nested, options?: { shape?: Shape }): Tensor<Shape, DType>;
export function tensor(data: Nested | HostStorage, options: { shape?: Shape; dtype?: DType } = {}): Tensor<Shape, DType> {
  const inferred = isHostStorage(data) ? { values: Array.from(data as any) as Scalar[], shape: [data.length] } : flatten(data);
  const dtype = options.dtype ?? (isHostStorage(data) ? storageDType(data) : inferred.values.reduce<DType>((type, value) => promoteTypes(type, scalarDType(value)), bool));
  return new Tensor(makeStorage(dtype, inferred.values), options.shape ?? inferred.shape, dtype);
}
export function zeros<const S extends Shape, D extends DType = typeof f32>(shape: S, options: { dtype?: D } = {}): Tensor<S, D> {
  const dtype = options.dtype ?? f32 as D; return new Tensor(makeStorage(dtype, sizeOf(shape)), shape, dtype);
}
export function ones<const S extends Shape, D extends DType = typeof f32>(shape: S, options: { dtype?: D } = {}): Tensor<S, D> {
  const dtype = options.dtype ?? f32 as D; return new Tensor(makeStorage(dtype, Array(sizeOf(shape)).fill(1)), shape, dtype);
}
export function randn<const S extends Shape, D extends DType = typeof f32>(shape: S, options: { seed?: number; scale?: number; dtype?: D } = {}): Tensor<S, D> {
  const dtype = options.dtype ?? f32 as D;
  if (!['f16', 'f32', 'f64'].includes(dtype)) throw new Error('randn requires a floating dtype');
  let state = options.seed ?? 42;
  const uniform = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
  const values = Array.from({ length: sizeOf(shape) }, () =>
    Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()) * (options.scale ?? 1));
  return new Tensor(makeStorage(dtype, values), shape, dtype);
}
