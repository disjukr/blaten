import type { KernelFunction } from './kernel.js';
import type { Shape, DType, Scalar } from './types.js';
import { Tensor } from './tensor.js';
import { flattenTree, type TensorTree } from './tree.js';
export type GradientTree<T> = T extends Tensor ? T : T extends readonly unknown[] ? { [K in keyof T]: GradientTree<T[K]> } : T extends object ? { [K in keyof T]: GradientTree<T[K]> } : Tensor<[], 'f32'>;
export type RuntimeResult<T> = T extends Tensor ? T : T extends number ? Tensor<[], 'f32'> : T extends boolean ? Tensor<[], 'bool'> : T extends bigint ? Tensor<[], 'i64'> : T extends readonly unknown[] ? { [K in keyof T]: RuntimeResult<T[K]> } : T extends object ? { [K in keyof T]: RuntimeResult<T[K]> } : T;
export interface Executable<T extends KernelFunction = KernelFunction> {
  (...inputs: Parameters<T>): RuntimeResult<ReturnType<T>>;
  dispose(): void;
}
export type TensorOrScalar = Tensor<Shape, DType> | Scalar;
export type KernelInput = TensorOrScalar | readonly KernelInput[] | { readonly [key: string]: KernelInput };
export interface CompileOptions { readonly inputs: readonly KernelInput[] }
export interface Backend {
  readonly capabilities: { readonly dtypes: readonly DType[] };
  compile<T extends KernelFunction>(fn: T, options?: CompileOptions): Promise<Executable<T>>;
  copyFrom<S extends Shape, D extends DType>(tensor: Tensor<S, D>): Promise<Tensor<S, D>>;
  copyFrom<T extends TensorTree>(tree: T): Promise<T>;
  scope<T>(fn: () => T): T;
  synchronize(): Promise<void>;
  dispose(): void;
}
export function compile<T extends KernelFunction>(fn: T, backend: Backend, options?: CompileOptions): Promise<Executable<T>> {
  return backend.compile(fn, options);
}
export function callable<T extends KernelFunction = KernelFunction>(run: (...inputs: Parameters<T>) => RuntimeResult<ReturnType<T>>, dispose: () => void): Executable<T> {
  return Object.assign((...inputs: Parameters<T>) => run(...inputs), { dispose });
}
export const isKernelLeaf = (value: unknown): value is TensorOrScalar => value instanceof Tensor || ['number', 'boolean', 'bigint'].includes(typeof value);
/** Bounded shape/dtype specialization cache. Failed compilations can be retried. */
export function compileLazy<T extends KernelFunction>(fn: T, backend: Backend, options: { cacheSize?: number } = {}) {
  const limit = options.cacheSize ?? 8;
  if (!Number.isSafeInteger(limit) || limit < 1) throw new Error('compileLazy cacheSize must be a positive integer');
  type Entry = { plan: Promise<Executable<T>>; users: number; retired: boolean };
  const cache = new Map<string, Entry>();
  let disposed = false;
  const retire = (entry: Entry) => {
    entry.retired = true;
    if (!entry.users) void entry.plan.then(plan => plan.dispose(), () => {});
  };
  const prepare = async (...inputs: Parameters<T>): Promise<Executable<T>> => {
    if (disposed) throw new Error('Lazy compiled function is disposed');
    const { leaves, layout } = flattenTree(inputs, isKernelLeaf);
    const key = JSON.stringify([layout, leaves.map(value => value instanceof Tensor ? (value.assertAlive(), [value.shape, value.dtype]) : typeof value)]);
    let entry = cache.get(key);
    if (entry) { cache.delete(key); cache.set(key, entry) }
    else {
      entry = { plan: backend.compile(fn, { inputs }), users: 0, retired: false };
      cache.set(key, entry);
    }
    entry.users++;
    while (cache.size > limit) { const oldest = cache.keys().next().value!; const evicted = cache.get(oldest)!; cache.delete(oldest); retire(evicted) }
    const release = () => { if (--entry.users === 0 && entry.retired) void entry.plan.then(plan => plan.dispose(), () => {}) };
    try {
      const result = await entry.plan;
      if (disposed) throw new Error('Lazy compiled function is disposed');
      let released = false;
      return callable<T>((...inputs) => {
        if (released || disposed) throw new Error('Prepared executable is disposed');
        return result(...inputs);
      }, () => { if (!released) { released = true; release() } });
    } catch (error) { if (cache.get(key) === entry) cache.delete(key); retire(entry); release(); throw error }
  };
  const run = async (...inputs: Parameters<T>) => {
    const plan = await prepare(...inputs);
    try { return plan(...inputs) } finally { plan.dispose() }
  };
  return Object.assign(run, { prepare, dispose: () => { if (disposed) return; disposed = true; cache.forEach(retire); cache.clear() } });
}
