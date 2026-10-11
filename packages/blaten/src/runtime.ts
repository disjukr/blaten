import type { KernelFunction } from './kernel.js';
import type { Shape, DType, Scalar } from './types.js';
import { Tensor } from './tensor.js';
import { flattenTree, type TensorTree } from './tree.js';
import { Resource, managed } from './resource.js';
export type GradientTree<T> = T extends Tensor ? T : T extends readonly unknown[] ? { [K in keyof T]: GradientTree<T[K]> } : T extends object ? { [K in keyof T]: GradientTree<T[K]> } : Tensor<[], 'f32'>;
export type RuntimeResult<T> = T extends Tensor ? T : T extends number ? Tensor<[], 'f32'> : T extends boolean ? Tensor<[], 'bool'> : T extends bigint ? Tensor<[], 'i64'> : T extends readonly unknown[] ? { [K in keyof T]: RuntimeResult<T[K]> } : T extends object ? { [K in keyof T]: RuntimeResult<T[K]> } : T;
export interface Executable<T extends KernelFunction = KernelFunction> extends Disposable {
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
export function callable<T extends KernelFunction = KernelFunction>(run: (...inputs: Parameters<T>) => RuntimeResult<ReturnType<T>>, dispose: (() => void) | Resource): Executable<T> {
  const resource = dispose instanceof Resource ? dispose : new Resource(dispose);
  return managed((...inputs: Parameters<T>) => {
    if (resource.isDisposed) throw new Error('Executable is disposed');
    return run(...inputs);
  }, resource);
}
export const isKernelLeaf = (value: unknown): value is TensorOrScalar => value instanceof Tensor || ['number', 'boolean', 'bigint'].includes(typeof value);
type Entry<T extends KernelFunction> = { plan: Promise<Executable<T>>; users: number; retired: boolean };
function retire<T extends KernelFunction>(entry: Entry<T>): void {
  entry.retired = true;
  if (!entry.users) void entry.plan.then(plan => plan.dispose(), () => {});
}
function release<T extends KernelFunction>(entry: Entry<T>): void {
  if (--entry.users === 0 && entry.retired) void entry.plan.then(plan => plan.dispose(), () => {});
}
// A lease retains only its plan, never the lazy function or its specialization cache.
function lease<T extends KernelFunction>(entry: Entry<T>, plan: Executable<T>): Executable<T> {
  return callable<T>((...inputs) => plan(...inputs), () => release(entry));
}
class LazyCache<T extends KernelFunction> {
  private readonly cache = new Map<string, Entry<T>>();
  private disposed = false;
  constructor(private readonly fn: T, private readonly backend: Backend, private readonly limit: number) {}
  async prepare(...inputs: Parameters<T>): Promise<Executable<T>> {
    if (this.disposed) throw new Error('Lazy compiled function is disposed');
    const { leaves, layout } = flattenTree(inputs, isKernelLeaf);
    const key = JSON.stringify([layout, leaves.map(value => value instanceof Tensor ? (value.assertAlive(), [value.shape, value.dtype]) : typeof value)]);
    let entry = this.cache.get(key);
    if (entry) { this.cache.delete(key); this.cache.set(key, entry) }
    else {
      entry = { plan: this.backend.compile(this.fn, { inputs }), users: 0, retired: false };
      this.cache.set(key, entry);
    }
    entry.users++;
    while (this.cache.size > this.limit) {
      const oldest = this.cache.keys().next().value!, evicted = this.cache.get(oldest)!;
      this.cache.delete(oldest); retire(evicted);
    }
    try {
      const plan = await entry.plan;
      // The acquired lease protects an in-flight compilation even if the
      // cache is disposed or collected while awaiting it.
      return lease(entry, plan);
    } catch (error) {
      if (this.cache.get(key) === entry) this.cache.delete(key);
      retire(entry); release(entry); throw error;
    }
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.cache.forEach(retire); this.cache.clear();
  }
}
/** Bounded shape/dtype specialization cache. Prepared leases own their lifetime. */
export function compileLazy<T extends KernelFunction>(fn: T, backend: Backend, options: { cacheSize?: number } = {}) {
  const limit = options.cacheSize ?? 8;
  if (!Number.isSafeInteger(limit) || limit < 1) throw new Error('compileLazy cacheSize must be a positive integer');
  const cache = new LazyCache(fn, backend, limit);
  const prepare = (...inputs: Parameters<T>) => cache.prepare(...inputs);
  const run = async (...inputs: Parameters<T>) => {
    using plan = await prepare(...inputs);
    return plan(...inputs);
  };
  return managed(Object.assign(run, { prepare }), new Resource(() => cache.dispose()));
}
