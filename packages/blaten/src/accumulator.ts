import { Tensor, storageOf } from './tensor.js';
import { execute, assign } from './execution.js';
import { isFloat, isHostStorage } from './types.js';
import { tensors, sameLayout, unflattenTree, type TensorTree } from './tree.js';
import { keepTensor, scope } from './scope.js';
import * as ir from './ir.js';

export interface Accumulator<Input, Output = Input> extends Disposable {
  readonly weight: number;
  add(value: Input, options?: { weight?: number }): void;
  compute(): Output;
  reset(): void;
  dispose(): void;
}

/** Owns a weighted sum of a floating tensor tree; the template supplies metadata only. */
export function meanAccumulator<T extends TensorTree>(template: T): Accumulator<T> {
  const { leaves, layout } = tensors(template);
  if (!leaves.length) throw new Error('Mean accumulator requires a nonempty tensor tree');
  const specifications = leaves.map(value => {
    const storage = storageOf(value);
    if (!isFloat(value.dtype)) throw new Error('Mean accumulator requires floating tensors');
    return { shape: value.shape, dtype: value.dtype, owner: isHostStorage(storage) ? null : storage.owner };
  });
  if (new Set(specifications.map(spec => spec.owner)).size !== 1) throw new Error('Accumulator tensors must be in the same environment');
  let buffers: Tensor[] = [], totalWeight = 0, disposed = false;
  const alive = () => { if (disposed) throw new Error('Mean accumulator is disposed') };
  return {
    get weight() { return totalWeight },
    add(value: T, options: { weight?: number } = {}): void {
      alive();
      const weight = options.weight ?? 1, tree = tensors(value);
      if (!Number.isFinite(weight) || weight <= 0 || !Number.isFinite(totalWeight + weight)) throw new Error('Accumulator weight must be positive and finite');
      if (!sameLayout(layout, tree.layout)) throw new Error('Accumulator tensor tree does not match template');
      tree.leaves.forEach((tensor, index) => {
        const spec = specifications[index]!, storage = storageOf(tensor);
        if (tensor.dtype !== spec.dtype || String(tensor.shape) !== String(spec.shape)) throw new Error('Requires matching floating tensor shapes and dtypes');
        if ((isHostStorage(storage) ? null : storage.owner) !== spec.owner) throw new Error('Accumulator values must be in the same environment');
      });
      if (!buffers.length) {
        buffers = scope(() => tree.leaves.map(tensor => tensor.mul(weight)));
        buffers.forEach(keepTensor);
      } else {
        const inputs = [...buffers, ...tree.leaves].map((tensor, index) => ir.node('input', [], tensor.shape, { index }, tensor.dtype));
        const outputs = buffers.map((buffer, index) => ir.binary('add', inputs[index]!, ir.binary('mul', inputs[index + buffers.length]!, ir.constant(weight, buffer.dtype))));
        const result = execute({ inputs, outputs, single: false }, [...buffers, ...tree.leaves]) as Tensor[];
        try { result.forEach((tensor, index) => assign(buffers[index]!, tensor)) }
        finally { result.forEach(tensor => tensor.dispose()) }
      }
      totalWeight += weight;
    },
    compute(): T {
      alive(); if (!totalWeight) throw new Error('Cannot compute an empty mean accumulator');
      return scope(() => unflattenTree(layout, buffers.map(tensor => tensor.div(totalWeight)))) as T;
    },
    reset(): void {
      alive();
      if (buffers.length) {
        const inputs = buffers.map((tensor, index) => ir.node('input', [], tensor.shape, { index }, tensor.dtype));
        const outputs = buffers.map(tensor => ir.broadcast(ir.constant(0, tensor.dtype), tensor.shape));
        const zeros = execute({ inputs, outputs, single: false }, buffers) as Tensor[];
        try { zeros.forEach((tensor, index) => assign(buffers[index]!, tensor)) }
        finally { zeros.forEach(tensor => tensor.dispose()) }
      }
      totalWeight = 0;
    },
    dispose(): void { if (disposed) return; buffers.forEach(tensor => tensor.dispose()); buffers = []; totalWeight = 0; disposed = true },
    [Symbol.dispose](): void { this.dispose() },
  };
}
