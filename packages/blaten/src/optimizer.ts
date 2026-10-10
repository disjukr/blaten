import { Tensor, storageOf } from './tensor.js';
import { execute, assign } from './execution.js';
import { isFloat, isHostStorage } from './types.js';
import { tensors, sameLayout, unflattenTree, type TensorTree } from './tree.js';
import { keepTensor, scope } from './scope.js';
import * as ir from './ir.js';

function validate(parameters: readonly Tensor[], gradients: readonly Tensor[]): void {
  if (gradients.length !== parameters.length) throw new Error('Gradient count does not match parameters');
  const locations = new Set<object | null>();
  [...parameters, ...gradients].forEach(value => { const storage = storageOf(value); locations.add(isHostStorage(storage) ? null : storage.owner) });
  if (locations.size !== 1) throw new Error('Parameters and gradients must be in the same environment');
  parameters.forEach((parameter, index) => {
    const gradient = gradients[index]!;
    if (!isFloat(parameter.dtype) || parameter.dtype !== gradient.dtype || String(parameter.shape) !== String(gradient.shape)) throw new Error('Requires matching floating parameters and gradients');
  });
}
export function sgd<P extends TensorTree>(parameters: P, options: { learningRate: number }) {
  const { leaves: parametersCopy, layout } = tensors(parameters), rate = options.learningRate;
  if (!parametersCopy.length || new Set(parametersCopy).size !== parametersCopy.length) throw new Error('SGD requires distinct parameters');
  if (!(rate > 0) || !Number.isFinite(rate)) throw new Error('SGD requires a positive finite learning rate');
  return { step(gradients: P): void {
    const tree = tensors(gradients);
    if (!sameLayout(layout, tree.layout)) throw new Error('SGD gradient tree does not match parameters');
    validate(parametersCopy, tree.leaves);
    const inputs = [...parametersCopy, ...tree.leaves].map((value, index) => ir.node('input', [], value.shape, { index }, value.dtype));
    const outputs = parametersCopy.map((parameter, index) => ir.binary('sub', inputs[index]!, ir.binary('mul', inputs[index + parametersCopy.length]!, ir.constant(rate, parameter.dtype))));
    const result = execute({ inputs, outputs, single: false }, [...parametersCopy, ...tree.leaves]) as Tensor[];
    try { result.forEach((value, index) => assign(parametersCopy[index]!, value)) }
    finally { result.forEach(value => value.dispose()) }
  } };
}

/** Owns persistent buffers in the parameters' environment. add() borrows its input. */
export function gradientAccumulator<P extends TensorTree>(parameters: P) {
  const { leaves, layout } = tensors(parameters);
  if (!leaves.length || new Set(leaves).size !== leaves.length) throw new Error('Accumulator requires distinct parameters');
  validate(leaves, leaves);
  let buffers: Tensor[] = [], totalWeight = 0, disposed = false;
  const alive = () => { if (disposed) throw new Error('Gradient accumulator is disposed') };
  return {
    get weight() { return totalWeight },
    add(gradients: P, options: { weight?: number } = {}): void {
      alive();
      const weight = options.weight ?? 1, tree = tensors(gradients);
      if (!Number.isFinite(weight) || weight <= 0 || !Number.isFinite(totalWeight + weight)) throw new Error('Gradient weight must be positive and finite');
      if (!sameLayout(layout, tree.layout)) throw new Error('Gradient tree does not match parameters');
      validate(leaves, tree.leaves);
      if (!buffers.length) {
        const created: Tensor[] = [];
        try { tree.leaves.forEach(g => created.push(g.mul(weight))) }
        catch (error) { created.forEach(g => g.dispose()); throw error }
        created.forEach(keepTensor); buffers = created;
      } else {
        validate(buffers, tree.leaves);
        const inputs = [...buffers, ...tree.leaves].map((value, index) => ir.node('input', [], value.shape, { index }, value.dtype));
        const outputs = buffers.map((buffer, index) => ir.binary('add', inputs[index]!, ir.binary('mul', inputs[index + buffers.length]!, ir.constant(weight, buffer.dtype))));
        const result = execute({ inputs, outputs, single: false }, [...buffers, ...tree.leaves]) as Tensor[];
        try { result.forEach((value, index) => assign(buffers[index]!, value)) }
        finally { result.forEach(value => value.dispose()) }
      }
      totalWeight += weight;
    },
    mean(): P {
      alive(); if (!totalWeight) throw new Error('Cannot average an empty gradient accumulator');
      return scope(() => unflattenTree(layout, buffers.map(value => value.div(totalWeight)))) as P;
    },
    clear(): void {
      alive();
      if (buffers.length) {
        const inputs = buffers.map((value, index) => ir.node('input', [], value.shape, { index }, value.dtype));
        const outputs = buffers.map(value => ir.broadcast(ir.constant(0, value.dtype), value.shape));
        const zeros = execute({ inputs, outputs, single: false }, buffers) as Tensor[];
        try { zeros.forEach((value, index) => assign(buffers[index]!, value)) }
        finally { zeros.forEach(value => value.dispose()) }
      }
      totalWeight = 0;
    },
    dispose(): void { if (disposed) return; buffers.forEach(value => value.dispose()); buffers = []; totalWeight = 0; disposed = true },
  };
}
