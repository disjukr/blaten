import { Tensor, storageOf } from './tensor.js';
import { execute, assign } from './execution.js';
import { isFloat, isHostStorage } from './types.js';
import { tensors, sameLayout, type TensorTree } from './tree.js';
import { scope } from './scope.js';
import { meanAccumulator, type Accumulator } from './accumulator.js';
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
export interface Optimizer<P extends TensorTree> {
  readonly parameters: P;
  step(gradients: P): void;
}
export function sgd<P extends TensorTree>(parameters: P, options: { learningRate: number }): Optimizer<P> {
  const { leaves: parametersCopy, layout } = tensors(parameters), rate = options.learningRate;
  if (!parametersCopy.length || new Set(parametersCopy).size !== parametersCopy.length) throw new Error('SGD requires distinct parameters');
  if (!(rate > 0) || !Number.isFinite(rate)) throw new Error('SGD requires a positive finite learning rate');
  return { parameters, step(gradients: P): void {
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

/** A weighted mean accumulator constrained to distinct floating parameters. */
export function gradientAccumulator<P extends TensorTree>(parameters: P): Accumulator<P> {
  const { leaves } = tensors(parameters);
  if (!leaves.length || new Set(leaves).size !== leaves.length) throw new Error('Accumulator requires distinct parameters');
  return meanAccumulator(parameters);
}

/** Borrows the optimizer and parameters; owns only its accumulation buffers. */
export function accumulate<P extends TensorTree>(optimizer: Optimizer<P>, options: { steps: number }) {
  const { steps } = options;
  if (!Number.isSafeInteger(steps) || steps <= 0) throw new Error('Accumulation steps must be a positive integer');
  const accumulator = gradientAccumulator(optimizer.parameters);
  let pendingSteps = 0, disposed = false;
  const alive = () => { if (disposed) throw new Error('Accumulating optimizer is disposed') };
  return {
    parameters: optimizer.parameters,
    get pendingSteps() { return pendingSteps },
    step(gradients: P, options: { weight?: number } = {}): void {
      alive();
      const weight = options.weight ?? 1;
      if (!Number.isFinite(weight) || weight <= 0) throw new Error('Gradient weight must be positive and finite');
      if (steps === 1) { optimizer.step(gradients); return }
      if (pendingSteps >= steps) this.flush();
      accumulator.add(gradients, { weight });
      pendingSteps++;
      if (pendingSteps === steps) this.flush();
    },
    flush(): void {
      alive();
      if (!pendingSteps) return;
      scope(() => optimizer.step(accumulator.compute()));
      accumulator.reset();
      pendingSteps = 0;
    },
    // Cleanup discards pending gradients; updates must be explicit, even on errors.
    dispose(): void { if (disposed) return; accumulator.dispose(); pendingSteps = 0; disposed = true },
    [Symbol.dispose](): void { this.dispose() },
  };
}
