import { Tensor, storageOf } from './tensor.js';
import { execute, assign } from './execution.js';
import { isFloat, isHostStorage, type DType, type Shape } from './types.js';
import * as ir from './ir.js';

export function sgd(parameters: readonly Tensor[], options: { learningRate: number }) {
  const parametersCopy = [...parameters], rate = options.learningRate;
  if (!parametersCopy.length || new Set(parametersCopy).size !== parametersCopy.length) throw new Error('SGD requires distinct parameters');
  if (!(rate > 0) || !Number.isFinite(rate)) throw new Error('SGD requires a positive finite learning rate');
  // Validation precedes execution and mutation, so a bad gradient cannot partially update weights.
  return { step(gradients: readonly Tensor<Shape, DType>[]): void {
    if (gradients.length !== parametersCopy.length) throw new Error('SGD gradient count does not match parameters');
    const locations = new Set<object | null>();
    [...parametersCopy, ...gradients].forEach(value => {
      const storage = storageOf(value);
      locations.add(isHostStorage(storage) ? null : storage.owner);
    });
    if (locations.size !== 1) throw new Error('SGD requires parameters and gradients in the same environment');
    parametersCopy.forEach((parameter, index) => {
      const gradient = gradients[index]!;
      if (!isFloat(parameter.dtype) || parameter.dtype !== gradient.dtype || String(parameter.shape) !== String(gradient.shape)) throw new Error('SGD requires matching floating parameters and gradients');
    });
    const inputs = [...parametersCopy, ...gradients].map((value, index) => ir.node('input', [], value.shape, { index }, value.dtype));
    const outputs = parametersCopy.map((parameter, index) => ir.binary('sub', inputs[index]!, ir.binary('mul', inputs[index + parametersCopy.length]!, ir.constant(rate, parameter.dtype))));
    const result = execute({ inputs, outputs, single: false }, [...parametersCopy, ...gradients]) as Tensor[];
    try { result.forEach((value, index) => assign(parametersCopy[index]!, value)) }
    finally { result.forEach(value => value.dispose()) }
  } };
}
