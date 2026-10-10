import type { KernelIR } from './kernel-ir.js';
import type { GradientTree, KernelInput } from './runtime.js';
import { dtypeOf, scalarDType, tensorType, type Signature, type SignatureConstraints, type ValueConstraint, type ValueType } from './types.js';
import { Tensor } from './tensor.js';
export type KernelFunction = (...args: any[]) => any;
export interface KernelMetadata {
  version: 2;
  name: string;
  ir: KernelIR;
  externals: Record<string, () => unknown>;
  signature?: Signature;
  signatureError?: string;
  constraints?: SignatureConstraints;
}
export type Definition =
  | { kind: 'kernel'; metadata: KernelMetadata }
  | { kind: 'grad'; source: KernelFunction; args: number[]; value: boolean }
  | { kind: 'vjp'; source: KernelFunction; backward: KernelFunction }
  | { kind: 'jvp'; source: KernelFunction; jvp: KernelFunction };
/** Shared across copies of blaten in the same JavaScript realm. */
export const kernelIr = Symbol.for('blaten.kernel');
function attachDefinition<T extends KernelFunction>(fn: T, definition: Definition): T {
  Object.defineProperty(fn, kernelIr, {
    value: Object.freeze(definition), enumerable: false, writable: false, configurable: false,
  });
  return fn;
}
function validateMetadata(metadata: KernelMetadata): void {
  if (metadata?.version !== 2 || !metadata.ir) throw new Error('Kernel registration requires IR version 2. Rebuild with @blaten/compiler.');
}
export function registerKernel<T extends KernelFunction>(fn: T, metadata: KernelMetadata): T {
  validateMetadata(metadata);
  return attachDefinition(fn, { kind: 'kernel', metadata });
}
export function definitionOf(fn: KernelFunction): Definition {
  // A new function must not accidentally inherit another function's definition.
  const definition: Definition | undefined = Object.getOwnPropertyDescriptor(fn, kernelIr)?.value;
  if (!definition) throw new Error(`Function ${fn.name || '<anonymous>'} is not a kernel. Add 'use kernel' and enable @blaten/compiler/vite.`);
  if (!['kernel', 'grad', 'vjp', 'jvp'].includes(definition.kind)) throw new Error('Unsupported kernel definition. Rebuild with @blaten/compiler.');
  if (definition.kind === 'kernel') validateMetadata(definition.metadata);
  return definition;
}
/** Derivatives and custom differentiation rules retain their source's input signature. */
export function signatureOf(fn: KernelFunction, inputs?: readonly KernelInput[]): Signature {
  const seen = new Set<KernelFunction>();
  while (true) {
    if (seen.has(fn)) throw new Error('Cyclic kernel signature definition');
    seen.add(fn);
    const definition = definitionOf(fn);
    if (definition.kind !== 'kernel') { fn = definition.source; continue }
    const { metadata } = definition;
    if (inputs) {
      if (inputs.length !== metadata.ir.parameterCount) throw new Error('Kernel signature input count does not match IR');
      const constraintOf = (type: ValueType): ValueConstraint => typeof type === 'string' ? type : type.kind === 'tensor' ? { kind: 'tensor', shape: type.shape, dtypes: [type.dtype] } : type.kind === 'tuple' ? { kind: 'tuple', items: type.items.map(constraintOf) } : { kind: 'record', fields: Object.fromEntries(Object.entries(type.fields).map(([key, child]) => [key, constraintOf(child)])) };
      const constraints = metadata.constraints?.inputs ?? metadata.signature?.inputs.map(constraintOf);
      if (!constraints) throw new Error(`Cannot compile ${metadata.name}: ${metadata.signatureError ?? 'input type constraints are missing'}`);
      const specialize = (input: any, constraint: ValueConstraint, path: string): ValueType => {
        if (typeof constraint === 'string') {
          if (typeof input === 'object' || scalarDType(input) !== constraint) throw new Error(`${path}: expected scalar ${constraint}`);
          return constraint;
        }
        if (constraint.kind === 'tuple') {
          if (!Array.isArray(input) || input.length !== constraint.items.length) throw new Error(`${path}: expected a fixed tuple`);
          return { kind: 'tuple', items: constraint.items.map((child, index) => specialize(input[index], child, `${path}[${index}]`)) };
        }
        if (constraint.kind === 'record') {
          const keys = Object.keys(constraint.fields).sort();
          if (!input || typeof input !== 'object' || (Object.getPrototypeOf(input) !== Object.prototype && Object.getPrototypeOf(input) !== null) || JSON.stringify(Object.keys(input).sort()) !== JSON.stringify(keys)) throw new Error(`${path}: expected record fields ${keys.join(', ')}`);
          return { kind: 'record', fields: Object.fromEntries(keys.map(key => [key, specialize(input[key], constraint.fields[key]!, `${path}.${key}`)])) };
        }
        if (!(input instanceof Tensor)) throw new Error(`${path}: expected a tensor`);
        input.assertAlive();
        if (!constraint.dtypes.includes(input.dtype)) throw new Error(`${path}: expected dtype ${constraint.dtypes.join(' | ')}, received ${input.dtype}`);
        if (constraint.shape && (constraint.shape.length !== input.rank || constraint.shape.some((dim, axis) => dim !== null && dim !== input.shape[axis]))) throw new Error(`${path}: expected shape [${constraint.shape.map(dim => dim ?? 'number')}], received [${input.shape}]`);
        return tensorType(input.shape as readonly number[], input.dtype);
      };
      return { inputs: inputs.map((input, index) => specialize(input, constraints[index]!, `Input ${index}`)) };
    }
    if (!metadata.signature) throw new Error(`Cannot compile ${metadata.name}: ${metadata.signatureError ?? 'concrete input signature is missing'}. Supply options.inputs or concrete shape and dtype annotations.`);
    if (metadata.signature.inputs.length !== metadata.ir.parameterCount) throw new Error('Kernel signature input count does not match IR');
    const validate = (type: ValueType): void => {
      if (typeof type === 'string' || type.kind === 'tensor') dtypeOf(type);
      else if (type.kind === 'tuple') type.items.forEach(validate);
      else Object.values(type.fields).forEach(validate);
    };
    metadata.signature.inputs.forEach(validate);
    return metadata.signature;
  }
}
type GradientSelection<T extends KernelFunction, A extends number | readonly number[]> =
  A extends number ? GradientTree<Parameters<T>[A]> :
  A extends readonly [infer I extends number] ? GradientTree<Parameters<T>[I]> :
  A extends readonly number[] ? { [K in keyof A]: GradientTree<Parameters<T>[A[K] & number]> } : never;
export function grad<T extends KernelFunction, const A extends number | readonly number[] = 0>(fn: T, options?: { arg?: A }): (...args: Parameters<T>) => GradientSelection<T, A>;
export function grad(fn: KernelFunction, options: { arg?: number | readonly number[] } = {}): KernelFunction {
  return derivative(fn, options, false);
}
export function valueAndGrad<T extends KernelFunction, const A extends number | readonly number[] = 0>(fn: T, options?: { arg?: A }): (...args: Parameters<T>) => { value: ReturnType<T>; grads: GradientSelection<T, A> };
export function valueAndGrad(fn: KernelFunction, options: { arg?: number | readonly number[] } = {}): KernelFunction {
  return derivative(fn, options, true);
}
function derivative(fn: KernelFunction, options: { arg?: number | readonly number[] }, value: boolean) {
  const args = typeof options.arg === 'number' ? [options.arg] : [...(options.arg ?? [0])];
  if (!args.length || new Set(args).size !== args.length || args.some(n => !Number.isSafeInteger(n) || n < 0)) {
    throw new Error('Gradient arguments must be distinct nonnegative integers');
  }
  const result = () => { throw new Error('Generated derivatives must be compiled with a backend') };
  return attachDefinition(result, { kind: 'grad', source: fn, args, value });
}
export function customVjp<T extends KernelFunction>(fn: T, rule: { backward: KernelFunction }): T {
  const wrapper = ((...args: any[]) => fn(...args)) as T;
  return attachDefinition(wrapper, { kind: 'vjp', source: fn, backward: rule.backward });
}
export function customJvp<T extends KernelFunction>(fn: T, rule: { jvp: KernelFunction }): T {
  const wrapper = ((...args: any[]) => fn(...args)) as T;
  return attachDefinition(wrapper, { kind: 'jvp', source: fn, jvp: rule.jvp });
}
