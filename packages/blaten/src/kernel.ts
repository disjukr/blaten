import type { KernelIR } from './kernel-ir.js';
import { dtypeOf, type Signature } from './types.js';
export type KernelFunction = (...args: any[]) => any;
export interface KernelMetadata {
  version: 2;
  name: string;
  ir: KernelIR;
  externals: Record<string, () => unknown>;
  signature?: Signature;
  signatureError?: string;
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
export function signatureOf(fn: KernelFunction): Signature {
  const seen = new Set<KernelFunction>();
  while (true) {
    if (seen.has(fn)) throw new Error('Cyclic kernel signature definition');
    seen.add(fn);
    const definition = definitionOf(fn);
    if (definition.kind !== 'kernel') { fn = definition.source; continue }
    const { metadata } = definition;
    if (!metadata.signature) throw new Error(`Cannot compile ${metadata.name}: ${metadata.signatureError ?? 'concrete input signature is missing'}. Wrap the kernel in a function with concrete shape and dtype annotations.`);
    if (metadata.signature.inputs.length !== metadata.ir.parameterCount) throw new Error('Kernel signature input count does not match IR');
    metadata.signature.inputs.forEach(dtypeOf);
    return metadata.signature;
  }
}
export function grad<T extends KernelFunction>(fn: T, options: { arg?: number | readonly number[] } = {}): KernelFunction {
  return derivative(fn, options, false);
}
export function valueAndGrad<T extends KernelFunction>(fn: T, options: { arg?: number | readonly number[] } = {}): KernelFunction {
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
