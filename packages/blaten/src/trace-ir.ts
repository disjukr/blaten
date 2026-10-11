import { definitionOf, kernelIr, type KernelFunction } from './kernel.js';
import { topological } from './ir.js';
import type { Program } from './lowering.js';
import type { Recorder, JsonValue, Diagnostic } from './trace.js';
import type { Signature } from './types.js';

// Tag values JSON cannot represent without loss. Never serialize runtime tensors.
function portable(value: unknown): JsonValue {
  if (typeof value === 'bigint') return { type: 'bigint', value: String(value) };
  if (typeof value === 'number' && (!Number.isFinite(value) || Object.is(value, -0))) return { type: 'number', value: Object.is(value, -0) ? '-0' : String(value) };
  if (value === null || typeof value === 'string' || typeof value === 'boolean' || typeof value === 'number') return value;
  if (Array.isArray(value) || ArrayBuffer.isView(value)) return Array.from(value as ArrayLike<unknown>, portable);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).filter(([, v]) => v !== undefined).map(([k, v]) => [k, portable(v)]));
  throw new Error(`Unsupported trace value: ${typeof value}`);
}
export function diagnostic(error: unknown): Diagnostic {
  return error instanceof Error ? { name: error.name, message: error.message, ...(error.stack ? { stack: error.stack } : {}) } : { name: 'Error', message: String(error) };
}
export function traceName(fn: KernelFunction, seen = new Set<KernelFunction>()): string {
  if (seen.has(fn)) return fn.name || '<anonymous>';
  seen.add(fn);
  const definition = Object.getOwnPropertyDescriptor(fn, kernelIr)?.value as import('./kernel.js').Definition | undefined;
  if (!definition) return fn.name || '<anonymous>';
  if (definition.kind === 'kernel') return definition.metadata.name;
  const name = traceName(definition.source, seen);
  return definition.kind === 'grad' ? `${definition.value ? 'valueAndGrad' : 'grad'}(${name})` : `${definition.kind}(${name})`;
}
export function traceKernel(record: Recorder, fn: KernelFunction): void {
  const kernels: JsonValue[] = [], ids = new Map<KernelFunction, string>();
  const visit = (current: KernelFunction): string => {
    const found = ids.get(current); if (found) return found;
    const id = `k${ids.size + 1}`; ids.set(current, id);
    const definition = definitionOf(current);
    if (definition.kind === 'kernel') {
      const { externals, source, ...metadata } = definition.metadata;
      const references: Record<string, JsonValue> = {};
      for (const [name, get] of Object.entries(externals)) {
        const value = get();
        if (typeof value === 'function' && Object.hasOwn(value, kernelIr)) references[name] = { kernel: visit(value as KernelFunction) };
        else references[name] = { kind: typeof value };
      }
      kernels.push(portable({ id, kind: definition.kind, ...metadata, ...(source ? { source } : {}), externals: references }));
    } else {
      kernels.push(portable({ id, kind: definition.kind, source: visit(definition.source),
        ...(definition.kind === 'grad' ? { args: definition.args, value: definition.value } : definition.kind === 'vjp' ? { backward: visit(definition.backward) } : { jvp: visit(definition.jvp) }) }));
    }
    return id;
  };
  const entry = visit(fn);
  record.addArtifact({ id: 'kernel-ir', kind: 'blaten.kernel-ir', schemaVersion: 1, content: { type: 'json', value: { entry, kernels } } });
  record.addStage({ id: 'kernel', name: 'Kernel IR', inputs: [], artifacts: ['kernel-ir'] });
}
export function traceProgram(record: Recorder, program: Program, signature?: Signature): Map<import('./ir.js').Node, string> {
  const nodes = [...new Set([...program.inputs, ...topological(program.outputs)])];
  const ids = new Map(nodes.map((n, i) => [n, `n${i + 1}`]));
  const value = portable({ ...(signature ? { signature } : {}), single: program.single,
    inputs: program.inputs.map(n => ids.get(n)), outputs: program.outputs.map(n => ids.get(n)),
    inputLayout: program.inputLayout, outputLayout: program.outputLayout,
    nodes: nodes.map(n => ({ id: ids.get(n), op: n.op, shape: n.shape, dtype: n.dtype, inputs: n.inputs.map(x => ids.get(x)), attrs: Object.fromEntries(Object.entries(n.attrs).filter(([key, v]) => key !== 'tensor' && typeof v !== 'function').map(([key, v]) => [key, ids.has(v) ? { node: ids.get(v) } : v])) })) });
  record.addArtifact({ id: 'tensor-ir', kind: 'blaten.tensor-ir', schemaVersion: 1, content: { type: 'json', value } });
  record.addStage({ id: 'tensor', name: 'Tensor IR', inputs: signature ? ['kernel'] : [], artifacts: ['tensor-ir'] });
  return ids;
}
