import { definitionOf, type KernelFunction } from './kernel.js';
import type { KernelInstruction } from './kernel-ir.js';
import { Tensor, hostData } from './tensor.js';
import { getIntrinsic, toNode, type Intrinsic } from './intrinsics.js';
import * as ir from './ir.js';
import { shapeOf, dtypeOf, isFloat, type Signature } from './types.js';
import { flattenTree, unflattenTree, sameLayout, type TreeLayout } from './tree.js';

export interface KernelOrigin { kernel: KernelFunction; instruction: number }
type Origins = Map<ir.Node, KernelOrigin[]>;
export interface Program { inputs: ir.Node[]; outputs: ir.Node[]; single: boolean; inputLayout?: TreeLayout; outputLayout?: TreeLayout; origins?: Origins }
const isExpression = (x: unknown): x is ir.Expression => x instanceof ir.Expression;
const asNode = (x: any) => toNode(x);
const isValue = (x: unknown): x is ir.Expression | number | boolean | bigint => isExpression(x) || ['number', 'boolean', 'bigint'].includes(typeof x);
const boxed = (x: ir.Node) => new ir.Expression(x);
// Method references carry their receiver locally, without registering temporary functions.
class IntrinsicMethod {
  constructor(readonly intrinsic: Intrinsic, readonly receiver: ir.Expression) {}
}
// Specialize the kernel value graph into shape-resolved tensor IR.
export function lower(fn: KernelFunction, signature: Signature, trace = false): Program {
  const origins: Origins | undefined = trace ? new Map() : undefined;
  const inputs: ir.Node[] = [];
  const argument = (type: Signature['inputs'][number]): any => {
    if (typeof type !== 'string' && type.kind === 'tuple') return type.items.map(argument);
    if (typeof type !== 'string' && type.kind === 'record') return Object.fromEntries(Object.keys(type.fields).sort().map(key => [key, argument(type.fields[key]!)]));
    const input = ir.node('input', [], shapeOf(type), { index: inputs.length }, dtypeOf(type));
    inputs.push(input); return boxed(input);
  };
  const args = signature.inputs.map(argument);
  const inputLayout = flattenTree(args, isExpression).layout;
  const value = invoke(fn, args, [], origins);
  const result = flattenTree(value, isValue);
  const outputs = result.leaves.map(asNode);
  if (signature.output) {
    const expand = (type: Signature['inputs'][number]): any => typeof type === 'string' || type.kind === 'tensor' ? type : type.kind === 'tuple' ? type.items.map(expand) : Object.fromEntries(Object.keys(type.fields).sort().map(key => [key, expand(type.fields[key]!)]));
    const expected = Array.isArray(signature.output) ? signature.output.map(expand) : expand(signature.output as Signature['inputs'][number]);
    const types = flattenTree(expected, (x): x is import('./types.js').ValueType => typeof x === 'string' || !!x && typeof x === 'object' && 'kind' in x && x.kind === 'tensor');
    if (!sameLayout(types.layout, result.layout) || types.leaves.length !== outputs.length || types.leaves.some((type, i) => String(shapeOf(type)) !== String(outputs[i]!.shape) || dtypeOf(type) !== outputs[i]!.dtype)) throw new Error('Output signature does not match inferred shape or dtype');
  }
  return { inputs, outputs, single: result.layout.kind === 'leaf', inputLayout, outputLayout: result.layout, ...(origins ? { origins } : {}) };
}
function invoke(fn: KernelFunction, args: any[], stack: KernelFunction[], origins?: Origins): any {
  if (stack.includes(fn)) throw new Error('Recursive kernels are not supported');
  const definition = definitionOf(fn);
  if (definition.kind === 'grad') {
    const output = invoke(definition.source, args, stack, origins);
    if ((!isExpression(output) && typeof output === 'object') || asNode(output).shape.length !== 0) throw new Error('grad requires a scalar output; use a reduction');
    const selected = definition.args.map(i => { if (i >= args.length) throw new Error('Gradient argument out of range'); return args[i] });
    const targetTree = flattenTree(selected, isValue);
    const targets = targetTree.leaves.map(asNode);
    if (!targets.length) throw new Error('Gradient targets cannot be an empty tree');
    const gradients = ir.vjp(asNode(output), targets).map(boxed);
    const structured = unflattenTree(targetTree.layout, gradients);
    const result = selected.length === 1 ? structured[0] : structured;
    return definition.value ? { value: output, grads: result } : result;
  }
  if (definition.kind === 'vjp') {
    const forward = asNode(invoke(definition.source, args, stack, origins));
    const tree = flattenTree(args, isValue);
    const n = ir.node('custom', tree.leaves.map(asNode), forward.shape, {
      forward,
      backward: (seed: ir.Node) => {
        const gradients = invoke(definition.backward, [...args, boxed(seed)], stack, origins);
        const result = flattenTree(gradients, isValue);
        if (!sameLayout(tree.layout, result.layout)) throw new Error('customVjp backward must return an array matching the input trees');
        return result.leaves.map(asNode);
      },
    }, forward.dtype);
    return boxed(n);
  }
  if (definition.kind === 'jvp') {
    const forward = asNode(invoke(definition.source, args, stack, origins));
    // Transpose the linear tangent map using the same reverse-mode IR transform.
    const tree = flattenTree(args, isValue);
    const tangents = tree.leaves.map((a, i) => ir.node('input', [], asNode(a).shape, { tangent: i }, asNode(a).dtype));
    const tangentArgs = unflattenTree(tree.layout, tangents.map(boxed));
    const tangentOutput = asNode(invoke(definition.jvp, [...args, ...tangentArgs], stack, origins));
    if (String(tangentOutput.shape) !== String(forward.shape)) throw new Error('customJvp output shape must match the forward output');
    assertLinearTangent(tangentOutput, tangents);
    return boxed(ir.node('custom', tree.leaves.map(asNode), forward.shape, {
      forward, backward: (seed: ir.Node) => {
        const gradients = ir.vjp(tangentOutput, tangents, seed);
        // Tangent values must not survive a valid linear JVP's transposition.
        if (ir.topological(gradients).some(n => tangents.includes(n))) throw new Error('customJvp must be linear in input tangents');
        return gradients;
      },
    }, forward.dtype));
  }
  const { metadata } = definition;
  if (metadata.ir.parameterCount !== args.length) throw new Error(`${metadata.name}: expected ${metadata.ir.parameterCount} inputs, received ${args.length}`);
  const values: any[] = [];
  const nextStack = [...stack, fn];
  const evaluate = (a: KernelInstruction): any => {
    switch (a.op) {
      case 'argument': return args[a.index];
      case 'external': {
        const getter = metadata.externals[a.name];
        if (!getter) throw new Error(`Missing kernel external ${a.name}`);
        const value = getter();
        // Captured host tensors are constants, snapshotted at compilation.
        if (value instanceof Tensor) {
          const tensor: Tensor = value;
          return boxed(ir.node('constant', [], tensor.shape, { data: hostData(tensor).slice() }, tensor.dtype));
        }
        return value;
      }
      case 'literal': return a.value;
      case 'bigint': return BigInt(a.value);
      case 'array': return a.values.map(index => values[index]);
      case 'object': return Object.fromEntries(Object.entries(a.fields).map(([key, index]) => [key, values[index]]));
      case 'binary': {
        const left = values[a.left], right = values[a.right];
        if (!isExpression(left) && !isExpression(right) && typeof left === 'number' && typeof right === 'number') {
          switch (a.operator) { case 'add': return left + right; case 'sub': return left - right; case 'mul': return left * right; case 'div': return left / right; case 'pow': return left ** right; case 'gt': return +(left > right) }
        }
        const exponent = typeof right === 'bigint' ? Number(right) : right;
        if (a.operator === 'pow' && Number.isInteger(exponent) && exponent >= 0 && exponent <= 16) {
          let result = ir.constant(1);
          for (let i = 0; i < exponent; i++) result = ir.binary('mul', result, asNode(left));
          return boxed(result);
        }
        if (a.operator === 'pow') throw new Error('Kernel tensor powers require an integer exponent between 0 and 16');
        return boxed(ir.binary(a.operator, asNode(left), asNode(right)));
      }
      case 'neg': {
        const value = values[a.value];
        return typeof value === 'number' ? -value : boxed(ir.unary('neg', asNode(value)));
      }
      case 'member': {
        const object = values[a.object], key = values[a.key];
        if (key === 'shape' && isExpression(object)) return object.shape;
        if (key === 'dtype' && isExpression(object)) return object.dtype;
        const intrinsic = typeof key === 'string' ? getIntrinsic(key) : undefined;
        if (isExpression(object) && intrinsic?.tensorMethod) return new IntrinsicMethod(intrinsic, object);
        if (Array.isArray(object) && typeof key === 'number') return object[key];
        if (object && Object.getPrototypeOf(object) === Object.prototype && typeof key === 'string' && Object.hasOwn(object, key)) return object[key];
        throw new Error('Kernel member access supports tensor methods, shape, dtype, and fixed tree indexing');
      }
      case 'call': {
        const target = values[a.target], callArgs = a.args.map(index => values[index]);
        if (typeof target !== 'function' && !(target instanceof IntrinsicMethod)) throw new Error('Kernel call target must be a registered function');
        const intrinsic = target instanceof IntrinsicMethod ? target.intrinsic : getIntrinsic(target);
        if (target instanceof IntrinsicMethod) callArgs.unshift(target.receiver);
        if (intrinsic?.name === 'map') {
          return boxed(mapKernel(callArgs[0], asNode(callArgs[1]), nextStack, origins));
        }
        return intrinsic ? intrinsic.fn(...callArgs) : invoke(target, callArgs, nextStack, origins);
      }
    }
  };
  for (const [instruction, operation] of metadata.ir.instructions.entries()) {
    const value = evaluate(operation);
    values.push(value);
    // Record only actual tensor-valued results, including aliases. Internal
    // intrinsic temporaries and generated gradients have no direct instruction.
    if (origins && isExpression(value)) {
      const references = origins.get(value.node) || [];
      references.push({ kernel: fn, instruction });
      origins.set(value.node, references);
    }
  }
  return values[metadata.ir.result];
}

/** Apply a scalar kernel with the input dtype preserved during specialization. */
export function mapKernel(fn: KernelFunction, value: ir.Node, stack: KernelFunction[] = [], origins?: Origins): ir.Node {
  const element = ir.node('input', [], [], { mapElement: true }, value.dtype);
  const body = asNode(invoke(fn, [boxed(element)], stack, origins));
  if (body.shape.length) throw new Error('map callback must return a scalar');
  if (origins?.has(element)) origins.set(value, [...(origins.get(value) || []), ...origins.get(element)!]);
  const replace = new Map<ir.Node, ir.Node>([[element, value]]);
  const clone = (n: ir.Node): ir.Node => {
    const old = replace.get(n); if (old) return old;
    if (!['constant', 'cast', 'and', 'or', 'not', 'add', 'sub', 'mul', 'div', 'neg', 'exp', 'log', 'tanh', 'relu', 'gt', 'eq'].includes(n.op)) throw new Error('map callback must be an elementwise scalar kernel');
    const children = n.inputs.map(clone);
    const shape = children.reduce<readonly number[]>((s, c) => ir.broadcastShape(s, c.shape), []);
    const result = ir.node(n.op, children, shape, n.attrs, n.dtype);
    if (origins?.has(n)) origins.set(result, [...origins.get(n)!]);
    replace.set(n, result); return result;
  };
  return ir.broadcast(clone(body), value.shape);
}

// Conservatively require a homogeneous linear expression in tangent arguments.
function assertLinearTangent(output: ir.Node, tangents: ir.Node[]): void {
  const memo = new Map<ir.Node, number>();
  const degree = (n: ir.Node): number => {
    if (memo.has(n)) return memo.get(n)!;
    let result: number;
    if (tangents.includes(n)) result = 1;
    else if (n.op === 'input') result = 0;
    else if (n.op === 'constant') result = n.attrs.value === 0 ? -1 : 0;
    else if (n.op === 'custom') result = degree(n.attrs.forward);
    else {
      const values = n.inputs.map(degree), [a, b] = values as [number, number];
      switch (n.op) {
        case 'add': case 'sub':
          if (a >= 0 && b >= 0 && a !== b) throw new Error('customJvp must be homogeneous and linear in input tangents');
          result = Math.max(a, b); break;
        case 'mul': case 'matmul': result = a < 0 || b < 0 ? -1 : a + b; break;
        case 'div': if (b > 0) throw new Error('customJvp must be linear in input tangents'); result = a; break;
        case 'cast': if (a > 0 && (!isFloat(n.dtype) || !isFloat(n.inputs[0]!.dtype))) throw new Error('customJvp tangent casts must be floating'); result = a; break;
        case 'neg': case 'sum': case 'reshape': case 'broadcast': case 'transpose': result = a; break;
        default: if (values.some(v => v > 0)) throw new Error('customJvp must be linear in input tangents'); result = 0;
      }
    }
    if (result > 1) throw new Error('customJvp must be linear in input tangents');
    memo.set(n, result); return result;
  };
  if (degree(output) === 0) throw new Error('customJvp must be homogeneous and linear in input tangents');
}
