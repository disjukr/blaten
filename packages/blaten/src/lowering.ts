import { definitionOf, type KernelFunction } from './kernel.js';
import type { KernelInstruction } from './kernel-ir.js';
import { getIntrinsic, toNode, type Intrinsic } from './intrinsics.js';
import * as ir from './ir.js';
import { shapeOf, dtypeOf, isFloat, type Signature } from './types.js';

export interface Program { inputs: ir.Node[]; outputs: ir.Node[]; single: boolean }
const isExpression = (x: unknown): x is ir.Expression => x instanceof ir.Expression;
const asNode = (x: any) => toNode(x);
const boxed = (x: ir.Node) => new ir.Expression(x);
// Method references carry their receiver locally, without registering temporary functions.
class IntrinsicMethod {
  constructor(readonly intrinsic: Intrinsic, readonly receiver: ir.Expression) {}
}
// Specialize the kernel value graph into shape-resolved tensor IR.
export function lower(fn: KernelFunction, signature: Signature): Program {
  const inputs = signature.inputs.map((type, index) => ir.node('input', [], shapeOf(type), { index }, dtypeOf(type)));
  const value = invoke(fn, inputs.map(boxed), []);
  const outputs = (Array.isArray(value) ? value.flat(Infinity) : [value]).map(asNode);
  if (signature.output) {
    const types = typeof signature.output === 'string' || !Array.isArray(signature.output) ? [signature.output] : signature.output;
    if (types.length !== outputs.length || types.some((type, i) => String(shapeOf(type as any)) !== String(outputs[i]!.shape) || dtypeOf(type as any) !== outputs[i]!.dtype)) throw new Error('Output signature does not match inferred shape or dtype');
  }
  return { inputs, outputs, single: !Array.isArray(value) };
}
function invoke(fn: KernelFunction, args: any[], stack: KernelFunction[]): any {
  if (stack.includes(fn)) throw new Error('Recursive kernels are not supported');
  const definition = definitionOf(fn);
  if (definition.kind === 'grad') {
    const output = invoke(definition.source, args, stack);
    if (Array.isArray(output) || asNode(output).shape.length !== 0) throw new Error('grad requires a scalar output; use a reduction');
    const targets = definition.args.map(i => { if (i >= args.length) throw new Error('Gradient argument out of range'); return asNode(args[i]) });
    const gradients = ir.vjp(asNode(output), targets).map(boxed);
    const result = gradients.length === 1 ? gradients[0] : gradients;
    return definition.value ? [output, ...gradients] : result;
  }
  if (definition.kind === 'vjp') {
    const forward = asNode(invoke(definition.source, args, stack));
    const n = ir.node('custom', args.map(asNode), forward.shape, {
      forward,
      backward: (seed: ir.Node) => {
        const gradients = invoke(definition.backward, [...args, boxed(seed)], stack);
        if (!Array.isArray(gradients)) throw new Error('customVjp backward must return an array');
        return gradients.map(asNode);
      },
    }, forward.dtype);
    return boxed(n);
  }
  if (definition.kind === 'jvp') {
    const forward = asNode(invoke(definition.source, args, stack));
    // Transpose the linear tangent map using the same reverse-mode IR transform.
    const tangents = args.map((a, i) => ir.node('input', [], asNode(a).shape, { tangent: i }, asNode(a).dtype));
    const tangentOutput = asNode(invoke(definition.jvp, [...args, ...tangents.map(boxed)], stack));
    if (String(tangentOutput.shape) !== String(forward.shape)) throw new Error('customJvp output shape must match the forward output');
    assertLinearTangent(tangentOutput, tangents);
    return boxed(ir.node('custom', args.map(asNode), forward.shape, {
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
        return getter();
      }
      case 'literal': return a.value;
      case 'bigint': return BigInt(a.value);
      case 'array': return a.values.map(index => values[index]);
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
        throw new Error('Kernel member access supports only shape and fixed array indexing');
      }
      case 'call': {
        const target = values[a.target], callArgs = a.args.map(index => values[index]);
        if (typeof target !== 'function' && !(target instanceof IntrinsicMethod)) throw new Error('Kernel call target must be a registered function');
        const intrinsic = target instanceof IntrinsicMethod ? target.intrinsic : getIntrinsic(target);
        if (target instanceof IntrinsicMethod) callArgs.unshift(target.receiver);
        if (intrinsic?.name === 'map') {
          return boxed(mapKernel(callArgs[0], asNode(callArgs[1]), nextStack));
        }
        return intrinsic ? intrinsic.fn(...callArgs) : invoke(target, callArgs, nextStack);
      }
    }
  };
  for (const instruction of metadata.ir.instructions) values.push(evaluate(instruction));
  return values[metadata.ir.result];
}

/** Apply a scalar kernel with the input dtype preserved during specialization. */
export function mapKernel(fn: KernelFunction, value: ir.Node, stack: KernelFunction[] = []): ir.Node {
  const element = ir.node('input', [], [], { mapElement: true }, value.dtype);
  const body = asNode(invoke(fn, [boxed(element)], stack));
  if (body.shape.length) throw new Error('map callback must return a scalar');
  const replace = new Map<ir.Node, ir.Node>([[element, value]]);
  const clone = (n: ir.Node): ir.Node => {
    const old = replace.get(n); if (old) return old;
    if (!['constant', 'cast', 'and', 'or', 'not', 'add', 'sub', 'mul', 'div', 'neg', 'exp', 'log', 'tanh', 'relu', 'gt'].includes(n.op)) throw new Error('map callback must be an elementwise scalar kernel');
    const children = n.inputs.map(clone);
    const shape = children.reduce<readonly number[]>((s, c) => ir.broadcastShape(s, c.shape), []);
    const result = ir.node(n.op, children, shape, n.attrs, n.dtype); replace.set(n, result); return result;
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
