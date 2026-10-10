import { Tensor } from './tensor.js';
import * as ir from './ir.js';
import { mapKernel } from './lowering.js';
import { kernelIr } from './kernel.js';
import { sizeOf, f32, bool, isFloat, isInteger, makeStorage, convertScalar, type DType, type FloatDType, type HostStorage, type Scalar, type ScalarOf, type Shape } from './types.js';
type Value = Tensor<Shape, DType> | Scalar;
export interface Intrinsic {
  readonly name: string;
  readonly fn: Function;
  readonly tensorMethod: boolean;
  /** Zero-based positions in the exported function that must receive kernels. */
  readonly kernelParameters: readonly number[];
}
const byName = new Map<string, Intrinsic>();
const byFunction = new WeakMap<Function, Intrinsic>();
const register = <T extends Function>(name: string, fn: T, kernelParameters: readonly number[] = []): T => {
  const intrinsic: Intrinsic = Object.freeze({
    name, fn, kernelParameters: Object.freeze([...kernelParameters]),
    // Tensor and this module import each other; inspect the prototype only on lookup.
    get tensorMethod() { return typeof Object.getOwnPropertyDescriptor(Tensor.prototype, name)?.value === 'function' },
  });
  byName.set(name, intrinsic);
  byFunction.set(fn, intrinsic);
  return fn;
};
/** List the actual registered operations without memoizing a derived representation. */
export function getIntrinsics(): readonly Intrinsic[] { return Object.freeze([...byName.values()]) }
/** Both lookup paths return the same registered definition. */
export function getIntrinsic(value: string | Function): Intrinsic | undefined {
  return typeof value === 'string' ? byName.get(value) : byFunction.get(value);
}
export function toNode(value: unknown): ir.Node {
  if (value instanceof ir.Expression) return value.node;
  if (typeof value === 'number' || typeof value === 'bigint' || typeof value === 'boolean') return ir.constant(value);
  if (value instanceof Tensor) {
    const tensor = value as Tensor;
    return ir.node('constant', [], tensor.shape, { data: tensor.data }, tensor.dtype);
  }
  throw new Error('Expected a tensor or number');
}
export function broadcastIndex(index: number, outShape: Shape, inShape: Shape): number {
  let result = 0, stride = 1;
  for (let i = outShape.length - 1; i >= 0; i--) {
    const coordinate = index % outShape[i]!; index = Math.floor(index / outShape[i]!);
    const dim = inShape[i - outShape.length + inShape.length];
    if (dim !== undefined) { result += (dim === 1 ? 0 : coordinate) * stride; stride *= dim }
  }
  return result;
}
export function reductionIndex(index: number, inputShape: Shape, axes: readonly number[], keepDims: boolean): number {
  const coordinates = inputShape.map(() => 0);
  for (let i = inputShape.length - 1; i >= 0; i--) {
    coordinates[i] = index % inputShape[i]!; index = Math.floor(index / inputShape[i]!);
  }
  let result = 0;
  inputShape.forEach((dim, i) => {
    if (!axes.includes(i)) result = result * dim + coordinates[i]!;
    else if (keepDims) result *= 1;
  });
  return result;
}
function arithmetic(op: 'add' | 'sub' | 'mul' | 'div', a: number | bigint, b: number | bigint, dtype: DType): number | bigint {
  if (isInteger(dtype)) {
    const x = BigInt(a), y = BigInt(b);
    switch (op) { case 'add': return x + y; case 'sub': return x - y; case 'mul': return x * y; case 'div': return x / y }
  }
  const x = Number(a), y = Number(b);
  switch (op) { case 'add': return x + y; case 'sub': return x - y; case 'mul': return x * y; case 'div': return x / y }
}
export function evaluateCpu(root: ir.Node): HostStorage {
  const cache = new Map<ir.Node, HostStorage>();
  const evaluate = (n: ir.Node): HostStorage => {
    const old = cache.get(n); if (old) return old;
    const out = makeStorage(n.dtype, sizeOf(n.shape)) as any;
    const [a, b] = n.inputs;
    const x = a && evaluate(a), y = b && evaluate(b);
    const at = (values: HostStorage, input: ir.Node, i: number) => values[broadcastIndex(i, n.shape, input.shape)]!;
    const write = (i: number, value: Scalar) => { out[i] = convertScalar(value, n.dtype) };
    for (let i = 0; i < out.length; i++) {
      switch (n.op) {
        case 'constant': write(i, n.attrs.data?.[i] ?? n.attrs.value); break;
        case 'cast': write(i, x![i]!); break;
        case 'and': write(i, Boolean(at(x!, a!, i)) && Boolean(at(y!, b!, i))); break;
        case 'or': write(i, Boolean(at(x!, a!, i)) || Boolean(at(y!, b!, i))); break;
        case 'not': write(i, !x![i]); break;
        case 'add': case 'sub': case 'mul': case 'div': write(i, arithmetic(n.op, at(x!, a!, i), at(y!, b!, i), n.dtype)); break;
        case 'gt': out[i] = +(at(x!, a!, i) > at(y!, b!, i)); break;
        case 'neg': write(i, -x![i]!); break;
        case 'exp': write(i, Math.exp(Number(x![i]!))); break;
        case 'log': write(i, Math.log(Number(x![i]!))); break;
        case 'tanh': write(i, Math.tanh(Number(x![i]!))); break;
        case 'relu': write(i, x![i]! > 0 ? x![i]! : 0); break;
        case 'reshape': out[i] = x![i]!; break;
        case 'broadcast': out[i] = at(x!, a!, i); break;
        case 'transpose': out[i] = x![(i % n.shape[1]!) * a!.shape[1]! + Math.floor(i / n.shape[1]!)]!; break;
        case 'matmul': {
          const row = Math.floor(i / n.shape[1]!), col = i % n.shape[1]!;
          let total: number | bigint = isInteger(n.dtype) ? 0n : 0;
          for (let k = 0; k < a!.shape[1]!; k++) {
            const product = arithmetic('mul', x![row * a!.shape[1]! + k]!, y![k * b!.shape[1]! + col]!, n.dtype);
            total = arithmetic('add', total, product, n.dtype);
            if (n.dtype === 'f16' || n.dtype === 'f32') total = Math.fround(Number(total));
          }
          write(i, total); break;
        }
        case 'sum': break;
        case 'softmax': case 'logsoftmax': {
          const width = n.shape.at(-1)!, start = Math.floor(i / width) * width;
          let max = -Infinity, total = 0;
          for (let k = 0; k < width; k++) max = Math.max(max, Number(x![start + k]!));
          for (let k = 0; k < width; k++) total += Math.exp(Number(x![start + k]!) - max);
          write(i, n.op === 'softmax' ? Math.exp(Number(x![i]!) - max) / total : Number(x![i]!) - max - Math.log(total)); break;
        }
        default: throw new Error(`No CPU implementation for ${n.op}`);
      }
    }
    if (n.op === 'sum') {
      const accumulatorDType = n.dtype === 'f16' ? f32 : n.dtype;
      const accumulated = makeStorage(accumulatorDType, out.length) as any;
      for (let i = 0; i < x!.length; i++) {
        const destination = reductionIndex(i, a!.shape, n.attrs.axes, n.attrs.keepDims);
        accumulated[destination] = convertScalar(arithmetic('add', accumulated[destination]!, x![i]!, accumulatorDType), accumulatorDType);
      }
      for (let i = 0; i < out.length; i++) write(i, accumulated[i]);
    }
    cache.set(n, out); return out;
  };
  return evaluate(root);
}
function finish(n: ir.Node, inputs: readonly unknown[]): any {
  if (inputs.some(x => x instanceof ir.Expression)) return new ir.Expression(n);
  const values = evaluateCpu(n);
  if (inputs.every(x => typeof x === 'number' || typeof x === 'bigint' || typeof x === 'boolean') && n.shape.length === 0) return n.dtype === bool ? Boolean(values[0]) : values[0];
  return new Tensor(values, n.shape, n.dtype);
}
interface BinaryOperation {
  (a: number, b: number): number;
  (a: bigint, b: bigint): Scalar;
  <D extends FloatDType>(a: Tensor<Shape, D>, b: Tensor<Shape, D> | Scalar): Tensor<Shape, D>;
  <D extends FloatDType>(a: Scalar, b: Tensor<Shape, D>): Tensor<Shape, D>;
  (a: Tensor<Shape, DType>, b: Value): Tensor<Shape, DType>;
  (a: Value, b: Tensor<Shape, DType>): Tensor<Shape, DType>;
}
interface UnaryOperation { (x: number): number; <D extends FloatDType>(x: Tensor<Shape, D>): Tensor<Shape, D>; (x: Tensor<Shape, DType>): Tensor<Shape, DType> }
const binary = (op: 'add' | 'sub' | 'mul' | 'div'): BinaryOperation => (a: Value, b: Value): any => finish(ir.binary(op, toNode(a), toNode(b)), [a, b]);
const unary = (op: 'neg' | 'exp' | 'log' | 'tanh' | 'relu'): UnaryOperation => (x: Value): any => finish(ir.unary(op, toNode(x)), [x]);
export const add = register('add', binary('add'));
export const sub = register('sub', binary('sub'));
export const mul = register('mul', binary('mul'));
export const div = register('div', binary('div'));
export const gt = register('gt', (a: Value, b: Value): any => finish(ir.binary('gt', toNode(a), toNode(b)), [a, b]));
export const and = register('and', (a: Value, b: Value): any => finish(ir.logical('and', toNode(a), toNode(b)), [a, b]));
export const or = register('or', (a: Value, b: Value): any => finish(ir.logical('or', toNode(a), toNode(b)), [a, b]));
export const not = register('not', (x: Value): any => finish(ir.logical('not', toNode(x)), [x]));
export const cast = register('cast', <D extends DType>(x: Tensor<Shape, DType>, dtype: D): Tensor<Shape, D> => finish(ir.cast(toNode(x), dtype), [x]));
export const neg = register('neg', unary('neg'));
export const exp = register('exp', unary('exp'));
export const log = register('log', unary('log'));
export const tanh = register('tanh', unary('tanh'));
export const relu = register('relu', unary('relu'));
interface MatmulOperation {
  <D extends FloatDType>(a: Tensor<Shape, D>, b: Tensor<Shape, D>): Tensor<Shape, D>;
  (a: Tensor<Shape, DType>, b: Tensor<Shape, DType>): Tensor<Shape, DType>;
}
export const matmul = register('matmul', ((a: Tensor<Shape, DType>, b: Tensor<Shape, DType>): Tensor<Shape, DType> => finish(ir.matmul(toNode(a), toNode(b)), [a, b])) as MatmulOperation);
export const transpose = register('transpose', (x: Tensor<Shape, DType>): Tensor<Shape, DType> => finish(ir.transpose(toNode(x)), [x]));
export const reshape = register('reshape', (x: Tensor<Shape, DType>, shape: Shape): Tensor<Shape, DType> => finish(ir.reshape(toNode(x), shape), [x]));
export const sum = register('sum', <D extends DType>(x: Tensor<Shape, D>, axis?: number | readonly number[], keepDims = false): Tensor<Shape, D extends 'bool' ? 'u32' : D> => finish(ir.reduce(toNode(x), axis, keepDims), [x]));
export const mean = register('mean', (x: Tensor<Shape, DType>, axis?: number | readonly number[], keepDims = false): Tensor<Shape, DType> => {
  const input = toNode(x), reduced = ir.reduce(input, axis, keepDims);
  return finish(ir.binary('div', reduced, ir.constant(sizeOf(input.shape) / sizeOf(reduced.shape))), [x]);
});
export const softmax = register('softmax', (x: Tensor<Shape, DType>): Tensor<Shape, DType> => {
  const input = ir.floating(toNode(x));
  if (!input.shape.length) throw new Error('softmax requires at least one axis');
  return finish(ir.node('softmax', [input], input.shape), [x]);
});
export const crossEntropy = register('crossEntropy', <D extends DType>(logits: Tensor<Shape, D>, targets: Tensor<Shape, DType>): Tensor<Shape, D extends FloatDType ? D : typeof f32> => {
  const a = ir.floating(toNode(logits)), b = ir.cast(toNode(targets), a.dtype);
  if (a.shape.length !== 2 || String(a.shape) !== String(b.shape)) throw new Error('crossEntropy expects [batch, classes] logits and one-hot targets');
  const logs = ir.node('logsoftmax', [a], a.shape);
  return finish(ir.binary('div', ir.unary('neg', ir.reduce(ir.binary('mul', logs, b))), ir.constant(a.shape[0]!)), [logits, targets]);
});
export const map = register('map', <D extends DType>(fn: (x: ScalarOf<D>) => Scalar, x: Tensor<Shape, D>): Tensor<Shape, DType> => {
  if (Object.hasOwn(fn, kernelIr)) {
    const result = mapKernel(fn, toNode(x));
    return new Tensor(evaluateCpu(result), result.shape, result.dtype);
  }
  const values = Array.from(x.data as any, (value: any) => fn((x.dtype === bool ? Boolean(value) : value) as ScalarOf<D>));
  const dtype = values.every(value => typeof value === 'boolean') ? bool : x.dtype === bool ? f32 : x.dtype;
  return new Tensor<Shape, DType>(makeStorage<DType>(dtype, values), x.shape, dtype);
}, [0]);
