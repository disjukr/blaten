import { sizeOf, f32, bool, u32, isDType, isFloat, scalarFitsInteger, scalarDType, promoteTypes, type DType, type Scalar, type Shape } from './types.js';
export type Op = 'input' | 'constant' | 'cast' | 'and' | 'or' | 'not' | 'add' | 'sub' | 'mul' | 'div' | 'neg' | 'exp' | 'log' | 'tanh' | 'relu' | 'gt' | 'matmul' | 'transpose' | 'sum' | 'broadcast' | 'reshape' | 'softmax' | 'logsoftmax' | 'custom';
let sequence = 0;
export interface Node {
  readonly id: number; readonly op: Op; readonly inputs: readonly Node[];
  readonly shape: Shape; readonly attrs: Record<string, any>;
  readonly dtype: DType;
}
export class Expression { constructor(readonly node: Node) {} get shape() { return this.node.shape } get dtype() { return this.node.dtype } }
export function node(op: Op, inputs: readonly Node[], shape: Shape, attrs: Record<string, any> = {}, dtype: DType = inputs[0]?.dtype ?? f32): Node {
  sizeOf(shape);
  if (!isDType(dtype)) throw new Error(`Unknown dtype ${dtype}`);
  return { id: sequence++, op, inputs, shape: [...shape], attrs, dtype };
}
export function constant(value: Scalar, dtype: DType = scalarDType(value), weak = true): Node { return node('constant', [], [], { value, weak }, dtype) }
export function cast(x: Node, dtype: DType): Node { return x.dtype === dtype ? x : node('cast', [x], x.shape, {}, dtype) }
export function floating(x: Node): Node { return isFloat(x.dtype) ? x : cast(x, f32) }
function operands(a: Node, b: Node): [Node, Node, DType] {
  // JS scalar literals are weakly typed and adopt the tensor's dtype when representable.
  const adopt = (value: Node, other: Node) => value.op === 'constant' && value.attrs.weak && !other.attrs.weak &&
    (isFloat(other.dtype) || scalarFitsInteger(value.attrs.value, other.dtype)) ? constant(value.attrs.value, other.dtype, false) : value;
  a = adopt(a, b); b = adopt(b, a);
  const dtype = promoteTypes(a.dtype, b.dtype);
  return [cast(a, dtype), cast(b, dtype), dtype];
}
export function broadcastShape(a: Shape, b: Shape): Shape {
  const rank = Math.max(a.length, b.length);
  return Array.from({ length: rank }, (_, i) => {
    const x = a[i - rank + a.length] ?? 1, y = b[i - rank + b.length] ?? 1;
    if (x !== y && x !== 1 && y !== 1) throw new Error(`Cannot broadcast [${a}] with [${b}]`);
    return Math.max(x, y);
  });
}
export function binary(op: 'add' | 'sub' | 'mul' | 'div' | 'gt', a: Node, b: Node): Node {
  let dtype: DType; [a, b, dtype] = operands(a, b);
  if (dtype === bool && op !== 'gt') throw new Error('Boolean arithmetic requires an explicit numeric cast');
  if (dtype === bool && op === 'gt') { a = cast(a, u32); b = cast(b, u32) }
  if (op === 'div' && !isFloat(dtype)) { dtype = f32; a = cast(a, dtype); b = cast(b, dtype) }
  return node(op, [a, b], broadcastShape(a.shape, b.shape), {}, op === 'gt' ? bool : dtype);
}
export function logical(op: 'and' | 'or' | 'not', a: Node, b?: Node): Node {
  if (a.dtype !== bool || (b && b.dtype !== bool)) throw new Error('Logical operations require bool inputs');
  return node(op, b ? [a, b] : [a], b ? broadcastShape(a.shape, b.shape) : a.shape, {}, bool);
}
export function unary(op: 'neg' | 'exp' | 'log' | 'tanh' | 'relu', x: Node): Node {
  if (['exp', 'log', 'tanh'].includes(op)) x = floating(x);
  else if (x.dtype === bool) throw new Error('Boolean arithmetic requires an explicit numeric cast');
  return node(op, [x], x.shape);
}
export function broadcast(x: Node, shape: Shape): Node {
  if (String(x.shape) === String(shape)) return x;
  if (String(broadcastShape(x.shape, shape)) !== String(shape)) throw new Error('Invalid broadcast target');
  return node('broadcast', [x], shape);
}
export function reduce(x: Node, axes?: number | readonly number[], keepDims = false): Node {
  if (x.dtype === bool) x = cast(x, u32);
  const selected = axes === undefined ? x.shape.map((_, i) => i) : typeof axes === 'number' ? [axes] : [...axes];
  const normalized = selected.map(i => i < 0 ? i + x.shape.length : i).sort((a, b) => a - b);
  if (new Set(normalized).size !== normalized.length || normalized.some(i => !Number.isInteger(i) || i < 0 || i >= x.shape.length)) throw new Error('Invalid reduction axes');
  return node('sum', [x], x.shape.flatMap((n, i) => normalized.includes(i) ? keepDims ? [1] : [] : [n]), { axes: normalized, keepDims });
}
export function reshape(x: Node, shape: Shape): Node {
  if (sizeOf(shape) !== sizeOf(x.shape)) throw new Error('reshape must preserve element count');
  return node('reshape', [x], shape);
}
export function transpose(x: Node): Node {
  if (x.shape.length !== 2) throw new Error('transpose currently requires rank 2');
  return node('transpose', [x], [x.shape[1]!, x.shape[0]!]);
}
export function matmul(a: Node, b: Node): Node {
  if (a.shape.length !== 2 || b.shape.length !== 2 || a.shape[1] !== b.shape[0]) throw new Error(`matmul requires compatible rank-2 tensors: [${a.shape}], [${b.shape}]`);
  let dtype: DType; [a, b, dtype] = operands(a, b);
  if (dtype === bool) throw new Error('Boolean matmul requires an explicit numeric cast');
  return node('matmul', [a, b], [a.shape[0]!, b.shape[1]!], {}, dtype);
}
export function unbroadcast(g: Node, shape: Shape): Node {
  const lead = g.shape.length - shape.length;
  const axes = g.shape.flatMap((n, i) => i < lead || (shape[i - lead] === 1 && n !== 1) ? [i] : []);
  return axes.length ? reshape(reduce(g, axes, true), shape) : reshape(g, shape);
}
export function topological(outputs: readonly Node[]): Node[] {
  const visited = new Set<Node>(), order: Node[] = [];
  const visit = (n: Node) => {
    if (visited.has(n)) return;
    visited.add(n); n.inputs.forEach(visit);
    if (n.op === 'custom') visit(n.attrs.forward);
    order.push(n);
  };
  outputs.forEach(visit); return order;
}
// Differentiation returns the same IR. It does not execute or compile any kernel.
export function vjp(output: Node, targets: readonly Node[], seed = constant(1, output.dtype)): Node[] {
  if (!isFloat(output.dtype) || targets.some(target => !isFloat(target.dtype))) throw new Error('Automatic differentiation requires floating outputs and targets');
  const gradients = new Map<Node, Node>([[output, broadcast(seed, output.shape)]]);
  const accumulate = (input: Node, value: Node) => {
    if (!isFloat(input.dtype)) return;
    const g = cast(unbroadcast(value, input.shape), input.dtype), old = gradients.get(input);
    gradients.set(input, old ? binary('add', old, g) : g);
  };
  for (const n of topological([output]).reverse()) {
    const g = gradients.get(n); if (!g) continue;
    const [a, b] = n.inputs as [Node, Node];
    switch (n.op) {
      case 'input': case 'constant': case 'gt': case 'and': case 'or': case 'not': break;
      case 'cast': if (isFloat(a.dtype)) accumulate(a, cast(g, a.dtype)); break;
      case 'add': accumulate(a, g); accumulate(b, g); break;
      case 'sub': accumulate(a, g); accumulate(b, unary('neg', g)); break;
      case 'mul': accumulate(a, binary('mul', g, b)); accumulate(b, binary('mul', g, a)); break;
      case 'div':
        accumulate(a, binary('div', g, b));
        accumulate(b, unary('neg', binary('div', binary('mul', g, a), binary('mul', b, b)))); break;
      case 'neg': accumulate(a, unary('neg', g)); break;
      case 'exp': accumulate(a, binary('mul', g, n)); break;
      case 'log': accumulate(a, binary('div', g, a)); break;
      case 'tanh': accumulate(a, binary('mul', g, binary('sub', constant(1), binary('mul', n, n)))); break;
      case 'relu': accumulate(a, binary('mul', g, binary('gt', a, constant(0)))); break;
      case 'reshape': accumulate(a, reshape(g, a.shape)); break;
      case 'broadcast': accumulate(a, g); break;
      case 'transpose': accumulate(a, transpose(g)); break;
      case 'matmul': accumulate(a, matmul(g, transpose(b))); accumulate(b, matmul(transpose(a), g)); break;
      case 'sum': {
        const expanded = a.shape.map((dim, i) => n.attrs.axes.includes(i) ? 1 : dim);
        accumulate(a, broadcast(reshape(g, expanded), a.shape)); break;
      }
      case 'softmax': {
        const product = binary('mul', g, n);
        accumulate(a, binary('mul', n, binary('sub', g, reduce(product, -1, true)))); break;
      }
      case 'logsoftmax':
        accumulate(a, binary('sub', g, binary('mul', unary('exp', n), reduce(g, -1, true)))); break;
      case 'custom': {
        const values: Node[] = n.attrs.backward(g);
        if (values.length !== n.inputs.length) throw new Error('customVjp must return one gradient per input');
        values.forEach((value, i) => accumulate(n.inputs[i]!, value)); break;
      }
    }
  }
  return targets.map(t => gradients.get(t) ?? broadcast(constant(0, t.dtype), t.shape));
}
