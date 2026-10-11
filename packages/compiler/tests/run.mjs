import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { stripTypeScriptTypes } from 'node:module';
import plugin from '@blaten/compiler';
import { definitionOf, registerKernel, kernelIr } from 'blaten/internal';
import { lower, signatureOf, topological } from 'blaten/backend';
import { compile, tensor, f32, f64, grad, valueAndGrad, tensorType, kernelIr as publicKernelIr } from 'blaten';

const compiler = plugin();
const spanSource = `// 원문 위치 확인 😀
function helper(x: number) { return x }
export function traced(x: number, obj: { value: number }) {
  'use kernel';
  const y = helper(x);
  return [y, y, helper(x), obj.value, obj['value'], -x, true, 3n];
}`;
async function spanIR(options) {
  const result = plugin(options).transform(spanSource, '/project/src/spans.ts');
  const code = stripTypeScriptTypes(result.code, { mode: 'strip' }).replace(/(['"])(blaten\/internal)\1/g, () => JSON.stringify(import.meta.resolve('blaten/internal')));
  const exported = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  return definitionOf(exported.traced).metadata.ir;
}
const untracedIR = await spanIR({});
assert.equal(Object.hasOwn(untracedIR, 'spans'), false);
assert.equal(Object.hasOwn(await spanIR({ trace: false }), 'spans'), false);
const tracedIR = await spanIR({ trace: true });
const { spans, ...plainIR } = tracedIR;
assert.deepEqual(plainIR, untracedIR, 'Tracing must not alter instructions or their indices');
assert.equal(spans.length, tracedIR.instructions.length);
for (const locations of spans) {
  assert.ok(locations.length > 0);
  assert.equal(new Set(locations.map(s => `${s.start}:${s.end}`)).size, locations.length);
  for (const { start, end } of locations) assert.ok(start >= 0 && end > start && end <= spanSource.length);
}
const excerpts = index => spans[index].map(s => spanSource.slice(s.start, s.end));
assert.equal(excerpts(0).filter(text => text === 'x').length, 3, 'Reused arguments record each occurrence');
const externalIndex = tracedIR.instructions.findIndex(i => i.op === 'external' && i.name === 'helper');
assert.deepEqual(excerpts(externalIndex), ['helper', 'helper']);
const aliasIndex = tracedIR.instructions.findIndex(i => i.op === 'call');
assert.deepEqual(excerpts(aliasIndex), ['helper(x)', 'y', 'y']);
const memberIndex = tracedIR.instructions.findIndex(i => i.op === 'member');
assert.deepEqual(excerpts(memberIndex), ['obj.value']);
const propertyIndex = tracedIR.instructions.findIndex(i => i.op === 'literal' && i.value === 'value');
assert.deepEqual(excerpts(propertyIndex), ['value']);
assert.deepEqual(JSON.parse(JSON.stringify(tracedIR)), tracedIR);
async function sourceMetadata(options, file, viteRoot) {
  const instance = plugin(options);
  if (viteRoot) instance.configResolved({ root: viteRoot });
  const result = instance.transform("export function identity(x: number) { 'use kernel'; return x }", file);
  const javascript = stripTypeScriptTypes(result.code, { mode: 'strip' }).replace(/(['"])(blaten\/internal)\1/g, () => JSON.stringify(import.meta.resolve('blaten/internal')));
  const exported = await import(`data:text/javascript;base64,${Buffer.from(javascript).toString('base64')}`);
  return definitionOf(exported.identity).metadata.source;
}
assert.equal((await sourceMetadata({}, '/project/src/model.ts?query', '/project')).file, 'src/model.ts');
assert.equal((await sourceMetadata({ sourceRoot: '/project' }, '/project/src/model.ts', '/other')).file, 'src/model.ts');
assert.equal((await sourceMetadata({ sourceRoot: '/project' }, '/shared/model.ts')).file, '../shared/model.ts');
assert.equal((await sourceMetadata({ sourceRoot: 'C:/project' }, 'C:\\project\\src\\model.ts')).file, 'src/model.ts');
assert.equal((await sourceMetadata({ sourceRoot: 'C:/project' }, 'D:\\shared\\model.ts')).file, 'external/model.ts');
assert.equal((await sourceMetadata({ sourceRoot: '/project' }, '\0virtual:/private/model.ts')).file, 'virtual/model.ts');
const location = await sourceMetadata({ sourceRoot: '/project' }, '/project/src/model.ts');
assert.deepEqual(location, { file: 'src/model.ts', start: 7, end: "export function identity(x: number) { 'use kernel'; return x }".length });
assert.equal(Object.hasOwn(location, 'text'), false);
const source = `
import { reshape, zeros, tensor, f32, bool, type Tensor as T } from 'blaten';
import type { Image as ImportedImage } from './fixtures/types.js';
type Image = T<readonly [2, 3], typeof f32>;
type Weights = { layer: readonly [Image]; bias: Image };
export function structured(p: Weights, x: Image) { 'use kernel'; return { prediction: x.add(p.layer[0]).add(p.bias), weights: p } }
export function structuredLoss(p: Weights) { 'use kernel'; return p.layer[0].mul(p.layer[0]).sum().add(p.bias.sum()) }
type Recursive = { child: Recursive };
export function recursive(p: Recursive) { 'use kernel'; return p }
export function concrete(x: Image) { 'use kernel'; return shaped(x) }
export function imported(x: ImportedImage) { 'use kernel'; return shaped(x) }
export const concreteArrow = (x: Image) => { 'use kernel'; return shaped(x) };
export const concreteExpression = function(x: Image) { 'use kernel'; return shaped(x) };
const created = zeros([2, 3]);
const reshaped = created.reshape([3, 2]).transpose().cast(f32);
const withShape = tensor([1, 2, 3, 4, 5, 6], { shape: [2, 3] });
export function inferred(x: typeof created) { 'use kernel'; return shaped(x) }
export function propagated(x: typeof reshaped) { 'use kernel'; return shaped(x) }
export function explicitShape(x: typeof withShape) { 'use kernel'; return shaped(x) }
export function captured(x: Image) { 'use kernel'; return x.add(withShape) }
export function dynamic(x: T<[number, 3], typeof f32>) { 'use kernel'; return shaped(x) }
export function ambiguous(x: T<[2, 3]>) { 'use kernel'; return shaped(x) }
export function mask(x: T<[2], typeof bool>) { 'use kernel'; return x.not() }
export function scalarBool(x: boolean) { 'use kernel'; return x }
export function optional(x?: number) { 'use kernel'; return x }
export function implicit(x) { 'use kernel'; return x }
export function generic<S extends readonly number[]>(x: T<S, typeof f32>) { 'use kernel'; return shaped(x) }
export function optionalShape(x: T<[2?], typeof f32>) { 'use kernel'; return shaped(x) }
export function restShape(x: T<[2, ...number[]], typeof f32>) { 'use kernel'; return shaped(x) }
export let exponent = 2;
export function setExponent(value: number) { exponent = value }
export function square(x: number) { 'use kernel'; return x ** exponent }
export function increment(x: bigint) { 'use kernel'; return x + 9007199254740993n }
export function shaped(x: T) {
  'use kernel';
  const count = x.shape[0] * x.shape[1];
  const flattened = reshape(x, [count]);
  return flattened.mul(flattened).sum();
}
export const alias = (x: number) => { 'use kernel'; const y = square(x); return [y, -y] };
`;
const transformed = compiler.transform(source, fileURLToPath(new URL('./contract.ts', import.meta.url)));
assert.ok(transformed?.code);
// Native import requires file URLs when evaluating the transformed module as a data URL.
const code = stripTypeScriptTypes(transformed.code, { mode: 'strip' }).replace(/(['"])(blaten(?:\/internal)?)\1/g, (_, quote, name) => JSON.stringify(import.meta.resolve(name)));
const module = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
assert.equal(kernelIr, Symbol.for('blaten.kernel'));
const captured = lower(module.captured, { inputs: [tensorType([2, 3])] });
const capturedConstant = topological(captured.outputs).find(node => node.op === 'constant' && node.attrs.data);
assert.deepEqual([...capturedConstant.attrs.data], [1, 2, 3, 4, 5, 6]);
assert.equal(publicKernelIr, kernelIr);
// Import a second runtime instance: it must read symbols written by the first.
const otherRuntime = await import(`${import.meta.resolve('blaten/internal')}?symbol-contract`);
for (const fn of [module.square, module.shaped, module.alias, module.increment]) {
  assert.equal(otherRuntime.definitionOf(fn), definitionOf(fn));
  const descriptor = Object.getOwnPropertyDescriptor(fn, kernelIr);
  assert.equal(descriptor.value, definitionOf(fn));
  assert.equal(descriptor.enumerable, false);
  assert.equal(descriptor.writable, false);
  assert.equal(descriptor.configurable, false);
  const { metadata } = definitionOf(fn);
  assert.equal(metadata.version, 2);
  assert.ok(!('ast' in metadata) && !('text' in metadata.source), 'Runtime metadata must not carry source AST or source text');
  assert.ok(metadata.ir.instructions.length > 0);
  for (const instruction of metadata.ir.instructions) assert.ok(!('type' in instruction), 'IR instructions must not be Babel nodes');
  assert.deepEqual(JSON.parse(JSON.stringify(metadata.ir)), metadata.ir, 'Kernel IR must be serializable');
}
assert.ok(definitionOf(module.increment).metadata.ir.instructions.some(instruction => instruction.op === 'bigint' && instruction.value === '9007199254740993'));
for (const shape of [[2, 3], [3, 4]]) {
  const program = lower(module.shaped, { inputs: [tensorType(shape)] });
  assert.deepEqual(program.outputs[0].shape, []);
  const product = program.outputs[0].inputs[0];
  assert.equal(product.op, 'mul');
  assert.deepEqual(product.shape, [shape[0] * shape[1]], 'Shape expressions specialize per signature');
}
assert.equal(lower(module.square, { inputs: [f32] }).outputs[0].op, 'mul');
assert.equal(topological(lower(module.square, { inputs: [f32] }).outputs).filter(node => node.op === 'mul').length, 2);
module.setExponent(3);
const cube = lower(module.square, { inputs: [f32] }).outputs[0];
assert.equal(topological([cube]).filter(node => node.op === 'mul').length, 3, 'External values remain late-bound');
const pair = lower(module.alias, { inputs: [f32] });
assert.equal(pair.single, false);
assert.equal(pair.outputs.length, 2);
assert.equal(pair.outputs[1].op, 'neg');
assert.equal(lower(grad(module.square), { inputs: [f32] }).outputs[0].op, 'add');
assert.equal(lower(otherRuntime.grad(module.square), { inputs: [f32] }).outputs[0].op, 'add');
for (const fn of [otherRuntime.grad(module.square), otherRuntime.valueAndGrad(module.square), otherRuntime.customVjp(module.square, { backward: module.alias }), otherRuntime.customJvp(module.square, { jvp: module.square })]) {
  assert.equal(definitionOf(fn), otherRuntime.definitionOf(fn), 'Generated derivative definitions share the symbol contract');
}
assert.throws(() => definitionOf(module.square.bind(null)), /not a kernel/);
const inherited = Object.setPrototypeOf(() => 0, module.square);
assert.throws(() => definitionOf(inherited), /not a kernel/);
const incompatible = () => 0;
Object.defineProperty(incompatible, kernelIr, { value: { kind: 'kernel', metadata: { version: 1 } } });
assert.throws(() => definitionOf(incompatible), /IR version 2/);
assert.throws(() => lower(module.square, { inputs: [] }), /expected 1 inputs/);

// A runtime-only consumer can register IR directly without a parser or compiler plugin.
const manual = registerKernel(() => { throw new Error('JS body must not execute') }, {
  version: 2, name: 'manual', externals: {}, ir: {
    parameterCount: 1,
    instructions: [{ op: 'argument', index: 0 }, { op: 'binary', operator: 'mul', left: 0, right: 0 }],
    result: 1,
  },
});
assert.equal(lower(manual, { inputs: [f32] }).outputs[0].op, 'mul');
assert.throws(() => registerKernel(() => {}, { version: 1 }), /IR version 2/);
for (const body of ['if (x > 0) return x; return -x', 'let y = x; return y', 'return Math.sin(x) % 2', 'const { a } = x; return a']) {
  assert.throws(() => compiler.transform(`function bad(x: any) { 'use kernel'; ${body} }`, '/bad.ts'), /Unsupported|initialized const/);
}
assert.deepEqual(signatureOf(module.square), { inputs: [f32] });
assert.deepEqual(signatureOf(module.increment), { inputs: ['i64'] });
assert.deepEqual(signatureOf(module.scalarBool), { inputs: ['bool'] });
assert.deepEqual(signatureOf(module.concrete), { inputs: [tensorType([2, 3])] });
for (const fn of [module.imported, module.concreteArrow, module.concreteExpression, module.inferred, module.propagated, module.explicitShape]) assert.deepEqual(signatureOf(fn), signatureOf(module.concrete));
assert.deepEqual(signatureOf(module.mask), { inputs: [tensorType([2], 'bool')] });
assert.deepEqual(signatureOf(grad(grad(module.concrete))), signatureOf(module.concrete));
for (const fn of [module.dynamic, module.ambiguous, module.optional, module.implicit, module.generic, module.optionalShape, module.restShape, manual]) assert.throws(() => signatureOf(fn), /Cannot compile/);
const backend = { async compile(fn) { const program = lower(fn, signatureOf(fn)); return Object.assign(() => {}, { dispatchCount: 1, program, dispose() {} }) } };
assert.deepEqual((await compile(module.concrete, backend)).program.inputs[0].shape, [2, 3]);
await assert.rejects(compile(module.dynamic, backend), /Cannot compile/);
const example = tensor([[1, 2, 3], [4, 5, 6]]);
assert.deepEqual(signatureOf(module.dynamic, [example]), { inputs: [tensorType([2, 3])] });
assert.deepEqual(signatureOf(grad(module.shaped), [example]), { inputs: [tensorType([2, 3])] });
assert.deepEqual(signatureOf(module.ambiguous, [tensor([[1, 2, 3], [4, 5, 6]], { dtype: f64 })]), { inputs: [tensorType([2, 3], f64)] });
assert.throws(() => signatureOf(module.dynamic, [tensor([[1, 2]])]), /expected shape/);
assert.throws(() => signatureOf(module.concrete, [tensor([[1, 2, 3], [4, 5, 6]], { dtype: f64 })]), /expected dtype/);
assert.throws(() => signatureOf(module.scalarBool, [1]), /expected scalar/);
assert.throws(() => signatureOf(module.implicit, [example]), /Cannot compile/);
const destroyed = tensor([[1, 2, 3]]); destroyed.dispose();
assert.throws(() => signatureOf(module.dynamic, [destroyed]), /disposed/);
const inputBackend = { ...backend, async compile(fn, options) { return { program: lower(fn, signatureOf(fn, options?.inputs)) } } };
assert.deepEqual((await compile(module.dynamic, inputBackend, { inputs: [example] })).program.inputs[0].shape, [2, 3]);
console.log('Passed: type-derived signatures, wrapper specialization, runtime signature errors, AST-to-IR build boundary, shape specialization, external references, helpers, direct IR registration, and build-time syntax validation.');

const weights = { layer: [example], bias: example };
const structuredSignature = signatureOf(module.structured, [weights, example]);
const structuredProgram = lower(module.structured, structuredSignature);
assert.equal(structuredProgram.inputs.length, 3);
assert.equal(structuredProgram.outputs.length, 3);
assert.equal(structuredProgram.outputLayout.kind, 'record');
const differentiated = lower(valueAndGrad(module.structuredLoss), signatureOf(module.structuredLoss, [weights]));
assert.equal(differentiated.outputs.length, 3);
assert.equal(differentiated.outputLayout.fields.grads.kind, 'record');
assert.throws(() => signatureOf(module.structuredLoss, [{ ...weights, extra: example }]), /record fields/);
assert.throws(() => signatureOf(module.structuredLoss, [{ ...weights, layer: [] }]), /fixed tuple/);
assert.throws(() => signatureOf(module.structuredLoss, [{ ...weights, bias: tensor([1]) }]), /expected shape/);
assert.throws(() => signatureOf(module.recursive), /Cannot compile/);
console.log('Passed: nested parameter signatures, structured results and gradients, and tree validation.');
