import { registerKernel } from 'blaten/internal';
import type { Backend, Signature, Shape } from 'blaten';
import type { KernelFunction } from 'blaten/backend';
import assert from 'node:assert/strict';
import { create, globals } from 'webgpu';
import { tensor, tensorType, bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, compile, grad, valueAndGrad, customVjp, customJvp, map, sum, mul, add, matmul, relu, crossEntropy, reshape, type Tensor, type DType } from 'blaten';
import { typegpu } from '@blaten/typegpu';
import { square } from './helper.js';

function concreteSquares(x: Tensor<[4], typeof f32>) { 'use kernel'; return squares(x) }
function unresolvedSquares(x: Tensor<[number], typeof f32>) { 'use kernel'; return squares(x) }
function squares(x: Tensor<Shape, typeof f32>) { 'use kernel'; return sum(map(square, x)) }
function quartic(x: number) { 'use kernel'; return x ** 4 }
function squareBackward(x: number, dy: number) { 'use kernel'; return [2 * x * dy] }
function squareTangent(x: number, dx: number) { 'use kernel'; return 2 * x * dx }
function nonlinearTangent(x: number, dx: number) { 'use kernel'; return x * dx * dx }
function loss(x: Tensor, w: Tensor, b: Tensor, y: Tensor<Shape, typeof f32>) {
  'use kernel';
  const logits = add(matmul(x, w), b);
  return crossEntropy(logits, y);
}
function shaped(x: Tensor<Shape, typeof f32>) { 'use kernel'; return sum(reshape(x, [2, 2]), 0) }
function inferredShape(x: Tensor<Shape, typeof f32>) { 'use kernel'; return x.reshape([x.shape[0] * x.shape[1]]).sum() }
function activation(x: Tensor<Shape, typeof f32>) { 'use kernel'; return sum(mul(relu(x), x)) }
function unmarked(x: number) { return x * 2 }
function invalid(x: number) { 'use kernel'; return unmarked(x) }
function invalidMethod(x: Tensor<Shape, typeof f32>) { 'use kernel'; return (x as any).map(square) }
const chained = (x: Tensor<Shape, typeof f32>) => { 'use kernel'; return x.mul(x).sum() };
function negativeAxis(x: Tensor<Shape, typeof f32>) { 'use kernel'; return x.sum(-1, true) }
function smooth(x: Tensor<Shape, typeof f32>) { 'use kernel'; return x.softmax().mul(x).sum() }
function affineTangent(x: number, dx: number) { 'use kernel'; return x * dx + 1 }
function dtypeArithmetic(x: Tensor<Shape, DType>) { 'use kernel'; return x.mul(x).add(1) }
function dtypeCompare(x: Tensor<Shape, DType>) { 'use kernel'; return x.gt(1) }
function dtypeLogic(x: Tensor<Shape, typeof bool>) { 'use kernel'; return x.not().and(x.or(true)) }
function dtypeCast(x: Tensor<Shape, DType>) { 'use kernel'; return x.cast(f32) }
function dtypeCastSigned32(x: Tensor<Shape, typeof f32>) { 'use kernel'; return x.cast(i32) }
function dtypeCastUnsigned32(x: Tensor<Shape, typeof f32>) { 'use kernel'; return x.cast(u32) }
function dtypeCastBool(x: Tensor<Shape, typeof f32>) { 'use kernel'; return x.cast(bool) }
function nonfiniteConstant(x: Tensor<Shape, typeof f32>) { 'use kernel'; return x.add(Infinity) }
function dtypeSoftmax(x: Tensor<Shape, DType>) { 'use kernel'; return x.softmax() }
function dtypeReduction(x: Tensor<Shape, DType>) { 'use kernel'; return x.sum() }
function dtypeMatmul(x: Tensor<Shape, DType>, y: Tensor<Shape, DType>) { 'use kernel'; return x.matmul(y) }
function bigintLiteral(x: Tensor<Shape, DType>) { 'use kernel'; return x.add(9007199254740993n) }
function secondOrderLoss(x: Tensor<Shape, typeof f32>) { 'use kernel'; return gradSquares(x).sum() }
const gradSquares = grad(chained);
function near(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 2e-4) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) assert.ok(Math.abs(actual[i]! - expected[i]!) < tolerance, `element ${i}: ${actual[i]} != ${expected[i]}`);
}
export async function main() {
  Object.assign(globalThis, globals);
  const gpu = create([]), adapter = await gpu.requestAdapter();
  assert.ok(adapter, 'A WebGPU adapter is required for GPU integration tests');
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [] });
  const errors: string[] = [];
  device.addEventListener('uncapturederror', e => errors.push(e.error.message));
  const backend = await typegpu({ device });
  try {
    await assert.rejects(compile(squares, backend), /Cannot compile.*shape/);
    await assert.rejects(compile(unresolvedSquares, backend), /Cannot compile.*shape/);
    const concrete = await compile(concreteSquares, backend);
    const concreteResult = concrete.run(tensor([1, 2, 3, 4])) as Tensor<Shape, typeof f32>;
    assert.equal(await concreteResult.item(), 30);
    assert.equal(concrete.dispatchCount, 1);
    concreteResult.dispose(); concrete.dispose();
    for (const dtype of backend.supportedDTypes) {
      const source = tensor(dtype === bool ? [true, false, true] : [1, 2, 3], { dtype });
      const uploaded = backend.upload(source);
      assert.equal(uploaded.dtype, dtype);
      assert.deepEqual(Array.from<number | bigint>(await uploaded.read()), Array.from<number | bigint>(source.data));
      const castProgram = await compileForTest(dtypeCast, backend, { inputs: [tensorType([3], dtype)] });
      const converted = castProgram.run(uploaded) as Tensor<Shape, typeof f32>;
      near(await converted.read(), dtype === bool ? [1, 0, 1] : [1, 2, 3]); converted.dispose();
      const reduction = await compileForTest(dtypeReduction, backend, { inputs: [tensorType([3], dtype)] });
      const reduced = reduction.run(uploaded) as Tensor<Shape, DType>;
      assert.equal(await reduced.item(), dtype === bool ? 2 : 6); reduced.dispose();
      if (dtype !== bool) {
        const arithmetic = await compileForTest(dtypeArithmetic, backend, { inputs: [tensorType([3], dtype)] });
        const result = arithmetic.run(uploaded) as Tensor<Shape, DType>;
        assert.equal(result.dtype, dtype);
        assert.deepEqual(Array.from<number | bigint>(await result.read()), [2, 5, 10]); result.dispose();
        assert.throws(() => arithmetic.run(tensor([1, 2, 3], { dtype: dtype === f32 ? i32 : f32 })), /expected dtype/);
        const compare = await compileForTest(dtypeCompare, backend, { inputs: [tensorType([3], dtype)] });
        const mask = compare.run(uploaded) as Tensor<Shape, typeof bool>;
        assert.equal(mask.dtype, bool); assert.deepEqual(Array.from<number | bigint>(await mask.read()), [0, 1, 1]); mask.dispose();
        const matrix = await compileForTest(dtypeMatmul, backend, { inputs: [tensorType([1, 2], dtype), tensorType([2, 1], dtype)] });
        const product = matrix.run(tensor([[1, 2]], { dtype }), tensor([[3], [4]], { dtype })) as Tensor<Shape, DType>;
        assert.equal(await product.item(), 11); product.dispose();
        const overflowInput = dtype === i8 ? [127, -128, 64] : dtype === u8 ? [255, 128, 0] : dtype === i16 ? [32767, -32768, 1000] : dtype === u16 ? [65535, 32768, 1000] : dtype === i32 ? [2147483647, -2147483648, 65537] : dtype === u32 ? [4294967295, 2147483648, 65537] : [0.1, -0.5, 1.25];
        const overflowSource = tensor(overflowInput, { dtype });
        const overflow = arithmetic.run(overflowSource) as Tensor<Shape, DType>;
        assert.deepEqual(Array.from<number | bigint>(await overflow.read()), Array.from<number | bigint>(dtypeArithmetic(overflowSource).data), `${dtype}: GPU and CPU overflow/rounding must agree`);
        overflow.dispose();
        if (dtype === f16) {
          const derivative = await compileForTest(grad(chained), backend, { inputs: [tensorType([3], f16)] });
          const gradient = derivative.run(uploaded) as Tensor<Shape, typeof f16>;
          assert.equal(gradient.dtype, f16); near(await gradient.read(), [2, 4, 6]); gradient.dispose();
          const softmax = await compileForTest(dtypeSoftmax, backend, { inputs: [tensorType([3], f16)] });
          const probabilities = softmax.run(overflowSource) as Tensor<Shape, typeof f16>;
          near(await probabilities.read(), dtypeSoftmax(overflowSource).data as ArrayLike<number>, 1e-3); probabilities.dispose();
        }
      } else {
        const logic = await compileForTest(dtypeLogic, backend, { inputs: [tensorType([3], bool)] });
        const result = logic.run(uploaded) as Tensor<Shape, typeof bool>;
        assert.deepEqual(Array.from<number | bigint>(await result.read()), [0, 1, 0]); result.dispose();
      }
      uploaded.dispose();
    }
    const castSource = tensor([3e9, -3e9, 4294967296, -2.9, 2.9, NaN, Infinity, -Infinity]);
    for (const fn of [dtypeCastSigned32, dtypeCastUnsigned32, dtypeCastBool]) {
      const casts = await compileForTest(fn, backend, { inputs: [tensorType([8])] });
      const output = casts.run(castSource) as Tensor<Shape, DType>;
      assert.deepEqual(Array.from<number | bigint>(await output.read()), Array.from<number | bigint>(fn(castSource).data)); output.dispose();
    }
    const infinity = await compileForTest(nonfiniteConstant, backend, { inputs: [tensorType([1])] });
    const infiniteValue = infinity.run(tensor([1])) as Tensor<Shape, typeof f32>;
    assert.equal(await infiniteValue.item(), Infinity); infiniteValue.dispose();
    const noHalfAdapter = await gpu.requestAdapter();
    assert.ok(noHalfAdapter);
    const noHalfDevice = await noHalfAdapter.requestDevice();
    const noHalfBackend = await typegpu({ device: noHalfDevice });
    try { await assert.rejects(compileForTest(dtypeArithmetic, noHalfBackend, { inputs: [tensorType([3], f16)] }), /shader-f16/) }
    finally { noHalfBackend.dispose(); noHalfDevice.destroy() }
    for (const dtype of [f64, i64, u64]) {
      await assert.rejects(compileForTest(dtypeArithmetic, backend, { inputs: [tensorType([3], dtype)] }), /does not support/);
      assert.throws(() => backend.upload(tensor([1, 2, 3], { dtype })), /does not support/);
    }
    await assert.rejects(compileForTest(bigintLiteral, backend, { inputs: [tensorType([3], i64)] }), /does not support/);
    const fused = await compileForTest(squares, backend, { inputs: [tensorType([4])] });
    assert.equal(fused.dispatchCount, 1, 'map + multiply + sum must be one compute dispatch');
    const value = fused.run(tensor([1, 2, 3, 4])) as Tensor<Shape, typeof f32>;
    assert.equal(await value.item(), 30); value.dispose();
    const methods = await compileForTest(chained, backend, { inputs: [tensorType([4])] });
    assert.equal(methods.dispatchCount, 1);
    near(await (methods.run(tensor([1, 2, 3, 4])) as Tensor<Shape, typeof f32>).read(), [30]);
    const differentiate = await compileForTest(grad(squares), backend, { inputs: [tensorType([4])] });
    near(await (differentiate.run(tensor([1, 2, 3, 4])) as Tensor<Shape, typeof f32>).read(), [2, 4, 6, 8]);
    let derivative = quartic;
    for (const expected of [32, 48, 48, 24, 0]) {
      derivative = grad(derivative) as typeof quartic;
      const executable = await compileForTest(derivative, backend, { inputs: [f32] });
      assert.equal(await (executable.run(2) as Tensor<Shape, typeof f32>).item(), expected);
    }
    for (const custom of [customVjp(square, { backward: squareBackward }), customJvp(square, { jvp: squareTangent })]) {
      const first = await compileForTest(grad(custom), backend, { inputs: [f32] });
      const second = await compileForTest(grad(grad(custom)), backend, { inputs: [f32] });
      near(await (first.run(3) as Tensor<Shape, typeof f32>).read(), [6]); near(await (second.run(3) as Tensor<Shape, typeof f32>).read(), [2]);
    }
    await assert.rejects(compileForTest(grad(customJvp(square, { jvp: nonlinearTangent })), backend, { inputs: [f32] }), /linear/);
    await assert.rejects(compileForTest(grad(customJvp(square, { jvp: affineTangent })), backend, { inputs: [f32] }), /linear/);
    await assert.rejects(compileForTest(invalid, backend, { inputs: [f32] }), /not a kernel/);
    await assert.rejects(compileForTest(invalidMethod, backend, { inputs: [tensorType([4])] }), /member access/);
    const shapeProgram = await compileForTest(shaped, backend, { inputs: [tensorType([4])] });
    near(await (shapeProgram.run(tensor([1, 2, 3, 4])) as Tensor<Shape, typeof f32>).read(), [4, 6]);
    assert.throws(() => shapeProgram.run(tensor([1, 2])), /expected/);
    for (const shape of [[2, 3], [3, 2]]) {
      const specialized = await compileForTest(inferredShape, backend, { inputs: [tensorType(shape)] });
      const result = specialized.run(tensor(new Float32Array([1, 2, 3, 4, 5, 6]), { shape })) as Tensor<Shape, typeof f32>;
      assert.equal(await result.item(), 21); result.dispose(); specialized.dispose();
    }
    const rows = await compileForTest(negativeAxis, backend, { inputs: [tensorType([2, 3])] });
    const rowValue = rows.run(tensor([[1, 2, 3], [4, 5, 6]])) as Tensor<Shape, typeof f32>;
    assert.deepEqual(rowValue.shape, [2, 1]); near(await rowValue.read(), [6, 15]);
    const second = await compileForTest(grad(secondOrderLoss), backend, { inputs: [tensorType([4])] });
    near(await (second.run(tensor([1, 2, 3, 4])) as Tensor<Shape, typeof f32>).read(), [2, 2, 2, 2]);
    const softmaxGrad = await compileForTest(grad(smooth), backend, { inputs: [tensorType([3])] });
    const z = tensor([0.3, -0.7, 1.2]), expectedSoftmax = new Float32Array(3);
    for (let i = 0; i < 3; i++) {
      const original = z.data[i]!; z.data[i] = original + 1e-3; const upper = await smooth(z).item();
      z.data[i] = original - 1e-3; const lower = await smooth(z).item(); z.data[i] = original;
      expectedSoftmax[i] = (upper - lower) / 2e-3;
    }
    near(await (softmaxGrad.run(z) as Tensor<Shape, typeof f32>).read(), expectedSoftmax, 3e-4);
    const act = await compileForTest(grad(activation), backend, { inputs: [tensorType([4])] });
    near(await (act.run(tensor([-2, -1, 1, 2])) as Tensor<Shape, typeof f32>).read(), [0, 0, 2, 4]);
    const x = tensor([[1, 2, -1], [0.5, -1, 3]]), w = tensor([[0.1, -0.2], [0.3, 0.1], [-0.1, 0.2]]), b = tensor([0.05, -0.05]), y = tensor([[1, 0], [0, 1]]);
    const training = await compileForTest(valueAndGrad(loss, { arg: [1, 2] }), backend, { inputs: [tensorType(x.shape), tensorType(w.shape), tensorType(b.shape), tensorType(y.shape)] });
    const [actualLoss, dw, db] = training.run(x, w, b, y) as Tensor<Shape, typeof f32>[];
    near(await actualLoss!.read(), await loss(x, w, b, y).read());
    for (const [parameter, gradient] of [[w, dw!], [b, db!]]) {
      const expected = new Float32Array(parameter!.size), epsilon = 1e-3;
      for (let i = 0; i < parameter!.size; i++) {
        const original = parameter!.data[i]!;
        parameter!.data[i] = original + epsilon; const upper = await loss(x, w, b, y).item();
        parameter!.data[i] = original - epsilon; const lower = await loss(x, w, b, y).item();
        parameter!.data[i] = original; expected[i] = (upper - lower) / (2 * epsilon);
      }
      near(await gradient!.read(), expected, 3e-4);
    }
    const uploaded = [backend.upload(w), backend.upload(b)];
    backend.sgd(uploaded, [dw!, db!], 0.1);
    const before = await actualLoss!.item();
    const after = training.run(x, uploaded[0]!, uploaded[1]!, y) as Tensor<Shape, typeof f32>[];
    assert.ok(await after[0]!.item() < before, 'GPU SGD should reduce loss');
    await device.queue.onSubmittedWorkDone();
    assert.deepEqual(errors, [], 'WebGPU validation must pass');
    console.log(`Passed GPU dtypes: ${backend.supportedDTypes.join(', ')}; unsupported f64/i64/u64 and missing shader-f16 are rejected.`);
    console.log('Passed: GPU fusion, imported helpers, gradients, 5th derivatives, custom VJP/JVP, finite differences, shapes, and GPU SGD.');
  } finally { backend.dispose(); device.destroy() }
}

function compileForTest(fn: KernelFunction, backend: Backend, signature: Signature) {
  const count = signature.inputs.length;
  const wrapper = registerKernel((...args: any[]) => fn(...args), {
    version: 2, name: 'backendTestFixture', signature, externals: { target: () => fn },
    ir: { parameterCount: count, instructions: [
      ...signature.inputs.map((_, index) => ({ op: 'argument' as const, index })),
      { op: 'external', name: 'target' },
      { op: 'call', target: count, args: signature.inputs.map((_, index) => index) },
    ], result: count + 1 },
  });
  return compile(wrapper, backend);
}
