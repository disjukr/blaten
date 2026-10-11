import { hostData, deviceStorageOf } from 'blaten/backend';
import { testTrace } from './trace.js';
import { registerKernel } from 'blaten/internal';
import type { Backend, Signature, Shape } from 'blaten';
import type { KernelFunction } from 'blaten/backend';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { create } from 'webgpu';
import { requestDevice } from '@blaten/typegpu/node';
import { tensor, tensorType, bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, compile, grad, valueAndGrad, customVjp, customJvp, map, sum, mul, add, matmul, relu, crossEntropy, reshape, type Tensor, type DType } from 'blaten';
import { typegpu, inspect } from '@blaten/typegpu';
import { sgd, gradientAccumulator, compileLazy, accumulate, meanMetric, meanAccumulator, multiclassAccuracy } from 'blaten';
import { square } from './helper.js';

type WeightTree = { layer: { weight: Tensor<[2], typeof f32> }; bias: Tensor<[], typeof f32> };
function classify(x: Tensor<[2, 3], typeof f32>, y: Tensor<[2], 'i32'>) {
  'use kernel';
  return x.argmax(-1).eq(y);
}
function treeLoss(p: WeightTree, x: Tensor<[2], typeof f32>) {
  'use kernel';
  return p.layer.weight.mul(x).mul(p.layer.weight).sum().add(p.bias.mul(p.bias));
}
const treeGradient = grad(treeLoss);
function treeSecondLoss(p: WeightTree, x: Tensor<[2], typeof f32>) { 'use kernel'; return treeGradient(p, x).layer.weight.sum() }
function treeBackward(p: WeightTree, x: Tensor<[2], typeof f32>, dy: Tensor<[], typeof f32>) {
  'use kernel';
  return [{ layer: { weight: p.layer.weight.mul(x).mul(2).mul(dy) }, bias: p.bias.mul(2).mul(dy) }, p.layer.weight.mul(p.layer.weight).mul(dy)];
}
function treeTangent(p: WeightTree, x: Tensor<[2], typeof f32>, dp: WeightTree, dx: Tensor<[2], typeof f32>) {
  'use kernel';
  return p.layer.weight.mul(x).mul(dp.layer.weight).mul(2).sum().add(p.layer.weight.mul(p.layer.weight).mul(dx).sum()).add(p.bias.mul(dp.bias).mul(2));
}
function treeOutput(p: WeightTree) { 'use kernel'; return { layer: [p.layer.weight], bias: p.bias } }
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
// eslint-disable-next-line blaten/kernel-contract -- Negative fixture: compilation must reject an unmarked helper.
function invalid(x: number) { 'use kernel'; return unmarked(x) }
// eslint-disable-next-line blaten/kernel-contract -- Negative fixture: compilation must reject unsupported member access.
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
  const device = await requestDevice(), gpu = create([]);
  const errors: string[] = [];
  device.addEventListener('uncapturederror', e => errors.push(e.error.message));
  const backend = await typegpu({ device });
  try {
      await testTrace(device);
    await assert.rejects(compile(squares, backend), /Cannot compile.*shape/);
    await assert.rejects(compile(unresolvedSquares, backend), /Cannot compile.*shape/);
    const concrete = await compile(concreteSquares, backend);
    const concreteResult = concrete(tensor([1, 2, 3, 4], { shape: [4] })) as Tensor<Shape, typeof f32>;
    assert.equal(await concreteResult.item(), 30);
    assert.equal(inspect(concrete).dispatchCount, 1);
    concreteResult.dispose(); concrete.dispose();
    const inputProgram = await compile(unresolvedSquares, backend, { inputs: [tensor([1, 2, 3])] });
    assert.equal('dispatchCount' in inputProgram, false);
    const inputResult = inputProgram(tensor([2, 3, 4], { shape: [3] })) as Tensor;
    assert.equal(await inputResult.item(), 29); inputResult.dispose();
    assert.throws(() => inputProgram(tensor([1, 2], { shape: [2] })), /expected/);
    await assert.rejects(compile(unresolvedSquares, backend, { inputs: [tensor([1, 2], { dtype: i32 })] }), /expected dtype/);
    inputProgram.dispose();
    const gpuInput = await backend.copyFrom(tensor([1, 2, 3]));
    assert.equal('storage' in gpuInput, false);
    assert.equal('data' in gpuInput, false);
    const savedRead = gpuInput.read;
    gpuInput.read = async () => { throw new Error('Direct GPU operations must not read host data') };
    let direct: Tensor;
    try { direct = gpuInput.mul(gpuInput).sum() }
    finally { gpuInput.read = savedRead }
    assert.equal(await direct.item(), 14);
    const changedInput = await backend.copyFrom(tensor([2, 3, 4]));
    const changedOutput = changedInput.mul(changedInput).sum();
    assert.equal(await changedOutput.item(), 29);
    assert.equal(await direct.item(), 14, 'Cached direct plans must preserve prior outputs');
    assert.throws(() => gpuInput.add(tensor([1, 2, 3])), /copyFrom/);
    const gpuMap = map(square, gpuInput);
    near(await gpuMap.read() as ArrayLike<number>, [1, 4, 9]); gpuMap.dispose();
    direct.dispose(); changedOutput.dispose(); changedInput.dispose(); gpuInput.dispose();
    await backend.synchronize();
    for (const dtype of backend.capabilities.dtypes) {
      const source = tensor(dtype === bool ? [true, false, true] : [1, 2, 3], { dtype });
      const uploaded = await backend.copyFrom(source);
      assert.notEqual(uploaded, source);
      assert.equal(uploaded.dtype, dtype);
      assert.deepEqual(Array.from<number | bigint>(await uploaded.read()), Array.from<number | bigint>(hostData(source)));
      const castProgram = await compileForTest(dtypeCast, backend, { inputs: [tensorType([3], dtype)] });
      const converted = castProgram(uploaded) as Tensor<Shape, typeof f32>;
      near(await converted.read(), dtype === bool ? [1, 0, 1] : [1, 2, 3]); converted.dispose();
      const reduction = await compileForTest(dtypeReduction, backend, { inputs: [tensorType([3], dtype)] });
      const reduced = reduction(uploaded) as Tensor<Shape, DType>;
      assert.equal(await reduced.item(), dtype === bool ? 2 : 6); reduced.dispose();
      if (dtype !== bool) {
        const arithmetic = await compileForTest(dtypeArithmetic, backend, { inputs: [tensorType([3], dtype)] });
        const result = arithmetic(uploaded) as Tensor<Shape, DType>;
        assert.equal(result.dtype, dtype);
        assert.deepEqual(Array.from<number | bigint>(await result.read()), [2, 5, 10]); result.dispose();
        assert.throws(() => arithmetic(tensor([1, 2, 3], { dtype: dtype === f32 ? i32 : f32 })), /expected dtype/);
        const compare = await compileForTest(dtypeCompare, backend, { inputs: [tensorType([3], dtype)] });
        const mask = compare(uploaded) as Tensor<Shape, typeof bool>;
        assert.equal(mask.dtype, bool); assert.deepEqual(Array.from<number | bigint>(await mask.read()), [0, 1, 1]); mask.dispose();
        const matrix = await compileForTest(dtypeMatmul, backend, { inputs: [tensorType([1, 2], dtype), tensorType([2, 1], dtype)] });
        const product = matrix(tensor([[1, 2]], { dtype }), tensor([[3], [4]], { dtype })) as Tensor<Shape, DType>;
        assert.equal(await product.item(), 11); product.dispose();
        const overflowInput = dtype === i8 ? [127, -128, 64] : dtype === u8 ? [255, 128, 0] : dtype === i16 ? [32767, -32768, 1000] : dtype === u16 ? [65535, 32768, 1000] : dtype === i32 ? [2147483647, -2147483648, 65537] : dtype === u32 ? [4294967295, 2147483648, 65537] : [0.1, -0.5, 1.25];
        const overflowSource = tensor(overflowInput, { dtype });
        const overflow = arithmetic(overflowSource) as Tensor<Shape, DType>;
        assert.deepEqual(Array.from<number | bigint>(await overflow.read()), Array.from<number | bigint>(hostData(dtypeArithmetic(overflowSource))), `${dtype}: GPU and CPU overflow/rounding must agree`);
        overflow.dispose();
        if (dtype === f16) {
          const derivative = await compileForTest(grad(chained), backend, { inputs: [tensorType([3], f16)] });
          const gradient = derivative(uploaded) as Tensor<Shape, typeof f16>;
          assert.equal(gradient.dtype, f16); near(await gradient.read(), [2, 4, 6]); gradient.dispose();
          const softmax = await compileForTest(dtypeSoftmax, backend, { inputs: [tensorType([3], f16)] });
          const probabilities = softmax(overflowSource) as Tensor<Shape, typeof f16>;
          near(await probabilities.read(), hostData(dtypeSoftmax(overflowSource)) as ArrayLike<number>, 1e-3); probabilities.dispose();
        }
      } else {
        const logic = await compileForTest(dtypeLogic, backend, { inputs: [tensorType([3], bool)] });
        const result = logic(uploaded) as Tensor<Shape, typeof bool>;
        assert.deepEqual(Array.from<number | bigint>(await result.read()), [0, 1, 0]); result.dispose();
      }
      const originalRead = uploaded.read;
      uploaded.read = async () => { throw new Error('Same-backend copying must stay on the GPU') };
      let copied: Tensor<Shape, DType>;
      try { copied = await backend.copyFrom(uploaded) }
      finally { uploaded.read = originalRead }
      assert.notEqual(copied, uploaded);
      uploaded.dispose();
      assert.deepEqual(Array.from<number | bigint>(await copied.read()), Array.from<number | bigint>(hostData(source)), 'Copies survive disposal of the source');
      copied.dispose();
      assert.equal(source.isDisposed, false);
    }
    const hostSource = tensor([1, 2, 3]);
    const hostCopy = await backend.copyFrom(hostSource);
    hostData(hostSource)[0] = 99;
    near(await hostCopy.read(), [1, 2, 3]);
    hostCopy.dispose();
    assert.equal(hostData(hostSource)[0], 99, 'Disposing a copy must not dispose its source');
    hostSource.dispose();
    await assert.rejects(backend.copyFrom(hostSource), /disposed/);
    const castSource = tensor([3e9, -3e9, 4294967296, -2.9, 2.9, NaN, Infinity, -Infinity]);
    for (const fn of [dtypeCastSigned32, dtypeCastUnsigned32, dtypeCastBool]) {
      const casts = await compileForTest(fn, backend, { inputs: [tensorType([8])] });
      const output = casts(castSource) as Tensor<Shape, DType>;
      assert.deepEqual(Array.from<number | bigint>(await output.read()), Array.from<number | bigint>(hostData(fn(castSource)))); output.dispose();
    }
    const infinity = await compileForTest(nonfiniteConstant, backend, { inputs: [tensorType([1])] });
    const infiniteValue = infinity(tensor([1])) as Tensor<Shape, typeof f32>;
    assert.equal(await infiniteValue.item(), Infinity); infiniteValue.dispose();
    const noHalfAdapter = await gpu.requestAdapter();
    assert.ok(noHalfAdapter);
    const noHalfDevice = await noHalfAdapter.requestDevice();
    const noHalfBackend = await typegpu({ device: noHalfDevice });
    try {
      await assert.rejects(compileForTest(dtypeArithmetic, noHalfBackend, { inputs: [tensorType([3], f16)] }), /shader-f16/);
      const source = await backend.copyFrom(tensor([4, 5, 6]));
      const copy = await noHalfBackend.copyFrom(source);
      assert.throws(() => source.add(copy), /copyFrom/);
      source.dispose();
      near(await copy.read(), [4, 5, 6]); copy.dispose();
    }
    finally { noHalfBackend.dispose(); noHalfDevice.destroy() }
    for (const dtype of [f64, i64, u64]) {
      await assert.rejects(compileForTest(dtypeArithmetic, backend, { inputs: [tensorType([3], dtype)] }), /does not support/);
      await assert.rejects(backend.copyFrom(tensor([1, 2, 3], { dtype })), /does not support/);
    }
    await assert.rejects(compileForTest(bigintLiteral, backend, { inputs: [tensorType([3], i64)] }), /does not support/);
    const fused = await compileForTest(squares, backend, { inputs: [tensorType([4])] });
    assert.equal(inspect(fused).dispatchCount, 1, 'map + multiply + sum must be one compute dispatch');
    const value = fused(tensor([1, 2, 3, 4])) as Tensor<Shape, typeof f32>;
    assert.equal(await value.item(), 30); value.dispose();
    const methods = await compileForTest(chained, backend, { inputs: [tensorType([4])] });
    assert.equal(inspect(methods).dispatchCount, 1);
    near(await (methods(tensor([1, 2, 3, 4])) as Tensor<Shape, typeof f32>).read(), [30]);
    const differentiate = await compileForTest(grad(squares), backend, { inputs: [tensorType([4])] });
    near(await (differentiate(tensor([1, 2, 3, 4])) as Tensor<Shape, typeof f32>).read(), [2, 4, 6, 8]);
    let derivative: KernelFunction = quartic;
    for (const expected of [32, 48, 48, 24, 0]) {
      derivative = grad(derivative);
      const executable = await compileForTest(derivative, backend, { inputs: [f32] });
      assert.equal(await (executable(2) as Tensor<Shape, typeof f32>).item(), expected);
    }
    for (const custom of [customVjp(square, { backward: squareBackward }), customJvp(square, { jvp: squareTangent })]) {
      const first = await compileForTest(grad(custom), backend, { inputs: [f32] });
      const second = await compileForTest(grad(grad(custom)), backend, { inputs: [f32] });
      near(await (first(3) as Tensor<Shape, typeof f32>).read(), [6]); near(await (second(3) as Tensor<Shape, typeof f32>).read(), [2]);
    }
    await assert.rejects(compileForTest(grad(customJvp(square, { jvp: nonlinearTangent })), backend, { inputs: [f32] }), /linear/);
    await assert.rejects(compileForTest(grad(customJvp(square, { jvp: affineTangent })), backend, { inputs: [f32] }), /linear/);
    await assert.rejects(compileForTest(invalid, backend, { inputs: [f32] }), /not a kernel/);
    await assert.rejects(compileForTest(invalidMethod, backend, { inputs: [tensorType([4])] }), /member access/);
    const shapeProgram = await compileForTest(shaped, backend, { inputs: [tensorType([4])] });
    near(await (shapeProgram(tensor([1, 2, 3, 4])) as Tensor<Shape, typeof f32>).read(), [4, 6]);
    assert.throws(() => shapeProgram(tensor([1, 2])), /expected/);
    for (const shape of [[2, 3], [3, 2]]) {
      const specialized = await compileForTest(inferredShape, backend, { inputs: [tensorType(shape)] });
      const result = specialized(tensor(new Float32Array([1, 2, 3, 4, 5, 6]), { shape })) as Tensor<Shape, typeof f32>;
      assert.equal(await result.item(), 21); result.dispose(); specialized.dispose();
    }
    const rows = await compileForTest(negativeAxis, backend, { inputs: [tensorType([2, 3])] });
    const rowValue = rows(tensor([[1, 2, 3], [4, 5, 6]])) as Tensor<Shape, typeof f32>;
    assert.deepEqual(rowValue.shape, [2, 1]); near(await rowValue.read(), [6, 15]);
    const second = await compileForTest(grad(secondOrderLoss), backend, { inputs: [tensorType([4])] });
    near(await (second(tensor([1, 2, 3, 4])) as Tensor<Shape, typeof f32>).read(), [2, 2, 2, 2]);
    const softmaxGrad = await compileForTest(grad(smooth), backend, { inputs: [tensorType([3])] });
    const z = tensor([0.3, -0.7, 1.2]), expectedSoftmax = new Float32Array(3);
    for (let i = 0; i < 3; i++) {
      const original = hostData(z)[i]!; hostData(z)[i] = original + 1e-3; const upper = await smooth(z).item();
      hostData(z)[i] = original - 1e-3; const lower = await smooth(z).item(); hostData(z)[i] = original;
      expectedSoftmax[i] = (upper - lower) / 2e-3;
    }
    near(await (softmaxGrad(z) as Tensor<Shape, typeof f32>).read(), expectedSoftmax, 3e-4);
    const act = await compileForTest(grad(activation), backend, { inputs: [tensorType([4])] });
    near(await (act(tensor([-2, -1, 1, 2])) as Tensor<Shape, typeof f32>).read(), [0, 0, 2, 4]);
    const x = tensor([[1, 2, -1], [0.5, -1, 3]]), w = tensor([[0.1, -0.2], [0.3, 0.1], [-0.1, 0.2]]), b = tensor([0.05, -0.05]), y = tensor([[1, 0], [0, 1]]);
    const training = await compileForTest(valueAndGrad(loss, { arg: [1, 2] }), backend, { inputs: [tensorType(x.shape), tensorType(w.shape), tensorType(b.shape), tensorType(y.shape)] });
    const { value: actualLoss, grads: [dw, db] } = training(x, w, b, y) as { value: Tensor<Shape, typeof f32>; grads: Tensor<Shape, typeof f32>[] };
    near(await actualLoss!.read(), await loss(x, w, b, y).read());
    for (const [parameter, gradient] of [[w, dw!], [b, db!]]) {
      const expected = new Float32Array(parameter!.size), epsilon = 1e-3;
      for (let i = 0; i < parameter!.size; i++) {
        const original = hostData(parameter!)[i]!;
        hostData(parameter!)[i] = original + epsilon; const upper = await loss(x, w, b, y).item();
        hostData(parameter!)[i] = original - epsilon; const lower = await loss(x, w, b, y).item();
        hostData(parameter!)[i] = original; expected[i] = (upper - lower) / (2 * epsilon);
      }
      near(await gradient!.read(), expected, 3e-4);
    }
    const uploaded = await Promise.all([backend.copyFrom(w), backend.copyFrom(b)]);
    sgd(uploaded, { learningRate: 0.1 }).step([dw!, db!]);
    const before = await actualLoss!.item();
    const after = training(x, uploaded[0]!, uploaded[1]!, y) as { value: Tensor<Shape, typeof f32> };
    assert.ok(await after.value.item() < before, 'GPU SGD should reduce loss');
    const treeParams = await backend.copyFrom({ layer: { weight: tensor([2, 3], { shape: [2] }) }, bias: tensor(1, { shape: [] }) });
    const treeInput = tensor([1, 2], { shape: [2] });
    const treeTraining = await backend.compile(valueAndGrad(treeLoss));
    for (const source of [customVjp(treeLoss, { backward: treeBackward }), customJvp(treeLoss, { jvp: treeTangent })]) {
      const custom = await backend.compile(valueAndGrad(source));
      const result = custom(treeParams, treeInput);
      near(await result.grads.layer.weight.read(), [4, 12]);
      near(await result.grads.bias.read(), [2]);
      result.value.dispose(); result.grads.layer.weight.dispose(); result.grads.bias.dispose(); custom.dispose();
    }
    const secondTree = await backend.compile(grad(treeSecondLoss));
    const secondGrads = secondTree(treeParams, treeInput);
    near(await secondGrads.layer.weight.read(), [2, 4]);
    near(await secondGrads.bias.read(), [0]);
    secondGrads.layer.weight.dispose(); secondGrads.bias.dispose(); secondTree.dispose();
    const accumulated = gradientAccumulator(treeParams);
    let firstTreeGradients: WeightTree | undefined;
    assert.equal(await backend.scope(() => {
      const { value, grads } = treeTraining(treeParams, treeInput);
      firstTreeGradients = grads;
      grads.layer.weight.read = async () => { throw new Error('Accumulation must stay on the GPU') };
      grads.bias.read = async () => { throw new Error('Accumulation must stay on the GPU') };
      accumulated.add(grads, { weight: 2 });
      return value.item();
    }), 23);
    assert.ok(firstTreeGradients!.layer.weight.isDisposed, 'Accumulator must own its buffers independently');
    backend.scope(() => {
      const { grads } = treeTraining(treeParams, tensor([3, 1], { shape: [2] }));
      accumulated.add(grads, { weight: 1 });
    });
    const averaged = accumulated.compute();
    near(await averaged.layer.weight.read(), [20 / 3, 10]);
    near(await averaged.bias.read(), [2]);
    sgd(treeParams, { learningRate: 0.1 }).step(averaged);
    near(await treeParams.layer.weight.read(), [2 - 2 / 3, 2]);
    near(await treeParams.bias.read(), [0.8]);
    averaged.layer.weight.dispose(); averaged.bias.dispose();
    accumulated.reset();
    assert.equal(accumulated.weight, 0);
    assert.throws(() => accumulated.compute(), /empty/);
    backend.scope(() => accumulated.add(treeTraining(treeParams, treeInput).grads));
    await backend.scope(() => accumulated.compute().bias.read());
    const nonfinite = await backend.copyFrom({ layer: { weight: tensor([Infinity, NaN], { shape: [2] }) }, bias: tensor(Infinity, { shape: [] }) });
    accumulated.add(nonfinite); accumulated.reset();
    backend.scope(() => accumulated.add(treeTraining(treeParams, treeInput).grads));
    const resetMean = accumulated.compute();
    near(await resetMean.bias.read(), [1.6]);
    resetMean.layer.weight.dispose(); resetMean.bias.dispose();
    nonfinite.layer.weight.dispose(); nonfinite.bias.dispose();
    accumulated.dispose();
    // Metrics and optimizer wrappers submit GPU work without reading their inputs.
    const metricInput = await backend.copyFrom(tensor(2, { shape: [] }));
    const wrapperParameters = await backend.copyFrom({ p: tensor(10, { shape: [] }) });
    using metric = meanMetric();
    using wrapped = accumulate(sgd(wrapperParameters, { learningRate: 1 }), { steps: 2 });
    metricInput.read = async () => { throw new Error('Must not read GPU metrics or gradients') };
    backend.scope(() => { metric.add(metricInput, { weight: 3 }); wrapped.step({ p: metricInput }, { weight: 3 }) });
    backend.scope(() => {
      const next = metricInput.mul(3);
      next.read = async () => { throw new Error('Must not read GPU metrics or gradients') };
      metric.add(next, { weight: 1 }); wrapped.step({ p: next }, { weight: 1 });
    });
    const snapshot = metric.compute();
    metric.reset(); metricInput.dispose();
    assert.equal(await snapshot.item(), 3); snapshot.dispose();
    assert.equal(await wrapperParameters.p.item(), 7);
    backend.scope(() => wrapped.step({ p: wrapperParameters.p.mul(0).add(1) }));
    wrapped.flush(); wrapped.flush();
    assert.equal(await wrapperParameters.p.item(), 6);
    wrapped.dispose(); wrapperParameters.p.dispose();
    const vectorTemplate = await backend.scope(() => backend.copyFrom(tensor([0, 0], { shape: [2] })));
    using vectors = meanAccumulator(vectorTemplate);
    vectorTemplate.dispose();
    const vectorInputs = await backend.copyFrom([tensor([2, 4], { shape: [2] }), tensor([6, 8], { shape: [2] })]);
    vectorInputs.forEach(value => { value.read = async () => { throw new Error('Must not read accumulator inputs') } });
    vectors.add(vectorInputs[0]!, { weight: 3 }); vectors.add(vectorInputs[1]!, { weight: 1 });
    vectorInputs.forEach(value => value.dispose());
    const vectorSnapshot = vectors.compute();
    vectors.reset(); vectors.dispose();
    near(await vectorSnapshot.read(), [3, 5]); vectorSnapshot.dispose();
    const scores = tensor([1, 5, 5, 3, 1, 9], { shape: [2, 3] });
    const gpuScores = await backend.copyFrom(scores);
    const classification = await backend.compile(classify);
    assert.deepEqual([...await backend.scope(() => classification(scores, tensor([1, 2], { shape: [2], dtype: 'i32' })).read())], [1, 1]);
    classification.dispose();
    near(await backend.scope(() => gpuScores.argmax(0).read()), [1, 0, 1]);
    const nonfiniteScores = await backend.copyFrom(tensor([-Infinity, -Infinity, 0, 0, NaN, Infinity], { shape: [2, 3] }));
    near(await backend.scope(() => nonfiniteScores.argmax(-1).read()), [2, 1]);
    assert.deepEqual([...await backend.scope(() => nonfiniteScores.eq(nonfiniteScores).read())], [1, 1, 1, 1, 0, 1]);
    nonfiniteScores.dispose();
    gpuScores.read = async () => { throw new Error('Accuracy must not read logits') };
    using accuracy = multiclassAccuracy();
    backend.scope(() => accuracy.add({ logits: gpuScores, targets: tensor([0, 1, 0, 1, 0, 0], { shape: [2, 3] }), validCount: 1 }));
    assert.equal(await backend.scope(() => accuracy.compute().item()), 1);
    backend.scope(() => accuracy.add({ logits: gpuScores, targets: tensor([0, 1, 0, 1, 0, 0], { shape: [2, 3] }) }));
    const accuracySnapshot = accuracy.compute();
    accuracy.reset(); gpuScores.dispose(); scores.dispose();
    assert.ok(Math.abs(await accuracySnapshot.item() - 2 / 3) < 1e-6); accuracySnapshot.dispose();
    const treePlan = await backend.compile(treeOutput);
    const returned = treePlan(treeParams);
    returned.layer[0].dispose(); returned.bias.dispose();
    const lazy = compileLazy(unresolvedSquares, backend, { cacheSize: 1 });
    const two = tensor([1, 2], { shape: [2] }), three = tensor([1, 2, 3], { shape: [3] });
    const lazyResults = await Promise.all([lazy(two), lazy(three)]);
    assert.equal(await lazyResults[0]!.item(), 5);
    assert.equal(await lazyResults[1]!.item(), 14);
    lazyResults.forEach(value => value.dispose()); lazy.dispose();
    let ownedSnapshot: Tensor;
    {
      using plan = await backend.compile(unresolvedSquares, { inputs: [tensor([1, 2, 3])] });
      ownedSnapshot = plan(tensor([1, 2, 3], { shape: [3] }));
    }
    assert.equal(await ownedSnapshot.item(), 14, 'Outputs survive executable using disposal');
    ownedSnapshot.dispose();
    let disposedTensor: Tensor;
    {
      using value = await backend.copyFrom(tensor([1, 2, 3]));
      disposedTensor = value;
    }
    assert.ok(disposedTensor.isDisposed);

    // The backend remains live throughout collection, so its resource tracking
    // and cleanup closures must not keep Tensor or Executable objects alive.
    const collected = { tensors: 0 };
    async function abandonTensor() {
      const value = await backend.copyFrom(tensor([4, 5, 6]));
      const buffer = deviceStorageOf(value).buffer as { destroy(): void };
      const destroy = buffer.destroy.bind(buffer);
      buffer.destroy = () => { collected.tensors++; destroy() };
      return new WeakRef(value);
    }
    const weakTensor = await abandonTensor();
    const weakPlan = new WeakRef(await backend.compile(unresolvedSquares, { inputs: [tensor([1, 2, 3])] }));
    for (let attempt = 0; attempt < 200; attempt++) {
      await setImmediate(); global.gc!(); await setImmediate();
      if (collected.tensors === 1 && !weakPlan.deref()) break;
    }
    assert.equal(weakTensor.deref(), undefined);
    assert.equal(weakPlan.deref(), undefined);
    assert.equal(collected.tensors, 1, 'GC destroys an unreachable GPU tensor buffer exactly once');

    const disposableBackend = await typegpu({ device });
    const liveTensor = await disposableBackend.copyFrom(tensor([1, 2, 3]));
    const livePlan = await disposableBackend.compile(unresolvedSquares, { inputs: [tensor([1, 2, 3])] });
    disposableBackend.dispose();
    assert.ok(liveTensor.isDisposed, 'Backend disposal invalidates live Tensor handles');
    await assert.rejects(liveTensor.read(), /disposed/);
    assert.throws(() => livePlan(tensor([1, 2, 3], { shape: [3] })), /disposed/);
    liveTensor.dispose(); livePlan[Symbol.dispose](); disposableBackend.dispose();
    await backend.synchronize();
    assert.deepEqual(errors, [], 'WebGPU validation must pass');
    console.log(`Passed GPU dtypes: ${backend.capabilities.dtypes.join(', ')}; unsupported f64/i64/u64 and missing shader-f16 are rejected.`);
    console.log('Passed: GPU fusion, imported helpers, gradients, 5th derivatives, custom VJP/JVP, finite differences, shapes, GPU SGD, structured gradients, GPU accumulation, and lazy compilation.');
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
