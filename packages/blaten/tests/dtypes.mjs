import assert from 'node:assert/strict';
import { tensor, zeros, ones, randn, bool, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64, grad, sum, map, log, tensorType } from 'blaten';
import { lower } from 'blaten/backend';
import { registerKernel } from 'blaten/internal';
import { evaluateCpu } from 'blaten/intrinsics';
const cases = [[bool, Uint8Array], [i8, Int8Array], [u8, Uint8Array], [i16, Int16Array], [u16, Uint16Array], [i32, Int32Array], [u32, Uint32Array], [i64, BigInt64Array], [u64, BigUint64Array], [f16, Float16Array], [f32, Float32Array], [f64, Float64Array]];
for (const [dtype, Constructor] of cases) {
  const x = tensor(dtype === bool ? [true, false] : [1, 2], { dtype });
  assert.equal(x.dtype, dtype);
  assert.ok(x.data instanceof Constructor);
  assert.ok(await x.read() instanceof Constructor);
  assert.equal(zeros([2], { dtype }).dtype, dtype);
  assert.equal(ones([2], { dtype }).dtype, dtype);
  assert.equal(x.reshape([1, 2]).dtype, dtype);
  if (dtype !== bool) {
    assert.deepEqual(Array.from(await x.mul(2).read()), dtype === i64 || dtype === u64 ? [2n, 4n] : [2, 4]);
    assert.equal(await x.sum().item(), dtype === i64 || dtype === u64 ? 3n : 3);
    assert.deepEqual(Array.from(x.gt(1).data), [0, 1]);
  }
}
assert.equal(tensor([true, false]).dtype, bool);
assert.equal(await tensor(true).item(), true);
assert.equal(tensor([1n, 2n]).dtype, i64);
assert.equal(tensor(new Float64Array([1, 2])).dtype, f64);
assert.equal(tensor(new Int32Array([1, 2])).dtype, i32);
assert.equal(tensor([1, 2]).dtype, f32);
assert.equal(await tensor([true, false, true]).sum().item(), 2);
assert.throws(() => tensor([true]).add(tensor([false])), /Boolean arithmetic/);
assert.deepEqual(Array.from(tensor([true, false]).cast(f32).data), [1, 0]);
assert.deepEqual(Array.from(tensor([0, 2, -1]).cast(bool).data), [0, 1, 1]);
assert.deepEqual(Array.from(tensor([NaN, Infinity, -Infinity, -0]).cast(bool).data), [1, 1, 1, 0]);
assert.deepEqual(Array.from(tensor([true, false]).not().data), [0, 1]);
assert.deepEqual(Array.from(tensor([true, false]).and(true).data), [1, 0]);
assert.deepEqual(Array.from(tensor([true, false]).or(true).data), [1, 1]);
assert.throws(() => tensor([1]).not(), /bool inputs/);
assert.equal(await tensor([0.1], { dtype: f16 }).item(), 0.0999755859375);
assert.equal(await tensor([1], { dtype: f64 }).add(2 ** -40).item(), 1 + 2 ** -40);
assert.equal(await tensor([127], { dtype: i8 }).add(1).item(), -128);
assert.equal(await tensor([255], { dtype: u8 }).add(1).item(), 0);
assert.equal(await tensor([2147483647], { dtype: i32 }).mul(2147483647).item(), 1);
assert.equal(await tensor([9007199254740993n], { dtype: i64 }).mul(3n).item(), 27021597764222979n);
assert.equal(await tensor([18446744073709551615n], { dtype: u64 }).add(1n).item(), 0n);
assert.equal(await tensor([-1.9], { dtype: f64 }).cast(i64).item(), -1n);
assert.equal(tensor([1], { dtype: i8 }).add(tensor([1], { dtype: u8 })).dtype, i16);
assert.equal(tensor([1], { dtype: i32 }).div(2).dtype, f32);
assert.equal(tensor([1], { dtype: i8 }).add(128).dtype, f32);
assert.equal(tensor([1], { dtype: f16 }).add(tensor([1], { dtype: f64 })).dtype, f64);
assert.equal(tensor([[1, 2]], { dtype: i64 }).matmul(tensor([[3], [4]], { dtype: i64 })).data[0], 11n);
assert.throws(() => tensor([9007199254740992], { dtype: i64 }), /safe integer/);
assert.throws(() => randn([2], { dtype: i32 }), /floating dtype/);
const logarithm = registerKernel(x => log(x), {
  version: 2, name: 'logarithm', externals: { log: () => log }, ir: {
    parameterCount: 1, instructions: [{ op: 'argument', index: 0 }, { op: 'external', name: 'log' }, { op: 'call', target: 1, args: [0] }], result: 2,
  },
});
const preciseLog = map(logarithm, tensor([1 + 2 ** -40], { dtype: f64 }));
assert.equal(preciseLog.dtype, f64);
assert.equal(await preciseLog.item(), Math.log(1 + 2 ** -40), 'Registered map callbacks retain the input dtype in their intermediate calculations');
assert.equal(map(logarithm, tensor([1, 2], { dtype: i32 })).dtype, f32);
const maskKernel = registerKernel(x => x > 1, {
  version: 2, name: 'mask', externals: {}, ir: {
    parameterCount: 1, instructions: [{ op: 'argument', index: 0 }, { op: 'literal', value: 1 }, { op: 'binary', operator: 'gt', left: 0, right: 1 }], result: 2,
  },
});
assert.equal(map(maskKernel, tensor([1, 2], { dtype: f64 })).dtype, bool);
assert.deepEqual(Array.from(map(maskKernel, tensor([1, 2], { dtype: f64 })).data), [0, 1]);

const squaredSum = registerKernel(x => x.mul(x).sum(), {
  version: 2, name: 'squaredSum', externals: { sum: () => sum }, ir: {
    parameterCount: 1, instructions: [{ op: 'argument', index: 0 }, { op: 'binary', operator: 'mul', left: 0, right: 0 }, { op: 'external', name: 'sum' }, { op: 'call', target: 2, args: [1] }], result: 3,
  },
});
for (const dtype of [f16, f32, f64]) {
  const program = lower(grad(squaredSum), { inputs: [tensorType([2], dtype)] });
  assert.equal(program.outputs[0].dtype, dtype);
  // Replace symbolic input by a typed constant to execute the differentiated graph on CPU.
  const input = program.inputs[0];
  Object.assign(input, { op: 'constant', attrs: { data: tensor([1.5, 2.5], { dtype }).data } });
  assert.deepEqual(Array.from(evaluateCpu(program.outputs[0])), [3, 5]);
}
assert.throws(() => lower(grad(squaredSum), { inputs: [tensorType([2], i32)] }), /floating outputs and targets/);
assert.throws(() => lower(squaredSum, { inputs: [tensorType([2], f64)], output: f32 }), /shape or dtype/);
console.log('Passed: all dtype storage, inference, casts, promotions, half rounding, exact 64-bit arithmetic, overflow, and float-only autodiff.');
