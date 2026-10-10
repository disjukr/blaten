import assert from 'node:assert/strict';
import { tensor, scope, disposeTree, sgd, gradientAccumulator, compileLazy, accumulate, meanMetric, meanAccumulator, multiclassAccuracy, argmax, eq, zeros } from 'blaten';
import { callable } from 'blaten/backend';

const parameters = { layer: [tensor([10, 20])], bias: tensor([5]) };
const accumulator = gradientAccumulator(parameters);
let borrowed;
scope(() => {
  borrowed = { bias: tensor([2]), layer: [tensor([1, 3])] };
  accumulator.add(borrowed, { weight: 2 });
});
assert.ok(borrowed.bias.isDisposed);
scope(() => accumulator.add({ layer: [tensor([7, 9])], bias: tensor([8]) }, { weight: 1 }));
assert.equal(accumulator.weight, 3);
const mean = accumulator.compute();
assert.deepEqual([...await mean.layer[0].read()], [3, 5]);
assert.deepEqual([...await mean.bias.read()], [4]);
sgd(parameters, { learningRate: 0.5 }).step(mean);
assert.deepEqual([...await parameters.layer[0].read()], [8.5, 17.5]);
assert.deepEqual([...await parameters.bias.read()], [3]);
disposeTree(mean);
assert.throws(() => accumulator.add({ layer: [tensor([1])], bias: tensor([1]) }), /matching/);
assert.equal(accumulator.weight, 3, 'Invalid add must leave accumulator state intact');
accumulator.reset();
assert.throws(() => accumulator.compute(), /empty/);
scope(() => accumulator.add({ layer: [tensor([6, 12])], bias: tensor([3]) }));
assert.deepEqual([...await scope(() => accumulator.compute().layer[0].read())], [6, 12]);
accumulator.reset();
scope(() => accumulator.add({ layer: [tensor([Infinity, NaN])], bias: tensor([Infinity]) }));
accumulator.reset();
scope(() => accumulator.add({ layer: [tensor([6, 12])], bias: tensor([3]) }));
assert.deepEqual([...await scope(() => accumulator.compute().layer[0].read())], [6, 12], 'reset must reset nonfinite values, not multiply them by zero');
accumulator.dispose(); accumulator.dispose();
assert.throws(() => accumulator.reset(), /disposed/);
disposeTree(parameters);

// The general accumulator preserves any floating tensor tree, independent of template lifetime.
let vectorAccumulator;
scope(() => { vectorAccumulator = meanAccumulator(tensor([0, 0], { shape: [2] })) });
assert.throws(() => vectorAccumulator.compute(), /empty/);
scope(() => vectorAccumulator.add(tensor([2, 4], { shape: [2] }), { weight: 3 }));
scope(() => vectorAccumulator.add(tensor([6, 8], { shape: [2] }), { weight: 1 }));
const vectorSnapshot = vectorAccumulator.compute();
vectorAccumulator.reset();
assert.equal(vectorAccumulator.weight, 0);
assert.throws(() => vectorAccumulator.compute(), /empty/);
scope(() => vectorAccumulator.add(tensor([10, 20], { shape: [2] })));
assert.deepEqual([...await scope(() => vectorAccumulator.compute().read())], [10, 20]);
assert.throws(() => scope(() => vectorAccumulator.add(tensor([1, 2], { shape: [2], dtype: 'f64' }))), /matching/);
assert.equal(vectorAccumulator.weight, 1);
vectorAccumulator[Symbol.dispose]();
assert.deepEqual([...await vectorSnapshot.read()], [3, 5], 'Computed trees must survive reset and disposal');
vectorSnapshot.dispose();
assert.throws(() => scope(() => meanAccumulator(tensor([1], { dtype: 'i32' }))), /floating/);
assert.throws(() => meanAccumulator({}), /nonempty/);
// Repeated leaves are valid data; the gradient-specific wrapper still rejects duplicate parameters.
scope(() => {
  const shared = tensor([0]);
  using pairs = meanAccumulator([shared, shared]);
  pairs.add([shared.add(2), shared.add(4)]);
  assert.deepEqual(pairs.compute().map(value => [...value.shape]), [[1], [1]]);
  assert.throws(() => gradientAccumulator([shared, shared]), /distinct/);
});

// Arbitrary argmax axes, ties, nonfinite values and exact integer comparisons.
scope(() => {
  const scores = tensor([1, 5, 5, -3, -1, -2], { shape: [2, 3] });
  assert.deepEqual([...argmax(scores, -1).shape], [2]);
  assert.deepEqual([...scores.argmax(0).shape], [3]);
  assert.throws(() => scores.argmax(2), /axis/);
  assert.throws(() => scores.argmax(0.5), /axis/);
  assert.throws(() => tensor(1, { shape: [] }).argmax(0), /axis/);
});
assert.deepEqual([...await scope(() => tensor([1, 5, 5, -3, -1, -2], { shape: [2, 3] }).argmax(-1).read())], [1, 1]);
assert.deepEqual([...await scope(() => tensor([1, 5, 5, 3, 1, 9], { shape: [2, 3] }).argmax(0).read())], [1, 0, 1]);
assert.deepEqual([...await scope(() => tensor([0, NaN, NaN, Infinity], { shape: [4] }).argmax(0).read())], [1]);
assert.deepEqual([...await scope(() => tensor([-Infinity, -Infinity], { shape: [2] }).argmax(0).read())], [0]);
assert.deepEqual([...await scope(() => tensor([9007199254740992n, 9007199254740993n], { shape: [2] }).argmax(0).read())], [1]);
assert.deepEqual([...await scope(() => tensor([false, true, true], { shape: [3] }).argmax(0).read())], [1]);
assert.deepEqual([...await scope(() => eq(tensor([1, 2], { shape: [2, 1] }), tensor([1, 2, 3], { shape: [3] })).read())], [1, 0, 0, 0, 1, 0]);
assert.equal(eq(true, false), false);
assert.deepEqual([...await scope(() => tensor([9007199254740992n, 9007199254740993n]).eq(9007199254740993n).read())], [0, 1]);
{
  using accuracy = multiclassAccuracy();
  assert.throws(() => accuracy.compute(), /empty/);
  scope(() => accuracy.add({ logits: tensor([2, 1, 1, 2, 2, 1], { shape: [3, 2] }), targets: tensor([1, 0, 1, 0, 1, 0], { shape: [3, 2] }), validCount: 2 }));
  // The final matching padded row must not count: 1/2, not 2/3.
  assert.equal(await scope(() => accuracy.compute().item()), 0.5);
  scope(() => accuracy.add({ logits: tensor([2, 1], { shape: [1, 2] }), targets: tensor([1, 0], { shape: [1, 2] }) }));
  assert.ok(Math.abs(await scope(() => accuracy.compute().item()) - 2 / 3) < 1e-6);
  assert.equal(accuracy.weight, 3);
  assert.throws(() => scope(() => accuracy.add({ logits: zeros([1, 2]), targets: zeros([1, 3]) })), /matching/);
  for (const validCount of [0, -1, 2, 0.5, NaN]) assert.throws(() => scope(() => accuracy.add({ logits: zeros([1, 2]), targets: zeros([1, 2]), validCount })), /validCount/);
  assert.equal(accuracy.weight, 3);
  using snapshot = accuracy.compute();
  accuracy.reset();
  assert.throws(() => accuracy.compute(), /empty/);
  assert.ok(Math.abs(await snapshot.item() - 2 / 3) < 1e-6);
}

// The optimizer wrapper owns buffers but borrows gradients and parameters.
const wrappedParameters = { weight: tensor([10]), bias: tensor(5, { shape: [] }) };
const wrapped = accumulate(sgd(wrappedParameters, { learningRate: 1 }), { steps: 2 });
assert.throws(() => accumulate(sgd(wrappedParameters, { learningRate: 1 }), { steps: 0 }), /positive integer/);
let gradient;
scope(() => {
  gradient = { weight: tensor([2]), bias: tensor(4, { shape: [] }) };
  gradient.weight.read = async () => { throw new Error('Must not read gradients') };
  wrapped.step(gradient, { weight: 3 });
});
assert.ok(gradient.weight.isDisposed);
assert.equal(wrapped.pendingSteps, 1);
assert.equal(await wrappedParameters.weight.item(), 10);
assert.throws(() => wrapped.step(wrappedParameters, { weight: 0 }), /positive/);
assert.throws(() => scope(() => wrapped.step({ weight: tensor([1, 2]), bias: tensor(1, { shape: [] }) })), /matching/);
assert.equal(wrapped.pendingSteps, 1, 'Rejected gradients must not advance accumulation');
scope(() => wrapped.step({ weight: tensor([6]), bias: tensor(8, { shape: [] }) }, { weight: 1 }));
assert.equal(wrapped.pendingSteps, 0);
assert.equal(await wrappedParameters.weight.item(), 7);
assert.equal(await wrappedParameters.bias.item(), 0);
scope(() => wrapped.step({ weight: tensor([1]), bias: tensor(2, { shape: [] }) }));
wrapped.flush(); wrapped.flush();
assert.equal(await wrappedParameters.weight.item(), 6);
assert.equal(await wrappedParameters.bias.item(), -2);
scope(() => wrapped.step({ weight: tensor([100]), bias: tensor(100, { shape: [] }) }));
wrapped[Symbol.dispose](); wrapped.dispose();
assert.equal(await wrappedParameters.weight.item(), 6, 'Disposal must discard pending gradients');
assert.ok(!wrappedParameters.weight.isDisposed);
assert.throws(() => wrapped.flush(), /disposed/);
{
  using immediate = accumulate(sgd(wrappedParameters, { learningRate: 1 }), { steps: 1 });
  scope(() => immediate.step({ weight: tensor([1]), bias: tensor(1, { shape: [] }) }, { weight: 7 }));
  assert.equal(immediate.pendingSteps, 0);
}
assert.equal(await wrappedParameters.weight.item(), 5);
disposeTree(wrappedParameters);

let metric;
let snapshot;
{
  using loss = meanMetric();
  metric = loss;
  assert.throws(() => loss.compute(), /empty/);
  scope(() => loss.add(tensor(2, { shape: [] }), { weight: 3 }));
  scope(() => loss.add(tensor(6, { shape: [] }), { weight: 1 }));
  assert.equal(loss.weight, 4);
  assert.equal(await scope(() => loss.compute().item()), 3);
  assert.throws(() => scope(() => loss.add(tensor(10, { shape: [] }), { weight: NaN })), /positive/);
  assert.throws(() => scope(() => loss.add(tensor([1]))), /scalar/);
  assert.equal(loss.weight, 4);
  snapshot = loss.compute();
  loss.reset();
  assert.throws(() => loss.compute(), /empty/);
  scope(() => loss.add(tensor(Infinity, { shape: [] })));
  loss.reset();
  scope(() => loss.add(tensor(7, { shape: [] })));
  assert.equal(await scope(() => loss.compute().item()), 7);
}
assert.equal(await snapshot.item(), 3, 'Reset and disposal must preserve computed snapshots');
snapshot.dispose();
assert.throws(() => metric.reset(), /disposed/);
assert.throws(() => scope(() => metric.add(tensor(1, { shape: [] }))), /disposed/);
let failedMetric, failedOptimizer, ownedTensor;
assert.throws(() => {
  using cleanup = meanMetric(); failedMetric = cleanup;
  using directTensor = zeros([]); ownedTensor = directTensor;
  using cleanupOptimizer = accumulate(sgd({ p: directTensor }, { learningRate: 1 }), { steps: 2 });
  failedOptimizer = cleanupOptimizer;
  scope(() => { cleanup.add(tensor(1, { shape: [] })); cleanupOptimizer.step({ p: tensor(2, { shape: [] }) }) });
  throw new Error('cleanup');
}, /cleanup/);
assert.throws(() => failedMetric.compute(), /disposed/);
assert.throws(() => failedOptimizer.flush(), /disposed/);
assert.ok(ownedTensor.isDisposed, 'using must clean up on errors');

const external = tensor([1]);
let temporary, childTemporary;
const kept = scope(() => {
  temporary = tensor([2]);
  return scope(() => { childTemporary = tensor([3]); return { kept: tensor([4]), external } });
});
assert.ok(temporary.isDisposed && childTemporary.isDisposed);
assert.ok(!kept.kept.isDisposed && !external.isDisposed);
disposeTree(kept);
let failed;
assert.throws(() => scope(() => { failed = tensor([5]); throw new Error('failure') }), /failure/);
assert.ok(failed.isDisposed);
assert.throws(() => scope(async () => tensor([1])), /synchronous callback/);
let pending, other;
let resolve, reject;
const first = scope(() => { pending = tensor([6]); return new Promise(r => { resolve = r }) });
const second = scope(() => { other = tensor([7]); return new Promise((_, r) => { reject = r }) });
assert.ok(!pending.isDisposed && !other.isDisposed);
resolve(42); assert.equal(await first, 42);
assert.ok(pending.isDisposed && !other.isDisposed, 'Overlapping read scopes must not share ownership');
reject(new Error('read failed')); await assert.rejects(second, /read failed/);
assert.ok(other.isDisposed);

let compilations = 0, disposals = 0, failNext = false;
const backend = { async compile() {
  compilations++;
  await Promise.resolve();
  if (failNext) { failNext = false; throw new Error('compile failure') }
  let disposed = false;
  return callable(x => { assert.ok(!disposed); return x.mul(2) }, () => { if (!disposed) { disposed = true; disposals++ } });
} };
const fn = compileLazy(x => x, backend, { cacheSize: 1 });
const x = tensor([1]), y = tensor([2, 3]);
let result = await fn(x); assert.deepEqual([...await result.read()], [2]); result.dispose();
result = await fn(x); result.dispose(); assert.equal(compilations, 1);
const lease = await fn.prepare(x);
result = await fn(y); result.dispose();
assert.equal(disposals, 0, 'A prepared plan must survive cache eviction while leased');
result = lease(x); result.dispose(); lease.dispose(); await Promise.resolve();
assert.equal(disposals, 1);
const concurrent = await Promise.all([fn(x), fn(y)]);
concurrent.forEach(result => result.dispose());
failNext = true;
await assert.rejects(fn(tensor([1, 2, 3])), /compile failure/);
result = await fn(tensor([1, 2, 3])); result.dispose();
fn.dispose(); await Promise.resolve();
await assert.rejects(fn(x), /disposed/);
x.dispose(); y.dispose();
console.log('Passed: weighted gradient accumulation, resource scopes, concurrent reads, and bounded lazy compilation caching.');
