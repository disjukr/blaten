import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { Tensor, tensor, compileLazy } from 'blaten';
import { Resource, callable } from 'blaten/backend';

// These integration checks run in Node with --expose-gc; finalizer timing is not
// an application guarantee. Yield between collections so finalizers can run.
async function eventually(predicate, message) {
  for (let attempt = 0; attempt < 200; attempt++) {
    await setImmediate();
    global.gc();
    await setImmediate();
    if (predicate()) return;
  }
  assert.fail(message);
}

const counts = { tensors: 0, plans: 0 };
const resources = new Set();
function deviceTensor() {
  const resource = new Resource(() => { counts.tensors++; resources.delete(resource) });
  resources.add(resource);
  return new Tensor({ owner: resources, buffer: {},
    get isDisposed() { return resource.isDisposed },
    read: async () => new Float32Array([3]), dispose: () => resource.dispose(),
  }, [], 'f32');
}
{
  using value = deviceTensor();
  assert.equal(await value.item(), 3);
}
assert.equal(counts.tensors, 1);
const explicit = deviceTensor();
explicit.dispose(); explicit[Symbol.dispose]();
assert.equal(counts.tensors, 2);
const forgottenTensor = new WeakRef(deviceTensor());
await eventually(() => counts.tensors === 3, 'An unreachable Tensor must release its device storage');
assert.equal(forgottenTensor.deref(), undefined);
assert.equal(resources.size, 0, 'Resource tracking must not keep Tensor objects alive');

const invalidated = deviceTensor();
resources.forEach(resource => resource.dispose());
assert.ok(invalidated.isDisposed);
await assert.rejects(invalidated.read(), /disposed/);
invalidated.dispose();
assert.equal(counts.tensors, 4);

// A pending read must keep its Tensor alive until it has completed.
let finishRead;
function pendingRead() {
  const value = new Tensor({ owner: resources, buffer: {},
    read: () => new Promise(resolve => { finishRead = resolve }),
    dispose: () => { counts.tensors++ },
  }, [], 'f32');
  return { weak: new WeakRef(value), promise: value.read() };
}
const reading = pendingRead();
for (let i = 0; i < 5; i++) { await setImmediate(); global.gc(); await setImmediate() }
assert.ok(reading.weak.deref());
assert.equal(counts.tensors, 4);
finishRead(new Float32Array([7]));
assert.deepEqual([...await reading.promise], [7]);
await eventually(() => counts.tensors === 5, 'A completed read must not retain its Tensor');

function plan() {
  const resource = new Resource(() => { counts.plans++; resources.delete(resource) });
  resources.add(resource);
  return callable(x => x.mul(2), resource);
}
const x = tensor([1]);
let snapshot;
{
  using compiled = plan();
  snapshot = compiled(x);
}
assert.deepEqual([...await snapshot.read()], [2]); snapshot.dispose();
assert.equal(counts.plans, 1);
const disposedPlan = plan();
disposedPlan.dispose(); disposedPlan[Symbol.dispose]();
assert.throws(() => disposedPlan(x), /disposed/);
assert.equal(counts.plans, 2);
const forgottenPlan = new WeakRef(plan());
await eventually(() => counts.plans === 3, 'An unreachable Executable must release its resources');
assert.equal(forgottenPlan.deref(), undefined);

const backend = { async compile() { return plan() } };
const lazy = compileLazy(x => x, backend, { cacheSize: 1 });
const prepared = await lazy.prepare(x);
const y = tensor([2, 3]);
(await lazy(y)).dispose(); // Evicts x while the prepared lease still uses it.
assert.equal(counts.plans, 3);
lazy[Symbol.dispose]();
await eventually(() => counts.plans === 4, 'Disposing the cache must release unleased plans');
using preparedResult = prepared(x);
assert.deepEqual([...await preparedResult.read()], [2], 'A lease survives cache disposal');
prepared[Symbol.dispose](); prepared.dispose();
await eventually(() => counts.plans === 5, 'The last lease releases an evicted plan');
await assert.rejects(lazy(x), /disposed/);

async function abandonedCache() {
  const lazy = compileLazy(x => x, backend);
  const lease = await lazy.prepare(x);
  return { weak: new WeakRef(lazy), lease };
}
const abandoned = await abandonedCache();
await eventually(() => abandoned.weak.deref() === undefined, 'A prepared lease must not retain its lazy cache');
using abandonedResult = abandoned.lease(x);
assert.deepEqual([...await abandonedResult.read()], [2]);
abandoned.lease.dispose();
await eventually(() => counts.plans === 6, 'A collected cache retires its leased plans');

async function abandonedLease() {
  const lazy = compileLazy(x => x, backend, { cacheSize: 1 });
  const lease = await lazy.prepare(x);
  (await lazy(y)).dispose();
  return { lazy, weak: new WeakRef(lease) };
}
const abandonedPrepared = await abandonedLease();
await eventually(() => counts.plans === 7, 'A collected prepared lease must release an evicted plan');
assert.equal(abandonedPrepared.weak.deref(), undefined);
abandonedPrepared.lazy.dispose();
await eventually(() => counts.plans === 8, 'Explicit disposal after GC remains safe');
x.dispose(); y.dispose();
assert.equal(resources.size, 0);
let completeCompilation;
const deferredBackend = { compile() { return new Promise(resolve => { completeCompilation = resolve }) } };
const inFlight = compileLazy(x => x, deferredBackend);
const pendingCall = inFlight(tensor([8]));
inFlight.dispose();
completeCompilation(plan());
using completedResult = await pendingCall;
assert.deepEqual([...await completedResult.read()], [16], 'An acquired lease protects an in-flight call');
await eventually(() => counts.plans === 9, 'In-flight calls release their retired plans on completion');
console.log('Passed: Tensor/Executable using, GC cleanup, pending reads, and independent cache/lease lifetimes.');
