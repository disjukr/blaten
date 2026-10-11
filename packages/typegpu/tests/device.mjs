import assert from 'node:assert/strict';
import { requestDevice } from '@blaten/typegpu';

function provider(features) {
  const requested = {};
  const device = { destroy() { requested.destroyed = true } };
  const gpu = { async requestAdapter(options) {
    requested.adapter = options;
    return { features: new Set(features), async requestDevice(descriptor) {
      requested.device = descriptor;
      return device;
    } };
  } };
  return { gpu, device, requested };
}
const half = provider(['shader-f16']);
assert.equal(await requestDevice({ gpu: half.gpu }), half.device);
assert.deepEqual(half.requested.device.requiredFeatures, ['shader-f16']);
const plain = provider([]);
await requestDevice({ gpu: plain.gpu });
assert.deepEqual(plain.requested.device.requiredFeatures, []);
const options = { gpu: half.gpu, adapterOptions: { powerPreference: 'high-performance' },
  label: 'test', requiredFeatures: [], requiredLimits: { maxStorageBuffersPerShaderStage: 8 } };
await requestDevice(options);
assert.deepEqual(half.requested.adapter, options.adapterOptions);
assert.deepEqual(half.requested.device, { label: 'test', requiredFeatures: [], requiredLimits: options.requiredLimits });
half.device[Symbol.dispose]();
assert.equal(half.requested.destroyed, true);
await assert.rejects(requestDevice({ gpu: { async requestAdapter() { return null } } }), /No WebGPU adapter/);
const failure = new Error('device request failed');
await assert.rejects(requestDevice({ gpu: { async requestAdapter() {
  return { features: new Set(), async requestDevice() { throw failure } };
} } }), error => error === failure);
console.log('Passed: device selection, feature defaults, descriptor overrides, and adapter/device request failures.');
