import assert from 'node:assert/strict';
import { tensor, compileLazy, valueAndGrad, type TensorF32 } from 'blaten';
import sourceText from './trace.ts?raw';
import { attachSources } from 'blaten/trace';
import { typegpu, inspect } from '@blaten/typegpu';

function squareSum(x: TensorF32<[number]>) { 'use kernel'; return x.mul(x).sum() }
const derivative = valueAndGrad(squareSum);

export async function testTrace(device: GPUDevice) {
  const backend = await typegpu({ device, trace: true }), trace = backend.trace!;
  const captured: string[] = [];
  const original = device.createShaderModule;
  device.createShaderModule = function (descriptor) {
    captured.push(descriptor.code);
    return original.call(this, descriptor);
  };
  try {
    using x = tensor([2, 3], { shape: [2] });
    using plan = await backend.compile(derivative, { inputs: [x] });
    const first = trace.toResult(), record = first.compilations[0]!;
    assert.equal(record.status, 'success');
    assert.equal(record.name, 'valueAndGrad(squareSum)');
    const shaders = record.artifacts.filter(a => a.content.type === 'text' && a.content.language === 'wgsl').map(a => (a.content as { value: string }).value);
    assert.deepEqual(shaders, captured, 'Record the exact WGSL submitted to GPUDevice');
    assert.equal(shaders.length, inspect(plan).dispatchCount);
    const kernel = record.artifacts.find(a => a.kind === 'blaten.kernel-ir')!.content;
    assert.equal(kernel.type, 'json');
    const kernels = (kernel as unknown as { value: { kernels: { kind: string; source?: { file: string; artifact?: string } }[] } }).value.kernels;
    assert.ok(kernels.some(k => k.kind === 'grad'));
    assert.ok(kernels.some(k => k.source?.file === 'tests/trace.ts'));
    assert.ok(!record.artifacts.some(a => a.content.type === 'text' && a.content.language === 'typescript'));
    attachSources(first, { 'tests/trace.ts': sourceText });
    const source = first.sources.find(a => a.content.type === 'text' && a.content.language === 'typescript')!;
    assert.ok(source);
    assert.equal((source.content as { value: string }).value, sourceText);
    assert.ok(kernels.some(k => k.source?.artifact === source.id));
    assert.equal(source.location!.end - source.location!.start, (source.content as { value: string }).value.length);
    assert.equal(source.location!.file, 'tests/trace.ts');
    assert.ok(record.stages.find(s => s.id === 'source')!.artifacts.includes(source.id));
    assert.deepEqual(trace.toResult().sources, [], 'Attaching sources does not change runtime records');
    const graph = (record.artifacts.find(a => a.kind === 'blaten.tensor-ir')!.content as unknown as { value: { nodes: { id: string }[] } }).value;
    const ids = new Set(graph.nodes.map(n => n.id));
    const execution = (record.artifacts.find(a => a.kind === 'typegpu.execution-plan')!.content as unknown as { value: { dispatches: { output: string; fusedNodes: string[]; shader: string }[] } }).value;
    for (const dispatch of execution.dispatches) {
      assert.ok(ids.has(dispatch.output));
      assert.ok(dispatch.fusedNodes.every(id => ids.has(id)));
      assert.ok(record.artifacts.some(a => a.id === dispatch.shader));
    }
    const output = plan(x);
    assert.equal(await output.value.item(), 13);
    assert.deepEqual(Array.from(await output.grads.read()), [4, 6]);
    output.value.dispose(); output.grads.dispose();
    await assert.rejects(backend.compile(squareSum), /shape/);
    assert.equal(trace.toResult().compilations.at(-1)!.status, 'failed');
    assert.ok(trace.toResult().compilations.at(-1)!.artifacts.some(a => a.kind === 'blaten.kernel-ir'));
    using lazy = compileLazy(squareSum, backend);
    const before = trace.toResult().compilations.length;
    using lease1 = await lazy.prepare(x);
    using lease2 = await lazy.prepare(x);
    using x3 = tensor([1, 2, 3], { shape: [3] });
    using lease3 = await lazy.prepare(x3);
    assert.equal(trace.toResult().compilations.length, before + 2, 'Cache hits do not create compilation records');
    const gpuX = await backend.copyFrom(x);
    const direct = gpuX.add(gpuX); direct.dispose(); gpuX.dispose();
    assert.ok(trace.toResult().compilations.some(c => c.cause === 'direct'));
    assert.equal(first.compilations.length, 1);
    const result = trace.toResult();
    assert.deepEqual(JSON.parse(JSON.stringify(result)), result);
    console.log('Passed: exact traced WGSL, gradient graphs, source metadata, failure artifacts, lazy specialization, direct plans, and JSON round trips.');
  } finally { device.createShaderModule = original; backend.dispose() }
}
