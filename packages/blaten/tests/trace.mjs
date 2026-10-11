import assert from 'node:assert/strict';
import { createTrace, attachSources } from 'blaten/trace';
import { traceProgram } from 'blaten/backend';

const trace = createTrace();
const first = trace.begin({ name: 'training', backend: 'test', cause: 'explicit' });
const second = trace.begin({ name: 'inference', backend: 'other', cause: 'explicit' });
const artifact = { id: 'ir', kind: 'test.ir', schemaVersion: 1, content: { type: 'json', value: { nodes: [1, 2] } } };
first.addArtifact(artifact);
artifact.content.value.nodes.push(3);
first.addStage({ id: 'lower', name: 'Lower', inputs: [], artifacts: ['ir'] });
const pending = trace.toResult();
assert.deepEqual(pending.compilations.map(c => c.status), ['pending', 'pending']);
first.complete();
second.fail({ name: 'Error', message: 'Lowering failed' });
const result = trace.toResult();
assert.deepEqual(result.compilations.map(c => c.status), ['success', 'failed']);
assert.deepEqual(result.compilations[0].artifacts[0].content.value.nodes, [1, 2]);
assert.deepEqual(pending.compilations.map(c => c.status), ['pending', 'pending']);
result.compilations[0].artifacts[0].content.value.nodes.push(99);
assert.deepEqual(trace.toResult().compilations[0].artifacts[0].content.value.nodes, [1, 2]);
assert.deepEqual(JSON.parse(JSON.stringify(trace.toResult())), trace.toResult());
const sourceTrace = createTrace();
const sourceRecord = sourceTrace.begin({ name: 'source', backend: 'test', cause: 'explicit' });
sourceRecord.addArtifact({ id: 'kernel-ir', kind: 'blaten.kernel-ir', schemaVersion: 1, content: { type: 'json', value: {
  kernels: [
    { id: 'k1', kind: 'kernel', source: { file: 'src/a.ts', start: 2, end: 5 } },
    { id: 'k2', kind: 'kernel', source: { file: 'src/b.ts', start: 0, end: 3 } },
    { id: 'k3', kind: 'grad', source: 'k1' },
    { id: 'k4', kind: 'kernel', source: { file: 'src/a.ts', start: 0, end: 2 } },
  ],
} } });
sourceRecord.addStage({ id: 'kernel', name: 'Kernel IR', inputs: [], artifacts: ['kernel-ir'] });
const sharedRecord = sourceTrace.begin({ name: 'shared', backend: 'test', cause: 'explicit' });
sharedRecord.addArtifact({ id: 'kernel-ir', kind: 'blaten.kernel-ir', schemaVersion: 1, content: { type: 'json', value: {
  kernels: [{ id: 'k1', kind: 'kernel', source: { file: 'src/a.ts', start: 2, end: 5 } }],
} } });
sharedRecord.addStage({ id: 'kernel', name: 'Kernel IR', inputs: [], artifacts: ['kernel-ir'] });
const enriched = JSON.parse(JSON.stringify(sourceTrace.toResult()));
attachSources(enriched, { 'src/a.ts': '00abc99' });
assert.equal(enriched.sources.length, 1);
assert.equal(enriched.sources[0].content.value, '00abc99');
const kernelSource = enriched.compilations[0].artifacts[0].content.value.kernels[0].source;
assert.equal(kernelSource.artifact, 's1');
assert.equal(enriched.sources[0].content.value.slice(kernelSource.start, kernelSource.end), 'abc');
assert.equal(enriched.compilations[0].artifacts[0].content.value.kernels[3].source.artifact, 's1');
assert.equal(enriched.compilations[1].artifacts[0].content.value.kernels[0].source.artifact, 's1');
assert.deepEqual(enriched.compilations[0].stages[0].artifacts, ['s1']);
assert.equal(enriched.compilations[0].artifacts[0].content.value.kernels[1].source.artifact, undefined);
attachSources(enriched, { 'src/b.ts': 'defghi' });
attachSources(enriched, { 'src/a.ts': '00xyz99' });
assert.equal(enriched.compilations[0].artifacts.length, 1);
assert.equal(enriched.sources.length, 2);
assert.equal(enriched.sources[0].content.value, '00xyz99');
assert.deepEqual(enriched.sources[0].location, { file: 'src/a.ts', start: 0, end: 7 });
assert.deepEqual(enriched.compilations[0].stages[0].artifacts, ['s1', 's2']);
assert.deepEqual(enriched.compilations[0].stages[1].inputs, ['source']);
assert.equal(sourceTrace.toResult().compilations[0].artifacts.length, 1);
assert.deepEqual(sourceTrace.toResult().sources, []);
assert.deepEqual(JSON.parse(JSON.stringify(enriched)), enriched);
const constants = trace.begin({ name: 'constants', backend: 'test', cause: 'direct' });
const node = { id: 123, op: 'constant', shape: [], dtype: 'i64', inputs: [], attrs: { value: 9007199254740993n, data: new BigInt64Array([9007199254740993n]), special: [-0, NaN, Infinity] } };
traceProgram(constants, { inputs: [], outputs: [node], single: true });
const attrs = trace.toResult().compilations.at(-1).artifacts[0].content.value.nodes[0].attrs;
assert.deepEqual(attrs.value, { type: 'bigint', value: '9007199254740993' });
assert.deepEqual(attrs.data, [{ type: 'bigint', value: '9007199254740993' }]);
assert.deepEqual(attrs.special.map(v => v.value), ['-0', 'NaN', 'Infinity']);
assert.deepEqual(JSON.parse(JSON.stringify(trace.toResult())), trace.toResult());
console.log('Passed: independent compilation records, detached results, payload ownership, and JSON round trips.');
