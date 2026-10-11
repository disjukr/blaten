import assert from 'node:assert/strict';
import { ImageDataset, dataLoader, imageClassification } from '@blaten/datasets';
import { tensor } from 'blaten';

const dataset = new ImageDataset({
  pixels: new Uint8Array([0, 255, 64, 128, 255, 0, 128, 64, 255, 255]),
  labels: new Uint8Array([0, 1, 1, 0, 1]), featureCount: 2, imageShape: [1, 2, 1],
  metadata: { name: 'fixture', split: 'train', classNames: ['zero', 'one'] },
});
const collate = imageClassification({ layout: 'flat', labels: 'oneHot' });
const collect = async options => Array.fromAsync(dataLoader({ dataset, batchSize: 2, collate, ...options }));
assert.equal(dataset.count, 5);
assert.deepEqual(dataset.imageShape, [1, 2, 1]);
assert.ok(Object.isFrozen(dataset.metadata) && Object.isFrozen(dataset.metadata.classNames));
assert.deepEqual(await dataset.get(1), { image: { data: new Uint8Array([64, 128]), width: 2, height: 1, channels: 1, featureCount: 2 }, label: 1 });
for (const index of [-1, 5, 0.5, NaN]) await assert.rejects(dataset.get(index), RangeError);
const batches = await collect({});
assert.deepEqual(batches.map(batch => [batch.size, batch.capacity]), [[2, 2], [2, 2], [1, 1]]);
assert.deepEqual(batches.map(batch => batch.indices), [[0, 1], [2, 3], [4]]);
assert.ok(Object.isFrozen(batches[0].indices));
assert.deepEqual([...await batches[0].data.images.read()], [0, 1, Math.fround(64 / 255), Math.fround(128 / 255)]);
assert.deepEqual([...await batches[0].data.targets.read()], [1, 0, 0, 1]);
assert.deepEqual(batches[2].data.images.shape, [1, 2]);
const firstAgain = (await collect({ size: 2 }))[0];
assert.notEqual(batches[0].data.images, firstAgain.data.images);
batches[0][Symbol.dispose](); batches[0].dispose();
assert.ok(batches[0].data.images.isDisposed && batches[0].data.targets.isDisposed);
assert.ok(!firstAgain.data.images.isDisposed && !batches[1].data.images.isDisposed, 'Disposal only releases this batch');
for (const batch of batches) batch.dispose();
firstAgain.dispose();
const padded = await collect({ lastBatch: 'pad' });
assert.deepEqual([padded[2].size, padded[2].capacity], [1, 2]);
assert.deepEqual([...await padded[2].data.images.read()], [1, 1, 0, 0]);
assert.deepEqual([...await padded[2].data.targets.read()], [0, 1, 0, 0]);
for (const batch of padded) batch.dispose();
const dropped = await collect({ lastBatch: 'drop' });
assert.equal(dropped.length, 2); dropped.forEach(batch => batch.dispose());
assert.equal((await collect({ batchSize: 9, lastBatch: 'drop' })).length, 0);
const subset = await collect({ size: 3 });
assert.deepEqual(subset.map(batch => batch.indices), [[0, 1], [2]]); subset.forEach(batch => batch.dispose());
const order = async seed => {
  const batches = await collect({ shuffleSeed: seed });
  try { return batches.flatMap(batch => batch.indices) } finally { batches.forEach(batch => batch.dispose()) }
};
assert.deepEqual(await order(101), await order(101));
assert.notDeepEqual(await order(101), await order(102));
assert.deepEqual((await order(101)).toSorted((a, b) => a - b), [0, 1, 2, 3, 4]);
assert.equal((await collect({ size: 0 })).length, 0);
for (const options of [{ batchSize: 0 }, { batchSize: 1.5 }, { size: 6 }, { size: -1 }, { shuffleSeed: NaN }, { lastBatch: 'invalid' }]) {
  assert.throws(() => dataLoader({ dataset, batchSize: 2, collate, ...options }), /Invalid|safe integer/);
}
assert.throws(() => new ImageDataset({ pixels: new Uint8Array(2), labels: new Uint8Array([2]), featureCount: 2, imageShape: [1, 2, 1], metadata: { name: 'invalid', split: 'train', classNames: ['zero', 'one'] } }), /Invalid/);
assert.throws(() => new ImageDataset({ pixels: new Uint8Array(2), labels: new Uint8Array([0]), featureCount: 2, imageShape: [2, 2, 1], metadata: { name: 'invalid', split: 'train', classNames: ['zero'] } }), /shape/);

const rgb = new ImageDataset({ pixels: new Uint8Array([1, 2, 3, 4, 5, 6]), labels: new Uint8Array([1]), featureCount: 6, imageShape: [1, 2, 3], metadata: { name: 'rgb', split: 'test', classNames: ['zero', 'one'] } });
for (const layout of ['Nchw', 'Nhwc']) {
  const [batch] = await Array.fromAsync(dataLoader({ dataset: rgb, batchSize: 2, lastBatch: 'pad', collate: imageClassification({ layout, scale: 1 }) }));
  assert.deepEqual(batch.data.images.shape, layout === 'Nchw' ? [2, 3, 1, 2] : [2, 1, 2, 3]);
  assert.deepEqual([...await batch.data.images.read()], [...(layout === 'Nchw' ? [1, 4, 2, 5, 3, 6] : [1, 2, 3, 4, 5, 6]), 0, 0, 0, 0, 0, 0]);
  assert.equal(batch.data.targets.dtype, 'u32');
  assert.deepEqual([...await batch.data.targets.read()], [1, 0]);
  batch.dispose();
}
for (const options of [{ scale: NaN }, { layout: 'invalid' }, { labels: 'invalid' }]) assert.throws(() => imageClassification(options), /Invalid/);
const context = { metadata: dataset.metadata, size: 1, capacity: 1 };
const sample = await dataset.get(0);
assert.throws(() => collate([{ ...sample, label: 2 }], context), /class label/);
assert.throws(() => collate([{ image: { ...sample.image, data: new Uint8Array(1) }, label: 0 }], context), /dimensions/);
const samples = [await dataset.get(0), await dataset.get(1)];
assert.throws(() => collate([samples[0], { ...samples[1], image: { ...samples[1].image, width: 1 } }], { ...context, size: 2, capacity: 2 }), /dimensions/);

// Generic records and async transforms/collators do not need image metadata or tensors.
const texts = { count: 3, metadata: { name: 'texts', split: 'train' }, async get(index) { return { text: ['a', 'bb', 'ccc'][index] } } };
let releases = 0;
const transformed = dataLoader({ dataset: texts, batchSize: 2, transform: async (sample, context) => ({ length: sample.text.length, seed: context.seed, index: context.index }), collate: async samples => ({ data: { lengths: samples.map(sample => sample.length), seeds: samples.map(sample => sample.seed), indices: samples.map(sample => sample.index) }, dispose: () => releases++ }) });
const one = await Array.fromAsync(transformed), two = await Array.fromAsync(transformed);
assert.deepEqual(one.map(batch => batch.data), two.map(batch => batch.data), 'Iteration repeats ordering and transform seeds');
assert.deepEqual(one.map(batch => batch.data.lengths), [[1, 2], [3]]);
assert.deepEqual(one.map(batch => batch.data.indices), [[0, 1], [2]]);
one[0].dispose(); one[0][Symbol.dispose](); assert.equal(releases, 1);
one.slice(1).forEach(batch => batch.dispose()); two.forEach(batch => batch.dispose()); assert.equal(releases, 4);
const borrowed = tensor([42]);
const [borrowBatch] = await Array.fromAsync(dataLoader({ dataset: texts, batchSize: 3, collate: () => ({ data: { borrowed } }) }));
borrowBatch.dispose(); assert.deepEqual([...await borrowed.read()], [42]); borrowed.dispose();
let retained;
for await (const batch of dataLoader({ dataset: rgb, batchSize: 1, collate: imageClassification() })) { retained = batch; break }
assert.ok(!retained.data.images.isDisposed, 'Early iterator exit does not dispose a returned batch'); retained.dispose();
await assert.rejects(Array.fromAsync(dataLoader({ dataset: { ...texts, async get() { throw new Error('read failed') } }, batchSize: 1, collate: samples => ({ data: samples }) })), /read failed/);
await assert.rejects(Array.fromAsync(dataLoader({ dataset: texts, batchSize: 1, transform() { throw new Error('transform failed') }, collate: samples => ({ data: samples }) })), /transform failed/);
await assert.rejects(Array.fromAsync(dataLoader({ dataset: texts, batchSize: 1, collate() { throw new Error('collate failed') } })), /collate failed/);

// first() reads only one batch, preserves returned ownership and leaves other iterations alone.
let firstReads = 0, firstReleases = 0;
const firstLoader = dataLoader({
  dataset: { ...texts, async get(index) { firstReads++; return texts.get(index) } },
  batchSize: 2,
  collate: samples => ({ data: samples.map(sample => sample.text), dispose: () => firstReleases++ }),
});
const ongoing = firstLoader[Symbol.asyncIterator]();
const initial = (await ongoing.next()).value;
assert.deepEqual(initial.data, ['a', 'bb']);
const first = await firstLoader.first();
assert.equal(firstReads, 4, 'first() reads only the first batch');
assert.deepEqual(first.data, initial.data);
assert.notEqual(first, initial);
assert.equal(firstReleases, 0, 'Closing the first() iterator preserves the returned batch');
const following = (await ongoing.next()).value;
assert.deepEqual(following.data, ['ccc'], 'first() does not advance an existing iterator');
const repeated = await firstLoader.first();
assert.deepEqual(repeated.data, first.data);
for (const batch of [initial, first, following, repeated]) batch.dispose();
assert.equal(firstReleases, 4);
await ongoing.return();
for (const settings of [{ size: 0 }, { batchSize: 9, lastBatch: 'drop' }]) {
  assert.equal(await dataLoader({ dataset, batchSize: 2, collate, ...settings }).first(), undefined);
}
const firstShuffled = dataLoader({ dataset, batchSize: 2, shuffleSeed: 101, collate });
const firstShuffledBatch = await firstShuffled.first();
assert.deepEqual(firstShuffledBatch.indices, (await order(101)).slice(0, 2)); firstShuffledBatch.dispose();
const firstPadded = await dataLoader({ dataset: rgb, batchSize: 2, lastBatch: 'pad', collate: imageClassification() }).first();
assert.deepEqual([firstPadded.size, firstPadded.capacity], [1, 2]);
assert.ok(!firstPadded.data.images.isDisposed); firstPadded.dispose();
await assert.rejects(dataLoader({ dataset: texts, batchSize: 1, collate() { throw new Error('first failed') } }).first(), /first failed/);
