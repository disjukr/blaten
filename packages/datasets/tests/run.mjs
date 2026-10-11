import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, writeFile, readdir, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import './batches.mjs';

import { cachedFile } from '../dist/cache.js';
import { parseIdx } from '../dist/idx.js';

await import('./catalog.mjs');
await import('./archives.mjs');

function idx() {
  const images = Buffer.alloc(16 + 2 * 784), labels = Buffer.alloc(8 + 2);
  images.writeUInt32BE(2051, 0); images.writeUInt32BE(2, 4); images.writeUInt32BE(28, 8); images.writeUInt32BE(28, 12);
  labels.writeUInt32BE(2049, 0); labels.writeUInt32BE(2, 4); labels[8] = 3; labels[9] = 7;
  return [images, labels];
}
const [images, labels] = idx();
assert.deepEqual([...parseIdx(images, labels, 2).labels], [3, 7]);
assert.throws(() => parseIdx(Buffer.alloc(2), labels, 2), /header/);
assert.throws(() => parseIdx(images, labels, 3), /dataset/);
assert.throws(() => parseIdx(images.subarray(0, -1), labels, 2), /dataset/);
const badLabels = Buffer.from(labels); badLabels[8] = 10;
assert.throws(() => parseIdx(images, badLabels, 2), /class label/);
const badImages = Buffer.from(images); badImages.writeUInt32BE(27, 8);
assert.throws(() => parseIdx(badImages, labels, 2), /dataset/);
const wrongCount = Buffer.from(labels); wrongCount.writeUInt32BE(1, 4);
assert.throws(() => parseIdx(images, wrongCount, 2), /dataset/);

// No internet in unit tests: a local server exercises real fetch and cache I/O.
const directory = await mkdtemp(join(tmpdir(), 'blaten-datasets-'));
const compressed = gzipSync(Buffer.from('dataset fixture'));
const md5 = createHash('md5').update(compressed).digest('hex');
let requests = 0, mode = 'valid';
const server = createServer((request, response) => {
  requests++;
  if (mode === 'missing') { response.writeHead(404); response.end(); return }
  response.end(mode === 'corrupt' ? Buffer.from('invalid gzip') : compressed);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/fixture.gz`;
const source = { url, path: join(directory, 'dataset', 'fixture.gz'), md5 };
try {
  await assert.rejects(cachedFile({ ...source, download: false }), /missing/);
  assert.equal(requests, 0, 'Offline loading never fetches');
  const results = await Promise.all([cachedFile(source), cachedFile(source)]);
  assert.equal(requests, 1, 'Concurrent callers share one download');
  assert.equal(results[0].toString(), 'dataset fixture');
  assert.deepEqual(await readFile(source.path), compressed);
  mode = 'missing';
  assert.equal((await cachedFile({ ...source, download: false })).toString(), 'dataset fixture');
  assert.equal(requests, 1, 'A valid cache is usable without the source');
  await writeFile(source.path, 'broken cache'); mode = 'valid';
  await assert.rejects(cachedFile({ ...source, download: false }), /invalid checksum/);
  assert.equal(requests, 1, 'Offline loading does not repair corrupt files');
  await cachedFile(source);
  assert.equal(requests, 2, 'A corrupt cache is replaced');
  assert.deepEqual(await readFile(source.path), compressed);
  const missing = { ...source, path: join(directory, 'dataset', 'missing.gz') };
  mode = 'missing'; await assert.rejects(cachedFile(missing), /404/);
  mode = 'corrupt'; await assert.rejects(cachedFile(missing), /checksum/);
  await assert.rejects(readFile(missing.path), { code: 'ENOENT' });
  mode = 'valid'; await cachedFile(missing);
  assert.deepEqual(await readFile(missing.path), compressed, 'Failed downloads can be retried');
  assert.deepEqual((await readdir(join(directory, 'dataset'))).toSorted(), ['fixture.gz', 'missing.gz']);
} finally {
  await new Promise(resolve => server.close(resolve));
  await rm(directory, { recursive: true, force: true });
}
console.log('Passed: image batches, seeded shuffling, normalization, one-hot labels, padding, IDX validation, atomic caches, concurrent downloads, and retries.');
