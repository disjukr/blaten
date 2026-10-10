import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { resolve } from 'node:path';
import { tensor } from 'blaten';
const origin = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
async function download(name: string): Promise<Buffer> {
  const directory = resolve('data'); await mkdir(directory, { recursive: true });
  const path = resolve(directory, name);
  let compressed: Buffer;
  try { compressed = await readFile(path) }
  catch (error: any) {
    if (error.code !== 'ENOENT') throw error;
    console.log(`Downloading ${name}`);
    const response = await fetch(origin + name);
    if (!response.ok) throw new Error(`MNIST download failed: ${response.status}`);
    compressed = Buffer.from(await response.arrayBuffer());
    gunzipSync(compressed); // Validate download before caching.
    await writeFile(path, compressed);
  }
  return gunzipSync(compressed);
}
export interface Dataset { pixels: Uint8Array; labels: Uint8Array; count: number }
export async function load(split: 'train' | 'test'): Promise<Dataset> {
  const prefix = split === 'train' ? 'train' : 't10k';
  const [images, labels] = await Promise.all([download(`${prefix}-images-idx3-ubyte.gz`), download(`${prefix}-labels-idx1-ubyte.gz`)]);
  const count = images.readUInt32BE(4);
  if (images.readUInt32BE(0) !== 2051 || images.readUInt32BE(8) !== 28 || images.readUInt32BE(12) !== 28 || labels.readUInt32BE(0) !== 2049 || labels.readUInt32BE(4) !== count || images.length !== 16 + count * 784 || labels.length !== 8 + count) throw new Error('Invalid MNIST IDX dataset');
  if (labels.subarray(8).some(x => x > 9)) throw new Error('Invalid MNIST class label');
  return { pixels: images.subarray(16), labels: labels.subarray(8), count };
}
export function batch(dataset: Dataset, indices: readonly number[], batchSize: number) {
  const x = new Float32Array(batchSize * 784), y = new Float32Array(batchSize * 10);
  indices.forEach((source, row) => {
    for (let pixel = 0; pixel < 784; pixel++) x[row * 784 + pixel] = dataset.pixels[source * 784 + pixel]! / 255;
    y[row * 10 + dataset.labels[source]!] = 1;
  });
  return [tensor(x, { shape: [batchSize, 784] }), tensor(y, { shape: [batchSize, 10] })] as const;
}
export function shuffle(indices: number[], seed: number): void {
  let state = seed;
  for (let i = indices.length - 1; i > 0; i--) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const j = state % (i + 1); [indices[i], indices[j]] = [indices[j]!, indices[i]!];
  }
}
