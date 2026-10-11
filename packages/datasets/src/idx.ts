import { resolve } from 'node:path';
import { cachedFile } from './cache.js';
import { ImageDataset } from './image.js';
import type { DatasetOptions } from './files.js';

/** Decode the 28x28 uint8 IDX format shared by MNIST and Fashion-MNIST. */
export function parseIdx(images: Buffer, labels: Buffer, expectedCount: number) {
  if (images.length < 16 || labels.length < 8) throw new Error('Invalid IDX header');
  const count = images.readUInt32BE(4);
  if (images.readUInt32BE(0) !== 2051 || images.readUInt32BE(8) !== 28 || images.readUInt32BE(12) !== 28 || labels.readUInt32BE(0) !== 2049 || count !== expectedCount || labels.readUInt32BE(4) !== count || images.length !== 16 + count * 784 || labels.length !== 8 + count) throw new Error('Invalid IDX dataset');
  const targets = labels.subarray(8);
  if (targets.some(label => label > 9)) throw new Error('Invalid IDX class label');
  return { pixels: images.subarray(16), labels: targets };
}

interface IdxSource {
  readonly name: string; readonly origin: string; readonly homepage: string; readonly classes: readonly string[] & { length: 10 };
  readonly checksums: readonly [string, string, string, string];
}
export async function loadIdx(source: IdxSource, options: DatasetOptions<'train' | 'test'>): Promise<ImageDataset<784, 10>> {
  if (options.split !== 'train' && options.split !== 'test') throw new Error(`Unsupported ${source.name} split: ${options.split}`);
  const directory = resolve(options.root ?? `.cache/blaten-datasets/${source.name}`);
  const training = options.split === 'train', prefix = training ? 'train' : 't10k', count = training ? 60000 : 10000, offset = training ? 0 : 2;
  const names = [`${prefix}-images-idx3-ubyte.gz`, `${prefix}-labels-idx1-ubyte.gz`];
  const [images, labels] = await Promise.all(names.map((name, i) => cachedFile({ url: source.origin + name, path: resolve(directory, name), md5: source.checksums[offset + i]!, download: options.download })));
  return new ImageDataset({ ...parseIdx(images!, labels!, count), featureCount: 784, imageShape: [28, 28, 1],
    metadata: { name: source.name, split: options.split, homepage: source.homepage, classNames: source.classes },
  });
}
