import { loadIdx } from './idx.js';
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import bzip from 'seek-bzip';
import { ImageDataset } from './image.js';
import { cachedFile } from './cache.js';
import { downloadFile, rootPath, inside, splitOf, prepareArchive, exists, nestedRoot, type DownloadOptions, type DatasetOptions } from './files.js';
import { parseMat, matNumber, matNumbers, type MatArray } from './mat.js';
import type { Dataset, ImageClassificationSample, ImageClassificationMetadata } from './dataset.js';
import { indexed } from './files.js';

const digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] as const;
export type MnistSplit = 'train' | 'test';
/** Load one MNIST split, downloading missing/corrupted files unless download is false. */
export function mnist(options: DatasetOptions<MnistSplit>): Promise<ImageDataset<784, 10>> {
  return loadIdx({ name: 'mnist', origin: 'https://storage.googleapis.com/cvdf-datasets/mnist/', homepage: 'http://yann.lecun.com/exdb/mnist/', classes: digits,
    checksums: ['f68b3c2dcbeaaa9fbdd348bbdeb94873', 'd53e105ee54ea40749a09fcbcd1e9432', '9fb629c4189551a2d022fa330f9573f3', 'ec29112dd5afa0611ce80d1b7f02629c'],
  }, options);
}
export interface EmnistOptions extends DownloadOptions<'train' | 'test'> { variant: 'byclass' | 'bymerge' | 'balanced' | 'letters' | 'digits' | 'mnist' }
function idxImages(buffer: Buffer): { pixels: Uint8Array; count: number } {
  if (buffer.length < 16 || buffer.readUInt32BE(0) !== 2051 || buffer.readUInt32BE(8) !== 28 || buffer.readUInt32BE(12) !== 28 || buffer.length !== 16 + buffer.readUInt32BE(4) * 784) throw new Error('Invalid 28x28 IDX images');
  return { pixels: buffer.subarray(16), count: buffer.readUInt32BE(4) };
}
async function readIdxFile(path: string): Promise<Buffer> {
  if (await exists(path)) return gunzipSync(await readFile(path));
  return readFile(path.replace(/\.gz$/, ''));
}
export async function kmnist(options: DownloadOptions<'train' | 'test'>): Promise<ImageDataset<784, 10>> {
  splitOf(options.split, ['train', 'test']);
  const root = rootPath('kmnist', options.root), train = options.split === 'train', prefix = train ? 'train' : 't10k';
  const hashes = train ? ['bdb82020997e1d708af4cf47b453dcf7', 'e144d726b3acfaa3e44228e80efcd344'] : ['5c965bf0a639b31b8f53240b1b52f4d7', '7320c461ea6c1c855c0b718fb2a4b134'];
  const [images, targets] = await Promise.all(['images-idx3-ubyte', 'labels-idx1-ubyte'].map((suffix, i) => {
    const name = `${prefix}-${suffix}.gz`;
    return cachedFile({ path: inside(root, name), url: `https://codh.rois.ac.jp/kmnist/dataset/kmnist/${name}`, md5: hashes[i]!, download: options.download });
  }));
  const { pixels, count } = idxImages(images!);
  if (targets!.length !== count + 8 || targets!.readUInt32BE(0) !== 2049 || targets!.readUInt32BE(4) !== count) throw new Error('Invalid KMNIST labels');
  return new ImageDataset({ pixels, labels: targets!.subarray(8), featureCount: 784, imageShape: [28, 28, 1], metadata: { name: 'kmnist', split: options.split, homepage: 'https://github.com/rois-codh/kmnist', classNames: ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] as const } });
}
export async function emnist(options: EmnistOptions): Promise<ImageDataset<784>> {
  splitOf(options.split, ['train', 'test']);
  splitOf(options.variant, ['byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist']);
  const root = rootPath('emnist', options.root), prefix = `emnist-${options.variant}-${options.split}`;
  if (!await exists(inside(root, `${prefix}-images-idx3-ubyte.gz`)) && !await exists(inside(root, `gzip/${prefix}-images-idx3-ubyte.gz`)) && !await exists(inside(root, `${prefix}-images-idx3-ubyte`))) {
    await prepareArchive(root, { url: 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip', md5: '58c8d27c78d21e728a6bc7b3cc06412e' }, `gzip/${prefix}-images-idx3-ubyte.gz`, options.download);
  }
  const directory = await exists(inside(root, 'gzip')) ? inside(root, 'gzip') : root;
  const [images, targets, mapping] = await Promise.all([readIdxFile(inside(directory, `${prefix}-images-idx3-ubyte.gz`)), readIdxFile(inside(directory, `${prefix}-labels-idx1-ubyte.gz`)), readFile(inside(directory, `emnist-${options.variant}-mapping.txt`), 'utf8')]);
  const raw = idxImages(images), classNames = mapping.trim().split(/\r?\n/).map(line => String.fromCodePoint(Number(line.trim().split(/\s+/)[1])));
  if (targets.length !== raw.count + 8 || targets.readUInt32BE(0) !== 2049 || targets.readUInt32BE(4) !== raw.count) throw new Error('Invalid EMNIST labels');
  // EMNIST is stored transposed relative to ordinary image coordinates.
  const pixels = new Uint8Array(raw.pixels.length);
  for (let n = 0; n < raw.count; n++) for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) pixels[n * 784 + y * 28 + x] = raw.pixels[n * 784 + x * 28 + y]!;
  const labels = Uint8Array.from(targets.subarray(8), label => options.variant === 'letters' ? label - 1 : label);
  return new ImageDataset({ pixels, labels, featureCount: 784, imageShape: [28, 28, 1], metadata: { name: 'emnist', split: options.split, homepage: 'https://www.nist.gov/itl/products-and-services/emnist-dataset', classNames } });
}
export async function qmnist(options: DownloadOptions<'train' | 'test' | 'test10k' | 'test50k'>): Promise<ImageDataset<784, 10>> {
  splitOf(options.split, ['train', 'test', 'test10k', 'test50k']);
  const root = rootPath('qmnist', options.root), split = options.split === 'train' ? 'train' : 'test';
  const hashes = split === 'train' ? ['ed72d4157d28c017586c42bc6afe6370', '0058f8dd561b90ffdd0f734c6a30e5e4'] : ['1394631089c404de565df7b7aeaf9412', '5b5b05890a5e13444e108efe57b788aa'];
  const [images, targets] = await Promise.all(['images-idx3-ubyte', 'labels-idx2-int'].map((suffix, i) => {
    const name = `qmnist-${split}-${suffix}.gz`;
    return cachedFile({ url: `https://raw.githubusercontent.com/facebookresearch/qmnist/master/${name}`, path: inside(root, name), md5: hashes[i]!, download: options.download });
  }));
  const raw = idxImages(images!);
  if (targets!.length !== 12 + raw.count * 32 || targets!.readUInt32BE(0) !== 3074 || targets!.readUInt32BE(4) !== raw.count || targets!.readUInt32BE(8) !== 8) throw new Error('Invalid QMNIST labels');
  const start = options.split === 'test50k' ? 10000 : 0, end = options.split === 'test10k' ? 10000 : raw.count;
  if (end < start || end > raw.count) throw new Error('Invalid QMNIST subset');
  const labels = Uint8Array.from({ length: end - start }, (_, i) => { const label = targets!.readInt32BE(12 + (start + i) * 32); if (label < 0 || label > 9) throw new Error('Invalid QMNIST class'); return label });
  return new ImageDataset({ pixels: raw.pixels.subarray(start * 784, end * 784), labels, featureCount: 784, imageShape: [28, 28, 1], metadata: { name: 'qmnist', split: options.split, homepage: 'https://github.com/facebookresearch/qmnist', classNames: digits } });
}
export async function usps(options: DownloadOptions<'train' | 'test'>): Promise<ImageDataset<256, 10>> {
  splitOf(options.split, ['train', 'test']);
  const name = options.split === 'train' ? 'usps.bz2' : 'usps.t.bz2';
  const path = await downloadFile(rootPath('usps', options.root), { url: `https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/${name}`, md5: options.split === 'train' ? 'ec16c51db3855ca6c91edd34d0e9b197' : '8ea070ee2aca1ac39742fdd1ef5ed118' }, options.download);
  const rows = bzip.decode(await readFile(path)).toString().trim().split(/\r?\n/), pixels = new Uint8Array(rows.length * 256), labels = new Uint8Array(rows.length);
  rows.forEach((row, n) => {
    const [label, ...features] = row.trim().split(/\s+/); labels[n] = Number(label) - 1;
    if (!Number.isInteger(Number(label)) || Number(label) < 1 || Number(label) > 10 || features.length !== 256) throw new Error('Invalid USPS record');
    const seen = new Set<number>();
    for (const feature of features) { const [index, value] = feature.split(':').map(Number); if (!Number.isInteger(index) || index! < 1 || index! > 256 || seen.has(index!) || !Number.isFinite(value) || value! < -1 || value! > 1) throw new Error('Invalid USPS feature'); seen.add(index!); pixels[n * 256 + index! - 1] = Math.floor((value! + 1) * 127.5) }
  });
  return new ImageDataset({ pixels, labels, featureCount: 256, imageShape: [16, 16, 1], metadata: { name: 'usps', split: options.split, homepage: 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps', classNames: digits } });
}
export async function svhn(options: DownloadOptions<'train' | 'test' | 'extra'>): Promise<Dataset<ImageClassificationSample<3072>, ImageClassificationMetadata<10>>> {
  splitOf(options.split, ['train', 'test', 'extra']);
  const hashes = { train: 'e26dedcc434d2e4c54c9b2d4a06d8373', test: 'eb5a983be6a315427106f1b164d9cef3', extra: 'a93ce644f1a588dc4d68dda5feec44a7' };
  const path = await downloadFile(rootPath('svhn', options.root), { url: `http://ufldl.stanford.edu/housenumbers/${options.split}_32x32.mat`, md5: hashes[options.split] }, options.download);
  const mat = parseMat(await readFile(path)), images = mat.X as MatArray, labels = matNumbers(mat.y);
  if (!images || images.shape.join(',') !== `32,32,3,${labels.length}` || labels.some(label => !Number.isInteger(label) || label < 1 || label > 10)) throw new Error('Invalid SVHN matrices');
  return indexed({ name: 'svhn', split: options.split, homepage: 'http://ufldl.stanford.edu/housenumbers/', classNames: digits }, labels.length, async n => {
    const data = new Uint8Array(3072);
    for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) for (let c = 0; c < 3; c++) data[(y * 32 + x) * 3 + c] = matNumber(images, y + x * 32 + c * 1024 + n * 3072);
    return { image: { data, width: 32, height: 32, channels: 3, featureCount: 3072 as const }, label: labels[n]! % 10 };
  });
}
