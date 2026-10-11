import { readFile } from 'node:fs/promises';
import { basename, dirname, relative } from 'node:path';
import { inflateSync } from 'node:zlib';
import type { Dataset } from './dataset.js';
import { readImage, readMask, type Flow, type FlowSample, type Mask } from './media.js';
import { indexed, inside, rootPath, nestedRoot, listFiles, lines, text, exists, splitOf, type LocalOptions } from './files.js';

export function parseFlo(buffer: Buffer): Flow {
  if (buffer.length < 12 || buffer.subarray(0, 4).toString() !== 'PIEH') throw new Error('Invalid .flo header');
  const width = buffer.readInt32LE(4), height = buffer.readInt32LE(8);
  if (width <= 0 || height <= 0 || buffer.length !== 12 + width * height * 8) throw new Error('Invalid .flo payload');
  const data = new Float32Array(width * height * 2); for (let i = 0; i < data.length; i++) data[i] = buffer.readFloatLE(12 + i * 4);
  return { data, width, height };
}
export function parsePfm(buffer: Buffer): Flow {
  let offset = 0;
  const line = () => { const end = buffer.indexOf(10, offset); if (end < 0) throw new Error('Truncated PFM header'); const result = buffer.subarray(offset, end).toString().trim(); offset = end + 1; return result };
  const magic = line(), [width, height] = line().split(/\s+/).map(Number), scale = Number(line());
  if (magic !== 'PF' || !Number.isInteger(width) || width! <= 0 || !Number.isInteger(height) || height! <= 0 || !Number.isFinite(scale) || scale === 0 || buffer.length - offset !== width! * height! * 12) throw new Error('Invalid flow PFM');
  const data = new Float32Array(width! * height! * 2);
  for (let y = 0; y < height!; y++) for (let x = 0; x < width!; x++) for (let c = 0; c < 2; c++) {
    const source = offset + ((height! - 1 - y) * width! * 3 + x * 3 + c) * 4;
    data[(y * width! + x) * 2 + c] = (scale < 0 ? buffer.readFloatLE(source) : buffer.readFloatBE(source)) * Math.abs(scale);
  }
  return { data, width: width!, height: height! };
}
/** Preserve the original 16-bit RGB channel values used by KITTI flow PNGs. */
export function parseKittiFlow(buffer: Buffer): { flow: Flow; validMask: Mask } {
  if (buffer.length < 33 || !buffer.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) throw new Error('Invalid PNG');
  let width = 0, height = 0, offset = 8; const chunks: Buffer[] = [];
  while (offset + 12 <= buffer.length) {
    const length = buffer.readUInt32BE(offset), name = buffer.subarray(offset + 4, offset + 8).toString();
    if (offset + length + 12 > buffer.length) throw new Error('Truncated PNG chunk');
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    if (name === 'IHDR') { width = data.readUInt32BE(0); height = data.readUInt32BE(4); if (data[8] !== 16 || data[9] !== 2 || data[10] !== 0 || data[11] !== 0 || data[12] !== 0) throw new Error('KITTI flow requires a non-interlaced 16-bit RGB PNG') }
    if (name === 'IDAT') chunks.push(data); offset += length + 12; if (name === 'IEND') break;
  }
  if (!width || !height || !chunks.length) throw new Error('Invalid KITTI flow PNG');
  const raw = inflateSync(Buffer.concat(chunks)), rowBytes = width * 6, decoded = Buffer.alloc(height * rowBytes);
  if (raw.length !== height * (rowBytes + 1)) throw new Error('Invalid KITTI flow PNG length');
  const paeth = (a: number, b: number, c: number) => { const p = a + b - c, x = Math.abs(p - a), y = Math.abs(p - b), z = Math.abs(p - c); return x <= y && x <= z ? a : y <= z ? b : c };
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (rowBytes + 1)]!; if (filter > 4) throw new Error('Invalid PNG filter');
    for (let x = 0; x < rowBytes; x++) {
      const a = x >= 6 ? decoded[y * rowBytes + x - 6]! : 0, b = y ? decoded[(y - 1) * rowBytes + x]! : 0, c = y && x >= 6 ? decoded[(y - 1) * rowBytes + x - 6]! : 0;
      const predictor = [0, a, b, Math.floor((a + b) / 2), paeth(a, b, c)][filter]!;
      decoded[y * rowBytes + x] = (raw[y * (rowBytes + 1) + x + 1]! + predictor) & 255;
    }
  }
  const data = new Float32Array(width * height * 2), valid = new Uint32Array(width * height);
  for (let i = 0; i < valid.length; i++) { data[i * 2] = (decoded.readUInt16BE(i * 6) - 32768) / 64; data[i * 2 + 1] = (decoded.readUInt16BE(i * 6 + 2) - 32768) / 64; valid[i] = Number(decoded.readUInt16BE(i * 6 + 4) !== 0) }
  return { flow: { data, width, height }, validMask: { data: valid, width, height } };
}
interface FlowRecord { first: string; second: string; flow?: string; invalid?: string }
function flowRecords(name: string, split: string, homepage: string, rows: readonly FlowRecord[]): Dataset<FlowSample> {
  return indexed({ name, split, homepage }, rows.length, async n => {
    const row = rows[n]!, [first, second] = await Promise.all([readImage(row.first), readImage(row.second)]);
    if (first.width !== second.width || first.height !== second.height) throw new Error('Flow image dimensions differ');
    if (!row.flow) return { first, second };
    const buffer = await readFile(row.flow), values = /\.png$/i.test(row.flow) ? parseKittiFlow(buffer) : { flow: /\.pfm$/i.test(row.flow) ? parsePfm(buffer) : parseFlo(buffer) };
    if (values.flow.width !== first.width || values.flow.height !== first.height) throw new Error('Flow and image dimensions differ');
    let validMask = 'validMask' in values ? values.validMask : undefined;
    if (row.invalid) validMask = await readMask(row.invalid, channels => Number(channels[0] === 0));
    return { first, second, ...values, ...(validMask ? { validMask } : {}) };
  });
}
export interface SintelOptions extends LocalOptions<'train' | 'test'> { pass?: 'clean' | 'final' | 'both' }
export async function sintel(options: SintelOptions): Promise<Dataset<FlowSample>> {
  splitOf(options.split, ['train', 'test']); const pass = options.pass ?? 'clean'; splitOf(pass, ['clean', 'final', 'both']);
  const split = options.split === 'train' ? 'training' : 'test', passes = pass === 'both' ? ['clean', 'final'] : [pass], root = await nestedRoot(rootPath('sintel', options.root), ['Sintel'], `${split}/${passes[0]}`), rows: FlowRecord[] = [];
  for (const pass of passes) {
    const images = await listFiles(inside(root, `${split}/${pass}`), /\.png$/), groups = new Map<string, string[]>();
    for (const path of images) { const scene = dirname(path); const group = groups.get(scene) ?? []; group.push(path); groups.set(scene, group) }
    for (const [scene, images] of groups) for (let i = 0; i < images.length - 1; i++) {
      const stem = basename(images[i]!, '.png'), invalid = inside(root, `training/invalid/${basename(scene)}/${stem}.png`);
      rows.push({ first: images[i]!, second: images[i + 1]!, ...(options.split === 'train' ? { flow: inside(root, `training/flow/${basename(scene)}/${stem}.flo`), ...(await exists(invalid) ? { invalid } : {}) } : {}) });
    }
  }
  return flowRecords('sintel', options.split, 'https://sintel.is.tue.mpg.de/', rows);
}
export async function flyingChairs(options: LocalOptions<'train' | 'val'>): Promise<Dataset<FlowSample>> {
  splitOf(options.split, ['train', 'val']); const root = await nestedRoot(rootPath('flying-chairs', options.root), ['FlyingChairs'], 'data');
  const flows = await listFiles(inside(root, 'data'), /_flow\.flo$/, false), splits = lines(await text(inside(root, 'FlyingChairs_train_val.txt'))).flatMap(line => line.split(/\s+/).map(Number));
  if (splits.length !== flows.length || splits.some(n => n !== 1 && n !== 2)) throw new Error('Invalid FlyingChairs split');
  const rows = flows.flatMap((path, i) => splits[i] === (options.split === 'train' ? 1 : 2) ? [{ first: path.replace('_flow.flo', '_img1.ppm'), second: path.replace('_flow.flo', '_img2.ppm'), flow: path }] : []);
  return flowRecords('flying-chairs', options.split, 'https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html', rows);
}
export interface FlyingThings3dOptions extends LocalOptions<'train' | 'test'> { pass?: 'clean' | 'final' | 'both'; camera?: 'left' | 'right' | 'both'; direction?: 'future' | 'past' | 'both' }
export async function flyingThings3d(options: FlyingThings3dOptions): Promise<Dataset<FlowSample>> {
  splitOf(options.split, ['train', 'test']); const pass = options.pass ?? 'clean', camera = options.camera ?? 'left', direction = options.direction ?? 'both';
  splitOf(pass, ['clean', 'final', 'both']); splitOf(camera, ['left', 'right', 'both']); splitOf(direction, ['future', 'past', 'both']);
  const split = options.split.toUpperCase(), passes = pass === 'both' ? ['clean', 'final'] : [pass], root = await nestedRoot(rootPath('flying-things3d', options.root), ['FlyingThings3D'], `frames_${passes[0]}pass/${split}`), rows: FlowRecord[] = [];
  for (const pass of passes) {
    const base = inside(root, `frames_${pass}pass/${split}`), groups = new Map<string, string[]>();
    for (const path of await listFiles(base, /\.png$/)) { const directory = dirname(path); if (camera !== 'both' && basename(directory) !== camera) continue; const group = groups.get(directory) ?? []; group.push(path); groups.set(directory, group) }
    for (const [directory, images] of groups) {
      const parts = relative(base, directory).split(/[\\/]/), eye = parts.pop()!, suffix = eye === 'left' ? 'L' : 'R';
      for (let i = 0; i < images.length - 1; i++) for (const dir of direction === 'both' ? ['future', 'past'] : [direction]) {
        const first = images[dir === 'future' ? i : i + 1]!, second = images[dir === 'future' ? i + 1 : i]!, frame = basename(first, '.png'), label = dir === 'future' ? 'Future' : 'Past';
        rows.push({ first, second, flow: inside(root, `optical_flow/${split}/${parts.join('/')}/into_${dir}/${eye}/OpticalFlowInto${label}_${frame}_${suffix}.pfm`) });
      }
    }
  }
  return flowRecords('flying-things3d', options.split, 'https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html', rows);
}
export async function kittiFlow(options: LocalOptions<'train' | 'test'>): Promise<Dataset<FlowSample>> {
  splitOf(options.split, ['train', 'test']); const split = options.split === 'train' ? 'training' : 'testing', root = await nestedRoot(rootPath('kitti-flow', options.root), ['KittiFlow', 'KITTI'], `${split}/image_2`);
  const images = await listFiles(inside(root, `${split}/image_2`), /_10\.png$/, false);
  return flowRecords('kitti-flow', options.split, 'https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow', images.map(path => ({ first: path, second: path.replace('_10.png', '_11.png'), ...(options.split === 'train' ? { flow: inside(root, `training/flow_occ/${basename(path)}`) } : {}) })));
}
