import { readFile } from 'node:fs/promises';
import sharp from 'sharp';
import type { Image } from './dataset.js';

export interface Mask { readonly data: Uint32Array; readonly width: number; readonly height: number; readonly ignoreIndex?: number }
export type Box = readonly [number, number, number, number]; // zero-based xyxy, exclusive maximum
export interface DetectionSample { readonly image: Image; readonly boxes: readonly Box[]; readonly labels: readonly number[]; readonly difficult?: readonly boolean[]; readonly isCrowd?: readonly boolean[] }
export interface SegmentationSample { readonly image: Image; readonly mask: Mask }
export interface CaptionSample { readonly image: Image; readonly captions: readonly string[] }
export interface PairSample { readonly first: Image; readonly second: Image; readonly same: boolean }
export interface Flow { readonly data: Float32Array; readonly width: number; readonly height: number }
export interface FlowSample { readonly first: Image; readonly second: Image; readonly flow?: Flow; readonly validMask?: Mask }
export interface VideoSample { readonly frames: readonly Image[]; readonly label?: number; readonly path: string; readonly fps: number }
export interface TextSample { readonly text: string; readonly label: number }
export interface TabularSample { readonly features: readonly number[]; readonly label: number }

export async function readImage(path: string): Promise<Image> {
  const input = await readFile(path);
  if (/^P[2356]\s/.test(input.subarray(0, 3).toString())) return readPpm(input);
  const { data, info } = await sharp(input).toColourspace('srgb').removeAlpha().raw().toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height, channels: info.channels, featureCount: data.length };
}
export async function readMask(path: string, map?: (channels: readonly number[]) => number, ignoreIndex?: number): Promise<Mask> {
  const { data, info } = await sharp(await readFile(path)).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const pixels = new Uint32Array(info.width * info.height);
  for (let i = 0; i < pixels.length; i++) {
    const channels = Array.from(data.subarray(i * info.channels, (i + 1) * info.channels));
    pixels[i] = map ? map(channels) : channels[0]!;
  }
  return { data: pixels, width: info.width, height: info.height, ignoreIndex };
}
export function readPpm(input: Buffer): Image {
  let offset = 0;
  const token = () => {
    while (offset < input.length) {
      if (input[offset] === 35) { while (offset < input.length && input[offset] !== 10) offset++; continue }
      if (input[offset]! <= 32) { offset++; continue } break;
    }
    const start = offset; while (offset < input.length && input[offset]! > 32 && input[offset] !== 35) offset++;
    return input.subarray(start, offset).toString();
  };
  const magic = token(), width = Number(token()), height = Number(token()), max = Number(token()), channels = magic === 'P5' || magic === 'P2' ? 1 : 3;
  if (!['P2', 'P3', 'P5', 'P6'].includes(magic) || !Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0 || !Number.isInteger(max) || max <= 0 || max > 65535) throw new Error('Invalid PPM header');
  const count = width * height * channels, data = new Uint8Array(count);
  if (magic === 'P2' || magic === 'P3') {
    for (let i = 0; i < count; i++) { const item = token(), value = Number(item); if (!item || !Number.isInteger(value) || value < 0 || value > max) throw new Error('Invalid PPM pixel'); data[i] = Math.round(value * 255 / max) }
  } else {
    if (input[offset] === 13 && input[offset + 1] === 10) offset += 2; else offset++;
    const bytes = max > 255 ? 2 : 1;
    if (input.length - offset !== count * bytes) throw new Error('Invalid PPM payload');
    for (let i = 0; i < count; i++) data[i] = Math.round((bytes === 1 ? input[offset + i]! : input.readUInt16BE(offset + i * 2)) * 255 / max);
  }
  return { data, width, height, channels, featureCount: count };
}
