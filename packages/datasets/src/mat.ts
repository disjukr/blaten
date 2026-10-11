import { inflateSync } from 'node:zlib';

export interface MatArray { readonly shape: readonly number[]; readonly type: number; readonly data: Buffer; readonly littleEndian: boolean }
export type MatValue = MatArray | string | MatValue[] | { [field: string]: MatValue };
interface Element { type: number; data: Buffer; next: number }
function element(buffer: Buffer, offset: number, little: boolean): Element {
  if (offset + 8 > buffer.length) throw new Error('Truncated MAT element');
  const tag = little ? buffer.readUInt32LE(offset) : buffer.readUInt32BE(offset);
  const small = little ? tag >>> 16 : tag > 0xffff ? tag & 0xffff : 0;
  if (small) {
    if (small > 4) throw new Error('Invalid MAT small element');
    return { type: little ? tag & 0xffff : tag >>> 16, data: buffer.subarray(offset + 4, offset + 4 + small), next: offset + 8 };
  }
  const size = little ? buffer.readUInt32LE(offset + 4) : buffer.readUInt32BE(offset + 4);
  if (offset + 8 + size > buffer.length) throw new Error('Truncated MAT data');
  // Compressed elements are not padded by scipy's writer.
  return { type: tag, data: buffer.subarray(offset + 8, offset + 8 + size), next: offset + 8 + (tag === 15 ? size : Math.ceil(size / 8) * 8) };
}
export function matNumber(array: MatArray, index: number): number {
  const sizes: Record<number, number> = { 1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 9: 8, 12: 8, 13: 8, 16: 1, 17: 2, 18: 4 };
  const size = sizes[array.type];
  if (!size || !Number.isSafeInteger(index) || index < 0 || (index + 1) * size > array.data.length) throw new Error('Invalid MAT numeric index');
  const view = new DataView(array.data.buffer, array.data.byteOffset, array.data.byteLength), o = index * size, l = array.littleEndian;
  switch (array.type) {
    case 1: return view.getInt8(o); case 2: case 16: return view.getUint8(o);
    case 3: return view.getInt16(o, l); case 4: case 17: return view.getUint16(o, l);
    case 5: return view.getInt32(o, l); case 6: case 18: return view.getUint32(o, l);
    case 7: return view.getFloat32(o, l); case 9: return view.getFloat64(o, l);
    case 12: return Number(view.getBigInt64(o, l)); case 13: return Number(view.getBigUint64(o, l));
    default: throw new Error(`Unsupported MAT type: ${array.type}`);
  }
}
export function matNumbers(value: MatValue | undefined): number[] {
  if (!value || typeof value === 'string' || Array.isArray(value) || !('shape' in value)) throw new Error('Expected MAT numeric array');
  const array = value as MatArray; return Array.from({ length: array.shape.reduce((a, b) => a * b, 1) }, (_, i) => matNumber(array, i));
}
export function matStrings(value: MatValue | undefined): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.flatMap(item => matStrings(item));
  throw new Error('Expected MAT strings');
}
function matrix(buffer: Buffer, little: boolean): { name: string; value: MatValue } {
  let offset = 0;
  const next = () => { const item = element(buffer, offset, little); offset = item.next; return item };
  const flags = next(); if (flags.data.length < 8) throw new Error('Invalid MAT flags');
  const kind = (little ? flags.data.readUInt32LE(0) : flags.data.readUInt32BE(0)) & 255;
  const dims = next(); const shape = matNumbers({ shape: [dims.data.length / 4], type: dims.type, data: dims.data, littleEndian: little });
  if (!shape.length || shape.some(n => !Number.isSafeInteger(n) || n < 0)) throw new Error('Invalid MAT shape');
  const name = next().data.toString('utf8'), count = shape.reduce((a, b) => a * b, 1);
  let value: MatValue;
  if (kind === 1) {
    value = Array.from({ length: count }, () => { const entry = next(); if (entry.type !== 14) throw new Error('Invalid MAT cell'); return matrix(entry.data, little).value });
  } else if (kind === 2) {
    const length = next(), fieldLength = little ? length.data.readInt32LE(0) : length.data.readInt32BE(0), names = next().data;
    if (fieldLength <= 0 || names.length % fieldLength) throw new Error('Invalid MAT struct fields');
    const fields = Array.from({ length: names.length / fieldLength }, (_, i) => names.subarray(i * fieldLength, (i + 1) * fieldLength).toString().replace(/\0.*$/, ''));
    const records: MatValue[] = [];
    for (let i = 0; i < count; i++) {
      const record: Record<string, MatValue> = {};
      for (const field of fields) { const entry = next(); if (entry.type !== 14) throw new Error('Invalid MAT struct'); record[field] = matrix(entry.data, little).value }
      records.push(record);
    }
    value = records;
  } else {
    const real = next(), array: MatArray = { shape, type: real.type, data: real.data, littleEndian: little };
    if (kind === 4) value = real.type === 16 ? real.data.toString('utf8') : String.fromCodePoint(...matNumbers(array).filter(n => n !== 0));
    else { if (count) matNumber(array, count - 1); value = array }
  }
  return { name, value };
}
/** MATLAB v5 numeric arrays, chars, cells, structs, and zlib-compressed matrices. */
export function parseMat(buffer: Buffer): Record<string, MatValue> {
  if (buffer.length < 128 || !['IM', 'MI'].includes(buffer.subarray(126, 128).toString())) throw new Error('Expected MATLAB v5 file');
  const little = buffer.subarray(126, 128).toString() === 'IM', result: Record<string, MatValue> = {};
  const parse = (data: Buffer, start: number) => {
    let offset = start;
    while (offset + 8 <= data.length) {
      const entry = element(data, offset, little); offset = entry.next;
      if (entry.type === 15) parse(inflateSync(entry.data), 0);
      else if (entry.type === 14) { const { name, value } = matrix(entry.data, little); result[name] = value }
      else throw new Error(`Unsupported MAT element: ${entry.type}`);
    }
  };
  parse(buffer, 128); return result;
}
