/// <reference lib="esnext.float16" preserve="true" />
export const bool = 'bool' as const;
export const i8 = 'i8' as const, u8 = 'u8' as const, i16 = 'i16' as const, u16 = 'u16' as const;
export const i32 = 'i32' as const, u32 = 'u32' as const, i64 = 'i64' as const, u64 = 'u64' as const;
export const f16 = 'f16' as const, f32 = 'f32' as const, f64 = 'f64' as const;
export type DType = typeof bool | typeof i8 | typeof u8 | typeof i16 | typeof u16 | typeof i32 | typeof u32 | typeof i64 | typeof u64 | typeof f16 | typeof f32 | typeof f64;
export type FloatDType = typeof f16 | typeof f32 | typeof f64;
export type Scalar = number | bigint | boolean;
export type ScalarOf<D extends DType> = D extends 'bool' ? boolean : D extends 'i64' | 'u64' ? bigint : number;
export interface StorageTypes {
  bool: Uint8Array; i8: Int8Array; u8: Uint8Array; i16: Int16Array; u16: Uint16Array;
  i32: Int32Array; u32: Uint32Array; i64: BigInt64Array; u64: BigUint64Array;
  f16: Float16Array; f32: Float32Array; f64: Float64Array;
}
export type HostStorage<D extends DType = DType> = StorageTypes[D];
const dtypeNames = new Set<string>(['bool', 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'f16', 'f32', 'f64']);
export const isDType = (value: unknown): value is DType => typeof value === 'string' && dtypeNames.has(value);
export const isFloat = (dtype: DType): dtype is FloatDType => dtype === f16 || dtype === f32 || dtype === f64;
export const isInteger = (dtype: DType) => dtype.startsWith('i') || dtype.startsWith('u');
export function isHostStorage(value: unknown): value is HostStorage { return ArrayBuffer.isView(value) && !(value instanceof DataView) }
export function storageDType(value: HostStorage): DType {
  if (value instanceof Float32Array) return f32;
  if (value instanceof Float64Array) return f64;
  if (typeof Float16Array !== 'undefined' && value instanceof Float16Array) return f16;
  if (value instanceof BigInt64Array) return i64;
  if (value instanceof BigUint64Array) return u64;
  if (value instanceof Int8Array) return i8;
  if (value instanceof Int16Array) return i16;
  if (value instanceof Int32Array) return i32;
  if (value instanceof Uint16Array) return u16;
  if (value instanceof Uint32Array) return u32;
  return u8;
}
export function scalarDType(value: Scalar): DType { return typeof value === 'boolean' ? bool : typeof value === 'bigint' ? i64 : f32 }
export function scalarFitsInteger(value: Scalar, dtype: DType): boolean {
  if (!isInteger(dtype) || typeof value === 'boolean' || (typeof value === 'number' && !Number.isSafeInteger(value))) return false;
  const width = BigInt(Number(dtype.slice(1))), integer = BigInt(value);
  return dtype[0] === 'i' ? integer >= -(1n << (width - 1n)) && integer < (1n << (width - 1n)) : integer >= 0n && integer < (1n << width);
}
export function convertScalar(value: Scalar, dtype: DType): number | bigint {
  if (dtype === bool) return +(typeof value === 'boolean' ? value : typeof value === 'bigint' ? value !== 0n : value !== 0);
  if (dtype === i64 || dtype === u64) {
    if (typeof value === 'number' && !Number.isSafeInteger(Math.trunc(value))) throw new Error('64-bit integer conversion requires a safe integer after truncation or bigint');
    const integer = BigInt(typeof value === 'number' ? Math.trunc(value) : value);
    return dtype === i64 ? BigInt.asIntN(64, integer) : BigInt.asUintN(64, integer);
  }
  if (isInteger(dtype) && typeof value === 'bigint') return Number(dtype[0] === 'i' ? BigInt.asIntN(Number(dtype.slice(1)), value) : BigInt.asUintN(Number(dtype.slice(1)), value));
  return Number(value);
}
export function makeStorage<D extends DType>(dtype: D, lengthOrValues: number | Iterable<Scalar>): HostStorage<D> {
  if (!isDType(dtype)) throw new Error(`Unknown dtype ${dtype}`);
  if (dtype === f16 && typeof Float16Array === 'undefined') throw new Error('f16 host storage requires Float16Array support');
  const constructors = { bool: Uint8Array, i8: Int8Array, u8: Uint8Array, i16: Int16Array, u16: Uint16Array, i32: Int32Array, u32: Uint32Array, i64: BigInt64Array, u64: BigUint64Array, f16: typeof Float16Array === 'undefined' ? undefined : Float16Array, f32: Float32Array, f64: Float64Array };
  const Constructor = constructors[dtype] as any;
  if (typeof lengthOrValues === 'number') return new Constructor(lengthOrValues);
  return Constructor.from(Array.from(lengthOrValues, value => convertScalar(value, dtype)));
}
/** Floating types dominate; mixed signed/unsigned integers widen to fit both ranges. */
export function promoteTypes(a: DType, b: DType): DType {
  if (a === b) return a;
  if (a === bool) return b;
  if (b === bool) return a;
  if (isFloat(a) || isFloat(b)) {
    const floats = [a, b].filter(isFloat);
    return floats.includes(f64) ? f64 : floats.includes(f32) ? f32 : f16;
  }
  const widthA = Number(a.slice(1)), widthB = Number(b.slice(1));
  if (a[0] === b[0]) return `${a[0]}${Math.max(widthA, widthB)}` as DType;
  const signed = a[0] === 'i' ? widthA : widthB, unsigned = a[0] === 'u' ? widthA : widthB;
  if (signed > unsigned) return `i${signed}` as DType;
  const wider = [8, 16, 32, 64].find(width => width > unsigned);
  return wider ? `i${wider}` as DType : f64;
}
