import { f32, isDType, type DType } from './dtype.js';
export * from './dtype.js';
export type Shape = readonly number[];
export interface TensorType { readonly kind: 'tensor'; readonly dtype: DType; readonly shape: Shape }
export type ValueType = DType | TensorType | { readonly kind: 'tuple'; readonly items: readonly ValueType[] } | { readonly kind: 'record'; readonly fields: Readonly<Record<string, ValueType>> };
export interface Signature { inputs: readonly ValueType[]; output?: ValueType | readonly ValueType[] }
export type ValueConstraint = DType | { readonly kind: 'tensor'; readonly shape: readonly (number | null)[] | null; readonly dtypes: readonly DType[] } | { readonly kind: 'tuple'; readonly items: readonly ValueConstraint[] } | { readonly kind: 'record'; readonly fields: Readonly<Record<string, ValueConstraint>> };
export interface SignatureConstraints { readonly inputs: readonly ValueConstraint[] }
export function sizeOf(shape: Shape): number {
  if (shape.some(n => !Number.isSafeInteger(n) || n <= 0)) {
    throw new Error('Shape dimensions must be positive integers');
  }
  const size = shape.reduce((a, b) => a * b, 1);
  if (!Number.isSafeInteger(size)) throw new Error('Tensor element count exceeds safe integer range');
  return size;
}
export function tensorType(shape: Shape, dtype: DType = f32): TensorType {
  if (!isDType(dtype)) throw new Error(`Unknown dtype ${dtype}`);
  sizeOf(shape);
  return Object.freeze({ kind: 'tensor', dtype, shape: Object.freeze([...shape]) });
}
export function shapeOf(type: ValueType): Shape {
  if (isDType(type)) return [];
  if (typeof type !== 'object' || type.kind !== 'tensor' || !isDType(type.dtype)) throw new Error('Signature types must be a dtype or tensorType(shape, dtype)');
  sizeOf(type.shape); return type.shape;
}
export function dtypeOf(type: ValueType): DType { shapeOf(type); return typeof type === 'string' ? type : (type as TensorType).dtype }
