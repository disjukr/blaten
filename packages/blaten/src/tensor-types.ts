import type { DType, FloatDType, Scalar, Shape } from './types.js';
import type { Tensor } from './tensor.js';

// Mirror promoteTypes. Distribute over unions so uncertain dtypes stay sound.
interface IntegerPromotion {
  i8:  { i8: 'i8'; u8: 'i16'; i16: 'i16'; u16: 'i32'; i32: 'i32'; u32: 'i64'; i64: 'i64'; u64: 'f64' };
  u8:  { i8: 'i16'; u8: 'u8'; i16: 'i16'; u16: 'u16'; i32: 'i32'; u32: 'u32'; i64: 'i64'; u64: 'u64' };
  i16: { i8: 'i16'; u8: 'i16'; i16: 'i16'; u16: 'i32'; i32: 'i32'; u32: 'i64'; i64: 'i64'; u64: 'f64' };
  u16: { i8: 'i32'; u8: 'u16'; i16: 'i32'; u16: 'u16'; i32: 'i32'; u32: 'u32'; i64: 'i64'; u64: 'u64' };
  i32: { i8: 'i32'; u8: 'i32'; i16: 'i32'; u16: 'i32'; i32: 'i32'; u32: 'i64'; i64: 'i64'; u64: 'f64' };
  u32: { i8: 'i64'; u8: 'u32'; i16: 'i64'; u16: 'u32'; i32: 'i64'; u32: 'u32'; i64: 'i64'; u64: 'u64' };
  i64: { i8: 'i64'; u8: 'i64'; i16: 'i64'; u16: 'i64'; i32: 'i64'; u32: 'i64'; i64: 'i64'; u64: 'f64' };
  u64: { i8: 'f64'; u8: 'u64'; i16: 'f64'; u16: 'u64'; i32: 'f64'; u32: 'u64'; i64: 'f64'; u64: 'u64' };
}
export type Promote<A extends DType, B extends DType> = A extends DType ? B extends DType ?
  A extends B ? A : A extends 'bool' ? B : B extends 'bool' ? A :
  'f64' extends A | B ? 'f64' : 'f32' extends A | B ? 'f32' : 'f16' extends A | B ? 'f16' :
  A extends keyof IntegerPromotion ? B extends keyof IntegerPromotion[A] ? IntegerPromotion[A][B] : never : never
  : never : never;
export type Floating<D extends DType> = D extends FloatDType ? D : 'f32';
export type Summed<D extends DType> = D extends 'bool' ? 'u32' : D;
export type Operand = Tensor | Scalar;
export type OperandShape<X extends Operand> = X extends Tensor<infer S, DType> ? S : readonly [];
type OperandDType<X extends Operand> = X extends Tensor<Shape, infer D> ? D : X extends boolean ? 'bool' : X extends bigint ? 'i64' : 'f32';
// Integer scalars can adopt an integer tensor's dtype only if their value fits.
// Their type alone cannot establish that, so retain both possible results.
type WithScalar<D extends DType, X extends Scalar> = D extends FloatDType ? D :
  X extends boolean ? Promote<D, 'bool'> : D extends 'bool' ? X extends bigint ? 'i64' : 'f32' : D | Promote<D, X extends bigint ? 'i64' : 'f32'>;
export type BinaryDType<A extends Operand, B extends Operand> =
  A extends Tensor<Shape, infer D> ? B extends Scalar ? WithScalar<D, B> : Promote<D, OperandDType<B>> :
  B extends Tensor<Shape, infer D> ? A extends Scalar ? WithScalar<D, A> : never : Promote<OperandDType<A>, OperandDType<B>>;

type BroadcastDim<A extends number, B extends number> = A extends 1 ? B : B extends 1 ? A :
  number extends A ? number : number extends B ? number : A extends B ? A : never;
export type Broadcast<A extends Shape, B extends Shape> = number extends A['length'] | B['length'] ? Shape :
  A extends readonly [...infer AT extends Shape, infer AL extends number] ?
    B extends readonly [...infer BT extends Shape, infer BL extends number] ? readonly [...Broadcast<AT, BT>, BroadcastDim<AL, BL>] : Readonly<A> : Readonly<B>;
export type MatmulShape<A extends Shape, B extends Shape> =
  A extends readonly [infer M extends number, number] ? B extends readonly [number, infer N extends number] ? readonly [M, N] : Shape : Shape;
export type Transposed<S extends Shape> = S extends readonly [infer A extends number, infer B extends number] ? readonly [B, A] : Shape;
type AxisValues<A> = A extends readonly number[] ? A[number] : A;
type ReduceTuple<S extends Shape, A, K extends boolean, I extends unknown[] = []> =
  S extends readonly [infer H extends number, ...infer T extends Shape] ?
    I['length'] extends A ? K extends true ? readonly [1, ...ReduceTuple<T, A, K, [...I, unknown]>] : ReduceTuple<T, A, K, [...I, unknown]> :
    `-${S['length']}` extends `${A & number}` ? K extends true ? readonly [1, ...ReduceTuple<T, A, K, [...I, unknown]>] : ReduceTuple<T, A, K, [...I, unknown]> :
    readonly [H, ...ReduceTuple<T, A, K, [...I, unknown]>] : readonly [];
type Ones<S extends Shape> = { readonly [P in keyof S]: 1 };
export type Reduced<S extends Shape, A extends number | readonly number[] | undefined, K extends boolean> =
  A extends undefined ? K extends true ? Ones<S> : readonly [] :
  number extends S['length'] ? Shape : number extends AxisValues<A> ? Shape : ReduceTuple<S, AxisValues<A>, K>;
