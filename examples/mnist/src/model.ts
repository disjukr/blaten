import { matmul, relu, crossEntropy, type f32, type Tensor, type Shape } from 'blaten';

export const hiddenSize = 32;
type F32<S extends Shape> = Tensor<S, typeof f32>;
export type Images<Batch extends number = number> = F32<[Batch, 784]>;
export type Labels<Batch extends number = number> = F32<[Batch, 10]>;
export type W1 = F32<[784, typeof hiddenSize]>;
export type B1 = F32<[typeof hiddenSize]>;
export type W2 = F32<[typeof hiddenSize, 10]>;
export type B2 = F32<[10]>;

export function predict(x: Images, w1: W1, b1: B1, w2: W2, b2: B2) {
  'use kernel';
  const hidden = relu(matmul(x, w1).add(b1));
  return matmul(hidden, w2).add(b2);
}
export function loss(x: Images, y: Labels, w1: W1, b1: B1, w2: W2, b2: B2) {
  'use kernel';
  return crossEntropy(predict(x, w1, b1, w2, b2), y);
}
