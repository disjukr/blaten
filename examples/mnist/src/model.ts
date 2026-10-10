import { crossEntropy, randn, zeros, type TensorF32 } from 'blaten';

export const hiddenSize = 32;
export type Images<Batch extends number = number> = TensorF32<[Batch, 784]>;
export type Labels<Batch extends number = number> = TensorF32<[Batch, 10]>;
export type Parameters = {
  w1: TensorF32<[784, typeof hiddenSize]>;
  b1: TensorF32<[typeof hiddenSize]>;
  w2: TensorF32<[typeof hiddenSize, 10]>;
  b2: TensorF32<[10]>;
};
export function initParameters(): Parameters {
  return {
    w1: randn([784, hiddenSize], { seed: 1, scale: Math.sqrt(2 / 784) }),
    b1: zeros([hiddenSize]),
    w2: randn([hiddenSize, 10], { seed: 2, scale: Math.sqrt(2 / hiddenSize) }),
    b2: zeros([10]),
  };
}
export function predict(p: Parameters, x: Images) {
  'use kernel';
  return x.matmul(p.w1).add(p.b1).relu().matmul(p.w2).add(p.b2);
}
export function loss(p: Parameters, x: Images, y: Labels) {
  'use kernel';
  return crossEntropy(predict(p, x), y);
}
