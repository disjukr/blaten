import { zeros, add, div, matmul, relu, neg, sum, mean, softmax, cast, reshape, transpose, crossEntropy, gt, and, not, argmax, eq, multiclassAccuracy, type MulticlassAccuracyInput, meanMetric, meanAccumulator, gradientAccumulator, accumulate, sgd, type Accumulator, type Tensor, type Shape } from '../src/index.js';

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false;
type Expect<T extends true> = T;
const x = zeros([2, 3]), w = zeros([3, 4]), bias = zeros([4]);
const prediction = x.matmul(w).add(bias).relu();
const functional = relu(add(matmul(x, w), bias));
const scalar = prediction.mul(2).div(2).neg();
const reduced = prediction.sum(), columnSum = prediction.sum(0), rowMean = prediction.mean(-1, true);
const axes = zeros([2, 3, 4]).sum([0, -1]);
const allKept = prediction.sum(undefined, true);
const reshaped = reshape(cast(prediction, 'f16'), [4, 2]);
const transposed = transpose(reshaped);
const loss = crossEntropy(prediction, zeros([2, 4]));
const mixed = zeros([2, 1], { dtype: 'i8' }).add(zeros([1, 3], { dtype: 'u8' }));
const floatMixed = zeros([2], { dtype: 'f16' }).add(zeros([2], { dtype: 'f32' }));
const integers = zeros([2], { dtype: 'u8' });
const uncertainScalar = integers.add(0.5), integerDivision = div(integers, integers);
const mask = prediction.gt(0).and(zeros([4], { dtype: 'bool' }));
const functionalMask = not(and(gt(prediction, 0), zeros([4], { dtype: 'bool' })));
const booleanSum = mask.sum();
const functions = [neg(x), softmax(x), sum(x, 1), mean(x)] as const;
const dynamic = zeros([] as number[]).add(x);
const unknownAxis = prediction.sum(0 as number);
export type Checks = [
  Expect<Equal<typeof prediction, Tensor<readonly [2, 4], 'f32'>>>,
  Expect<Equal<typeof functional, typeof prediction>>,
  Expect<Equal<typeof scalar, typeof prediction>>,
  Expect<Equal<typeof reduced, Tensor<readonly [], 'f32'>>>,
  Expect<Equal<typeof columnSum, Tensor<readonly [4], 'f32'>>>,
  Expect<Equal<typeof rowMean, Tensor<readonly [2, 1], 'f32'>>>,
  Expect<Equal<typeof axes, Tensor<readonly [3], 'f32'>>>,
  Expect<Equal<typeof allKept, Tensor<readonly [1, 1], 'f32'>>>,
  Expect<Equal<typeof reshaped, Tensor<readonly [4, 2], 'f16'>>>,
  Expect<Equal<typeof transposed, Tensor<readonly [2, 4], 'f16'>>>,
  Expect<Equal<typeof loss, Tensor<readonly [], 'f32'>>>,
  Expect<Equal<typeof mixed, Tensor<readonly [2, 3], 'i16'>>>,
  Expect<Equal<typeof floatMixed, Tensor<readonly [2], 'f32'>>>,
  Expect<Equal<typeof uncertainScalar, Tensor<readonly [2], 'u8' | 'f32'>>>,
  Expect<Equal<typeof integerDivision, Tensor<readonly [2], 'f32'>>>,
  Expect<Equal<typeof mask, Tensor<readonly [2, 4], 'bool'>>>,
  Expect<Equal<typeof functionalMask, typeof mask>>,
  Expect<Equal<typeof booleanSum, Tensor<readonly [], 'u32'>>>,
  Expect<Equal<typeof functions, readonly [Tensor<readonly [2, 3], 'f32'>, Tensor<readonly [2, 3], 'f32'>, Tensor<readonly [2], 'f32'>, Tensor<readonly [], 'f32'>]>>,
  Expect<Equal<typeof dynamic, Tensor<Shape, 'f32'>>>,
  Expect<Equal<typeof unknownAxis, Tensor<Shape, 'f32'>>>
];
// A chained result must not claim an unrelated shape or dtype.
// @ts-expect-error prediction has four output columns
const wrongShape: Tensor<readonly [2, 3], 'f32'> = prediction;
// @ts-expect-error mixed integer tensors promote to i16
const wrongDType: Tensor<readonly [2, 3], 'i8'> = mixed;
void wrongShape; void wrongDType;

using metric = meanMetric();
const metricValue = metric.compute();
using halfMetric = meanMetric({ dtype: 'f16' });
const halfValue = halfMetric.compute();
using optimizer = accumulate(sgd({ weights: w, bias }, { learningRate: 0.1 }), { steps: 4 });
optimizer.step({ weights: w, bias });
// @ts-expect-error gradients must preserve parameter shapes
optimizer.step({ weights: x, bias });
export type MetricChecks = [
  Expect<Equal<typeof metricValue, Tensor<readonly [], 'f32'>>>,
  Expect<Equal<typeof halfValue, Tensor<readonly [], 'f16'>>>
];

const tree = { layer: [w, bias] as const };
using treeMean = meanAccumulator(tree);
using gradients = gradientAccumulator(tree);
function collect<I, O>(accumulator: Accumulator<I, O>, input: I): O {
  accumulator.add(input);
  return accumulator.compute();
}
const averagedTree = collect(treeMean, tree);
const averagedGradient = collect(gradients, tree);
const averagedMetric = collect(metric, zeros([]));
// @ts-expect-error input shape must match the template
collect(treeMean, { layer: [x, bias] as const });
// @ts-expect-error legacy method names have been replaced by compute/reset
gradients.mean();
export type AccumulatorChecks = [
  Expect<Equal<typeof treeMean, Accumulator<typeof tree>>>,
  Expect<Equal<typeof gradients, Accumulator<typeof tree>>>,
  Expect<Equal<typeof averagedTree, typeof tree>>,
  Expect<Equal<typeof averagedGradient, typeof tree>>,
  Expect<Equal<typeof averagedMetric, Tensor<readonly [], 'f32'>>>
];

const predicted = prediction.argmax(-1);
const predictedFunction = argmax(prediction, 0);
const compared = predicted.eq(zeros([1], { dtype: 'i32' }));
const comparedFunction = eq(predicted, 0);
using accuracy = multiclassAccuracy();
const accuracyValue = accuracy.compute();
export type ClassificationChecks = [
  Expect<Equal<typeof predicted, Tensor<readonly [2], 'i32'>>>,
  Expect<Equal<typeof predictedFunction, Tensor<readonly [4], 'i32'>>>,
  Expect<Equal<typeof compared, Tensor<readonly [2], 'bool'>>>,
  Expect<Equal<typeof comparedFunction, typeof compared>>,
  Expect<Equal<typeof accuracyValue, Tensor<readonly [], 'f32'>>>,
  Expect<Equal<typeof accuracy, Accumulator<MulticlassAccuracyInput, Tensor<readonly [], 'f32'>>>>
];
