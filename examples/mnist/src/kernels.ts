import { loss, predict, type Images, type Labels, type W1, type B1, type W2, type B2 } from './model.js';

// Batch size belongs to these compiled entry points, not to the model.
export const batchSize = 64;
type BatchImages = Images<typeof batchSize>;
type BatchLabels = Labels<typeof batchSize>;

export function trainingLoss(x: BatchImages, y: BatchLabels, w1: W1, b1: B1, w2: W2, b2: B2) {
  'use kernel';
  return loss(x, y, w1, b1, w2, b2);
}

export function inference(x: BatchImages, w1: W1, b1: B1, w2: W2, b2: B2) {
  'use kernel';
  return predict(x, w1, b1, w2, b2);
}
