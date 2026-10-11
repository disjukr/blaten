import { sgd, valueAndGrad, accumulate, meanMetric, multiclassAccuracy, type Executable } from 'blaten';
import { inspect, type TypeGpuBackend } from '@blaten/typegpu';
import { hiddenSize, initParameters, loss, predict, type Parameters } from './model.js';
import { dataLoader, imageClassification, type ImageDataset } from '@blaten/datasets';

export interface TrainingOptions {
  trainSize: number; testSize: number; epochs: number; batchSize: number;
  accumulationSteps: number; learningRate: number;
}
interface TrainingData { train: ImageDataset<784, 10>; test: ImageDataset<784, 10> }
const collate = imageClassification({ layout: 'flat', labels: 'oneHot' });

async function evaluate(backend: TypeGpuBackend, inference: Executable<typeof predict>, parameters: Parameters, data: ImageDataset<784, 10>, size: number, batchSize: number) {
  using accuracy = multiclassAccuracy();
  for await (const batch of dataLoader({ dataset: data, size, batchSize, lastBatch: 'pad', collate })) {
    using ownedBatch = batch;
    backend.scope(() => {
      const { images: x, targets: y } = batch.data;
      accuracy.add({ logits: inference(parameters, x), targets: y, validCount: batch.size });
    });
  }
  return backend.scope(() => accuracy.compute().item());
}
export async function train(backend: TypeGpuBackend, data: TrainingData, options: TrainingOptions) {
  const { trainSize, testSize, epochs, batchSize, accumulationSteps, learningRate } = options;
  const { train: trainData, test: testData } = data;
  const parameters = await backend.scope(() => backend.copyFrom(initParameters()));
  const examples = dataLoader({ dataset: trainData, size: batchSize, batchSize, collate });
  using example = (await examples.first())!;
  const [training, inference] = await backend.scope(() => {
    const { images: x, targets: y } = example.data;
    return Promise.all([
      backend.compile(valueAndGrad(loss), { inputs: [parameters, x, y] }),
      backend.compile(predict, { inputs: [parameters, x] }),
    ]);
  });
  using optimizer = accumulate(sgd(parameters, { learningRate }), { steps: accumulationSteps });
  using lossMetric = meanMetric();
  const accuracy = () => evaluate(backend, inference, parameters, testData, testSize, batchSize);
  console.log(`MNIST 784 → ${hiddenSize} → 10; ${inspect(training).dispatchCount} training dispatches/batch`);
  const initialAccuracy = await accuracy(), history: { epoch: number; loss: number; accuracy: number }[] = [];
  console.log(`Before training: ${(initialAccuracy * 100).toFixed(2)}% test accuracy`);
  for (let epoch = 1; epoch <= epochs; epoch++) {
    lossMetric.reset();
    for await (const batch of dataLoader({ dataset: trainData, size: trainSize, batchSize, shuffleSeed: 100 + epoch, lastBatch: 'drop', collate })) {
      using ownedBatch = batch;
      backend.scope(() => {
        const { images: x, targets: y } = batch.data, { value, grads } = training(parameters, x, y);
        optimizer.step(grads, { weight: batch.size });
        lossMetric.add(value, { weight: batch.size });
      });
    }
    optimizer.flush();
    // Reporting is the synchronization boundary, independent of batch/update size.
    const averageLoss = await backend.scope(() => lossMetric.compute().item());
    const measured = { epoch, loss: averageLoss, accuracy: await accuracy() };
    history.push(measured);
    console.log(`Epoch ${epoch}/${epochs}: loss=${measured.loss.toFixed(4)}, test accuracy=${(measured.accuracy * 100).toFixed(2)}%`);
  }
  return { initialAccuracy, history };
}
