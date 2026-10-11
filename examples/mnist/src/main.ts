import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { hiddenSize, initParameters, loss, predict, type Parameters } from './model.js';
import { mnist, dataLoader, imageClassification, type ImageDataset } from '@blaten/datasets';
import { typegpu, type TypeGpuBackend } from '@blaten/typegpu';
import { sgd, valueAndGrad, compileLazy, zeros, accumulate, meanMetric, multiclassAccuracy, type Executable } from 'blaten';
import { requestDevice } from '@blaten/typegpu/node';
import { attachSources } from 'blaten/trace';
import { object } from '@optique/core/constructs';
import { message } from '@optique/core/message';
import { option } from '@optique/core/primitives';
import { float, integer } from '@optique/core/valueparser';
import { bindEnv, createEnvContext } from '@optique/env';
import { runAsync } from '@optique/run';

const env = createEnvContext();
const positiveInteger = integer({ min: 1 });
const count = (name: `--${string}`, key: string, fallback: number) => bindEnv(option(name, positiveInteger, {
  description: message`Environment: ${key}; default: ${String(fallback)}.`,
}), {
  context: env, key, parser: positiveInteger, default: fallback,
});

const parser = object({
  trainSize: count('--train-size', 'TRAIN_SIZE', 12000),
  testSize: count('--test-size', 'TEST_SIZE', 2000),
  epochs: count('--epochs', 'EPOCHS', 3),
  batchSize: count('--batch-size', 'BATCH_SIZE', 64),
  accumulationSteps: count('--accumulation-steps', 'ACCUMULATION_STEPS', 1),
  learningRate: bindEnv(option('--learning-rate', float({ min: Number.MIN_VALUE }), {
    description: message`Environment: LEARNING_RATE; default: 0.1.`,
  }), {
    context: env, key: 'LEARNING_RATE', parser: float({ min: Number.MIN_VALUE }), default: 0.1,
  }),
  compileOnly: option('--compile-only', '--inspect', { description: message`Compile model kernels without loading data or training; save results/trace.json.` }),
  trace: option('--trace', { description: message`Save compilation traces from training to results/trace.json.` }),
});

interface TrainingOptions {
  trainSize: number; testSize: number; epochs: number; batchSize: number;
  accumulationSteps: number; learningRate: number;
  compileOnly?: boolean;
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
async function train(backend: TypeGpuBackend, loadData: () => Promise<TrainingData>, options: TrainingOptions) {
  const { trainSize, testSize, epochs, batchSize, accumulationSteps, learningRate } = options;
  const parameters = await backend.scope(() => backend.copyFrom(initParameters()));
  using trainingKernel = compileLazy(valueAndGrad(loss), backend);
  using inferenceKernel = compileLazy(predict, backend);
  using x = zeros([batchSize, 784]);
  using y = zeros([batchSize, 10]);
  using training = await trainingKernel.prepare(parameters, x, y);
  using inference = await inferenceKernel.prepare(parameters, x);
  console.log(`MNIST 784 → ${hiddenSize} → 10`);
  if (options.compileOnly) {
    console.log('Compiled training and inference without loading data or executing either kernel.');
    return;
  }
  const { train: trainData, test: testData } = await loadData();
  using optimizer = accumulate(sgd(parameters, { learningRate }), { steps: accumulationSteps });
  using lossMetric = meanMetric();
  const accuracy = () => evaluate(backend, inference, parameters, testData, testSize, batchSize);
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

export async function main() {
  const { trace: requestedTrace, ...options } = await runAsync(parser, {
    programName: 'mnist', help: 'option', contexts: [env],
    brief: message`Train MNIST or inspect its compiled kernels. CLI values override environment variables.`,
  });
  if (options.trainSize < options.batchSize) throw new Error('Training size (--train-size / TRAIN_SIZE) must be at least batch size (--batch-size / BATCH_SIZE)');
  const trace = options.compileOnly || requestedTrace, started = performance.now();
  const loadData = async () => {
    const [trainData, testData] = await Promise.all([
      mnist({ split: 'train', root: 'data/mnist' }),
      mnist({ split: 'test', root: 'data/mnist' }),
    ]);
    if (options.trainSize > trainData.count || options.testSize > testData.count) throw new Error('Requested subset exceeds MNIST size');
    return { train: trainData, test: testData };
  };
  using device = await requestDevice();
  const errors: string[] = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  using backend = await typegpu({ device, trace });
  const info = device.adapterInfo;
  console.log(`Device: ${info.description || info.device || 'WebGPU'}`);
  const learning = await train(backend, loadData, options);
  await backend.synchronize();
  if (errors.length) throw new Error(errors.join('\n'));
  if (backend.trace) {
    const compilation = backend.trace.toResult();
    attachSources(compilation, { 'src/model.ts': await readFile('src/model.ts', 'utf8') });
    await mkdir('results', { recursive: true });
    await writeFile('results/trace.json', JSON.stringify(compilation, null, 2) + '\n');
    console.log(`Saved ${compilation.compilations.length} compilation records to results/trace.json`);
  }
  if (!learning) return;
  const result = {
    verifiedAt: new Date().toISOString(), dataset: 'MNIST (CVDF IDX)', backend: 'typegpu', typegpuVersion: createRequire(import.meta.url)('typegpu/package.json').version,
    node: process.version, platform: process.platform, architecture: process.arch,
    adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
    ...options, hidden: hiddenSize, ...learning, seconds: (performance.now() - started) / 1000,
  };
  await mkdir('results', { recursive: true });
  await writeFile('results/latest.json', JSON.stringify(result, null, 2) + '\n');
  const last = result.history.at(-1)!;
  if (last.accuracy < 0.75 || last.accuracy <= result.initialAccuracy + 0.3 || (options.epochs > 1 && last.loss >= result.history[0]!.loss)) throw new Error('MNIST learning verification failed');
  console.log(`Learning verified in ${result.seconds.toFixed(1)}s. Saved results/latest.json`);
}
