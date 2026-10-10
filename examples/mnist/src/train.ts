import { mkdir, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { sgd, tensor, valueAndGrad, gradientAccumulator, type Executable, type Tensor } from 'blaten';
import { inspect, type TypeGpuBackend } from '@blaten/typegpu';
import { hiddenSize, initParameters, loss, predict, type Parameters } from './model.js';
import { load, batches, type Dataset } from './data.js';
import { withBackend } from './runtime.js';

const setting = (name: string, fallback: number) => {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`);
  return value;
};
async function evaluate(backend: TypeGpuBackend, inference: Executable<typeof predict>, parameters: Parameters, data: Dataset, size: number, batchSize: number) {
  const results: Promise<number>[] = [];
  for (const batch of batches(data, { size, batchSize })) {
    results.push(backend.scope(() => {
      const [x] = batch.toTensors();
      return inference(parameters, x).read().then(logits => {
        let matches = 0;
        batch.indices.forEach((source, row) => {
          let best = 0;
          for (let c = 1; c < 10; c++) if (logits[row * 10 + c]! > logits[row * 10 + best]!) best = c;
          if (best === data.labels[source]) matches++;
        });
        return matches;
      });
    }));
  }
  return (await Promise.all(results)).reduce((sum, correct) => sum + correct, 0) / size;
}
export async function main() {
  const trainSize = setting('TRAIN_SIZE', 12000), testSize = setting('TEST_SIZE', 2000), epochs = setting('EPOCHS', 3), batchSize = setting('BATCH_SIZE', 64);
  const accumulationSteps = setting('ACCUMULATION_STEPS', 1), learningRate = Number(process.env.LEARNING_RATE ?? 0.1);
  if (setting('HIDDEN', hiddenSize) !== hiddenSize) throw new Error('Change hiddenSize in model.ts to compile other model widths.');
  if (!(learningRate > 0) || !Number.isFinite(learningRate)) throw new Error('LEARNING_RATE must be a positive finite number');
  if (trainSize < batchSize) throw new Error('TRAIN_SIZE must be at least BATCH_SIZE');
  const started = performance.now(), [trainData, testData] = await Promise.all([load('train'), load('test')]);
  if (trainSize > trainData.count || testSize > testData.count) throw new Error('Requested subset exceeds MNIST size');
  const result = await withBackend(async (backend, adapter) => {
    const parameters = await backend.scope(() => backend.copyFrom(initParameters()));
    const [training, inference] = await backend.scope(() => {
      const [x, y] = batches(trainData, { size: batchSize, batchSize }).next().value!.toTensors();
      return Promise.all([
        backend.compile(valueAndGrad(loss), { inputs: [parameters, x, y] }),
        backend.compile(predict, { inputs: [parameters, x] }),
      ]);
    });
    const optimizer = sgd(parameters, { learningRate }), accumulator = gradientAccumulator(parameters);
    const update = () => backend.scope(() => { optimizer.step(accumulator.mean()); accumulator.clear() });
    const accuracy = () => evaluate(backend, inference, parameters, testData, testSize, batchSize);
    console.log(`MNIST 784 → ${hiddenSize} → 10 on ${adapter.info.description || adapter.info.device || 'WebGPU'}; ${inspect(training).dispatchCount} training dispatches/batch`);
    const initialAccuracy = await accuracy(), history: { epoch: number; loss: number; accuracy: number }[] = [];
    console.log(`Before training: ${(initialAccuracy * 100).toFixed(2)}% test accuracy`);
    try {
      for (let epoch = 1; epoch <= epochs; epoch++) {
        let totalLoss: Tensor = await backend.scope(() => backend.copyFrom(tensor(0, { shape: [] })));
        let count = 0;
        try {
          for (const batch of batches(trainData, { size: trainSize, batchSize, shuffleSeed: 100 + epoch, dropLast: true })) {
            const previousLoss = totalLoss;
            totalLoss = backend.scope(() => {
              const [x, y] = batch.toTensors(), { value, grads } = training(parameters, x, y);
              if (accumulationSteps === 1) optimizer.step(grads);
              else accumulator.add(grads, { weight: batch.size });
              return totalLoss.add(value);
            });
            previousLoss.dispose();
            if (++count % accumulationSteps === 0 && accumulator.weight) update();
          }
          if (accumulator.weight) update();
          // Reporting is the synchronization boundary, independent of batch/update size.
          const averageLoss = await backend.scope(() => totalLoss.div(count).item());
          const measured = { epoch, loss: averageLoss, accuracy: await accuracy() };
          history.push(measured);
          console.log(`Epoch ${epoch}/${epochs}: loss=${measured.loss.toFixed(4)}, test accuracy=${(measured.accuracy * 100).toFixed(2)}%`);
        } finally { totalLoss.dispose() }
      }
    } finally { accumulator.dispose() }
    return {
      verifiedAt: new Date().toISOString(), dataset: 'MNIST (CVDF IDX)', backend: 'typegpu', typegpuVersion: createRequire(import.meta.url)('typegpu/package.json').version,
      node: process.version, platform: process.platform, architecture: process.arch,
      adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description },
      trainSize, testSize, batchSize, hidden: hiddenSize, learningRate, accumulationSteps, initialAccuracy, history, seconds: (performance.now() - started) / 1000,
    };
  });
  await mkdir('results', { recursive: true });
  await writeFile('results/latest.json', JSON.stringify(result, null, 2) + '\n');
  const last = result.history.at(-1)!;
  if (last.accuracy < 0.75 || last.accuracy <= result.initialAccuracy + 0.3 || (epochs > 1 && last.loss >= result.history[0]!.loss)) throw new Error('MNIST learning verification failed');
  console.log(`Learning verified in ${result.seconds.toFixed(1)}s. Saved results/latest.json`);
}
