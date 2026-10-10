import { f32 } from 'blaten';
import { mkdir, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { create, globals } from 'webgpu';
import { compile, randn, zeros, sgd, valueAndGrad, type Tensor, type Shape } from 'blaten';
import { typegpu, inspect } from '@blaten/typegpu';
import { hiddenSize, loss, predict } from './model.js';
import { load, batch, shuffle } from './data.js';
const setting = (name: string, fallback: number) => {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`);
  return value;
};
export async function main() {
  const trainSize = setting('TRAIN_SIZE', 12000), testSize = setting('TEST_SIZE', 2000), epochs = setting('EPOCHS', 3), batchSize = setting('BATCH_SIZE', 64), hidden = setting('HIDDEN', hiddenSize);
  const learningRate = Number(process.env.LEARNING_RATE ?? 0.1);
  if (hidden !== hiddenSize) throw new Error('Change hiddenSize in model.ts to compile other model widths.');
  if (!(learningRate > 0) || !Number.isFinite(learningRate)) throw new Error('LEARNING_RATE must be a positive finite number');
  if (trainSize < batchSize) throw new Error('TRAIN_SIZE must be at least BATCH_SIZE');
  const started = performance.now();
  const [trainData, testData] = await Promise.all([load('train'), load('test')]);
  if (trainSize > trainData.count || testSize > testData.count) throw new Error('Requested subset exceeds MNIST size');
  Object.assign(globalThis, globals);
  const gpu = create([]), adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter available');
  const device = await adapter.requestDevice();
  const gpuErrors: string[] = [];
  device.addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
  const backend = await typegpu({ device });
  try {
    const initialParameters: Tensor<Shape, typeof f32>[] = [randn([784, hidden], { seed: 1, scale: Math.sqrt(2 / 784) }), zeros([hidden]), randn([hidden, 10], { seed: 2, scale: Math.sqrt(2 / hidden) }), zeros([10])];
    const parameters = await Promise.all(initialParameters.map(x => backend.copyFrom(x)));
    const examples = batch(trainData, Array.from({ length: batchSize }, (_, i) => i), batchSize);
    const training = await compile(valueAndGrad(loss, { arg: [2, 3, 4, 5] }), backend, { inputs: [...examples, ...parameters] });
    const inference = await compile(predict, backend, { inputs: [examples[0], ...parameters] });
    examples.forEach(tensor => tensor.dispose());
    const optimizer = sgd(parameters, { learningRate });
    console.log(`MNIST 784 → ${hidden} → 10 on ${adapter.info.description || adapter.info.device || 'WebGPU'}; ${inspect(training).dispatchCount} training dispatches/batch`);
    const evaluate = async () => {
      let correct = 0;
      for (let start = 0; start < testSize; start += batchSize) {
        const indices = Array.from({ length: Math.min(batchSize, testSize - start) }, (_, i) => start + i);
        const [x, y] = batch(testData, indices, batchSize);
        const output = inference.run(x, ...parameters) as Tensor<Shape, typeof f32>;
        const logits = await output.read();
        indices.forEach((source, row) => {
          let best = 0;
          for (let c = 1; c < 10; c++) if (logits[row * 10 + c]! > logits[row * 10 + best]!) best = c;
          if (best === testData.labels[source]) correct++;
        });
        output.dispose(); x.dispose(); y.dispose();
      }
      return correct / testSize;
    };
    const initialAccuracy = await evaluate();
    console.log(`Before training: ${(initialAccuracy * 100).toFixed(2)}% test accuracy`);
    const history: { epoch: number; loss: number; accuracy: number }[] = [];
    const indices = Array.from({ length: trainSize }, (_, i) => i);
    for (let epoch = 1; epoch <= epochs; epoch++) {
      shuffle(indices, 100 + epoch);
      let totalLoss = 0, batches = 0;
      // Discard a partial training batch; evaluation still uses every requested sample.
      for (let start = 0; start + batchSize <= trainSize; start += batchSize) {
        const [x, y] = batch(trainData, indices.slice(start, start + batchSize), batchSize);
        const [value, ...gradients] = training.run(x, y, ...parameters) as Tensor<Shape, typeof f32>[];
        optimizer.step(gradients);
        totalLoss += await value!.item(); batches++;
        value!.dispose(); gradients.forEach(g => g.dispose()); x.dispose(); y.dispose();
      }
      if (!batches) throw new Error('TRAIN_SIZE must be at least BATCH_SIZE');
      const accuracy = await evaluate(), averageLoss = totalLoss / batches;
      history.push({ epoch, loss: averageLoss, accuracy });
      console.log(`Epoch ${epoch}/${epochs}: loss=${averageLoss.toFixed(4)}, test accuracy=${(accuracy * 100).toFixed(2)}%`);
    }
    await backend.synchronize();
    if (gpuErrors.length) throw new Error(gpuErrors.join('\n'));
    const result = {
      verifiedAt: new Date().toISOString(), dataset: 'MNIST (CVDF IDX)', backend: 'typegpu', typegpuVersion: createRequire(import.meta.url)('typegpu/package.json').version,
      node: process.version, platform: process.platform, architecture: process.arch,
      adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description },
      trainSize, testSize, batchSize, hidden, learningRate, initialAccuracy, history, seconds: (performance.now() - started) / 1000,
    };
    await mkdir('results', { recursive: true });
    await writeFile('results/latest.json', JSON.stringify(result, null, 2) + '\n');
    if (history.at(-1)!.accuracy < 0.75 || history.at(-1)!.accuracy <= initialAccuracy + 0.3 || (epochs > 1 && history.at(-1)!.loss >= history[0]!.loss)) throw new Error('MNIST learning verification failed');
    console.log(`Learning verified in ${result.seconds.toFixed(1)}s. Saved results/latest.json`);
  } finally { backend.dispose(); device.destroy() }
}
