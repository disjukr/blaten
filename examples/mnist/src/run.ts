import { mkdir, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { hiddenSize } from './model.js';
import { load } from './data.js';
import { train, type TrainingOptions } from './train.js';
import { withBackend } from './runtime.js';

function setting(name: string, fallback: number) {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`);
  return value;
}
function settings(): TrainingOptions {
  const options = {
    trainSize: setting('TRAIN_SIZE', 12000), testSize: setting('TEST_SIZE', 2000),
    epochs: setting('EPOCHS', 3), batchSize: setting('BATCH_SIZE', 64),
    accumulationSteps: setting('ACCUMULATION_STEPS', 1), learningRate: Number(process.env.LEARNING_RATE ?? 0.1),
  };
  if (setting('HIDDEN', hiddenSize) !== hiddenSize) throw new Error('Change hiddenSize in model.ts to compile other model widths.');
  if (!(options.learningRate > 0) || !Number.isFinite(options.learningRate)) throw new Error('LEARNING_RATE must be a positive finite number');
  if (options.trainSize < options.batchSize) throw new Error('TRAIN_SIZE must be at least BATCH_SIZE');
  return options;
}
export async function main() {
  const options = settings(), started = performance.now();
  const [trainData, testData] = await Promise.all([load('train'), load('test')]);
  if (options.trainSize > trainData.count || options.testSize > testData.count) throw new Error('Requested subset exceeds MNIST size');
  const result = await withBackend(async (backend, adapter) => {
    console.log(`Device: ${adapter.info.description || adapter.info.device || 'WebGPU'}`);
    const learning = await train(backend, { train: trainData, test: testData }, options);
    return {
      verifiedAt: new Date().toISOString(), dataset: 'MNIST (CVDF IDX)', backend: 'typegpu', typegpuVersion: createRequire(import.meta.url)('typegpu/package.json').version,
      node: process.version, platform: process.platform, architecture: process.arch,
      adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description },
      ...options, hidden: hiddenSize, ...learning, seconds: (performance.now() - started) / 1000,
    };
  });
  await mkdir('results', { recursive: true });
  await writeFile('results/latest.json', JSON.stringify(result, null, 2) + '\n');
  const last = result.history.at(-1)!;
  if (last.accuracy < 0.75 || last.accuracy <= result.initialAccuracy + 0.3 || (options.epochs > 1 && last.loss >= result.history[0]!.loss)) throw new Error('MNIST learning verification failed');
  console.log(`Learning verified in ${result.seconds.toFixed(1)}s. Saved results/latest.json`);
}
