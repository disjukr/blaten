import { compileLazy, valueAndGrad, zeros, tensor, sgd, meanMetric, multiclassAccuracy } from 'blaten';
import { requestDevice, typegpu } from '@blaten/typegpu';
import { attachSources, type TraceResult } from 'blaten/trace';
import { initParameters, loss, predict } from './model';
import modelSource from './model.ts?raw';
import { loadSplit } from './data';

export interface Options { epochs: number; trainSize: number; testSize: number; batchSize: number; learningRate: number }
export interface Weights { w1: number[]; b1: number[]; w2: number[]; b2: number[] }
export interface Measurement { epoch: number; loss?: number; accuracy: number }
export interface Update { status?: string; progress?: number; weights?: Weights; trace?: TraceResult; measurement?: Measurement; device?: string; trained?: boolean }
const yieldToBrowser = () => new Promise<void>(resolve => setTimeout(resolve, 0));

export async function createTrainer() {
  const device = await requestDevice();
  try {
    const backend = await typegpu({ device, trace: true });
    try {
      const gpuErrors: string[] = [];
      device.addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
      const trainingKernel = compileLazy(valueAndGrad(loss), backend);
      const inferenceKernel = compileLazy(predict, backend);
      let parameters = await backend.scope(() => backend.copyFrom(initParameters()));
      let epochCount = 0;
      let preparedBatchSize = 64;
      async function run(options: Options, signal: AbortSignal, update: (value: Update) => void, compileOnly = false, resetWeights = false) {
        const { epochs, trainSize, testSize, batchSize, learningRate } = options;
        if (![epochs, trainSize, testSize, batchSize].every(n => Number.isSafeInteger(n) && n > 0) || trainSize > 60000 || testSize > 10000 || batchSize > trainSize || !Number.isFinite(learningRate) || learningRate <= 0) throw new Error('Check the training settings.');
        const check = () => signal.throwIfAborted();
        update({ device: device.adapterInfo.description || device.adapterInfo.vendor || 'WebGPU', status: 'Preparing model', progress: 0 });
        if (resetWeights) {
          const fresh = await backend.scope(() => backend.copyFrom(initParameters()));
          Object.values(parameters).forEach(value => value.dispose());
          parameters = fresh; epochCount = 0;
          update({ trained: false });
        }
        try {
          using dummyX = zeros([batchSize, 784]);
          using dummyY = zeros([batchSize, 10]);
          using training = await trainingKernel.prepare(parameters, dummyX, dummyY);
          using inference = await inferenceKernel.prepare(parameters, dummyX);
          preparedBatchSize = batchSize;
          const publishTrace = () => {
            const result = backend.trace!.toResult();
            attachSources(result, { 'src/model.ts': modelSource });
            update({ trace: result });
          };
          const publishWeights = async () => {
            const entries = await Promise.all(Object.entries(parameters).map(async ([key, value]) => [key, Array.from(await value.read())]));
            update({ weights: Object.fromEntries(entries) as unknown as Weights });
            if (gpuErrors.length) throw new Error(gpuErrors.join('\n'));
            check();
          };
          await publishWeights(); publishTrace(); check();
          if (compileOnly) { update({ status: 'Compilation ready', progress: 1 }); return; }
          update({ status: 'Downloading and decompressing MNIST' });
          const [train, test] = await Promise.all([loadSplit('train', signal), loadSplit('test', signal)]);
          check();
          const batch = (split: typeof train, indices: readonly number[]) => {
            const images = new Float32Array(batchSize * 784), targets = new Float32Array(batchSize * 10);
            indices.forEach((index, row) => {
              for (let p = 0; p < 784; p++) images[row * 784 + p] = split.pixels[index * 784 + p]! / 255;
              targets[row * 10 + split.labels[index]!] = 1;
            });
            const x = tensor(images, { shape: [batchSize, 784] }), y = tensor(targets, { shape: [batchSize, 10] });
            return { x, y, [Symbol.dispose]() { x.dispose(); y.dispose(); } };
          };
          const evaluate = async () => {
            using metric = multiclassAccuracy();
            for (let start = 0; start < testSize; start += batchSize) {
              check();
              using data = batch(test, Array.from({ length: Math.min(batchSize, testSize - start) }, (_, i) => start + i));
              backend.scope(() => metric.add({ logits: inference(parameters, data.x), targets: data.y, validCount: Math.min(batchSize, testSize - start) }));
              if ((start / batchSize) % 8 === 0) { await backend.synchronize(); await yieldToBrowser(); }
            }
            return backend.scope(() => metric.compute().item());
          };
          if (epochCount === 0) {
            update({ status: 'Evaluating initial accuracy' });
            update({ measurement: { epoch: 0, accuracy: await evaluate() } });
          }
          const optimizer = sgd(parameters, { learningRate });
          const steps = Math.floor(trainSize / batchSize);
          for (let runEpoch = 1; runEpoch <= epochs; runEpoch++) {
            const epoch = epochCount + 1;
            using metric = meanMetric();
            const order = Array.from({ length: trainSize }, (_, i) => i);
            // A reproducible shuffle, matching the example's seeded initialization.
            let seed = 100 + epoch;
            for (let i = order.length - 1; i > 0; i--) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; const j = seed % (i + 1); [order[i], order[j]] = [order[j]!, order[i]!]; }
            for (let step = 0; step < steps; step++) {
              check();
              using data = batch(train, order.slice(step * batchSize, (step + 1) * batchSize));
              backend.scope(() => {
                const { value, grads } = training(parameters, data.x, data.y);
                optimizer.step(grads); metric.add(value, { weight: batchSize });
              });
              if (runEpoch === 1 && step === 0) update({ trained: true });
              if (step % 8 === 0 || step === steps - 1) {
                await backend.synchronize();
                update({ status: `Epoch ${epoch} · batch ${step + 1}/${steps}`, progress: ((runEpoch - 1) * steps + step + 1) / (epochs * steps) });
                if (step % 64 === 0) { await publishWeights(); publishTrace(); }
                await yieldToBrowser();
              }
            }
            update({ status: `Epoch ${epoch} · Evaluating accuracy` });
            const value = await backend.scope(() => metric.compute().item());
            const accuracy = await evaluate();
            epochCount = epoch;
            update({ measurement: { epoch, loss: value, accuracy } });
            await publishWeights(); publishTrace();
          }
          update({ status: 'Training complete', progress: 1 });
        } finally {
          const result = backend.trace!.toResult();
          attachSources(result, { 'src/model.ts': modelSource });
          update({ trace: result });
        }
      }
      let disposed = false;
      return {
        run,
        async predictDigit(image: Float32Array) {
          if (image.length !== 784) throw new Error('Expected a 28 × 28 image.');
          const data = new Float32Array(preparedBatchSize * 784);
          data.set(image);
          using input = tensor(data, { shape: [preparedBatchSize, 784] });
          using inference = await inferenceKernel.prepare(parameters, input);
          using output = backend.scope(() => inference(parameters, input));
          const logits = Array.from((await output.read()).slice(0, 10), Number);
          const max = Math.max(...logits), exp = logits.map(value => Math.exp(value - max));
          const total = exp.reduce((sum, value) => sum + value, 0);
          return exp.map(value => value / total);
        },
        dispose() {
          if (disposed) return;
          disposed = true;
          Object.values(parameters).forEach(value => value.dispose());
          inferenceKernel.dispose(); trainingKernel.dispose(); backend.dispose(); device.destroy();
        },
      };
    } catch (error) { backend.dispose(); throw error; }
  } catch (error) { device.destroy(); throw error; }
}
