import { Tensor } from './tensor.js';
import { f32, isFloat, type FloatDType } from './types.js';
import { scope } from './scope.js';
import { meanAccumulator, type Accumulator } from './accumulator.js';
import { execute } from './execution.js';
import * as ir from './ir.js';

type FloatScalar = Tensor<readonly [], FloatDType>;

/** A scalar metric with an explicit accumulation dtype; compute() performs no host read. */
export function meanMetric<D extends FloatDType = typeof f32>(options: { dtype?: D } = {}): Accumulator<FloatScalar, Tensor<readonly [], D>> {
  const dtype = options.dtype ?? f32 as D;
  if (!isFloat(dtype)) throw new Error('Mean metric requires a floating dtype');
  let accumulator: Accumulator<Tensor<readonly [], D>> | undefined, disposed = false;
  const alive = () => { if (disposed) throw new Error('Mean metric is disposed') };
  return {
    get weight() { return accumulator?.weight ?? 0 },
    add(value: FloatScalar, options: { weight?: number } = {}): void {
      alive();
      const weight = options.weight ?? 1;
      if (!Number.isFinite(weight) || weight <= 0 || !Number.isFinite((accumulator?.weight ?? 0) + weight)) throw new Error('Metric weight must be positive and finite');
      if (value.rank !== 0 || !isFloat(value.dtype)) throw new Error('Mean metric requires a floating scalar tensor');
      scope(() => {
        const converted = value.cast(dtype);
        accumulator ??= meanAccumulator(converted);
        accumulator.add(converted, { weight });
      });
    },
    compute(): Tensor<readonly [], D> {
      alive();
      if (!accumulator) throw new Error('Cannot compute an empty mean metric');
      return accumulator.compute();
    },
    reset(): void { alive(); accumulator?.dispose(); accumulator = undefined },
    dispose(): void { if (disposed) return; accumulator?.dispose(); accumulator = undefined; disposed = true },
    [Symbol.dispose](): void { this.dispose() },
  };
}

export interface MulticlassAccuracyInput {
  logits: Tensor<readonly [number, number], FloatDType>;
  targets: Tensor<readonly [number, number]>;
  validCount?: number;
}
type Counts = { correct: Tensor<readonly [], typeof f32>; total: Tensor<readonly [], typeof f32> };

/** One-hot multiclass accuracy; only the first validCount rows contribute. */
export function multiclassAccuracy(): Accumulator<MulticlassAccuracyInput, Tensor<readonly [], typeof f32>> {
  let counts: Accumulator<Counts> | undefined, totalWeight = 0, disposed = false;
  const alive = () => { if (disposed) throw new Error('Multiclass accuracy is disposed') };
  return {
    get weight() { return totalWeight },
    add(input: MulticlassAccuracyInput, options: { weight?: number } = {}): void {
      alive();
      const { logits, targets, validCount = logits.shape[0] } = input, weight = options.weight ?? 1;
      if (logits.rank !== 2 || targets.rank !== 2 || String(logits.shape) !== String(targets.shape) || !isFloat(logits.dtype)) throw new Error('Accuracy requires matching [batch, classes] logits and one-hot targets');
      if (!Number.isSafeInteger(validCount) || validCount <= 0 || validCount > logits.shape[0]) throw new Error('validCount must be a positive integer within the batch');
      if (!Number.isFinite(weight) || weight <= 0 || !Number.isFinite(totalWeight + validCount * weight)) throw new Error('Accuracy weight must be positive and finite');
      scope(() => {
        const inputs = [logits, targets].map((value, index) => ir.node('input', [], value.shape, { index }, value.dtype));
        const matches = ir.binary('eq', ir.argmax(inputs[0]!, -1), ir.argmax(inputs[1]!, -1));
        const outputs = [ir.cast(ir.countTrue(matches, validCount), f32), ir.constant(validCount, f32)];
        // Like a compiled kernel, the metric uploads host inputs to the selected device.
        const [correct, total] = execute({ inputs, outputs, single: false }, [logits, targets], { uploadHostInputs: true }) as [Counts['correct'], Counts['total']];
        const values = { correct, total };
        counts ??= meanAccumulator(values);
        counts.add(values, { weight });
      });
      totalWeight += validCount * weight;
    },
    compute(): Tensor<readonly [], typeof f32> {
      alive(); if (!counts) throw new Error('Cannot compute empty multiclass accuracy');
      // Both means have the same denominator, which cancels in their ratio.
      return scope(() => { const value = counts!.compute(); return value.correct.div(value.total) });
    },
    reset(): void { alive(); counts?.dispose(); counts = undefined; totalWeight = 0 },
    dispose(): void { if (disposed) return; counts?.dispose(); counts = undefined; totalWeight = 0; disposed = true },
    [Symbol.dispose](): void { this.dispose() },
  };
}
