import { Tensor } from './tensor.js';
import { f32, isFloat, type FloatDType } from './types.js';
import { scope } from './scope.js';
import { meanAccumulator, type Accumulator } from './accumulator.js';

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
