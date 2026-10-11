import type { Dataset, DatasetMetadata, ImageClassificationMetadata, ImageClassificationSample } from './dataset.js';
import type { ImageClassificationCollator, ImageClassificationData, ImageLayout, LabelEncoding } from './classification.js';

export interface CollateContext<Metadata extends DatasetMetadata = DatasetMetadata> {
  readonly metadata: Metadata;
  readonly size: number;
  readonly capacity: number;
}
export interface Collated<Data> {
  readonly data: Data;
  /** Only release owned resources, never borrowed sample resources. */
  readonly dispose?: () => void;
}
export type Collator<Sample, Metadata extends DatasetMetadata, Data> = (
  samples: readonly Sample[], context: CollateContext<Metadata>,
) => Collated<Data> | Promise<Collated<Data>>;
export interface Batch<Data> extends Disposable {
  readonly data: Data;
  readonly indices: readonly number[];
  readonly size: number;
  readonly capacity: number;
  dispose(): void;
}
export interface DataLoader<Data> extends AsyncIterable<Batch<Data>, void> {
  /** Read the first batch of a fresh iteration, or undefined if there are no batches. */
  first(): Promise<Batch<Data> | undefined>;
}
export interface TransformContext {
  readonly index: number;
  readonly seed: number;
}
interface LoaderSettings<Sample, Metadata extends DatasetMetadata> {
  readonly dataset: Dataset<Sample, Metadata>;
  readonly batchSize: number;
  /** Read the first N samples before shuffling. Default: the whole dataset. */
  readonly size?: number;
  readonly shuffleSeed?: number;
  readonly lastBatch?: 'keep' | 'drop' | 'pad';
}
export type DataLoaderOptions<Sample, Metadata extends DatasetMetadata = DatasetMetadata, Transformed = Sample, Data = unknown> = LoaderSettings<Sample, Metadata> & (
  | { readonly transform?: undefined; readonly collate: Collator<Sample, Metadata, Data> }
  | { readonly transform: (sample: Sample, context: TransformContext) => Transformed | Promise<Transformed>; readonly collate: Collator<Transformed, Metadata, Data> }
);

// Preserve shape and class count through the generic image classification collator.
export function dataLoader<F extends number, C extends number, L extends ImageLayout, E extends LabelEncoding>(
  options: LoaderSettings<ImageClassificationSample<F>, ImageClassificationMetadata<C>> & { transform?: undefined; collate: ImageClassificationCollator<L, E> },
): DataLoader<ImageClassificationData<F, C, L, E>>;
export function dataLoader<S, C extends number, F extends number, L extends ImageLayout, E extends LabelEncoding>(
  options: LoaderSettings<S, ImageClassificationMetadata<C>> & { transform: (sample: S, context: TransformContext) => ImageClassificationSample<F> | Promise<ImageClassificationSample<F>>; collate: ImageClassificationCollator<L, E> },
): DataLoader<ImageClassificationData<F, C, L, E>>;
export function dataLoader<S, M extends DatasetMetadata, D>(options: LoaderSettings<S, M> & { transform?: undefined; collate: Collator<S, M, D> }): DataLoader<D>;
export function dataLoader<S, M extends DatasetMetadata, T, D>(options: LoaderSettings<S, M> & { transform: (sample: S, context: TransformContext) => T | Promise<T>; collate: Collator<T, M, D> }): DataLoader<D>;
export function dataLoader<S, M extends DatasetMetadata, T = S, D = unknown>(options: DataLoaderOptions<S, M, T, D>): DataLoader<D> {
  const { dataset, batchSize, size = dataset.count, shuffleSeed, lastBatch = 'keep', transform, collate } = options;
  if (!Number.isSafeInteger(dataset.count) || dataset.count < 0 || !Number.isSafeInteger(size) || size < 0 || size > dataset.count || !Number.isSafeInteger(batchSize) || batchSize <= 0) throw new Error('Invalid dataset size or batch size');
  if (shuffleSeed !== undefined && !Number.isSafeInteger(shuffleSeed)) throw new Error('shuffleSeed must be a safe integer');
  if (!['keep', 'drop', 'pad'].includes(lastBatch)) throw new Error('Invalid lastBatch policy');
  return {
    async first() {
      for await (const batch of this) return batch;
      return undefined;
    },
    async *[Symbol.asyncIterator](): AsyncGenerator<Batch<D>, void> {
      const order = Array.from({ length: size }, (_, i) => i);
      let state = (shuffleSeed ?? 0) >>> 0;
      const random = () => state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      if (shuffleSeed !== undefined) {
        for (let i = order.length - 1; i > 0; i--) {
          const j = random() % (i + 1); [order[i], order[j]] = [order[j]!, order[i]!];
        }
      }
      for (let start = 0; start < size; start += batchSize) {
        const indices = Object.freeze(order.slice(start, start + batchSize));
        if (indices.length < batchSize && lastBatch === 'drop') break;
        const capacity = lastBatch === 'pad' ? batchSize : indices.length;
        const context = { metadata: dataset.metadata, size: indices.length, capacity };
        const samples: S[] = [];
        for (const index of indices) {
          samples.push(await dataset.get(index));
        }
        let result: Collated<D>;
        if (transform) {
          const transformed: T[] = [];
          for (let i = 0; i < samples.length; i++) transformed.push(await transform(samples[i]!, { index: indices[i]!, seed: random() }));
          result = await collate(transformed, context);
        } else result = await collate(samples, context);
        let cleanup = result.dispose;
        const dispose = () => { const release = cleanup; cleanup = undefined; release?.() };
        yield { data: result.data, indices, size: indices.length, capacity, dispose, [Symbol.dispose]: dispose };
      }
    },
  };
}
