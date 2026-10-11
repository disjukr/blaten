import { tensor, type TensorF32, type TensorU32 } from 'blaten';
import type { ImageClassificationMetadata, ImageClassificationSample } from './dataset.js';
import type { Collated, CollateContext } from './loader.js';

export type ImageLayout = 'flat' | 'Nhwc' | 'Nchw';
export type LabelEncoding = 'index' | 'oneHot';
export interface ImageClassificationOptions<L extends ImageLayout = 'Nchw', E extends LabelEncoding = 'index'> {
  readonly layout?: L;
  readonly scale?: number;
  readonly labels?: E;
}
export interface ImageClassificationData<F extends number, C extends number, L extends ImageLayout, E extends LabelEncoding> {
  readonly images: TensorF32<L extends 'flat' ? [number, F] : [number, number, number, number]>;
  readonly targets: E extends 'oneHot' ? TensorF32<[number, C]> : TensorU32<[number]>;
}
export interface ImageClassificationCollator<L extends ImageLayout, E extends LabelEncoding> {
  <F extends number, C extends number>(samples: readonly ImageClassificationSample<F>[], context: CollateContext<ImageClassificationMetadata<C>>): Collated<ImageClassificationData<F, C, L, E>>;
}

/** Convert Hwc byte images to owned host tensors. Defaults: Nchw, scale 1/255, class IDs. */
export function imageClassification<L extends ImageLayout = 'Nchw', E extends LabelEncoding = 'index'>(options: ImageClassificationOptions<L, E> = {}): ImageClassificationCollator<L, E> {
  const { layout = 'Nchw', scale = 1 / 255, labels = 'index' } = options;
  if (!['flat', 'Nhwc', 'Nchw'].includes(layout) || !['index', 'oneHot'].includes(labels) || !Number.isFinite(scale)) throw new Error('Invalid image classification options');
  return <F extends number, C extends number>(samples: readonly ImageClassificationSample<F>[], { metadata, size, capacity }: CollateContext<ImageClassificationMetadata<C>>) => {
    if (!samples.length || size !== samples.length || !Number.isSafeInteger(capacity) || capacity < size) throw new Error('Invalid collation size');
    const { height, width, channels, featureCount } = samples[0]!.image;
    const classes = metadata.classNames.length;
    if (!classes || [height, width, channels, featureCount].some(n => !Number.isSafeInteger(n) || n <= 0) || height * width * channels !== featureCount) throw new Error('Invalid classification dimensions');
    const pixels = new Float32Array(capacity * featureCount);
    const targets = labels === 'oneHot' ? new Float32Array(capacity * classes) : new Uint32Array(capacity);
    samples.forEach(({ image, label }, row) => {
      if (image.height !== height || image.width !== width || image.channels !== channels || image.featureCount !== featureCount || image.data.length !== featureCount) throw new Error('Images must have equal dimensions; resize them with transform before collation');
      if (!Number.isSafeInteger(label) || label < 0 || label >= classes) throw new Error('Invalid class label');
      for (let pixel = 0; pixel < featureCount; pixel++) {
        const destination = layout === 'Nchw' ? (pixel % channels) * height * width + Math.floor(pixel / channels) : pixel;
        pixels[row * featureCount + destination] = image.data[pixel]! * scale;
      }
      if (labels === 'oneHot') targets[row * classes + label] = 1;
      else targets[row] = label;
    });
    const shape = layout === 'flat' ? [capacity, featureCount] : layout === 'Nchw' ? [capacity, channels, height, width] : [capacity, height, width, channels];
    const images = tensor(pixels, { shape });
    try {
      const targetTensor = labels === 'oneHot' ? tensor(targets, { shape: [capacity, classes], dtype: 'f32' }) : tensor(targets, { shape: [capacity], dtype: 'u32' });
      const data = { images, targets: targetTensor } as ImageClassificationData<F, C, L, E>;
      return { data, dispose: () => { images.dispose(); targetTensor.dispose() } };
    } catch (error) { images.dispose(); throw error }
  };
}
