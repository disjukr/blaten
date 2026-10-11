import type { Dataset, ImageClassificationMetadata, ImageClassificationSample } from './dataset.js';

/** In-memory uint8 image classification data. Arrays are borrowed, not copied. */
export class ImageDataset<Features extends number = number, Classes extends number = number>
implements Dataset<ImageClassificationSample<Features>, ImageClassificationMetadata<Classes>> {
  readonly pixels: Uint8Array;
  readonly labels: Uint8Array;
  readonly count: number;
  readonly featureCount: Features;
  readonly imageShape: readonly [number, number, number];
  readonly metadata: ImageClassificationMetadata<Classes>;
  constructor(options: {
    pixels: Uint8Array; labels: Uint8Array; featureCount: Features;
    imageShape: readonly [number, number, number];
    metadata: ImageClassificationMetadata<Classes>;
  }) {
    const { pixels, labels, featureCount, imageShape, metadata } = options;
    if (!Number.isSafeInteger(featureCount) || featureCount <= 0 || imageShape.length !== 3 || imageShape.some(n => !Number.isSafeInteger(n) || n <= 0) || imageShape.reduce((a, b) => a * b, 1) !== featureCount) throw new Error('Invalid image shape');
    if (pixels.length !== labels.length * featureCount || !metadata.classNames.length || labels.some(label => label >= metadata.classNames.length)) throw new Error('Invalid image classification data');
    this.pixels = pixels; this.labels = labels; this.count = labels.length; this.featureCount = featureCount;
    this.imageShape = Object.freeze([...imageShape]) as typeof imageShape;
    this.metadata = Object.freeze({ ...metadata, classNames: Object.freeze([...metadata.classNames]) }) as typeof metadata;
  }
  async get(index: number): Promise<ImageClassificationSample<Features>> {
    if (!Number.isSafeInteger(index) || index < 0 || index >= this.count) throw new RangeError(`Invalid sample index: ${index}`);
    const [height, width, channels] = this.imageShape;
    return {
      image: { data: this.pixels.subarray(index * this.featureCount, (index + 1) * this.featureCount), height, width, channels, featureCount: this.featureCount },
      label: this.labels[index]!,
    };
  }
}
