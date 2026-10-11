import { mnist, fashionMnist, ImageDataset, dataLoader, imageClassification, type Dataset } from '../src/index.js';
import type { TensorF32, TensorU32 } from 'blaten';

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false;
type Expect<T extends true> = T;
type Mnist = Expect<Equal<Awaited<ReturnType<typeof mnist>>, ImageDataset<784, 10>>>;
type FashionMnist = Expect<Equal<Awaited<ReturnType<typeof fashionMnist>>, ImageDataset<784, 10>>>;
async function imageBatches(data: ImageDataset<784, 10>) {
  for await (const batch of dataLoader({ dataset: data, batchSize: 64, collate: imageClassification({ layout: 'flat', labels: 'oneHot' }) })) {
    const { images, targets } = batch.data;
    type Images = Expect<Equal<typeof images, TensorF32<[number, 784]>>>;
    type Targets = Expect<Equal<typeof targets, TensorF32<[number, 10]>>>;
  }
  const loader = dataLoader({ dataset: data, batchSize: 64, collate: imageClassification() });
  const batch = (await loader[Symbol.asyncIterator]().next()).value!;
  const first = await loader.first();
  type First = Expect<Equal<typeof first, typeof batch | undefined>>;
  type Images = Expect<Equal<typeof batch.data.images, TensorF32<[number, number, number, number]>>>;
  type Targets = Expect<Equal<typeof batch.data.targets, TensorU32<[number]>>>;
}
const custom = new ImageDataset({ pixels: new Uint8Array([0, 255]), labels: new Uint8Array([1]), featureCount: 2, imageShape: [1, 2, 1], metadata: { name: 'custom', split: 'train', classNames: ['zero', 'one'] as const } });
async function customBatches() {
  for await (const batch of dataLoader({ dataset: custom, batchSize: 1, collate: imageClassification({ layout: 'flat', labels: 'oneHot' }) })) {
    type Images = Expect<Equal<typeof batch.data.images, TensorF32<[number, 2]>>>;
    type Targets = Expect<Equal<typeof batch.data.targets, TensorF32<[number, 2]>>>;
  }
}
const texts: Dataset<{ text: string }> = { count: 1, metadata: { name: 'texts', split: 'train' }, async get() { return { text: 'hello' } } };
async function textBatches() {
  for await (const batch of dataLoader({ dataset: texts, batchSize: 1, transform: sample => sample.text.length, collate: lengths => ({ data: { lengths } }) })) {
    type Lengths = Expect<Equal<typeof batch.data.lengths, readonly number[]>>;
  }
  for await (const batch of dataLoader({ dataset: texts, batchSize: 1, collate: samples => ({ data: samples.map(sample => sample.text) }) })) {
    type Texts = Expect<Equal<typeof batch.data, string[]>>;
  }
}
// @ts-expect-error Split selection is required.
mnist();
// @ts-expect-error Only supported splits can be loaded.
mnist({ split: 'validation' });
// @ts-expect-error root is the single dataset path option.
mnist({ split: 'train', cacheDir: 'data' });
// @ts-expect-error A collator cannot consume a different sample type without a transform.
dataLoader({ dataset: texts, batchSize: 1, collate: (values: readonly number[]) => ({ data: values }) });

import { cifar10, cifar100, svhn, stl10, imageNet, voc, coco, oxfordIiitPet, cityscapes, kitti, widerFace, lfw, videoClips, ucf101, type ClassificationDataset, type DetectionSample, type CaptionSample, type PairSample, type SegmentationSample } from '../src/index.js';
type Cifar10 = Expect<Equal<Awaited<ReturnType<typeof cifar10>>, ClassificationDataset<3072, 10>>>;
type Svhn = Expect<Equal<Awaited<ReturnType<typeof svhn>>, ClassificationDataset<3072, 10>>>;
async function catalogTypes() {
  const fine = await cifar100({ split: 'test' });
  const coarse = await cifar100({ split: 'test', labels: 'coarse' });
  type Fine = Expect<Equal<typeof fine, ClassificationDataset<3072, 100>>>;
  type Coarse = Expect<Equal<typeof coarse, ClassificationDataset<3072, 20>>>;
  const cifar = await cifar10({ split: 'train' });
  const iterator = dataLoader({ dataset: cifar, batchSize: 2, collate: imageClassification({ layout: 'flat', labels: 'oneHot' }) })[Symbol.asyncIterator]();
  const batch = (await iterator.next()).value!;
  const first = await dataLoader({ dataset: cifar, batchSize: 2, collate: imageClassification({ layout: 'flat', labels: 'oneHot' }) }).first();
  type First = Expect<Equal<typeof first, typeof batch | undefined>>;
  type Images = Expect<Equal<typeof batch.data.images, TensorF32<[number, 3072]>>>;
  type Targets = Expect<Equal<typeof batch.data.targets, TensorF32<[number, 10]>>>;
  const detection = await (await voc({ split: 'train', year: 2012 })).get(0);
  const segmentation = await (await voc({ split: 'train', year: 2012, task: 'segmentation' })).get(0);
  const captions = await (await coco({ root: 'data', split: 'train', task: 'captions' })).get(0);
  const pet = await (await oxfordIiitPet({ split: 'test', task: 'segmentation' })).get(0);
  const pair = await (await lfw({ root: 'data', split: 'train', task: 'pairs' })).get(0);
  type Detection = Expect<Equal<typeof detection, DetectionSample>>;
  type Segmentation = Expect<Equal<typeof segmentation, SegmentationSample>>;
  type Captions = Expect<Equal<typeof captions, CaptionSample>>;
  type Pet = Expect<Equal<typeof pet, SegmentationSample>>;
  type Pair = Expect<Equal<typeof pair, PairSample>>;
  const unlabeled = await (await stl10({ split: 'unlabeled' })).get(0);
  // @ts-expect-error Unlabeled samples do not guarantee a classification target.
  const label: number = unlabeled.label;
  const test = await (await kitti({ root: 'data', split: 'test' })).get(0);
  // @ts-expect-error Test images do not expose annotations.
  test.boxes;
  const city = await (await cityscapes({ root: 'data', split: 'test' })).get(0);
  // @ts-expect-error Test images do not expose masks.
  city.mask;
  const face = await (await widerFace({ root: 'data', split: 'test' })).get(0);
  // @ts-expect-error Test images do not expose annotations.
  face.labels;
  await videoClips({ dataset: await ucf101({ root: 'data', split: 'train' }), framesPerClip: 16 });
}
// @ts-expect-error ImageNet requires a user-prepared root.
imageNet({ split: 'train' });
// @ts-expect-error COCO does not download provider archives.
coco({ root: 'data', split: 'train', download: true });
