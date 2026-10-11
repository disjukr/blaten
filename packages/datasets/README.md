# @blaten/datasets

Node.js loaders for well-known datasets and a generic asynchronous batch loader.
The package ships loader code only. Data is fetched directly from its provider or
read from a directory prepared by the user. All loaders below return the same
indexed `Dataset` interface: `count`, `metadata`, and async `get(index)`.
`dataLoader` also supports custom samples. This package requires Node.js; images
are decoded on the host with Sharp, and videos use external ffmpeg/ffprobe.

## Load a split

```ts
import { mnist, fashionMnist } from '@blaten/datasets';

const train = await mnist({ split: 'train' });
const test = await mnist({ split: 'test' });
const clothing = await fashionMnist({ split: 'train', root: './data/fashion-mnist' });
console.log(train.count, test.count); // 60000, 10000
console.log(clothing.metadata.classNames); // ['T-shirt/top', 'Trouser', ...]
console.log(await train.get(0)); // { image: { data, width, height, channels, featureCount }, label }
```

For MNIST and Fashion-MNIST, `split` is required and is either `'train'` or `'test'`.
Each call only loads files needed for that split. `root` is the dataset's own directory, regardless of whether
the loader or the user put the files there. By default it is
`.cache/blaten-datasets/mnist` or `.cache/blaten-datasets/fashion-mnist`, relative to
the working directory. There is no separate `cacheDir` option or extra dataset
subdirectory under an explicitly supplied `root`.

`download` defaults to `true`: valid files are reused, and missing or checksum-invalid
files are fetched, validated, and atomically saved. Set `download: false` to require
existing local files, with an error naming a missing or corrupt file and no network
requests. Downloaded files retain the provider's gzip filenames. Sources are the
[CVDF MNIST mirror](https://github.com/cvdfoundation/mnist) and the
[official Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist).
The loaders validate compressed-file checksums, IDX headers, split counts,
dimensions, payload lengths, and class IDs. Concurrent downloads of a file are
coalesced; offline reads never join a request that could download.

`metadata` contains `name`, `split`, `homepage`, and `classNames`. A label is a
zero-based index into `classNames`; these names are for display, not model input.
Loading creates host byte arrays, not tensors. `ImageDataset<784, 10>` exposes
`pixels`, `labels`, `featureCount`, and an Hwc `imageShape` of `[28, 28, 1]`.
Metadata and shape arrays are frozen. Raw arrays and per-sample pixel views are
borrowed and mutable; transforms should copy them before editing.

## Make batches

```ts
import { dataLoader, imageClassification, mnist } from '@blaten/datasets';

const dataset = await mnist({ split: 'train' });
const batches = dataLoader({
  dataset,
  batchSize: 64,
  size: 12000,
  shuffleSeed: 101,
  lastBatch: 'drop',
  collate: imageClassification({ layout: 'flat', labels: 'oneHot' }),
});

for await (const batch of batches) {
  using ownedBatch = batch;
  const { images, targets } = batch.data;
  // images: TensorF32<[number, 784]>, scaled to [0, 1]
  // targets: TensorF32<[number, 10]>, one-hot labels
  // Pass these host tensors to your compiled training function.
}
```

`dataLoader` accepts one configuration object and returns a reusable async iterable.
A new iteration repeats the same sample order and transform seeds. Change
`shuffleSeed` between epochs for a new deterministic order. Without it, samples
are visited in dataset order. `size` optionally selects the first N samples before
shuffling; by default all samples are read. Empty datasets/subsets yield no batches.

`await batches.first()` returns the first batch of a fresh iteration, or `undefined`
when no batches are available (including when `lastBatch: 'drop'` drops all samples).
It uses the same shuffling, transforms, and collation as normal iteration, reads
only that batch, and does not advance other iterators. Each call returns a newly
owned batch; use `using` or dispose it explicitly.

```ts
using example = await batches.first();
if (example) console.log(example.data.images.shape);
```

`transform(sample, { index, seed })` optionally processes individual samples before
collation and may be asynchronous. The seed is a reproducible uint32 value; a
transform can use its own seeded random generator for augmentations. `collate`
combines transformed samples and may also be asynchronous. Its output type becomes
`batch.data`, including arbitrary records, arrays, masks, or tensors. No assumption
about image classification or TensorTree is made by the generic loader.

`imageClassification()` accepts:

- `layout`: `'Nchw'` (default), `'Nhwc'`, or `'flat'`.
- `scale`: pixel multiplier, default `1 / 255`.
- `labels`: `'index'` (default, u32 class IDs) or `'oneHot'` (f32).

It expects Hwc uint8 images with equal dimensions and classification metadata with
`classNames`. Resize or crop variable-size images in `transform` first. Samples
remain byte images with numeric labels until collation. Each collation allocates
independently owned host tensors; device transfer remains a backend responsibility.

`lastBatch` is `'keep'` by default (a smaller final batch), `'drop'` (omit it), or
`'pad'` (fixed capacity). `batch.size` counts real samples; `batch.capacity` includes
padding. `batch.indices` only lists real sample indices. The built-in collator pads
images and one-hot targets with zero, and index targets with class ID zero. Padding
is not automatically ignored by loss or metrics: pass `batch.size` as `validCount`
where supported, or use `'drop'` for training when the loss does not mask padding.

## Custom datasets and collation

The indexed interface requires only `count`, `metadata`, and async `get(index)`:

```ts
import { dataLoader, type Dataset } from '@blaten/datasets';

const texts: Dataset<{ text: string; label: number }> = {
  count: 2,
  metadata: { name: 'reviews', split: 'train' },
  async get(index) {
    return [
      { text: 'great', label: 1 },
      { text: 'bad', label: 0 },
    ][index]!;
  },
};

const batches = dataLoader({
  dataset: texts,
  batchSize: 2,
  transform: sample => ({ ...sample, tokens: tokenize(sample.text) }),
  collate: samples => ({
    data: { tokens: samples.map(sample => sample.tokens), labels: samples.map(sample => sample.label) },
  }),
});
```

`tokenize` above is application-provided. The same interface can describe detection
samples (`{ image, boxes, labels }`), segmentation (`{ image, mask }`), captioning
(`{ image, captions }`), video (`{ frames, label }`), flow, and tabular data.
The catalog supplies specialized loaders; application-specific transforms and
collators for these tasks remain application code. Loaders requiring manually
prepared data have a required `root` and omit `download`.

A collator returns `{ data, dispose? }` and receives `{ metadata, size, capacity }`.
A custom collator must implement its own padding when `capacity > size` and return
a synchronous cleanup callback for any resources it owns. The loader does not
traverse `data` or dispose borrowed resources. Built-in batches own their tensors;
`using`, `batch.dispose()`, and `batch[Symbol.dispose]()` release them idempotently.
Unreferenced tensors still receive the core library's GC cleanup. A custom cleanup
callback is only run by explicit batch disposal, so use `using` for other resources.
Moving to the next batch or closing an iterator does not dispose already returned
batches. Loading and collation do not synchronize GPU work.


## Dataset catalog

Download-capable loaders accept `{ split, root?, download? }`: `download` defaults
to `true` and `root` defaults to `.cache/blaten-datasets/<dataset-name>`.
Files come from the provider URLs recorded in each loader; none are hosted or
redistributed by this package. Original archives stay in `root`. Downloads stream
to temporary files and are published atomically after checksum validation when
the provider supplies a checksum. Extracted files are reused; an extraction
completion marker allows interrupted archive extraction to be retried.
Archive-based downloads can include other splits in the same provider archive.
User-prepared extracted directories are accepted without requiring an archive.

| Loader | Splits | Options / original format | Provider |
| --- | --- | --- | --- |
| `mnist` | train, test | gzip IDX, 28×28 grayscale | Yann LeCun / CVDF mirror |
| `fashionMnist` | train, test | gzip IDX, 28×28 grayscale | Zalando Research |
| `kmnist` | train, test | gzip IDX, 28×28 grayscale | ROIS CODH |
| `emnist` | train, test | required `variant`: byclass, bymerge, balanced, letters, digits, mnist; gzip ZIP or extracted IDX + mapping | NIST |
| `qmnist` | train, test, test10k, test50k | gzip IDX; returns digit class from the extended label record | Meta Research |
| `usps` | train, test | bzip2 LIBSVM, 16×16 grayscale | USPS / NTU LIBSVM mirror |
| `svhn` | train, test, extra | MATLAB v5, 32×32 RGB | Stanford |
| `cifar10` | train, test | official binary archive, 32×32 RGB | University of Toronto |
| `cifar100` | train, test | `labels`: fine (default, 100) or coarse (20) | University of Toronto |
| `stl10` | train, test, unlabeled, train+unlabeled | binary archive, 96×96 RGB; optional train `fold`: 0–9 | Stanford |
| `caltech101` | all | ZIP containing category tar archives; excludes background | Caltech |
| `caltech256` | all | category tar; 257 classes including clutter | Caltech |
| `flowers102` | train, val, test | image TGZ + MAT split/labels; class names are numeric IDs 1–102 | Oxford VGG |
| `food101` | train, test | image tar + native metadata JSON | ETH Zurich |
| `oxfordIiitPet` | trainval, test | `task`: classification (default) or segmentation | Oxford / IIIT |
| `gtsrb` | train, test | image ZIPs + semicolon-delimited test labels | German Traffic Sign Benchmark |
| `imagenette` | train, val | `size`: 160px (default), 320px, full | fast.ai |
| `voc` | train, val, trainval, test (2007 only) | required `year`: 2007–2012; `task`: detection (default) or segmentation | PASCAL VOC / Oxford |
| `iris` | all | CSV; 4 features, 3 classes | UCI |
| `wine` | all | CSV; 13 features, 3 classes | UCI |
| `breastCancerWisconsin` | all | diagnostic WDBC CSV; 30 features, 2 classes; excludes patient ID | UCI |
| `imdb` | train, test | sentiment tar; supervised positive/negative reviews | Stanford |
| `agNews` | train, test | benchmark CSV; title + newline + description | AG corpus / Zhang et al. benchmark mirror |

Local loaders require `{ root, split, ... }` and have no `download` option. Prepare
provider files yourself, preserving the directory layout described below. There
is no downloader for licensed, authenticated, multi-part, or externally collected
sources in this group.

| Loader | Splits | Expected paths below root / options |
| --- | --- | --- |
| `imageNet` | train, val | `train/<wnid>/`, `val/<wnid>/`, or user-supplied `ILSVRC2012_img_train.tar` / `ILSVRC2012_img_val.tar`; flat validation images also require `ILSVRC2012_devkit_t12/data/meta.mat` and ground truth |
| `stanfordCars` | train, test | `devkit/cars_meta.mat`, `devkit/cars_train_annos.mat`, `cars_train/`; test uses `cars_test_annos_withlabels.mat`, `cars_test/` |
| `celebA` | train, val, test, all | `img_align_celeba/`, `list_attr_celeba.txt`, `list_eval_partition.txt`; returns named boolean attributes |
| `lfw` | train, test, all | `lfw_funneled/` or `lfw/`; peopleDevTrain/Test.txt for classification; `task: 'pairs'` uses pairsDevTrain/Test.txt or pairs.txt |
| `coco` | train, val | `train2017/` or `images/train2017/`, `annotations/instances_train2017.json`; `year`: 2014 or 2017 (default); `task: 'captions'` uses captions JSON; optional `annotationsFile` |
| `cityscapes` | train, val, test, train_extra | `leftImg8bit/<split>/<city>/`, `gtFine/` or `gtCoarse/`; `quality`: fine (default) or coarse; train_extra requires coarse; test has images only |
| `kitti` | train, test | `training/image_2/`, `training/label_2/` or `testing/image_2/`; test has images only |
| `widerFace` | train, val, test | `WIDER_<split>/images/`, `wider_face_split/wider_face_<split>_bbx_gt.txt`; test has images only |
| `flickr8k` | all, train, val, test | `Flicker8k_Dataset/` or `images/`, Flickr8k.token.txt; default split files Flickr_8k.train/dev/testImages.txt |
| `flickr30k` | all, train, val, test | `flickr30k-images/` or `images/`, results_20130124.token; train/val/test.txt splits must be supplied |
| `sbu` | all | `dataset/sbu_images/` with aligned SBU_captioned_photo_dataset_urls.txt and captions.txt; missing original photos are skipped |
| `kinetics` | train, val, test | `annotations/<split>.csv`, `<split>/<label>/<youtube_id>_<start:06>_<end:06>.mp4`; `version`: 400 (default), 600, 700; optional `annotationsFile` |
| `ucf101` | train, test | `UCF-101/<action>/`, `ucfTrainTestlist/classInd.txt`, trainlist01.txt / testlist01.txt; `fold`: 1–3; optional `annotationsDir` |
| `hmdb51` | train, test | `<action>/` or `hmdb51_org/<action>/`, testTrainMulti_7030_splits/<action>_test_split1.txt; `fold`: 1–3; optional `annotationsDir`, `videosDir` |
| `sintel` | train, test | `training/clean/<scene>/`, `training/final/<scene>/`, `training/flow/<scene>/*.flo`, optional invalid masks; `pass`: clean (default), final, both; test flow absent |
| `flyingChairs` | train, val | `data/*_img1.ppm`, *_img2.ppm, *_flow.flo and FlyingChairs_train_val.txt |
| `flyingThings3d` | train, test | `frames_cleanpass/` or `frames_finalpass/`, `optical_flow/`; `pass`: clean (default), final, both; `camera`: left (default), right, both; `direction`: future, past, both (default) |
| `kittiFlow` | train, test | `training/image_2/*_10.png`, *_11.png, `training/flow_occ/*_10.png`; test uses testing/image_2 and has no flow |
| `imageFolder` | all, train, val, test | all: `<class>/<image>`; other splits: `<split>/<class>/<image>`; sorted directory names become classes |

Flickr loaders also accept `annotationsFile`, `imagesDir`, `splitFile`. Captions
may be native token files, `image,caption` CSV, or the original Flickr8k HTML.
`fakeData({ split: 'all', size, imageShape?, classes?, seed? })` generates
reproducible byte images without disk or network access. Default Hwc image shape
is `[28, 28, 1]`, classes is 10, and seed is 0.

## Task samples

Images are uint8 Hwc arrays. File-based color images are decoded to RGB; digit
IDX datasets retain one channel. Lazy file loaders decode requested samples,
while IDX, SVHN and the small tabular datasets keep source arrays in host memory.
Tensor allocation remains in collation. Class labels are zero-based indices into
`metadata.classNames`; COCO category IDs are sorted and mapped to dense labels.
Unlabeled STL10 samples omit `label`. Class names follow source metadata: some
sources only supply numeric IDs or ImageNet wnids.

- Detection: `{ image, boxes, labels }`, optionally `difficult` or `isCrowd`. Boxes
  are zero-based `[xmin, ymin, xmax, ymax]` with exclusive upper bounds. VOC
  one-based coordinates and COCO/WIDER xywh coordinates are converted.
- Segmentation: `{ image, mask: { data: Uint32Array, width, height, ignoreIndex? } }`.
  VOC uses background 0, classes 1–20, ignore 255; pets use pet 0, background 1,
  border 2; Cityscapes preserves original label IDs (0–33, ignore 255).
- Captions: `{ image, captions: string[] }`, one sample per image, all captions.
- Face pairs: `{ first, second, same }`; CelebA returns `{ image, attributes }`.
- Flow: `{ first, second, flow?, validMask? }`; flow has Float32 Hwc2 data in pixel
  units. FLO/PFM orientation and KITTI 16-bit PNG encoding are decoded explicitly.
- Text: `{ text, label }`; text is un-tokenized, including original IMDb HTML.
- Tabular: `{ features, label }`; names/order are in `metadata.featureNames`.
- Video: `{ frames, label?, path, fps }`; each frame is an RGB image. `get(index)`
  decodes the whole video. Prefer `getClip(index, startFrame, frameCount)` or a
  `videoClips` view for bounded clips. Install ffmpeg and ffprobe on PATH, or
  supply `ffmpegPath` and `ffprobePath`. These programs are not bundled.

```ts
import { coco, dataLoader, ucf101, videoClips } from '@blaten/datasets';

const detections = await coco({ root: './data/coco', split: 'train' });
const batches = dataLoader({
  dataset: detections,
  batchSize: 4,
  collate: samples => ({ data: samples }), // variable image sizes and box counts
});

const videos = await ucf101({ root: './data/ucf101', split: 'train', fold: 1 });
const clips = await videoClips({ dataset: videos, framesPerClip: 16, stride: 8 });
console.log(clips.count, await clips.get(0));
```

`videoClips` probes frame counts without decoding every video up front, drops
incomplete final clips, and decodes only the selected clip when read. It does not
resample frame rates. Built-in resizing, augmentation, tokenization, variable-size
padding, and task-specific tensor collation are deliberately left to applications.

Tests use provider-shaped small fixtures without internet access for classification,
detection, masks, captions, text, tables, MATLAB, flow, and archive extraction.
Native video decode tests run when ffmpeg/ffprobe are available. Large original
corpora are not part of the test suite or package.
