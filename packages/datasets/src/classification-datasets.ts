import { open, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { basename } from 'node:path';
import { loadIdx } from './idx.js';
import type { ImageDataset } from './image.js';
import { parse as csv } from 'csv-parse/sync';
import type { Dataset, ImageClassificationMetadata, ImageClassificationSample, Image } from './dataset.js';
import type { SegmentationSample } from './media.js';
import { readImage, readMask } from './media.js';
import { indexed, rootPath, inside, exists, nestedRoot, prepareArchive, extractArchive, downloadFile, lines, text, listFiles, splitOf, type DatasetOptions, type DownloadOptions, type LocalOptions } from './files.js';
import { parseMat, matNumbers, matStrings, type MatValue } from './mat.js';

const clothing = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] as const;
/** Fashion-MNIST uses the same image shape and split sizes as MNIST. */
export function fashionMnist(options: DatasetOptions<'train' | 'test'>): Promise<ImageDataset<784, 10>> {
  return loadIdx({ name: 'fashion-mnist', origin: 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/', homepage: 'https://github.com/zalandoresearch/fashion-mnist', classes: clothing,
    checksums: ['8d4fb7e6c68d591d4c3dfef9ec88bf0d', '25c81989df183df01b3e8a0aad5dffbe', 'bef4ecab320f06d8554ea6380940ec79', 'bb300cfdad3c16e7a12a480ee83cd310'],
  }, options);
}

export const imageExtension = /\.(jpg|jpeg|png|ppm|pgm|bmp|webp|tiff?)$/i;
export type ClassificationDataset<F extends number = number, C extends number = number> = Dataset<ImageClassificationSample<F>, ImageClassificationMetadata<C>>;
export function namesWithCount<C extends number>(names: readonly string[], count: C): readonly string[] & { length: C } {
  if (names.length !== count) throw new Error(`Expected ${count} classes, received ${names.length}`);
  return Object.freeze([...names]) as readonly string[] & { length: C };
}
export function imageRecords<C extends number>(name: string, split: string, classNames: readonly string[] & { length: C }, rows: readonly { path: string; label: number }[], homepage?: string): ClassificationDataset<number, C> {
  if (rows.some(row => !Number.isSafeInteger(row.label) || row.label < 0 || row.label >= classNames.length)) throw new Error(`Invalid ${name} class label`);
  return indexed({ name, split, classNames: Object.freeze([...classNames]) as typeof classNames, homepage }, rows.length, async n => ({ image: await readImage(rows[n]!.path), label: rows[n]!.label }));
}
async function folderRows(root: string, classes: readonly string[]): Promise<{ path: string; label: number }[]> {
  const rows: { path: string; label: number }[] = [];
  for (let label = 0; label < classes.length; label++) for (const path of await listFiles(inside(root, classes[label]!), imageExtension)) rows.push({ path, label });
  return rows;
}
async function classFolders(root: string): Promise<string[]> { return (await readdir(root, { withFileTypes: true })).filter(entry => entry.isDirectory() && !entry.name.startsWith('.')).map(entry => entry.name).sort() }
export async function imageFolder(options: LocalOptions<'all' | 'train' | 'val' | 'test'>): Promise<ClassificationDataset> {
  splitOf(options.split, ['all', 'train', 'val', 'test']);
  const root = options.split === 'all' ? rootPath('image-folder', options.root) : inside(rootPath('image-folder', options.root), options.split);
  const classes = await classFolders(root); if (!classes.length) throw new Error(`No image classes under ${root}`);
  return imageRecords('image-folder', options.split, classes, await folderRows(root, classes));
}
async function recordBytes(path: string, offset: number, size: number): Promise<Buffer> {
  const file = await open(path, 'r'); const data = Buffer.alloc(size);
  try { const { bytesRead } = await file.read(data, 0, size, offset); if (bytesRead !== size) throw new Error(`Truncated binary sample: ${path}`); return data } finally { await file.close() }
}
async function cifar<C extends number>(options: DownloadOptions<'train' | 'test'>, classes: C, coarse = false): Promise<ClassificationDataset<3072, C>> {
  splitOf(options.split, ['train', 'test']);
  const hundred = classes !== 10, name = hundred ? 'cifar100' : 'cifar10', root = rootPath(name, options.root), subdir = hundred ? 'cifar-100-binary' : 'cifar-10-batches-bin';
  const files = hundred ? [`${options.split}.bin`] : options.split === 'train' ? [1, 2, 3, 4, 5].map(n => `data_batch_${n}.bin`) : ['test_batch.bin'];
  if (!await exists(inside(root, files[0]!))) await prepareArchive(root, { url: `https://cave.cs.toronto.edu/kriz/cifar-${hundred ? 100 : 10}-binary.tar.gz`, md5: hundred ? '03b5dce01913d631647c71ecec9e9cb8' : 'c32a1d4ab5d03f1284b67883e8d87530' }, `${subdir}/${files[0]}`, options.download);
  const dir = await nestedRoot(root, [subdir], files[0]!), bytes = hundred ? 3074 : 3073, entries: { path: string; first: number; count: number }[] = [];
  let count = 0;
  for (const file of files) { const path = inside(dir, file), size = (await stat(path)).size; if (size % bytes) throw new Error(`Invalid CIFAR binary length: ${path}`); entries.push({ path, first: count, count: size / bytes }); count += size / bytes }
  const classNames = namesWithCount(lines(await text(inside(dir, hundred ? `${coarse ? 'coarse' : 'fine'}_label_names.txt` : 'batches.meta.txt'))), classes);
  return indexed({ name, split: options.split, homepage: 'https://cave.cs.toronto.edu/kriz/cifar.html', classNames }, count, async n => {
    const entry = entries.find(entry => n < entry.first + entry.count)!;
    const record = await recordBytes(entry.path, (n - entry.first) * bytes, bytes), label = record[hundred && !coarse ? 1 : 0]!;
    if (label >= classes) throw new Error('Invalid CIFAR label');
    const data = new Uint8Array(3072), start = hundred ? 2 : 1;
    for (let pixel = 0; pixel < 1024; pixel++) for (let c = 0; c < 3; c++) data[pixel * 3 + c] = record[start + c * 1024 + pixel]!;
    return { image: { data, width: 32, height: 32, channels: 3, featureCount: 3072 as const }, label };
  });
}
export function cifar10(options: DownloadOptions<'train' | 'test'>): Promise<ClassificationDataset<3072, 10>> { return cifar(options, 10) }
export function cifar100(options: DownloadOptions<'train' | 'test'> & { labels: 'coarse' }): Promise<ClassificationDataset<3072, 20>>;
export function cifar100(options: DownloadOptions<'train' | 'test'> & { labels?: 'fine' }): Promise<ClassificationDataset<3072, 100>>;
export function cifar100(options: DownloadOptions<'train' | 'test'> & { labels?: 'fine' | 'coarse' }): Promise<ClassificationDataset<3072>> { return options.labels === 'coarse' ? cifar(options, 20, true) : cifar(options, 100) }
export interface Stl10Options extends DownloadOptions<'train' | 'test' | 'unlabeled' | 'train+unlabeled'> { fold?: number }
export function stl10(options: Stl10Options & { split: 'train' | 'test' }): Promise<ClassificationDataset<27648, 10>>;
export function stl10(options: Stl10Options): Promise<Dataset<{ image: Image<27648>; label?: number }, ImageClassificationMetadata<10>>>;
export async function stl10(options: Stl10Options): Promise<Dataset<{ image: Image<27648>; label?: number }, ImageClassificationMetadata<10>>> {
  splitOf(options.split, ['train', 'test', 'unlabeled', 'train+unlabeled']);
  if (options.fold !== undefined && (options.split !== 'train' || !Number.isInteger(options.fold) || options.fold < 0 || options.fold > 9)) throw new Error('STL10 fold must be 0–9 and requires the train split');
  const root = rootPath('stl10', options.root), splits = options.split === 'train+unlabeled' ? ['train', 'unlabeled'] : [options.split];
  if (!await exists(inside(root, `${splits[0]}_X.bin`))) await prepareArchive(root, { url: 'https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz', md5: '91f7769df0f17e558f3565bffb0c7dfb' }, `stl10_binary/${splits[0]}_X.bin`, options.download);
  const dir = await nestedRoot(root, ['stl10_binary'], `${splits[0]}_X.bin`), records: { path: string; count: number; labels?: Buffer; first: number }[] = [];
  let count = 0;
  for (const split of splits) {
    const path = inside(dir, `${split}_X.bin`), size = (await stat(path)).size;
    if (size % 27648) throw new Error('Invalid STL10 image file');
    const labels = split === 'unlabeled' ? undefined : await readFile(inside(dir, `${split}_y.bin`));
    if (labels && (labels.length !== size / 27648 || labels.some(label => label < 1 || label > 10))) throw new Error('Invalid STL10 labels');
    records.push({ path, count: size / 27648, labels, first: count }); count += size / 27648;
  }
  let order = Array.from({ length: count }, (_, n) => n);
  if (options.fold !== undefined) { order = lines(await text(inside(dir, 'fold_indices.txt')))[options.fold]!.split(/\s+/).map(Number); if (order.some(n => !Number.isInteger(n) || n < 0 || n >= count)) throw new Error('Invalid STL10 fold') }
  const classNames = namesWithCount(lines(await text(inside(dir, 'class_names.txt'))), 10);
  return indexed({ name: 'stl10', split: options.split, homepage: 'https://cs.stanford.edu/~acoates/stl10/', classNames }, order.length, async n => {
    const index = order[n]!, record = records.find(record => index < record.first + record.count)!, position = index - record.first;
    const raw = await recordBytes(record.path, position * 27648, 27648), data = new Uint8Array(27648);
    for (let y = 0; y < 96; y++) for (let x = 0; x < 96; x++) for (let c = 0; c < 3; c++) data[(y * 96 + x) * 3 + c] = raw[c * 9216 + x * 96 + y]!;
    return { image: { data, width: 96, height: 96, channels: 3, featureCount: 27648 as const }, ...(record.labels ? { label: record.labels[position]! - 1 } : {}) };
  });
}
export async function caltech101(options: DownloadOptions<'all'>): Promise<ClassificationDataset<number, 101>> {
  splitOf(options.split, ['all']); const root = rootPath('caltech101', options.root);
  const archive = inside(root, 'caltech-101.zip'), marker = `${archive}.extracted`;
  if (!await exists(inside(root, '101_ObjectCategories')) || await exists(archive) && !await exists(marker)) {
    const zip = await downloadFile(root, { url: 'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip', md5: '3138e1922a9193bfa496528edbbc45d0' }, options.download);
    await extractArchive(zip, root);
    for (const path of await listFiles(inside(root, 'caltech-101'), /\.tar\.gz$/)) await extractArchive(path, root);
    await writeFile(marker, 'complete\n');
  }
  const dir = inside(root, '101_ObjectCategories'), classes = (await classFolders(dir)).filter(name => name !== 'BACKGROUND_Google');
  return imageRecords('caltech101', options.split, namesWithCount(classes, 101), await folderRows(dir, classes), 'https://www.vision.caltech.edu/datasets/');
}
export async function caltech256(options: DownloadOptions<'all'>): Promise<ClassificationDataset<number, 257>> {
  splitOf(options.split, ['all']); const root = rootPath('caltech256', options.root);
  await prepareArchive(root, { url: 'https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar', md5: '67b4f42ca05d46448c6bb8ecd2220f6d' }, '256_ObjectCategories', options.download);
  const dir = inside(root, '256_ObjectCategories'), classes = await classFolders(dir);
  return imageRecords('caltech256', options.split, namesWithCount(classes.map(name => name.replace(/^\d+\./, '')), 257), await folderRows(dir, classes), 'https://www.vision.caltech.edu/datasets/');
}
export async function flowers102(options: DownloadOptions<'train' | 'val' | 'test'>): Promise<ClassificationDataset<number, 102>> {
  splitOf(options.split, ['train', 'val', 'test']); const root = rootPath('flowers102', options.root), base = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/';
  await prepareArchive(root, { url: base + '102flowers.tgz', md5: '52808999861908f626f3c1f4e79d11fa' }, 'jpg', options.download);
  const [labelsPath, setsPath] = await Promise.all([downloadFile(root, { url: base + 'imagelabels.mat', md5: 'e0620be6f572b9609742df49c70aed4d' }, options.download), downloadFile(root, { url: base + 'setid.mat', md5: 'a5357ecc9cb78c4bef273ce3793fc85c' }, options.download)]);
  const labels = matNumbers(parseMat(await readFile(labelsPath)).labels), sets = parseMat(await readFile(setsPath)), ids = matNumbers(sets[{ train: 'trnid', val: 'valid', test: 'tstid' }[options.split]]);
  const rows = ids.map(id => { if (!Number.isInteger(id) || id < 1 || id > labels.length) throw new Error('Invalid Flowers102 split ID'); return { path: inside(root, `jpg/image_${String(id).padStart(5, '0')}.jpg`), label: labels[id - 1]! - 1 } });
  return imageRecords('flowers102', options.split, namesWithCount(Array.from({ length: 102 }, (_, i) => String(i + 1)), 102), rows, base);
}
export async function food101(options: DownloadOptions<'train' | 'test'>): Promise<ClassificationDataset<number, 101>> {
  splitOf(options.split, ['train', 'test']); const root = rootPath('food101', options.root);
  if (!await exists(inside(root, `meta/${options.split}.json`))) await prepareArchive(root, { url: 'https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz', md5: '85eeb15f3717b99a5da872d97d918f87' }, `food-101/meta/${options.split}.json`, options.download);
  const dir = await nestedRoot(root, ['food-101'], `meta/${options.split}.json`), classes = lines(await text(inside(dir, 'meta/classes.txt'))), groups = JSON.parse(await text(inside(dir, `meta/${options.split}.json`))) as Record<string, string[]>;
  const rows = classes.flatMap((name, label) => (groups[name] ?? []).map(file => ({ path: inside(dir, `images/${file}.jpg`), label })));
  return imageRecords('food101', options.split, namesWithCount(classes, 101), rows, 'https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/');
}
const petNames = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier'] as const;
export type PetOptions = DownloadOptions<'trainval' | 'test'>;
export function oxfordIiitPet(options: PetOptions & { task: 'segmentation' }): Promise<Dataset<SegmentationSample>>;
export function oxfordIiitPet(options: PetOptions & { task?: 'classification' }): Promise<ClassificationDataset<number, 37>>;
export async function oxfordIiitPet(options: PetOptions & { task?: 'classification' | 'segmentation' }): Promise<Dataset<SegmentationSample | ImageClassificationSample>> {
  splitOf(options.split, ['trainval', 'test']); const root = rootPath('oxford-iiit-pet', options.root), base = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/';
  await prepareArchive(root, { url: base + 'images.tar.gz', md5: '5c4f3ee8e5d25df40f4fd59a7f44e54c' }, 'images', options.download);
  await prepareArchive(root, { url: base + 'annotations.tar.gz', md5: '95a8c909bbe2e81eed6a22bccdf3f68f' }, `annotations/${options.split}.txt`, options.download);
  const rows = lines(await text(inside(root, `annotations/${options.split}.txt`))).filter(line => !line.startsWith('#')).map(line => line.split(/\s+/));
  if (rows.some(row => !Number.isInteger(Number(row[1])) || Number(row[1]) < 1 || Number(row[1]) > 37)) throw new Error('Invalid pet label');
  return indexed({ name: 'oxford-iiit-pet', split: options.split, homepage: 'https://www.robots.ox.ac.uk/~vgg/data/pets/', classNames: options.task === 'segmentation' ? ['pet', 'background', 'border'] : petNames }, rows.length, async n => {
    const row = rows[n]!, image = await readImage(inside(root, `images/${row[0]}.jpg`));
    return options.task === 'segmentation' ? { image, mask: await readMask(inside(root, `annotations/trimaps/${row[0]}.png`), channels => channels[0]! - 1) } : { image, label: Number(row[1]) - 1 };
  });
}
export async function stanfordCars(options: LocalOptions<'train' | 'test'>): Promise<ClassificationDataset<number, 196>> {
  splitOf(options.split, ['train', 'test']); const root = rootPath('stanford-cars', options.root), dir = await nestedRoot(root, ['stanford_cars'], 'devkit/cars_meta.mat');
  const classes = namesWithCount(matStrings(parseMat(await readFile(inside(dir, 'devkit/cars_meta.mat'))).class_names), 196);
  const annotations = parseMat(await readFile(inside(dir, options.split === 'train' ? 'devkit/cars_train_annos.mat' : 'cars_test_annos_withlabels.mat'))).annotations;
  if (!Array.isArray(annotations)) throw new Error('Invalid Stanford Cars annotations');
  const rows = annotations.map(annotation => { const record = annotation as Record<string, MatValue>; return { path: inside(dir, `cars_${options.split}/${matStrings(record.fname)[0]}`), label: matNumbers(record.class)[0]! - 1 } });
  return imageRecords('stanford-cars', options.split, classes, rows, 'https://ai.stanford.edu/~jkrause/cars/car_dataset.html');
}
const trafficNames = ['20 km/h', '30 km/h', '50 km/h', '60 km/h', '70 km/h', '80 km/h', 'End 80 km/h', '100 km/h', '120 km/h', 'No passing', 'No passing for trucks', 'Right of way at next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles', 'No trucks', 'No entry', 'General caution', 'Dangerous curve left', 'Dangerous curve right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware ice/snow', 'Wild animals', 'End all restrictions', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Straight or right', 'Straight or left', 'Keep right', 'Keep left', 'Roundabout', 'End no passing', 'End no passing for trucks'] as const;
export async function gtsrb(options: DownloadOptions<'train' | 'test'>): Promise<ClassificationDataset<number, 43>> {
  splitOf(options.split, ['train', 'test']); const root = rootPath('gtsrb', options.root), base = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/';
  const required = options.split === 'train' ? 'GTSRB/Training' : 'GTSRB/Final_Test/Images';
  if (options.split === 'train' && !await exists(inside(root, 'GTSRB/Final_Training/Images'))) await prepareArchive(root, { url: base + 'GTSRB-Training_fixed.zip', md5: '513f3c79a4c5141765e10e952eaa2478' }, required, options.download);
  if (options.split === 'test') { await prepareArchive(root, { url: base + 'GTSRB_Final_Test_Images.zip', md5: 'c7e4e6327067d32654124b0fe9e82185' }, required, options.download); await prepareArchive(root, { url: base + 'GTSRB_Final_Test_GT.zip', md5: 'fe31e9c9270bbcd7b84b7f21a9d9d9e5' }, 'GT-final_test.csv', options.download) }
  const rows: { path: string; label: number }[] = [];
  if (options.split === 'train') {
    const dir = await exists(inside(root, required)) ? inside(root, required) : inside(root, 'GTSRB/Final_Training/Images');
    for (const name of await classFolders(dir)) for (const path of await listFiles(inside(dir, name), imageExtension)) rows.push({ path, label: Number(name) });
  } else {
    const records = csv(await text(inside(root, 'GT-final_test.csv')), { columns: true, delimiter: ';', skip_empty_lines: true }) as { Filename: string; ClassId: string }[];
    for (const row of records) rows.push({ path: inside(root, `GTSRB/Final_Test/Images/${row.Filename}`), label: Number(row.ClassId) });
  }
  return imageRecords('gtsrb', options.split, trafficNames, rows, 'https://benchmark.ini.rub.de/');
}
export interface ImagenetteOptions extends DownloadOptions<'train' | 'val'> { size?: 'full' | '320px' | '160px' }
export async function imagenette(options: ImagenetteOptions): Promise<ClassificationDataset<number, 10>> {
  splitOf(options.split, ['train', 'val']); const size = options.size ?? '160px'; splitOf(size, ['full', '320px', '160px']);
  const folder = size === 'full' ? 'imagenette2' : `imagenette2-${size.replace('px', '')}`, root = rootPath(`imagenette-${size}`, options.root);
  if (!await exists(inside(root, options.split))) await prepareArchive(root, { url: `https://s3.amazonaws.com/fast-ai-imageclas/${folder}.tgz`, md5: { full: 'fe2fc210e6bb7c5664d602c3cd71e612', '320px': '3df6f0d01a2c9592104656642f5e78a3', '160px': 'e793b78cc4c9e9a4ccc0c1155377a412' }[size] }, `${folder}/${options.split}`, options.download);
  const dir = await nestedRoot(root, [folder], options.split), wnids = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257'];
  const names = namesWithCount(['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute'], 10);
  return imageRecords('imagenette', options.split, names, await folderRows(inside(dir, options.split), wnids), 'https://github.com/fastai/imagenette');
}
export function fakeData(options: { split: 'all'; size: number; imageShape?: readonly [number, number, number]; classes?: number; seed?: number }): ClassificationDataset {
  splitOf(options.split, ['all']); const shape = options.imageShape ?? [28, 28, 1], classes = options.classes ?? 10, seed = options.seed ?? 0;
  if (!Number.isSafeInteger(options.size) || options.size < 0 || !Number.isSafeInteger(classes) || classes <= 0 || shape.some(n => !Number.isSafeInteger(n) || n <= 0) || !Number.isSafeInteger(seed)) throw new Error('Invalid FakeData options');
  const [height, width, channels] = shape, featureCount = height * width * channels;
  return indexed({ name: 'fake-data', split: 'all', classNames: Array.from({ length: classes }, (_, i) => String(i)) }, options.size, async index => {
    let state = (seed + index) >>> 0; const random = () => state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const data = Uint8Array.from({ length: featureCount }, () => random() >>> 24);
    return { image: { data, height, width, channels, featureCount }, label: random() % classes };
  });
}
