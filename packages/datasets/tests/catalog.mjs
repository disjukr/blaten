import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm, copyFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';
import { deflateSync } from 'node:zlib';
import sharp from 'sharp';
import * as datasets from '../dist/index.js';
import { parseMat, matNumbers, matStrings } from '../dist/mat.js';
import { readPpm } from '../dist/media.js';
import { parseFlo, parsePfm, parseKittiFlow } from '../dist/flow-datasets.js';

const root = await mkdtemp(join(tmpdir(), 'blaten-catalog-'));
const originalFetch = globalThis.fetch;
globalThis.fetch = () => { throw new Error('Catalog fixtures must never access the internet') };
const path = (...parts) => join(root, ...parts);
async function put(name, data) { await mkdir(join(path(name), '..'), { recursive: true }); await writeFile(path(name), data) }
const rgb = Buffer.from([255, 0, 0, 0, 255, 0]);
const png = await sharp(rgb, { raw: { width: 2, height: 1, channels: 3 } }).png().toBuffer();
const jpeg = await sharp(rgb, { raw: { width: 2, height: 1, channels: 3 } }).jpeg().toBuffer();
const ppm = Buffer.concat([Buffer.from('P6\n# comment\n2 1\n255\n'), rgb]);
const names = n => Array.from({ length: n }, (_, i) => `class${String(i).padStart(3, '0')}`);
const options = (name, split, extra = {}) => ({ root: path(name), split, download: false, ...extra });
async function sample(name, split, extra) { return (await datasets[name](options(name, split, extra))).get(0) }

// Small native MATLAB v5 files: numeric, UTF-8 char, cell and struct matrices.
function matElement(type, bytes, little = true, small = false) {
  const data = Buffer.from(bytes), header = Buffer.alloc(8);
  if (small) { header[little ? 'writeUInt16LE' : 'writeUInt16BE'](type, 0); header[little ? 'writeUInt16LE' : 'writeUInt16BE'](data.length, 2); data.copy(header, 4); return header }
  header[little ? 'writeUInt32LE' : 'writeUInt32BE'](type, 0); header[little ? 'writeUInt32LE' : 'writeUInt32BE'](data.length, 4);
  return Buffer.concat([header, data, Buffer.alloc(type === 15 ? 0 : (8 - data.length % 8) % 8)]);
}
function ints(values, little = true) { const b = Buffer.alloc(values.length * 4); values.forEach((v, i) => b[little ? 'writeInt32LE' : 'writeInt32BE'](v, i * 4)); return b }
function matMatrix(name, value, little = true) {
  const kind = typeof value === 'string' ? 4 : Array.isArray(value) ? 1 : value.struct ? 2 : 6;
  const shape = kind === 4 ? [1, value.length] : kind === 1 ? [1, value.length] : kind === 2 ? [1, value.struct.length] : value.shape ?? [1, value.numbers.length];
  const parts = [matElement(6, ints([kind, 0], little), little), matElement(5, ints(shape, little), little), matElement(1, Buffer.from(name), little, name.length > 0 && name.length <= 4)];
  if (kind === 4) parts.push(matElement(16, Buffer.from(value), little));
  else if (kind === 1) parts.push(...value.map(v => matMatrix('', v, little)));
  else if (kind === 2) {
    const fields = Object.keys(value.struct[0]), width = Math.max(...fields.map(s => s.length)) + 1, b = Buffer.alloc(fields.length * width);
    fields.forEach((field, i) => b.write(field, i * width)); parts.push(matElement(5, ints([width], little), little, true), matElement(1, b, little));
    for (const row of value.struct) for (const field of fields) parts.push(matMatrix('', row[field], little));
  } else { const b = Buffer.alloc(value.numbers.length * 8); value.numbers.forEach((v, i) => b[little ? 'writeDoubleLE' : 'writeDoubleBE'](v, i * 8)); parts.push(matElement(9, b, little)) }
  return matElement(14, Buffer.concat(parts), little);
}
function matFile(values, little = true, compressed = false) {
  const header = Buffer.alloc(128); header.write('MATLAB 5.0 MAT-file'); header.write(little ? 'IM' : 'MI', 126);
  const matrices = Buffer.concat(Object.entries(values).map(([name, value]) => matMatrix(name, value, little)));
  return Buffer.concat([header, compressed ? matElement(15, deflateSync(matrices), little) : matrices]);
}
try {
  for (const little of [true, false]) for (const compressed of [true, false]) {
    const values = parseMat(matFile({ y: { numbers: [1, 10] }, classes: ['cat', 'dog'], annotations: { struct: [{ fname: 'a.jpg', class: { numbers: [2] } }] } }, little, compressed));
    assert.deepEqual(matNumbers(values.y), [1, 10]); assert.deepEqual(matStrings(values.classes), ['cat', 'dog']); assert.equal(values.annotations[0].fname, 'a.jpg');
  }
  assert.throws(() => parseMat(matFile({ y: { numbers: [1], shape: [1, 2] } })), /numeric index/);
  assert.throws(() => parseMat(Buffer.alloc(3)), /MATLAB/);
  assert.deepEqual([...readPpm(ppm).data], [...rgb]);
  assert.deepEqual([...readPpm(Buffer.from('P3\n2 1\n15\n15 0 0 0 15 0')).data], [...rgb]);

  for (const [name, n, directory] of [['caltech101', 101, '101_ObjectCategories'], ['caltech256', 257, '256_ObjectCategories']]) {
    for (const value of names(n)) await mkdir(path(name, directory, value), { recursive: true });
    await put(`${name}/${directory}/class000/a.png`, png); assert.equal((await sample(name, 'all')).label, 0);
  }
  await put('imageFolder/zebra/a.png', png); await put('imageFolder/apple/a.png', png);
  const folders = await datasets.imageFolder(options('imageFolder', 'all')); assert.deepEqual(folders.metadata.classNames, ['apple', 'zebra']); assert.equal(folders.count, 2);
  const fake = datasets.fakeData({ split: 'all', size: 3, seed: 7 }); assert.deepEqual(await fake.get(1), await fake.get(1)); assert.notDeepEqual(await fake.get(1), await fake.get(2)); await assert.rejects(fake.get(-1), /index/);

  const c10 = Buffer.alloc(3073); c10[0] = 3; c10[1] = 10; c10[1025] = 20; c10[2049] = 30;
  await put('cifar10/batches.meta.txt', names(10).join('\n')); await put('cifar10/test_batch.bin', c10);
  for (let i = 1; i <= 5; i++) await put(`cifar10/data_batch_${i}.bin`, c10);
  assert.deepEqual([...(await sample('cifar10', 'test')).image.data.slice(0, 3)], [10, 20, 30]); assert.equal((await datasets.cifar10(options('cifar10', 'train'))).count, 5);
  const c100 = Buffer.alloc(3074); c100[0] = 4; c100[1] = 70;
  await put('cifar100/fine_label_names.txt', names(100).join('\n')); await put('cifar100/coarse_label_names.txt', names(20).join('\n')); await put('cifar100/test.bin', c100);
  assert.equal((await sample('cifar100', 'test')).label, 70); assert.equal((await sample('cifar100', 'test', { labels: 'coarse' })).label, 4);
  const stl = Buffer.alloc(27648); stl[96] = 71;
  for (const split of ['train', 'test', 'unlabeled']) await put(`stl10/${split}_X.bin`, stl);
  for (const split of ['train', 'test']) await put(`stl10/${split}_y.bin`, Buffer.from([10]));
  await put('stl10/class_names.txt', names(10).join('\n')); await put('stl10/fold_indices.txt', Array(10).fill('0').join('\n'));
  assert.equal((await sample('stl10', 'train', { fold: 0 })).image.data[3], 71); assert.equal((await sample('stl10', 'unlabeled')).label, undefined);

  const emImages = Buffer.alloc(800), emLabels = Buffer.alloc(9); emImages.writeUInt32BE(2051); emImages.writeUInt32BE(1, 4); emImages.writeUInt32BE(28, 8); emImages.writeUInt32BE(28, 12); emImages[16 + 28] = 9;
  emLabels.writeUInt32BE(2049); emLabels.writeUInt32BE(1, 4); emLabels[8] = 1;
  await put('emnist/emnist-letters-train-images-idx3-ubyte', emImages); await put('emnist/emnist-letters-train-labels-idx1-ubyte', emLabels); await put('emnist/emnist-letters-mapping.txt', '1 65\n2 66');
  const em = await sample('emnist', 'train', { variant: 'letters' }); assert.equal(em.image.data[1], 9); assert.equal(em.label, 0);

  await put('food101/meta/classes.txt', names(101).join('\n')); await put('food101/meta/train.json', JSON.stringify({ class000: ['class000/a'] })); await put('food101/images/class000/a.jpg', jpeg); assert.equal((await sample('food101', 'train')).label, 0);
  await put('oxfordIiitPet/images/Abyssinian_1.jpg', jpeg); await put('oxfordIiitPet/annotations/trainval.txt', 'Abyssinian_1 1 1 1');
  const trimap = await sharp(Buffer.from([1, 2, 3]), { raw: { width: 3, height: 1, channels: 1 } }).png().toBuffer();
  await put('oxfordIiitPet/annotations/trimaps/Abyssinian_1.png', trimap);
  assert.equal((await sample('oxfordIiitPet', 'trainval')).label, 0); assert.deepEqual([...(await sample('oxfordIiitPet', 'trainval', { task: 'segmentation' })).mask.data], [0, 1, 2]);
  await put('stanfordCars/devkit/cars_meta.mat', matFile({ class_names: names(196) }));
  await put('stanfordCars/devkit/cars_train_annos.mat', matFile({ annotations: { struct: [{ fname: 'a.jpg', class: { numbers: [196] } }] } }, true, true)); await put('stanfordCars/cars_train/a.jpg', jpeg); assert.equal((await sample('stanfordCars', 'train')).label, 195);
  await put('gtsrb/GTSRB/Training/00042/a.ppm', ppm); assert.equal((await sample('gtsrb', 'train')).label, 42);
  await put('gtsrb/GTSRB/Final_Test/Images/a.ppm', ppm); await put('gtsrb/GT-final_test.csv', 'Filename;ClassId\na.ppm;3'); assert.equal((await sample('gtsrb', 'test')).label, 3);
  const wnids = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257'];
  for (const wnid of wnids) await mkdir(path('imagenette', 'val', wnid), { recursive: true }); await put('imagenette/val/n01440764/a.png', png); assert.equal((await sample('imagenette', 'val')).label, 0);
  for (let i = 0; i < 1000; i++) await mkdir(path('imageNet', 'train', `n${String(i).padStart(8, '0')}`), { recursive: true }); await put('imageNet/train/n00000000/a.png', png); assert.equal((await sample('imageNet', 'train')).label, 0);

  await put('voc/ImageSets/Main/train.txt', 'a'); await put('voc/ImageSets/Segmentation/train.txt', 'a'); await put('voc/JPEGImages/a.jpg', jpeg);
  await put('voc/Annotations/a.xml', '<annotation><object><name>cat</name><difficult>1</difficult><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox></object></annotation>');
  const detection = await sample('voc', 'train', { year: 2012 }); assert.deepEqual(detection.boxes, [[0, 1, 3, 4]]); assert.deepEqual(detection.labels, [7]); assert.deepEqual(detection.difficult, [true]);
  await put('voc/SegmentationClass/a.png', await sharp(Buffer.from([0, 0, 0, 128, 0, 0]), { raw: { width: 2, height: 1, channels: 3 } }).png().toBuffer());
  assert.deepEqual([...(await sample('voc', 'train', { year: 2012, task: 'segmentation' })).mask.data], [0, 1]);
  await put('coco/train2017/a.png', png); const images = [{ id: 9, file_name: 'a.png' }];
  await put('coco/annotations/instances_train2017.json', JSON.stringify({ images, categories: [{ id: 70, name: 'dog' }, { id: 1, name: 'cat' }], annotations: [{ image_id: 9, category_id: 70, bbox: [1, 2, 3, 4], iscrowd: 1 }] }));
  const co = await sample('coco', 'train'); assert.deepEqual(co.boxes, [[1, 2, 4, 6]]); assert.deepEqual(co.labels, [1]); assert.deepEqual(co.isCrowd, [true]);
  await put('coco/annotations/captions_train2017.json', JSON.stringify({ images, annotations: [{ image_id: 9, caption: 'a cat' }, { image_id: 9, caption: 'a dog' }] })); assert.deepEqual((await sample('coco', 'train', { task: 'captions' })).captions, ['a cat', 'a dog']);
  await put('celebA/list_attr_celeba.txt', '1\nSmiling Young\na.jpg 1 -1'); await put('celebA/list_eval_partition.txt', 'a.jpg 0'); await put('celebA/img_align_celeba/a.jpg', jpeg); assert.deepEqual((await sample('celebA', 'train')).attributes, { Smiling: true, Young: false });
  await put('lfw/lfw_funneled/A/A_0001.jpg', jpeg); await put('lfw/lfw_funneled/A/A_0002.jpg', jpeg); await put('lfw/lfw_funneled/B/B_0001.jpg', jpeg); await put('lfw/peopleDevTrain.txt', '1\nA 2'); await put('lfw/pairsDevTrain.txt', '2\nA 1 2\nA 1 B 1');
  assert.equal((await datasets.lfw(options('lfw', 'train'))).count, 2); const pairs = await datasets.lfw(options('lfw', 'train', { task: 'pairs' })); assert.equal((await pairs.get(0)).same, true); assert.equal((await pairs.get(1)).same, false);
  await put('cityscapes/leftImg8bit/train/city/a_leftImg8bit.png', png); await put('cityscapes/gtFine/train/city/a_gtFine_labelIds.png', trimap); assert.deepEqual([...(await sample('cityscapes', 'train')).mask.data], [1, 2, 3]);
  await put('kitti/training/image_2/a.png', png); await put('kitti/training/label_2/a.txt', 'Car 0 0 0 1 2 3 4 0 0 0 0 0 0 0\nDontCare 0 0 0 5 6 7 8'); assert.deepEqual((await sample('kitti', 'train')).boxes, [[1, 2, 3, 4]]);
  await put('widerFace/WIDER_train/images/event/a.jpg', jpeg); await put('widerFace/wider_face_split/wider_face_train_bbx_gt.txt', 'event/a.jpg\n2\n1 2 3 4 0 0 0 1 0 0\n5 6 7 8 0 0 0 0 0 1'); assert.deepEqual((await sample('widerFace', 'train')).boxes, [[1, 2, 4, 6]]);
  for (const name of ['flickr8k', 'flickr30k']) { await put(`${name}/images/123_abc.jpg`, jpeg); await put(`${name}/${name === 'flickr8k' ? 'Flickr8k.token.txt' : 'results_20130124.token'}`, '123_abc.jpg#0\ta cat\n123_abc.jpg#1\ta dog'); assert.deepEqual((await sample(name, 'all')).captions, ['a cat', 'a dog']) }
  await put('flickr8k/captions.html', '<html><table><tr><td><a href="https://flickr.com/photos/x/123/">image</a><ul><li>A cat &amp; dog</li></ul></td></tr></table></html>'); assert.deepEqual((await sample('flickr8k', 'all', { annotationsFile: 'captions.html' })).captions, ['A cat & dog']);
  await put('sbu/dataset/SBU_captioned_photo_dataset_urls.txt', 'https://example.com/a.jpg\nhttps://example.com/missing.jpg'); await put('sbu/dataset/SBU_captioned_photo_dataset_captions.txt', 'a cat\na missing photo'); await put('sbu/dataset/sbu_images/a.jpg', jpeg); assert.deepEqual((await sample('sbu', 'all')).captions, ['a cat']);

  await put('iris/iris.data', '1,2,3,4,Iris-setosa'); assert.deepEqual((await sample('iris', 'all')).features, [1, 2, 3, 4]);
  await put('wine/wine.data', `3,${Array(13).fill(1).join(',')}`); assert.equal((await sample('wine', 'all')).label, 2);
  await put('breastCancerWisconsin/wdbc.data', `123,B,${Array(30).fill(1).join(',')}`); assert.equal((await sample('breastCancerWisconsin', 'all')).features.length, 30);
  await put('imdb/train/neg/a.txt', 'bad <br /> movie'); await put('imdb/train/pos/a.txt', 'good'); assert.deepEqual(await sample('imdb', 'train'), { text: 'bad <br /> movie', label: 0 });
  await put('agNews/test.csv', '"4","title, with comma","description"'); assert.deepEqual(await sample('agNews', 'test'), { text: 'title, with comma\ndescription', label: 3 });

  const flo = Buffer.alloc(28); flo.write('PIEH'); flo.writeInt32LE(2, 4); flo.writeInt32LE(1, 8); [1, -2, 3, 4].forEach((v, i) => flo.writeFloatLE(v, 12 + i * 4)); assert.deepEqual([...parseFlo(flo).data], [1, -2, 3, 4]); assert.throws(() => parseFlo(flo.subarray(0, -1)), /Invalid/);
  const pfmRaw = Buffer.alloc(24); [3, 4, 0, 1, 2, 0].forEach((v, i) => pfmRaw.writeFloatLE(v, i * 4)); const pfm = Buffer.concat([Buffer.from('PF\n1 2\n-1\n'), pfmRaw]); assert.deepEqual([...parsePfm(pfm).data], [1, 2, 3, 4]);
  // 16-bit RGB flow PNG with every PNG row filter, independent of image decoder quantization.
  const pngChunk = (name, data) => { const b = Buffer.alloc(data.length + 12); b.writeUInt32BE(data.length); b.write(name, 4); data.copy(b, 8); return b };
  const header = Buffer.alloc(13); header.writeUInt32BE(1); header.writeUInt32BE(5, 4); header[8] = 16; header[9] = 2;
  const rawFlow = Buffer.alloc(5 * 7), row = Buffer.alloc(6); row.writeUInt16BE(32832); row.writeUInt16BE(32640, 2); row.writeUInt16BE(1, 4);
  for (let y = 0; y < 5; y++) { rawFlow[y * 7] = y; for (let i = 0; i < 6; i++) { const up = y ? row[i] : 0, predictor = y === 2 || y === 4 ? up : y === 3 ? Math.floor(up / 2) : 0; rawFlow[y * 7 + i + 1] = row[i] - predictor } }
  const flowPng = Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), pngChunk('IHDR', header), pngChunk('IDAT', deflateSync(rawFlow)), pngChunk('IEND', Buffer.alloc(0))]);
  const decoded = parseKittiFlow(flowPng); assert.deepEqual([...decoded.flow.data], Array(5).fill([1, -2]).flat()); assert.deepEqual([...decoded.validMask.data], [1, 1, 1, 1, 1]);
  await put('sintel/training/clean/scene/frame_0001.png', png); await put('sintel/training/clean/scene/frame_0002.png', png); await put('sintel/training/flow/scene/frame_0001.flo', flo); assert.deepEqual([...(await sample('sintel', 'train')).flow.data], [1, -2, 3, 4]);
  await put('flyingChairs/data/00001_img1.ppm', ppm); await put('flyingChairs/data/00001_img2.ppm', ppm); await put('flyingChairs/data/00001_flow.flo', flo); await put('flyingChairs/FlyingChairs_train_val.txt', '1'); assert.equal((await sample('flyingChairs', 'train')).flow.width, 2);
  await put('flyingThings3d/frames_cleanpass/TRAIN/A/0000/left/0006.png', png); await put('flyingThings3d/frames_cleanpass/TRAIN/A/0000/left/0007.png', png);
  const thingsRaw = Buffer.alloc(24); [1, 2, 0, 3, 4, 0].forEach((v, i) => thingsRaw.writeFloatLE(v, i * 4)); const thingsPfm = Buffer.concat([Buffer.from('PF\n2 1\n-1\n'), thingsRaw]);
  await put('flyingThings3d/optical_flow/TRAIN/A/0000/into_future/left/OpticalFlowIntoFuture_0006_L.pfm', thingsPfm); assert.deepEqual([...(await sample('flyingThings3d', 'train', { camera: 'left', direction: 'future' })).flow.data], [1, 2, 3, 4]);
  const flowImage = await sharp({ create: { width: 1, height: 5, channels: 3, background: 'red' } }).png().toBuffer(); await put('kittiFlow/training/image_2/a_10.png', flowImage); await put('kittiFlow/training/image_2/a_11.png', flowImage); await put('kittiFlow/training/flow_occ/a_10.png', flowPng); assert.equal((await sample('kittiFlow', 'train')).flow.height, 5);

  let ffmpegAvailable = true;
  try { await promisify(execFile)('ffmpeg', ['-version']); await promisify(execFile)('ffprobe', ['-version']) } catch (error) { if (error.code !== 'ENOENT') throw error; ffmpegAvailable = false }
  if (ffmpegAvailable) {
    const video = path('fixture.mp4'); await promisify(execFile)('ffmpeg', ['-v', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=4x4:r=4', '-frames:v', '4', '-c:v', 'mpeg4', video]);
    await put('ucf101/ucfTrainTestlist/classInd.txt', '1 Action'); await put('ucf101/ucfTrainTestlist/trainlist01.txt', 'Action/a.avi 1'); await put('ucf101/UCF-101/Action/a.avi', await import('node:fs/promises').then(fs => fs.readFile(video)));
    await put('hmdb51/testTrainMulti_7030_splits/action_test_split1.txt', 'a.avi 1'); await mkdir(path('hmdb51/action'), { recursive: true }); await copyFile(video, path('hmdb51/action/a.avi'));
    await put('kinetics/annotations/train.csv', 'label,youtube_id,time_start,time_end\naction,id,0,1'); await mkdir(path('kinetics/train/action'), { recursive: true }); await copyFile(video, path('kinetics/train/action/id_000000_000001.mp4'));
    for (const name of ['ucf101', 'hmdb51', 'kinetics']) { const dataset = await datasets[name](options(name, 'train')); assert.equal(await dataset.frameCount(0), 4); assert.equal((await dataset.getClip(0, 1, 2)).frames.length, 2); assert.equal((await dataset.get(0)).frames.length, 4); const clips = await datasets.videoClips({ dataset, framesPerClip: 2, stride: 1 }); assert.equal(clips.count, 3); assert.equal((await clips.get(2)).frames.length, 2); await assert.rejects(dataset.getClip(0, 3, 2), /range/) }
  } else console.log('Skipped native video decode: ffmpeg/ffprobe unavailable.');
  console.log('Passed: native catalog fixtures, MAT endian/compression, classification, detection, masks, captions, text, tables, optical flow, and video clips.');
} finally { globalThis.fetch = originalFetch; await rm(root, { recursive: true, force: true }) }
