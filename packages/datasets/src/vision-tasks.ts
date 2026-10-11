import { readFile, readdir } from 'node:fs/promises';
import { basename, dirname } from 'node:path';
import { Parser } from 'htmlparser2';
import { XMLParser } from 'fast-xml-parser';
import { parse as csv } from 'csv-parse/sync';
import type { Dataset, DatasetMetadata, Image, ImageClassificationSample } from './dataset.js';
import { readImage, readMask, type Box, type DetectionSample, type SegmentationSample, type CaptionSample, type PairSample } from './media.js';
import { indexed, inside, rootPath, nestedRoot, exists, text, lines, listFiles, prepareArchive, splitOf, type DownloadOptions, type LocalOptions } from './files.js';
import { imageRecords, imageExtension, type ClassificationDataset } from './classification-datasets.js';

const vocNames = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] as const;
const xml = new XMLParser({ ignoreAttributes: false });
export interface VocOptions extends DownloadOptions<'train' | 'val' | 'trainval' | 'test'> { year: 2007 | 2008 | 2009 | 2010 | 2011 | 2012 }
const vocArchives = {
  2007: ['VOCtrainval_06-Nov-2007.tar', 'c52e279531787c972589f7e41ab4ae64'],
  2008: ['VOCtrainval_14-Jul-2008.tar', '2629fa636546599198acfcfbfcf1904a'],
  2009: ['VOCtrainval_11-May-2009.tar', 'a3e00b113cfcfebf17e343f59da3caa1'],
  2010: ['VOCtrainval_03-May-2010.tar', 'da459979d0c395079b5c75ee67908abb'],
  2011: ['VOCtrainval_25-May-2011.tar', '6c3384ef61512963050cb5d687e5bf1e'],
  2012: ['VOCtrainval_11-May-2012.tar', '6cd6e144f989b92b3379bac3b3de84fd'],
} as const;
export function voc(options: VocOptions & { task: 'segmentation' }): Promise<Dataset<SegmentationSample>>;
export function voc(options: VocOptions & { task?: 'detection' }): Promise<Dataset<DetectionSample>>;
export async function voc(options: VocOptions & { task?: 'detection' | 'segmentation' }): Promise<Dataset<DetectionSample | SegmentationSample>> {
  splitOf(options.split, ['train', 'val', 'trainval', 'test']);
  if (!(options.year in vocArchives) || options.split === 'test' && options.year !== 2007) throw new Error('Only VOC2007 has a public annotated test split');
  const task = options.task ?? 'detection', root = rootPath(`voc${options.year}`, options.root), subset = task === 'segmentation' ? 'Segmentation' : 'Main', list = `ImageSets/${subset}/${options.split}.txt`;
  const [name, md5] = options.split === 'test' ? ['VOCtest_06-Nov-2007.tar', 'b6e924de25625d8de591ea690078ad9f'] : vocArchives[options.year];
  if (!await exists(inside(root, list))) await prepareArchive(root, { url: `https://thor.robots.ox.ac.uk/pascal/VOC/voc${options.year}/${name}`, md5 }, `VOCdevkit/VOC${options.year}/${list}`, options.download);
  const dir = await nestedRoot(root, [`VOCdevkit/VOC${options.year}`, `VOC${options.year}`], list), ids = lines(await text(inside(dir, list))).map(line => line.split(/\s+/)[0]!);
  const palette = new Map<number, number>();
  for (let id = 0; id < 256; id++) { let value = id, r = 0, g = 0, b = 0; for (let bit = 0; bit < 8; bit++) { r |= (value & 1) << (7 - bit); g |= ((value >> 1) & 1) << (7 - bit); b |= ((value >> 2) & 1) << (7 - bit); value >>= 3 } palette.set(r * 65536 + g * 256 + b, id) }
  return indexed({ name: 'voc', split: options.split, homepage: 'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/', classNames: task === 'segmentation' ? ['background', ...vocNames] : vocNames }, ids.length, async n => {
    const id = ids[n]!, image = await readImage(inside(dir, `JPEGImages/${id}.jpg`));
    if (task === 'segmentation') return { image, mask: await readMask(inside(dir, `SegmentationClass/${id}.png`), channels => channels.length === 1 ? channels[0]! : palette.get(channels[0]! * 65536 + channels[1]! * 256 + channels[2]!) ?? 255, 255) };
    const annotation = xml.parse(await text(inside(dir, `Annotations/${id}.xml`))).annotation;
    const objects = annotation.object ? Array.isArray(annotation.object) ? annotation.object : [annotation.object] : [];
    const boxes: Box[] = [], labels: number[] = [], difficult: boolean[] = [];
    for (const object of objects) {
      const label = vocNames.indexOf(object.name); if (label < 0) throw new Error(`Unknown VOC class: ${object.name}`);
      const { xmin, ymin, xmax, ymax } = object.bndbox;
      const box: Box = [Number(xmin) - 1, Number(ymin) - 1, Number(xmax), Number(ymax)];
      if (box.some(n => !Number.isFinite(n)) || box[2] < box[0] || box[3] < box[1]) throw new Error('Invalid VOC box');
      boxes.push(box); labels.push(label); difficult.push(Number(object.difficult ?? 0) === 1);
    }
    return { image, boxes, labels, difficult };
  });
}
export interface CocoOptions extends LocalOptions<'train' | 'val'> { year?: 2014 | 2017; annotationsFile?: string }
export interface CocoDetectionMetadata extends DatasetMetadata { readonly classNames: readonly string[]; readonly categoryIds: readonly number[] }
interface CocoImage { id: number; file_name: string }
interface CocoAnnotation { image_id: number; category_id?: number; bbox?: number[]; caption?: string; iscrowd?: number }
interface CocoFile { images: CocoImage[]; annotations: CocoAnnotation[]; categories?: { id: number; name: string }[] }
export function coco(options: CocoOptions & { task: 'captions' }): Promise<Dataset<CaptionSample>>;
export function coco(options: CocoOptions & { task?: 'detection' }): Promise<Dataset<DetectionSample, CocoDetectionMetadata>>;
export async function coco(options: CocoOptions & { task?: 'detection' | 'captions' }): Promise<Dataset<DetectionSample | CaptionSample>> {
  splitOf(options.split, ['train', 'val']); const year = options.year ?? 2017; if (year !== 2014 && year !== 2017) throw new Error('Unsupported COCO year');
  const root = rootPath('coco', options.root), task = options.task ?? 'detection', prefix = `${options.split}${year}`;
  const file = JSON.parse(await text(options.annotationsFile ? inside(root, options.annotationsFile) : inside(root, `annotations/${task === 'captions' ? 'captions' : 'instances'}_${prefix}.json`))) as CocoFile;
  if (!Array.isArray(file.images) || !Array.isArray(file.annotations) || new Set(file.images.map(image => image.id)).size !== file.images.length) throw new Error('Invalid COCO annotations');
  const categories = [...file.categories ?? []].sort((a, b) => a.id - b.id), categoryIds = categories.map(category => category.id), classNames = categories.map(category => category.name);
  const groups = new Map<number, CocoAnnotation[]>();
  for (const annotation of file.annotations) { const group = groups.get(annotation.image_id) ?? []; group.push(annotation); groups.set(annotation.image_id, group) }
  const images = [...file.images].sort((a, b) => a.id - b.id), imageDir = await nestedRoot(root, ['images'], prefix);
  return indexed({ name: 'coco', split: options.split, homepage: 'https://cocodataset.org/', ...(task === 'detection' ? { classNames, categoryIds: Object.freeze(categoryIds) } : {}) }, images.length, async n => {
    const source = images[n]!, image = await readImage(inside(imageDir, `${prefix}/${source.file_name}`)), annotations = groups.get(source.id) ?? [];
    if (task === 'captions') return { image, captions: annotations.map(annotation => { if (typeof annotation.caption !== 'string') throw new Error('Invalid COCO caption'); return annotation.caption }) };
    const boxes: Box[] = [], labels: number[] = [], isCrowd: boolean[] = [];
    for (const annotation of annotations) { const box = annotation.bbox, label = categoryIds.indexOf(annotation.category_id!); if (!box || box.length !== 4 || box.some(n => !Number.isFinite(n)) || box[2]! < 0 || box[3]! < 0 || label < 0) throw new Error('Invalid COCO detection'); boxes.push([box[0]!, box[1]!, box[0]! + box[2]!, box[1]! + box[3]!]); labels.push(label); isCrowd.push(annotation.iscrowd === 1) }
    return { image, boxes, labels, isCrowd };
  });
}
export interface CelebASample { readonly image: Image; readonly attributes: Readonly<Record<string, boolean>> }
export interface CelebAMetadata extends DatasetMetadata { readonly attributeNames: readonly string[] }
export async function celebA(options: LocalOptions<'train' | 'val' | 'test' | 'all'>): Promise<Dataset<CelebASample, CelebAMetadata>> {
  splitOf(options.split, ['train', 'val', 'test', 'all']); const root = await nestedRoot(rootPath('celeba', options.root), ['celeba'], 'list_attr_celeba.txt');
  const attr = lines(await text(inside(root, 'list_attr_celeba.txt'))), attributeNames = attr[1]!.split(/\s+/), partitions = new Map(lines(await text(inside(root, 'list_eval_partition.txt'))).map(line => { const [name, split] = line.split(/\s+/); return [name!, Number(split)] }));
  if (Number(attr[0]) !== attr.length - 2) throw new Error('Invalid CelebA attribute count');
  const wanted = { train: 0, val: 1, test: 2, all: -1 }[options.split], rows = attr.slice(2).map(line => line.split(/\s+/)).filter(row => wanted < 0 || partitions.get(row[0]!) === wanted);
  if (rows.some(row => row.length !== attributeNames.length + 1 || row.slice(1).some(value => value !== '-1' && value !== '1'))) throw new Error('Invalid CelebA attributes');
  return indexed({ name: 'celeba', split: options.split, homepage: 'https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html', attributeNames }, rows.length, async n => ({ image: await readImage(inside(root, `img_align_celeba/${rows[n]![0]}`)), attributes: Object.fromEntries(attributeNames.map((name, i) => [name, rows[n]![i + 1] === '1'])) }));
}
export type LfwOptions = LocalOptions<'train' | 'test' | 'all'>;
export function lfw(options: LfwOptions & { task: 'pairs' }): Promise<Dataset<PairSample>>;
export function lfw(options: LfwOptions & { task?: 'classification' }): Promise<ClassificationDataset>;
export async function lfw(options: LfwOptions & { task?: 'pairs' | 'classification' }): Promise<Dataset<PairSample | ImageClassificationSample>> {
  splitOf(options.split, ['train', 'test', 'all']); const root = rootPath('lfw', options.root), folder = await exists(inside(root, 'lfw_funneled')) ? 'lfw_funneled' : await exists(inside(root, 'lfw')) ? 'lfw' : '';
  const dir = inside(root, folder), imagePath = (name: string, index: string) => inside(dir, `${name}/${name}_${index.padStart(4, '0')}.jpg`);
  if (options.task === 'pairs') {
    const filename = options.split === 'all' ? 'pairs.txt' : `pairsDev${options.split === 'train' ? 'Train' : 'Test'}.txt`;
    const rows = lines(await text(inside(root, filename))).slice(1).map(line => line.split(/\s+/));
    if (rows.some(row => row.length !== 3 && row.length !== 4)) throw new Error('Invalid LFW pairs');
    return indexed({ name: 'lfw-pairs', split: options.split, homepage: 'https://vis-www.cs.umass.edu/lfw/' }, rows.length, async n => {
      const row = rows[n]!, same = row.length === 3;
      return { first: await readImage(imagePath(row[0]!, row[1]!)), second: await readImage(imagePath(same ? row[0]! : row[2]!, same ? row[2]! : row[3]!)), same };
    });
  }
  const classes = (await readdir(dir, { withFileTypes: true })).filter(entry => entry.isDirectory()).map(entry => entry.name).sort(), rows: { path: string; label: number }[] = [];
  if (options.split === 'all') for (let label = 0; label < classes.length; label++) for (const path of await listFiles(inside(dir, classes[label]!), imageExtension)) rows.push({ path, label });
  else {
    const source = lines(await text(inside(root, `peopleDev${options.split === 'train' ? 'Train' : 'Test'}.txt`))).slice(1);
    for (const line of source) { const [name, count] = line.split(/\s+/), label = classes.indexOf(name!); for (let i = 1; i <= Number(count); i++) rows.push({ path: imagePath(name!, String(i)), label }) }
  }
  return imageRecords('lfw', options.split, classes, rows, 'https://vis-www.cs.umass.edu/lfw/');
}
const cityNames = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle'];
export interface CityscapesOptions extends LocalOptions<'train' | 'val' | 'test' | 'train_extra'> { quality?: 'fine' | 'coarse' }
export function cityscapes(options: CityscapesOptions & { split: 'test' }): Promise<Dataset<{ image: Image }>>;
export function cityscapes(options: CityscapesOptions & { split: 'train' | 'val' | 'train_extra' }): Promise<Dataset<SegmentationSample>>;
export async function cityscapes(options: CityscapesOptions): Promise<Dataset<{ image: Image; mask?: SegmentationSample['mask'] }>> {
  splitOf(options.split, ['train', 'val', 'test', 'train_extra']); const quality = options.quality ?? 'fine'; splitOf(quality, ['fine', 'coarse']);
  if (options.split === 'train_extra' && quality !== 'coarse' || options.split === 'test' && quality !== 'fine') throw new Error('Unsupported Cityscapes quality/split combination');
  const root = await nestedRoot(rootPath('cityscapes', options.root), ['cityscapes'], `leftImg8bit/${options.split}`), images = await listFiles(inside(root, `leftImg8bit/${options.split}`), /_leftImg8bit\.png$/), target = quality === 'fine' ? 'gtFine' : 'gtCoarse';
  return indexed({ name: 'cityscapes', split: options.split, homepage: 'https://www.cityscapes-dataset.com/', classNames: cityNames }, images.length, async n => {
    const path = images[n]!, image = await readImage(path); if (options.split === 'test') return { image };
    const filename = basename(path).replace('_leftImg8bit.png', `_${target}_labelIds.png`), city = basename(dirname(path));
    return { image, mask: await readMask(inside(root, `${target}/${options.split}/${city}/${filename}`), undefined, 255) };
  });
}
const kittiNames = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc'];
export function kitti(options: LocalOptions<'test'>): Promise<Dataset<{ image: Image }>>;
export function kitti(options: LocalOptions<'train'>): Promise<Dataset<DetectionSample>>;
export async function kitti(options: LocalOptions<'train' | 'test'>): Promise<Dataset<{ image: Image; boxes?: readonly Box[]; labels?: readonly number[] }>> {
  splitOf(options.split, ['train', 'test']); const split = options.split === 'train' ? 'training' : 'testing';
  const root = await nestedRoot(rootPath('kitti', options.root), ['Kitti', 'KITTI'], `${split}/image_2`), images = await listFiles(inside(root, `${split}/image_2`), /\.png$/, false);
  return indexed({ name: 'kitti', split: options.split, homepage: 'https://www.cvlibs.net/datasets/kitti/', classNames: kittiNames }, images.length, async n => {
    const image = await readImage(images[n]!); if (options.split === 'test') return { image };
    const rows = lines(await text(inside(root, `training/label_2/${basename(images[n]!, '.png')}.txt`))).map(line => line.split(/\s+/)).filter(row => row[0] !== 'DontCare');
    const boxes: Box[] = [], labels: number[] = [];
    for (const row of rows) { const label = kittiNames.indexOf(row[0]!); const box = row.slice(4, 8).map(Number); if (label < 0 || box.length !== 4 || box.some(n => !Number.isFinite(n))) throw new Error('Invalid KITTI annotation'); labels.push(label); boxes.push(box as unknown as Box) }
    return { image, boxes, labels };
  });
}
export function widerFace(options: LocalOptions<'test'>): Promise<Dataset<{ image: Image }>>;
export function widerFace(options: LocalOptions<'train' | 'val'>): Promise<Dataset<DetectionSample>>;
export async function widerFace(options: LocalOptions<'train' | 'val' | 'test'>): Promise<Dataset<{ image: Image; boxes?: readonly Box[]; labels?: readonly number[] }>> {
  splitOf(options.split, ['train', 'val', 'test']); const root = await nestedRoot(rootPath('wider-face', options.root), ['widerface'], `WIDER_${options.split}/images`), rows: { path: string; boxes?: Box[] }[] = [];
  if (options.split === 'test') for (const path of await listFiles(inside(root, 'WIDER_test/images'), imageExtension)) rows.push({ path });
  else {
    const source = lines(await text(inside(root, `wider_face_split/wider_face_${options.split}_bbx_gt.txt`))); let i = 0;
    while (i < source.length) {
      const name = source[i++]!, count = Number(source[i++]); if (!Number.isSafeInteger(count) || count < 0) throw new Error('Invalid WIDER Face count');
      const boxes: Box[] = [];
      for (let n = 0; n < Math.max(count, 1); n++) {
        const row = source[i++]?.split(/\s+/).map(Number); if (!row || row.length < 4 || row.some(n => !Number.isFinite(n))) throw new Error('Invalid WIDER Face box');
        if (count && row[9] !== 1 && row[2]! > 0 && row[3]! > 0) boxes.push([row[0]!, row[1]!, row[0]! + row[2]!, row[1]! + row[3]!]);
      }
      rows.push({ path: inside(root, `WIDER_${options.split}/images/${name}`), boxes });
    }
  }
  return indexed({ name: 'wider-face', split: options.split, homepage: 'https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/', classNames: ['face'] }, rows.length, async n => ({ image: await readImage(rows[n]!.path), ...(rows[n]!.boxes ? { boxes: rows[n]!.boxes!, labels: rows[n]!.boxes!.map(() => 0) } : {}) }));
}
export interface FlickrOptions extends LocalOptions<'all' | 'train' | 'val' | 'test'> { annotationsFile?: string; imagesDir?: string; splitFile?: string }
async function flickr(name: 'flickr8k' | 'flickr30k', options: FlickrOptions): Promise<Dataset<CaptionSample>> {
  splitOf(options.split, ['all', 'train', 'val', 'test']); const root = rootPath(name, options.root), defaultFile = name === 'flickr8k' ? 'Flickr8k.token.txt' : 'results_20130124.token';
  const annotations = await text(inside(root, options.annotationsFile ?? defaultFile)), groups = new Map<string, string[]>();
  if (/<html|<table/i.test(annotations)) {
    const imageDir = options.imagesDir ? inside(root, options.imagesDir) : await exists(inside(root, 'Flicker8k_Dataset')) ? inside(root, 'Flicker8k_Dataset') : inside(root, 'images');
    const images = await listFiles(imageDir, imageExtension), byId = new Map(images.map(path => [basename(path).split('_')[0]!, basename(path)]));
    let inTable = false, inLink = false, inCaption = false, link = '', caption = '', current: string | undefined;
    const identify = (value: string) => { const match = value.trim().match(/\/(\d+)\/?$/); if (match) current = byId.get(match[1]!) };
    const parser = new Parser({
      onopentag(name, attributes) {
        if (name === 'table') inTable = true;
        if (inTable && name === 'a') { inLink = true; link = ''; if (attributes.href) identify(attributes.href) }
        if (inTable && name === 'li') { inCaption = true; caption = '' }
      },
      ontext(value) { if (inLink) link += value; if (inCaption) caption += value; if (value.includes('Image Not Found')) current = undefined },
      onclosetag(name) {
        if (name === 'a' && inLink) { identify(link); inLink = false }
        if (name === 'li' && inCaption) { if (current) { const group = groups.get(current) ?? []; group.push(caption.trim()); groups.set(current, group) } inCaption = false }
        if (name === 'table') { inTable = false; current = undefined }
      },
    });
    parser.end(annotations);
  } else if (/^image,caption/i.test(annotations)) {
    for (const row of csv(annotations, { columns: true, skip_empty_lines: true }) as { image: string; caption: string }[]) { const group = groups.get(row.image) ?? []; group.push(row.caption); groups.set(row.image, group) }
  } else for (const line of annotations.split(/\r?\n/).filter(Boolean)) {
    const tab = line.indexOf('\t'); if (tab < 0) throw new Error('Invalid Flickr caption line');
    const filename = line.slice(0, tab).replace(/#\d+$/, ''), group = groups.get(filename) ?? []; group.push(line.slice(tab + 1)); groups.set(filename, group);
  }
  let files = [...groups.keys()].sort();
  if (options.split !== 'all') {
    const filename = options.splitFile ?? (name === 'flickr8k' ? `Flickr_8k.${options.split === 'val' ? 'dev' : options.split}Images.txt` : `${options.split}.txt`);
    files = lines(await text(inside(root, filename))); if (files.some(file => !groups.has(file))) throw new Error('Flickr split references missing captions');
  }
  const imageDir = options.imagesDir ? inside(root, options.imagesDir) : await exists(inside(root, 'Flicker8k_Dataset')) ? inside(root, 'Flicker8k_Dataset') : await exists(inside(root, 'flickr30k-images')) ? inside(root, 'flickr30k-images') : inside(root, 'images');
  return indexed({ name, split: options.split, homepage: 'https://hockenmaier.cs.illinois.edu/' }, files.length, async n => ({ image: await readImage(inside(imageDir, files[n]!)), captions: groups.get(files[n]!)! }));
}
export const flickr8k = (options: FlickrOptions) => flickr('flickr8k', options);
export const flickr30k = (options: FlickrOptions) => flickr('flickr30k', options);
export async function sbu(options: LocalOptions<'all'>): Promise<Dataset<CaptionSample>> {
  splitOf(options.split, ['all']); const root = await nestedRoot(rootPath('sbu', options.root), ['dataset'], 'SBU_captioned_photo_dataset_urls.txt');
  const urls = lines(await text(inside(root, 'SBU_captioned_photo_dataset_urls.txt'))), captions = (await text(inside(root, 'SBU_captioned_photo_dataset_captions.txt'))).split(/\r?\n/); if (captions.at(-1) === '') captions.pop();
  if (urls.length !== captions.length) throw new Error('SBU URLs and captions have different lengths');
  const rows: { path: string; caption: string }[] = [];
  for (let i = 0; i < urls.length; i++) { const path = inside(root, `sbu_images/${basename(new URL(urls[i]!).pathname)}`); if (await exists(path)) rows.push({ path, caption: captions[i]! }) }
  return indexed({ name: 'sbu', split: 'all', homepage: 'https://www.cs.rice.edu/~vo9/sbucaptions/' }, rows.length, async n => ({ image: await readImage(rows[n]!.path), captions: [rows[n]!.caption] }));
}
