import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { basename } from 'node:path';
import { rootPath, inside, exists, extractArchive, listFiles, lines, text, splitOf, type LocalOptions } from './files.js';
import { imageRecords, namesWithCount, imageExtension, type ClassificationDataset } from './classification-datasets.js';
import { parseMat, matNumbers, matStrings, type MatValue } from './mat.js';

async function unpackOnce(archive: string, directory: string): Promise<void> {
  const marker = `${archive}.extracted`;
  if (await exists(directory) && await exists(marker)) return;
  await mkdir(directory, { recursive: true }); await extractArchive(archive, directory);
  await writeFile(marker, 'complete\n');
}

/** Read prepared class folders, or unpack the user's original ILSVRC2012 archives. */
export async function imageNet(options: LocalOptions<'train' | 'val'>): Promise<ClassificationDataset<number, 1000>> {
  splitOf(options.split, ['train', 'val']); const root = rootPath('imagenet', options.root), directory = inside(root, options.split);
  const archive = inside(root, `ILSVRC2012_img_${options.split}.tar`);
  if (await exists(archive)) await unpackOnce(archive, directory);
  else if (!await exists(directory)) throw new Error(`Prepare ${directory} or ${archive}; ImageNet is not downloaded automatically`);
  if (options.split === 'train') {
    for (const archive of await listFiles(directory, /\.tar$/, false)) {
      const target = inside(directory, basename(archive, '.tar'));
      await unpackOnce(archive, target);
    }
  }
  const devkit = inside(root, 'ILSVRC2012_devkit_t12');
  if (await exists(inside(root, 'ILSVRC2012_devkit_t12.tar.gz')) && (!await exists(inside(devkit, 'data/meta.mat')) || !await exists(inside(root, 'ILSVRC2012_devkit_t12.tar.gz.extracted')))) await unpackOnce(inside(root, 'ILSVRC2012_devkit_t12.tar.gz'), root);
  let records: { id: number; wnid: string; words: string }[] = [];
  if (await exists(inside(devkit, 'data/meta.mat'))) {
    const synsets = parseMat(await readFile(inside(devkit, 'data/meta.mat'))).synsets;
    if (!Array.isArray(synsets)) throw new Error('Invalid ImageNet synset metadata');
    for (const value of synsets) {
      const record = value as Record<string, MatValue>;
      if (matNumbers(record.num_children)[0] === 0) records.push({ id: matNumbers(record.ILSVRC2012_ID)[0]!, wnid: matStrings(record.WNID)[0]!, words: matStrings(record.words)[0]! });
    }
    records.sort((a, b) => a.wnid.localeCompare(b.wnid));
  }
  const folders = (await readdir(directory, { withFileTypes: true })).filter(entry => entry.isDirectory() && /^n\d+$/.test(entry.name)).map(entry => entry.name).sort();
  const wnids = records.length ? records.map(record => record.wnid) : folders;
  const classNames = namesWithCount(records.length ? records.map(record => record.words) : wnids, 1000), rows: { path: string; label: number }[] = [];
  if (folders.length) {
    for (const folder of folders) {
      const label = wnids.indexOf(folder); if (label < 0) throw new Error(`Unknown ImageNet synset: ${folder}`);
      for (const path of await listFiles(inside(directory, folder), imageExtension)) rows.push({ path, label });
    }
  } else {
    if (options.split !== 'val' || !records.length) throw new Error('Flat ImageNet validation images require ILSVRC2012_devkit_t12/data/meta.mat and ground-truth labels');
    const targets = lines(await text(inside(devkit, 'data/ILSVRC2012_validation_ground_truth.txt'))).map(Number), files = await listFiles(directory, imageExtension, false), ids = new Map(records.map((record, label) => [record.id, label]));
    if (targets.length !== files.length) throw new Error('ImageNet validation image and label counts differ');
    files.forEach((path, i) => { const label = ids.get(targets[i]!); if (label === undefined) throw new Error('Unknown ImageNet validation label'); rows.push({ path, label }) });
  }
  return imageRecords('imagenet', options.split, classNames, rows, 'https://www.image-net.org/');
}
