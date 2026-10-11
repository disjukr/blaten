import { readFile } from 'node:fs/promises';
import { parse as csv } from 'csv-parse/sync';
import { indexed, rootPath, inside, downloadFile, prepareArchive, nestedRoot, listFiles, lines, text, splitOf, exists, type DownloadOptions } from './files.js';
import type { Dataset, DatasetMetadata } from './dataset.js';
import type { TextSample, TabularSample } from './media.js';

export interface TabularMetadata extends DatasetMetadata { readonly featureNames: readonly string[]; readonly classNames: readonly string[] }
const tabularSources = {
  iris: { file: 'iris/iris.data', homepage: 'https://archive.ics.uci.edu/dataset/53/iris', classNames: ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], features: ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'] },
  wine: { file: 'wine/wine.data', homepage: 'https://archive.ics.uci.edu/dataset/109/wine', classNames: ['1', '2', '3'], features: ['alcohol', 'malicAcid', 'ash', 'alkalinityOfAsh', 'magnesium', 'totalPhenols', 'flavanoids', 'nonflavanoidPhenols', 'proanthocyanins', 'colorIntensity', 'hue', 'od280Od315', 'proline'] },
  'breast-cancer-wisconsin': { file: 'breast-cancer-wisconsin/wdbc.data', homepage: 'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic', classNames: ['malignant', 'benign'], features: ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concavePoints', 'symmetry', 'fractalDimension'] as string[] },
};
// WDBC columns are ten measurements, then their standard errors, then worst values.
const breastMeasures = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concavePoints', 'symmetry', 'fractalDimension'];
tabularSources['breast-cancer-wisconsin'].features = ['mean', 'standardError', 'worst'].flatMap(statistic => breastMeasures.map(name => `${name}.${statistic}`));
async function tabular(name: keyof typeof tabularSources, options: DownloadOptions<'all'>): Promise<Dataset<TabularSample, TabularMetadata>> {
  splitOf(options.split, ['all']); const source = tabularSources[name], root = rootPath(name, options.root);
  const path = await downloadFile(root, { url: `https://archive.ics.uci.edu/ml/machine-learning-databases/${source.file}` }, options.download);
  const records = csv(await text(path), { skip_empty_lines: true }) as string[][];
  const rows = records.map(row => {
    const label = name === 'iris' ? source.classNames.indexOf(row.at(-1)!) : name === 'wine' ? Number(row[0]) - 1 : row[1] === 'M' ? 0 : row[1] === 'B' ? 1 : -1;
    const features = (name === 'iris' ? row.slice(0, -1) : name === 'wine' ? row.slice(1) : row.slice(2)).map(Number);
    if (!Number.isInteger(label) || label < 0 || label >= source.classNames.length || features.length !== source.features.length || features.some(n => !Number.isFinite(n))) throw new Error(`Invalid ${name} record`);
    return { features: Object.freeze(features), label };
  });
  return indexed({ name, split: 'all', homepage: source.homepage, classNames: Object.freeze(source.classNames), featureNames: Object.freeze(source.features) }, rows.length, async n => rows[n]!);
}
export const iris = (options: DownloadOptions<'all'>) => tabular('iris', options);
export const wine = (options: DownloadOptions<'all'>) => tabular('wine', options);
export const breastCancerWisconsin = (options: DownloadOptions<'all'>) => tabular('breast-cancer-wisconsin', options);
export async function imdb(options: DownloadOptions<'train' | 'test'>): Promise<Dataset<TextSample>> {
  splitOf(options.split, ['train', 'test']); const root = rootPath('imdb', options.root);
  if (!await exists(inside(root, options.split))) await prepareArchive(root, { url: 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', md5: '7c2ac02c03563afcf9b574c7e56c153a' }, `aclImdb/${options.split}`, options.download);
  const dir = await nestedRoot(root, ['aclImdb'], options.split), rows: { path: string; label: number }[] = [];
  for (const [label, folder] of ['neg', 'pos'].entries()) for (const path of await listFiles(inside(dir, `${options.split}/${folder}`), /\.txt$/)) rows.push({ path, label });
  return indexed({ name: 'imdb', split: options.split, homepage: 'https://ai.stanford.edu/~amaas/data/sentiment/', classNames: ['negative', 'positive'] }, rows.length, async n => ({ text: await text(rows[n]!.path), label: rows[n]!.label }));
}
export async function agNews(options: DownloadOptions<'train' | 'test'>): Promise<Dataset<TextSample>> {
  splitOf(options.split, ['train', 'test']); const root = rootPath('ag-news', options.root);
  const path = await downloadFile(root, { url: `https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/${options.split}.csv` }, options.download);
  const rows = (csv(await text(path), { skip_empty_lines: true }) as string[][]).map(row => {
    const label = Number(row[0]) - 1; if (row.length !== 3 || !Number.isInteger(label) || label < 0 || label > 3) throw new Error('Invalid AG News record');
    return { text: `${row[1]}\n${row[2]}`, label };
  });
  return indexed({ name: 'ag-news', split: options.split, homepage: 'https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv', classNames: ['World', 'Sports', 'Business', 'Sci/Tech'] }, rows.length, async n => rows[n]!);
}
