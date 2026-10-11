import { spawn } from 'node:child_process';
import { readdir } from 'node:fs/promises';
import { basename, dirname } from 'node:path';
import { parse as csv } from 'csv-parse/sync';
import type { Dataset, DatasetMetadata } from './dataset.js';
import type { VideoSample } from './media.js';
import { indexed, inside, rootPath, nestedRoot, exists, lines, text, listFiles, splitOf, type LocalOptions } from './files.js';

export interface VideoTools { readonly ffmpegPath?: string; readonly ffprobePath?: string }
export interface VideoMetadata extends DatasetMetadata { readonly classNames: readonly string[] }
export interface VideoDataset extends Dataset<VideoSample, VideoMetadata> {
  frameCount(index: number): Promise<number>;
  getClip(index: number, start: number, count: number): Promise<VideoSample>;
}
interface VideoRecord { path: string; label?: number }
interface VideoInfo { width: number; height: number; fps: number; count: number }
async function command(executable: string, args: readonly string[]): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, [...args], { stdio: ['ignore', 'pipe', 'pipe'] }), output: Buffer[] = []; let errorText = '';
    child.stdout.on('data', chunk => output.push(chunk));
    child.stderr.on('data', chunk => { if (errorText.length < 8192) errorText += chunk.toString() });
    child.on('error', error => reject(new Error(`Cannot run ${executable}. Video decoding requires ffmpeg and ffprobe on PATH, or explicit paths.`, { cause: error })));
    child.on('close', code => code === 0 ? resolve(Buffer.concat(output)) : reject(new Error(`${executable} failed (${code}): ${errorText.trim()}`)));
  });
}
function videoRecords(name: string, split: string, homepage: string, classNames: readonly string[], rows: readonly VideoRecord[], tools: VideoTools): VideoDataset {
  const ffmpeg = tools.ffmpegPath ?? 'ffmpeg', ffprobe = tools.ffprobePath ?? 'ffprobe', probes = new Map<number, Promise<VideoInfo>>();
  const info = (index: number): Promise<VideoInfo> => {
    if (!Number.isSafeInteger(index) || index < 0 || index >= rows.length) return Promise.reject(new RangeError(`Invalid video index: ${index}`));
    const previous = probes.get(index); if (previous) return previous;
    const request = (async () => {
      const args = ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,nb_read_frames', '-of', 'json'];
      let stream = JSON.parse((await command(ffprobe, [...args, rows[index]!.path])).toString()).streams?.[0];
      if (!stream) throw new Error(`No video stream: ${rows[index]!.path}`);
      if (!Number.isSafeInteger(Number(stream.nb_frames)) || Number(stream.nb_frames) <= 0) stream = JSON.parse((await command(ffprobe, ['-count_frames', ...args, rows[index]!.path])).toString()).streams?.[0];
      const [numerator, denominator] = String(stream.avg_frame_rate).split('/').map(Number), fps = numerator! / denominator!, count = Number.isSafeInteger(Number(stream.nb_frames)) && Number(stream.nb_frames) > 0 ? Number(stream.nb_frames) : Number(stream.nb_read_frames);
      if (![stream.width, stream.height, count].every(n => Number.isSafeInteger(Number(n)) && Number(n) > 0) || !Number.isFinite(fps) || fps <= 0) throw new Error(`Invalid video metadata: ${rows[index]!.path}`);
      return { width: Number(stream.width), height: Number(stream.height), count, fps };
    })();
    probes.set(index, request); request.catch(() => probes.delete(index)); return request;
  };
  const getClip = async (index: number, start: number, count: number): Promise<VideoSample> => {
    const metadata = await info(index);
    if (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(count) || count <= 0 || start + count > metadata.count) throw new RangeError('Invalid video clip range');
    const row = rows[index]!, filter = `select=between(n\\,${start}\\,${start + count - 1})`;
    const raw = await command(ffmpeg, ['-v', 'error', '-threads', '1', '-i', row.path, '-an', '-sn', '-vf', filter, '-fps_mode', 'passthrough', '-frames:v', String(count), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']);
    const frameBytes = metadata.width * metadata.height * 3;
    if (raw.length !== count * frameBytes) throw new Error(`Video decoder returned an incomplete clip: ${row.path}`);
    const frames = Array.from({ length: count }, (_, n) => ({ data: raw.subarray(n * frameBytes, (n + 1) * frameBytes), width: metadata.width, height: metadata.height, channels: 3, featureCount: frameBytes }));
    return { frames, path: row.path, fps: metadata.fps, ...(row.label === undefined ? {} : { label: row.label }) };
  };
  return { ...indexed({ name, split, homepage, classNames: Object.freeze([...classNames]) }, rows.length, async index => getClip(index, 0, (await info(index)).count)), frameCount: async index => (await info(index)).count, getClip };
}
export interface Ucf101Options extends LocalOptions<'train' | 'test'>, VideoTools { fold?: 1 | 2 | 3; annotationsDir?: string }
export async function ucf101(options: Ucf101Options): Promise<VideoDataset> {
  splitOf(options.split, ['train', 'test']); const fold = options.fold ?? 1; if (![1, 2, 3].includes(fold)) throw new Error('UCF101 fold must be 1–3');
  const root = rootPath('ucf101', options.root), annotations = inside(root, options.annotationsDir ?? 'ucfTrainTestlist');
  const classes = lines(await text(inside(annotations, 'classInd.txt'))).map(line => { const [id, name] = line.split(/\s+/); return { id: Number(id), name: name! } }).sort((a, b) => a.id - b.id);
  if (classes.some((entry, index) => entry.id !== index + 1)) throw new Error('Invalid UCF101 class indices');
  const names = classes.map(entry => entry.name), dir = await exists(inside(root, 'UCF-101')) ? inside(root, 'UCF-101') : root;
  const rows = lines(await text(inside(annotations, `${options.split === 'train' ? 'trainlist' : 'testlist'}${String(fold).padStart(2, '0')}.txt`))).map(line => {
    const [path, id] = line.split(/\s+/), label = id ? Number(id) - 1 : names.indexOf(path!.split('/')[0]!);
    if (!Number.isInteger(label) || label < 0 || label >= names.length) throw new Error('Invalid UCF101 label');
    return { path: inside(dir, path!), label };
  });
  return videoRecords('ucf101', options.split, 'https://www.crcv.ucf.edu/data/UCF101/UCF101.htm', names, rows, options);
}
export interface Hmdb51Options extends LocalOptions<'train' | 'test'>, VideoTools { fold?: 1 | 2 | 3; annotationsDir?: string; videosDir?: string }
export async function hmdb51(options: Hmdb51Options): Promise<VideoDataset> {
  splitOf(options.split, ['train', 'test']); const fold = options.fold ?? 1; if (![1, 2, 3].includes(fold)) throw new Error('HMDB51 fold must be 1–3');
  const root = rootPath('hmdb51', options.root), annotations = inside(root, options.annotationsDir ?? 'testTrainMulti_7030_splits'), dir = options.videosDir ? inside(root, options.videosDir) : await exists(inside(root, 'hmdb51_org')) ? inside(root, 'hmdb51_org') : root;
  const files = await listFiles(annotations, new RegExp(`_test_split${fold}\\.txt$`), false), classNames = files.map(file => basename(file).replace(`_test_split${fold}.txt`, '')).sort(), rows: VideoRecord[] = [];
  for (const name of classNames) for (const line of lines(await text(inside(annotations, `${name}_test_split${fold}.txt`)))) {
    const [file, tag] = line.split(/\s+/); if (Number(tag) === (options.split === 'train' ? 1 : 2)) rows.push({ path: inside(dir, `${name}/${file}`), label: classNames.indexOf(name) });
  }
  return videoRecords('hmdb51', options.split, 'https://serre.lab.brown.edu/', classNames, rows, options);
}
export interface KineticsOptions extends LocalOptions<'train' | 'val' | 'test'>, VideoTools { version?: 400 | 600 | 700; annotationsFile?: string }
export async function kinetics(options: KineticsOptions): Promise<VideoDataset> {
  splitOf(options.split, ['train', 'val', 'test']); const version = options.version ?? 400; if (![400, 600, 700].includes(version)) throw new Error('Unsupported Kinetics version');
  const root = rootPath(`kinetics${version}`, options.root), candidates = options.annotationsFile ? [options.annotationsFile] : [`annotations/${options.split}.csv`, `annotations/kinetics-${version}_${options.split}.csv`, `kinetics-${version}_${options.split}.csv`, `kinetics_${options.split}.csv`, `${options.split}.csv`];
  let file: string | undefined; for (const candidate of candidates) if (await exists(inside(root, candidate))) { file = inside(root, candidate); break }
  if (!file) throw new Error(`Prepare Kinetics CSV annotations under ${root}, or specify annotationsFile`);
  const records = csv(await text(file), { columns: true, skip_empty_lines: true }) as { label?: string; youtube_id: string; time_start: string; time_end: string }[];
  const names = [...new Set(records.map(record => record.label).filter((label): label is string => !!label))].sort(), rows: VideoRecord[] = [];
  for (const row of records) {
    const start = Number(row.time_start), end = Number(row.time_end);
    if (!row.youtube_id || !Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start) throw new Error('Invalid Kinetics CSV record');
    const filename = `${row.youtube_id}_${String(start).padStart(6, '0')}_${String(end).padStart(6, '0')}.mp4`;
    const folders = row.label ? [`${options.split}/${row.label}`, `${options.split}/${row.label.replaceAll(' ', '_')}`, row.label, row.label.replaceAll(' ', '_')] : [options.split, ''];
    let path: string | undefined; for (const folder of folders) { const candidate = inside(root, `${folder}/${filename}`); if (await exists(candidate)) { path = candidate; break } }
    if (!path) throw new Error(`Missing Kinetics clip: ${filename}`);
    rows.push({ path, ...(row.label ? { label: names.indexOf(row.label) } : {}) });
  }
  return videoRecords(`kinetics${version}`, options.split, 'https://github.com/google-deepmind/kinetics-i3d', names, rows, options);
}
/** A clip view over videos, using frame metadata without decoding every full video. */
export async function videoClips(options: { dataset: VideoDataset; framesPerClip: number; stride?: number }): Promise<Dataset<VideoSample, VideoMetadata>> {
  const { dataset, framesPerClip, stride = framesPerClip } = options;
  if (!Number.isSafeInteger(framesPerClip) || framesPerClip <= 0 || !Number.isSafeInteger(stride) || stride <= 0) throw new Error('Invalid clip size or stride');
  const entries: { video: number; first: number; count: number }[] = []; let count = 0;
  for (let video = 0; video < dataset.count; video++) {
    const frames = await dataset.frameCount(video), clips = Math.max(0, Math.floor((frames - framesPerClip) / stride) + 1);
    entries.push({ video, first: count, count: clips }); count += clips;
  }
  return indexed({ ...dataset.metadata, name: `${dataset.metadata.name}-clips` }, count, async index => {
    let low = 0, high = entries.length; while (low < high) { const mid = Math.floor((low + high) / 2), entry = entries[mid]!; if (index >= entry.first + entry.count) low = mid + 1; else high = mid }
    const entry = entries[low]!; return dataset.getClip(entry.video, (index - entry.first) * stride, framesPerClip);
  });
}
