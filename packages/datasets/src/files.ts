import { createHash, randomUUID } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { access, lstat, mkdir, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, resolve, relative, isAbsolute } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { x as extractTar } from 'tar';
import { Open } from 'unzipper';
import type { Dataset, DatasetMetadata } from './dataset.js';

export interface DatasetOptions<Split extends string = string> {
  readonly split: Split;
  /** This dataset's directory. Default .cache/blaten-datasets/<name>. */
  readonly root?: string;
  /** Download missing/corrupted files from the provider. Default true. */
  readonly download?: boolean;
}

export interface DownloadOptions<Split extends string = string> { split: Split; root?: string; download?: boolean }
export interface LocalOptions<Split extends string = string> { split: Split; root: string }
export interface DownloadFile { url: string; name?: string; md5?: string }
export const rootPath = (name: string, root?: string) => resolve(root ?? `.cache/blaten-datasets/${name}`);
export async function exists(path: string): Promise<boolean> {
  try { await access(path); return true } catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false; throw error }
}
export function inside(root: string, name: string): string {
  const path = resolve(root, name), rel = relative(root, path);
  if ((rel === '..' || rel.startsWith('../') || rel.startsWith('..\\')) || isAbsolute(rel) || name.includes('\0')) throw new Error(`Invalid dataset path: ${name}`);
  return path;
}
export function splitOf<S extends string>(split: S, choices: readonly S[]): S {
  if (!choices.includes(split)) throw new Error(`Unsupported split: ${split}. Expected ${choices.join(', ')}`);
  return split;
}
export async function text(path: string): Promise<string> { return readFile(path, 'utf8') }
export const lines = (value: string) => value.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
export async function listFiles(root: string, pattern: RegExp, recursive = true): Promise<string[]> {
  const result: string[] = [];
  for (const entry of (await readdir(root, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name, 'en'))) {
    const path = inside(root, entry.name);
    if (entry.isDirectory() && recursive) result.push(...await listFiles(path, pattern));
    else if (entry.isFile() && pattern.test(entry.name)) result.push(path);
  }
  return result;
}
export async function nestedRoot(root: string, candidates: readonly string[], required: string): Promise<string> {
  for (const prefix of ['', ...candidates]) { const path = inside(root, prefix); if (await exists(inside(path, required))) return path }
  throw new Error(`Dataset is missing ${required} under ${root}. Prepare the provider's files there or enable download.`);
}
export function indexed<Sample, Metadata extends DatasetMetadata>(metadata: Metadata, count: number, get: (index: number) => Promise<Sample>): Dataset<Sample, Metadata> {
  return { metadata: Object.freeze(metadata.classNames ? { ...metadata, classNames: Object.freeze([...metadata.classNames]) } : metadata) as Metadata, count, async get(index) {
    if (!Number.isSafeInteger(index) || index < 0 || index >= count) throw new RangeError(`Invalid sample index: ${index}`);
    return get(index);
  } };
}
async function hash(path: string): Promise<string> {
  const digest = createHash('md5'); for await (const chunk of createReadStream(path)) digest.update(chunk); return digest.digest('hex');
}
const downloads = new Map<string, Promise<string>>();
/** Stream provider files to disk; validate hashes before publishing. No dataset hosting. */
export async function downloadFile(root: string, file: DownloadFile, download = true): Promise<string> {
  const path = inside(root, file.name ?? new URL(file.url).pathname.split('/').at(-1)!);
  const key = JSON.stringify([path, file.url, file.md5, download]);
  const previous = downloads.get(key); if (previous) return previous;
  const request = (async () => {
    if (await exists(path) && (!file.md5 || await hash(path) === file.md5)) return path;
    if (!download) throw new Error(`Dataset file missing or invalid: ${path}`);
    await mkdir(dirname(path), { recursive: true });
    const response = await fetch(file.url);
    if (!response.ok || !response.body) throw new Error(`Dataset download failed (${response.status}): ${file.url}`);
    const temporary = `${path}.${randomUUID()}.tmp`;
    try {
      await pipeline(Readable.fromWeb(response.body as import('node:stream/web').ReadableStream<Uint8Array>), createWriteStream(temporary, { flags: 'wx' }));
      if (file.md5 && await hash(temporary) !== file.md5) throw new Error(`Dataset checksum mismatch: ${file.url}`);
      if (!(await stat(temporary)).size) throw new Error(`Empty dataset download: ${file.url}`);
      await rename(temporary, path); return path;
    } finally { await rm(temporary, { force: true }) }
  })();
  downloads.set(key, request); try { return await request } finally { downloads.delete(key) }
}
const extractions = new Map<string, Promise<void>>();
export async function extractArchive(path: string, root: string): Promise<void> {
  const key = JSON.stringify([path, root]); const previous = extractions.get(key); if (previous) return previous;
  const request = (async () => {
    await mkdir(root, { recursive: true });
    if (/\.zip$/i.test(path)) {
      const zip = await Open.file(path);
      for (const entry of zip.files) {
        const target = inside(root, entry.path);
        let ancestor = target;
        while (ancestor !== resolve(root)) {
          if (await exists(ancestor) && (await lstat(ancestor)).isSymbolicLink()) throw new Error(`Archive path traverses a symlink: ${entry.path}`);
          ancestor = dirname(ancestor);
        }
        if (entry.type === 'Directory') { await mkdir(target, { recursive: true }); continue }
        // Unix symlinks are not followed or created.
        if (((entry.externalFileAttributes >>> 16) & 0xf000) === 0xa000) throw new Error(`Archive symlink is not supported: ${entry.path}`);
        await mkdir(dirname(target), { recursive: true });
        const temporary = `${target}.${randomUUID()}.tmp`;
        try { await pipeline(entry.stream(), createWriteStream(temporary, { flags: 'wx' })); await rename(temporary, target) }
        finally { await rm(temporary, { force: true }) }
      }
    } else {
      await extractTar({ file: path, cwd: root, strict: true, filter: (name, entry) => {
        inside(root, name);
        if ('type' in entry && (entry.type === 'SymbolicLink' || entry.type === 'Link')) throw new Error(`Archive links are not supported: ${name}`);
        return true;
      } });
    }
  })();
  extractions.set(key, request); try { await request } finally { extractions.delete(key) }
}
export async function prepareArchive(root: string, file: DownloadFile, required: string, download = true): Promise<void> {
  const archivePath = inside(root, file.name ?? new URL(file.url).pathname.split('/').at(-1)!);
  const marker = `${archivePath}.extracted`;
  if (await exists(inside(root, required)) && (!await exists(archivePath) || await exists(marker))) return;
  const archive = await downloadFile(root, file, download);
  await extractArchive(archive, root);
  if (!await exists(inside(root, required))) throw new Error(`Archive does not contain ${required}: ${archive}`);
  await writeFile(marker, 'complete\n');
}
